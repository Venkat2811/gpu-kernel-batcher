#![deny(missing_docs)]
//! # Consumer Process
//!
//! The `consumer` binary acts as the dedicated GPU worker in the system. It connects
//! to a shared memory (SHM) queue created by a producer, continuously polls for
//! `GemmCommand`s, and executes them on a specified GPU device.
//!
//! ## Execution Modes
//!
//! - **`none` (Single Mode)**: Fetches and executes one command at a time. This serves
//!   as the baseline for performance comparison, representing the naive approach with
//!   high kernel launch overhead.
//! - **`shape-batched` (Batched Mode)**: The optimized approach. It reads commands and
//!   groups them into buckets based on their `ShapeKey`. Buckets are flushed to the
//!   GPU using a single strided-batched kernel launch when they are full or when a
//!   short idle period is detected. This significantly reduces launch overhead.

use anyhow::Result;
use clap::Parser;
use std::collections::HashMap;
use std::thread;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use cklo::{model::ModelConfig, GemmCommand, GemmExecutor, ShapeKey, ShmQueue};

#[derive(Parser, Debug)]
#[command(author, version, about = "Consumer process for GEMM execution")]
/// Command-line arguments for the consumer process.
struct Args {
    /// Shared memory queue name
    #[arg(long, default_value = "/cklo_queue")]
    shm_name: String,

    /// Queue capacity (for opening existing queue)
    #[arg(long, default_value = "65536")]
    capacity: usize,

    /// GPU device ID
    #[arg(long, default_value = "0")]
    gpu_id: usize,

    /// Execution mode
    #[arg(long, default_value = "none")]
    mode: String,

    /// Maximum batch size for bucketing
    #[arg(long, default_value = "1024")]
    max_batch: usize,

    /// Idle timeout in microseconds
    #[arg(long, default_value = "150")]
    idle_us: u64,

    /// Enable CUDA events (when CUDA is added)
    #[arg(long)]
    events: bool,

    /// Output file for results
    #[arg(long)]
    out: Option<String>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Ready signal file (for orchestrator)
    #[arg(long)]
    ready_file: Option<String>,

    /// Done signal file (for orchestrator)
    #[arg(long)]
    done_file: Option<String>,

    /// Number of warmup operations
    #[arg(long, default_value = "3")]
    warmup_ops: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize tracing with custom format that includes process name
    let filter = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false) // Don't show module target
        .init();

    info!("[CONSUMER:INIT] Starting consumer process");
    info!("[CONSUMER:INIT] SHM: {}", args.shm_name);
    info!("[CONSUMER:INIT] Mode: {}, GPU: {}", args.mode, args.gpu_id);

    // Open shared memory queue
    let queue = ShmQueue::open(&args.shm_name)?;
    info!(
        "[CONSUMER:INIT] Opened SHM queue with capacity {}",
        queue.capacity()
    );

    // Signal ready if orchestrator is managing us
    if let Some(ref ready_file) = args.ready_file {
        std::fs::File::create(ready_file)?;
        info!("[CONSUMER:INIT] Signaled ready via {}", ready_file);
    }

    // Run based on mode
    let start_time = Instant::now();
    let stats = match args.mode.as_str() {
        "none" => run_single_mode(&queue, &args)?,
        "shape-batched" => run_batched_mode(&queue, &args)?,
        _ => {
            warn!("[CONSUMER:INIT] Unknown mode '{}', using none", args.mode);
            run_single_mode(&queue, &args)?
        }
    };

    let elapsed = start_time.elapsed();

    // Print results
    info!("[CONSUMER:DONE] Consumer finished");
    info!("[CONSUMER:DONE] Total dequeued: {}", stats.total_dequeued);
    info!("[CONSUMER:DONE] Total launches: {}", stats.total_launches);
    info!(
        "[CONSUMER:DONE] Queue empty events: {}",
        stats.queue_empty_count
    );
    info!(
        "[CONSUMER:DONE] Wall time: {:.2} ms",
        elapsed.as_secs_f64() * 1000.0
    );
    info!(
        "[CONSUMER:DONE] Throughput: {:.2} commands/sec",
        stats.total_dequeued as f64 / elapsed.as_secs_f64()
    );

    if stats.total_launches > 0 {
        let reduction = 100.0 * (1.0 - (stats.total_launches as f64 / stats.total_dequeued as f64));
        info!("[CONSUMER:DONE] Launch reduction: {:.1}%", reduction);
    }

    // Save results if requested
    if let Some(ref out_file) = args.out {
        save_results(out_file, &stats, elapsed)?;
    }

    // Signal done if orchestrator is managing us
    if let Some(ref done_file) = args.done_file {
        std::fs::File::create(done_file)?;
        info!("[CONSUMER:DONE] Signaled done via {}", done_file);
    }

    Ok(())
}

/// Holds statistics about the consumer's execution run.
#[derive(Debug, Default)]
struct ConsumerStats {
    total_dequeued: u64,
    total_launches: u64,
    queue_empty_count: u64,
    shape_counts: HashMap<ShapeKey, usize>,
    gpu_time_ms: f64,
    wall_time_ms: f64,
}

/// Runs the consumer in **single mode**, executing one command at a time.
fn run_single_mode(queue: &ShmQueue, args: &Args) -> Result<ConsumerStats> {
    let mut stats = ConsumerStats::default();

    info!("[CONSUMER:MODE] Running in single mode (one launch per command)");

    // Initialize CUDA executor with timing
    let mut executor = GemmExecutor::new_with_timing(args.gpu_id, args.events)?;
    info!(
        "[CONSUMER:CUDA] Initialized CUDA device {} (timing: {})",
        args.gpu_id, args.events
    );

    // Warmup phase - run a few operations to initialize CUDA
    let warmup_start = Instant::now();

    // Pre-allocate buffers for common decode shapes (batch=1, hidden=1024)
    // This matches the benchmark configuration
    let warmup_config = ModelConfig::new(1024, 8, 1);
    let decode_shapes = cklo::get_decode_shapes(&warmup_config);

    info!(
        "[CONSUMER:WARMUP] Starting warmup phase ({} ops)",
        args.warmup_ops + decode_shapes.len()
    );

    for shape in &decode_shapes {
        let cmd = GemmCommand::new(shape.m, shape.n, shape.k);
        executor.execute_single(&cmd)?;
    }

    // General warmup with small ops
    for _ in 0..args.warmup_ops {
        let warmup_cmd = GemmCommand::new(64, 64, 64);
        executor.execute_single(&warmup_cmd)?;
    }
    executor.synchronize()?;
    executor.reset_stats(); // Don't count warmup in results
    let warmup_elapsed = warmup_start.elapsed();
    info!(
        "[CONSUMER:WARMUP] Warmup complete ({:.1} ms)",
        warmup_elapsed.as_secs_f64() * 1000.0
    );

    // Start GPU timing before processing
    executor.record_start()?;

    loop {
        // Check stop condition
        if queue.should_stop() && queue.is_empty() {
            info!("[CONSUMER:PROCESS] Stop flag set and queue empty, exiting");
            break;
        }

        // Try to pop command
        match queue.pop() {
            Ok(cmd) => {
                stats.total_dequeued += 1;

                // Track shape
                let shape = cmd.shape_key();
                *stats.shape_counts.entry(shape).or_insert(0) += 1;

                // Execute real GEMM on GPU
                executor.execute_single(&cmd)?;

                if stats.total_dequeued % 10000 == 0 {
                    debug!(
                        "[CONSUMER:PROCESS] Processed {} commands",
                        stats.total_dequeued
                    );
                }
            }
            Err(_) => {
                stats.queue_empty_count += 1;
                thread::sleep(Duration::from_micros(10));
            }
        }
    }

    // Stop GPU timing after processing
    executor.record_stop()?;

    // Synchronize GPU before finishing
    executor.synchronize()?;

    // Compute GPU time from events
    executor.compute_gpu_time()?;

    // Get statistics from executor
    let (kernels, ops, _) = executor.get_stats();
    stats.total_launches = kernels;
    stats.gpu_time_ms = executor.get_gpu_time_ms();
    stats.wall_time_ms = executor.wall_time_ms;

    info!(
        "[CONSUMER:STATS] GPU executed {} kernels for {} operations",
        kernels, ops
    );
    if executor.timing_enabled {
        info!(
            "[CONSUMER:STATS] GPU time: {:.2} ms, Wall time: {:.2} ms",
            stats.gpu_time_ms, stats.wall_time_ms
        );
    }

    Ok(stats)
}

/// Runs the consumer in **batched mode**, grouping commands by shape for efficient execution.
fn run_batched_mode(queue: &ShmQueue, args: &Args) -> Result<ConsumerStats> {
    let mut stats = ConsumerStats::default();

    info!("[CONSUMER:MODE] Running in shape-batched mode");
    info!(
        "[CONSUMER:MODE] Max batch: {}, Idle timeout: {} μs",
        args.max_batch, args.idle_us
    );

    // Initialize CUDA executor with timing
    let mut executor = GemmExecutor::new_with_timing(args.gpu_id, args.events)?;
    info!(
        "[CONSUMER:CUDA] Initialized CUDA device {} (timing: {})",
        args.gpu_id, args.events
    );

    // Warmup phase - run a few operations to initialize CUDA
    let warmup_start = Instant::now();

    // Pre-allocate buffers for common decode shapes (batch=1, hidden=1024)
    // This matches the benchmark configuration
    let warmup_config = ModelConfig::new(1024, 8, 1);
    let decode_shapes = cklo::get_decode_shapes(&warmup_config);

    info!(
        "[CONSUMER:WARMUP] Starting warmup phase ({} ops)",
        args.warmup_ops + decode_shapes.len()
    );

    for shape in &decode_shapes {
        let cmd = GemmCommand::new(shape.m, shape.n, shape.k);
        executor.execute_single(&cmd)?;
    }

    // General warmup with small ops
    for _ in 0..args.warmup_ops {
        let warmup_cmd = GemmCommand::new(64, 64, 64);
        executor.execute_single(&warmup_cmd)?;
    }
    executor.synchronize()?;
    executor.reset_stats(); // Don't count warmup in results
    let warmup_elapsed = warmup_start.elapsed();
    info!(
        "[CONSUMER:WARMUP] Warmup complete ({:.1} ms)",
        warmup_elapsed.as_secs_f64() * 1000.0
    );

    // Start GPU timing before processing
    executor.record_start()?;

    let mut buckets: HashMap<ShapeKey, Vec<GemmCommand>> = HashMap::new();
    let mut last_activity = Instant::now();

    loop {
        // Check stop condition
        if queue.should_stop() && queue.is_empty() {
            info!("[CONSUMER:PROCESS] Stop flag set and queue empty, flushing and exiting");
            flush_all_buckets(&mut buckets, &mut stats, &mut executor)?;
            break;
        }

        // Try to pop command
        match queue.pop() {
            Ok(cmd) => {
                stats.total_dequeued += 1;
                last_activity = Instant::now();

                // Add to bucket
                let shape = cmd.shape_key();
                buckets.entry(shape).or_default().push(cmd);

                // Check if any bucket is full and needs flushing
                let full_shape = buckets
                    .iter()
                    .find(|(_, cmds)| cmds.len() >= args.max_batch)
                    .map(|(shape, _)| *shape);

                if let Some(shape_to_flush) = full_shape {
                    if let Some(mut cmds) = buckets.remove(&shape_to_flush) {
                        while !cmds.is_empty() {
                            let batch_size = cmds.len().min(args.max_batch);
                            let batch: Vec<_> = cmds.drain(..batch_size).collect();
                            executor.execute_batched(&batch)?;
                            *stats.shape_counts.entry(shape_to_flush).or_insert(0) += batch.len();
                        }
                    }
                }

                if stats.total_dequeued % 10000 == 0 {
                    debug!(
                        "[CONSUMER:PROCESS] Processed {} commands, {} launches",
                        stats.total_dequeued, stats.total_launches
                    );
                }
            }
            Err(_) => {
                stats.queue_empty_count += 1;

                // Check idle timeout for flush - flush all buckets when idle
                if !buckets.is_empty() && last_activity.elapsed().as_micros() > args.idle_us as u128
                {
                    for (shape, cmds) in buckets.iter_mut() {
                        if !cmds.is_empty() {
                            while !cmds.is_empty() {
                                let batch_size = cmds.len().min(args.max_batch);
                                let batch: Vec<_> = cmds.drain(..batch_size).collect();
                                executor.execute_batched(&batch)?;
                                *stats.shape_counts.entry(*shape).or_insert(0) += batch.len();
                            }
                        }
                    }
                    buckets.clear();
                    last_activity = Instant::now();
                }

                thread::sleep(Duration::from_micros(10));
            }
        }
    }

    // Stop GPU timing after processing
    executor.record_stop()?;

    // Synchronize GPU before finishing
    executor.synchronize()?;

    // Compute GPU time from events
    executor.compute_gpu_time()?;

    // Get statistics from executor
    let (kernels, ops, _) = executor.get_stats();
    stats.total_launches = kernels;
    stats.gpu_time_ms = executor.get_gpu_time_ms();
    stats.wall_time_ms = executor.wall_time_ms;

    info!(
        "[CONSUMER:STATS] GPU executed {} kernels for {} operations",
        kernels, ops
    );
    if executor.timing_enabled {
        info!(
            "[CONSUMER:STATS] GPU time: {:.2} ms, Wall time: {:.2} ms",
            stats.gpu_time_ms, stats.wall_time_ms
        );
    }

    Ok(stats)
}

fn flush_all_buckets(
    buckets: &mut HashMap<ShapeKey, Vec<GemmCommand>>,
    stats: &mut ConsumerStats,
    executor: &mut GemmExecutor,
) -> Result<()> {
    for (shape, cmds) in buckets.drain() {
        if !cmds.is_empty() {
            // Execute batched GEMM on GPU
            executor.execute_batched(&cmds)?;

            *stats.shape_counts.entry(shape).or_insert(0) += cmds.len();
            debug!(
                "[CONSUMER:FLUSH] Final flush: batch of {} for shape {:?}",
                cmds.len(),
                shape
            );
        }
    }
    Ok(())
}

fn save_results(path: &str, stats: &ConsumerStats, elapsed: Duration) -> Result<()> {
    use serde_json::json;
    use std::fs::File;
    use std::io::Write;

    let reduction = if stats.total_launches > 0 {
        100.0 * (1.0 - (stats.total_launches as f64 / stats.total_dequeued as f64))
    } else {
        0.0
    };

    let results = json!({
        "consumer": {
            "total_dequeued": stats.total_dequeued,
            "total_launches": stats.total_launches,
            "queue_empty_count": stats.queue_empty_count,
            "wall_time_ms": elapsed.as_secs_f64() * 1000.0,
            "gpu_time_ms": stats.gpu_time_ms,
            "executor_wall_time_ms": stats.wall_time_ms,
            "throughput_per_sec": stats.total_dequeued as f64 / elapsed.as_secs_f64(),
            "launch_reduction_percent": reduction,
            "unique_shapes": stats.shape_counts.len(),
        }
    });

    let mut file = File::create(path)?;
    writeln!(file, "{}", serde_json::to_string_pretty(&results)?)?;
    info!("[CONSUMER:DONE] Saved results to {}", path);

    Ok(())
}
