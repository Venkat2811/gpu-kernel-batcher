#![deny(missing_docs)]
//! # Producer Process
//!
//! The `producer` binary simulates a client application or inference engine that
//! generates GPU work. It creates a shared memory (SHM) queue, populates it with
//! a sequence of `GemmCommand`s based on a specified workload profile, and then
//! signals the consumer to stop processing.
//!
//! This process has no CUDA dependencies and is responsible for creating the
//! different scenarios used to benchmark the consumer's performance.
//!
//! ## Workload Generation
//!
//! The producer can generate several kinds of workloads to simulate realistic
//! LLM inference patterns:
//! - **`prefill`**: A small number of large, compute-bound GEMMs.
//! - **`decode`**: A large number of small, launch-bound GEMMs.
//! - **`gemv-micro`**: A synthetic workload to purely measure launch overhead.

use anyhow::Result;
use clap::Parser;
use std::thread;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use cklo::{model::ModelConfig, ShmQueue};

#[derive(Parser, Debug)]
#[command(author, version, about = "Producer process for GEMM commands")]
/// Command-line arguments for the producer process.
struct Args {
    /// Shared memory queue name
    #[arg(long, default_value = "/cklo_queue")]
    shm_name: String,

    /// Queue capacity (must be power of 2)
    #[arg(long, default_value = "65536")]
    capacity: usize,

    /// Workload type
    #[arg(long, default_value = "micro")]
    workload: String,

    /// Number of commands to generate
    #[arg(long, default_value = "1000")]
    count: usize,

    /// Hidden dimension (for decode workload)
    #[arg(long, default_value = "4096")]
    hidden: u32,

    /// Number of layers (for decode/prefill workload)
    #[arg(long, default_value = "32")]
    layers: usize,

    /// Batch size
    #[arg(long, default_value = "1")]
    batch: usize,

    /// Sequence length (for prefill workload)
    #[arg(long, default_value = "512")]
    seq_len: usize,

    /// M dimension (for single/gemv-micro workload)
    #[arg(long, default_value = "64")]
    m: u32,

    /// N dimension (for single workload)
    #[arg(long, default_value = "256")]
    n: u32,

    /// K dimension (for single/gemv-micro workload)
    #[arg(long, default_value = "4096")]
    k: u32,

    /// Output file for statistics
    #[arg(long)]
    out: Option<String>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Keep SHM alive (for orchestrator)
    #[arg(long)]
    keep_shm: bool,

    /// Data type (fp32 or fp16)
    #[arg(long, default_value = "fp32")]
    dtype: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize tracing with custom format that includes process name
    let filter = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false) // Don't show module target
        .init();

    info!("[PRODUCER:INIT] Starting producer process");
    info!(
        "[PRODUCER:INIT] SHM: {}, Capacity: {}",
        args.shm_name, args.capacity
    );
    info!(
        "[PRODUCER:INIT] Workload: {}, Count: {}",
        args.workload, args.count
    );

    // Create shared memory queue
    let queue = ShmQueue::create(&args.shm_name, args.capacity)?;
    info!("[PRODUCER:INIT] Created SHM queue successfully");

    // Statistics
    let start_time = Instant::now();
    let mut total_enqueued = 0u64;
    let mut queue_full_count = 0u64;

    // Create model config for realistic workloads
    let model_config = ModelConfig::new(args.hidden, args.layers, args.batch);

    // Check if FP16 mode
    let use_fp16 = args.dtype.to_lowercase() == "fp16";

    // Generate commands based on workload type
    let mut commands = match args.workload.as_str() {
        "prefill" => {
            info!("[PRODUCER:GEN] Generating realistic prefill workload");
            info!("[PRODUCER:GEN] Type: Compute-bound (expects 0-3% speedup)");
            info!(
                "[PRODUCER:GEN] Model config: hidden={}, layers={}, batch={}, seq_len={}",
                model_config.hidden, model_config.layers, model_config.batch, args.seq_len
            );
            info!(
                "[PRODUCER:GEN] Total operations: {} (4 ops × {} layers)",
                4 * model_config.layers,
                model_config.layers
            );
            cklo::generate_prefill(&model_config, args.seq_len)
        }
        "decode" => {
            info!("[PRODUCER:GEN] Generating realistic decode workload");
            info!("[PRODUCER:GEN] Type: Autoregressive (expects 10-20% speedup)");
            // Calculate number of tokens to generate
            let ops_per_token = 4 * model_config.layers; // 4 ops per layer
            let token_count = args.count.div_ceil(ops_per_token);
            info!(
                "[PRODUCER:GEN] Model config: hidden={}, layers={}, batch={}",
                model_config.hidden, model_config.layers, model_config.batch
            );
            info!(
                "[PRODUCER:GEN] Token generation: {} tokens × {} ops/token = {} total ops",
                token_count,
                ops_per_token,
                token_count * ops_per_token
            );
            cklo::generate_decode(&model_config, token_count)
        }
        "gemv-micro" => {
            info!("[PRODUCER:GEN] Generating GEMV-micro workload");
            info!("[PRODUCER:GEN] Type: Launch-dominated (expects 70%+ speedup)");
            info!(
                "[PRODUCER:GEN] Configuration: M={}, K={}, N=1, Count={}",
                args.m, args.k, args.count
            );
            cklo::generate_gemv_micro(args.m, args.k, args.count)
        }
        "micro" => {
            info!("[PRODUCER:GEN] Generating micro workload");
            info!("[PRODUCER:GEN] Count: {}", args.count);
            cklo::generate_micro(args.count)
        }
        "single" => {
            info!("[PRODUCER:GEN] Generating single shape workload");
            info!(
                "[PRODUCER:GEN] M: {}, N: {}, K: {}, Count: {}",
                args.m, args.n, args.k, args.count
            );
            cklo::generate_single(args.m, args.n, args.k, args.count)
        }
        _ => {
            warn!(
                "[PRODUCER:GEN] Unknown workload type '{}', using micro",
                args.workload
            );
            cklo::generate_micro(args.count)
        }
    };

    info!(
        "[PRODUCER:GEN] Workload generation complete: {} commands",
        commands.len()
    );

    // If FP16 mode, mark all commands
    if use_fp16 {
        info!("[PRODUCER:GEN] Precision: FP16 (half precision)");
        for cmd in &mut commands {
            *cmd = cmd.with_fp16();
        }
    } else {
        info!("[PRODUCER:GEN] Precision: FP32 (single precision)");
    }

    // Push all commands to queue
    for cmd in commands {
        loop {
            match queue.push(cmd) {
                Ok(_) => {
                    total_enqueued += 1;
                    if total_enqueued % 10000 == 0 {
                        debug!("Enqueued {} commands", total_enqueued);
                    }
                    break;
                }
                Err(_) => {
                    queue_full_count += 1;
                    thread::sleep(Duration::from_micros(10));
                }
            }
        }
    }

    // Set stop flag
    queue.set_stop();
    info!("[PRODUCER:SEND] Set stop flag");

    // Keep queue alive if orchestrated
    if args.keep_shm {
        info!("[PRODUCER:SEND] Keeping SHM queue alive for consumer");
        thread::sleep(Duration::from_secs(5)); // Give consumer time to finish
    }

    let elapsed = start_time.elapsed();

    // Print statistics
    info!("[PRODUCER:DONE] Producer finished");
    info!("[PRODUCER:DONE] Total enqueued: {}", total_enqueued);
    info!("[PRODUCER:DONE] Queue full events: {}", queue_full_count);
    info!(
        "[PRODUCER:DONE] Wall time: {:.2} ms",
        elapsed.as_secs_f64() * 1000.0
    );
    info!(
        "[PRODUCER:DONE] Throughput: {:.2} commands/sec",
        total_enqueued as f64 / elapsed.as_secs_f64()
    );

    // Save statistics if requested
    if let Some(out_file) = args.out {
        save_stats(&out_file, total_enqueued, queue_full_count, elapsed)?;
    }

    Ok(())
}

/// Saves the producer's run statistics to a JSON file.
fn save_stats(path: &str, enqueued: u64, full_count: u64, elapsed: Duration) -> Result<()> {
    use serde_json::json;
    use std::fs::File;
    use std::io::Write;

    let stats = json!({
        "producer": {
            "total_enqueued": enqueued,
            "queue_full_count": full_count,
            "wall_time_ms": elapsed.as_secs_f64() * 1000.0,
            "throughput_per_sec": enqueued as f64 / elapsed.as_secs_f64(),
        }
    });

    let mut file = File::create(path)?;
    writeln!(file, "{}", serde_json::to_string_pretty(&stats)?)?;
    info!("[PRODUCER:DONE] Statistics saved to {}", path);

    Ok(())
}
