// Orchestrator - Manages producer and consumer lifecycle
// Eliminates time-based coordination and race conditions

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;
use std::fs;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{debug, error, info};

#[derive(Parser, Debug, Serialize)]
#[command(
    author,
    version,
    about = "Orchestrator for producer-consumer benchmark"
)]
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

    /// Consumer mode (none, shape-batched)
    #[arg(long, default_value = "shape-batched")]
    mode: String,

    /// Max batch size for consumer
    #[arg(long, default_value = "512")]
    max_batch: usize,

    /// Idle timeout in microseconds
    #[arg(long, default_value = "150")]
    idle_us: u64,

    /// GPU device ID
    #[arg(long, default_value = "0")]
    gpu_id: u32,

    /// Results directory
    #[arg(long, default_value = "results/orchestrated")]
    out_dir: String,

    /// Enable CUDA events timing
    #[arg(long)]
    cuda_events: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Hidden dimension (for decode workload)
    #[arg(long, default_value = "8192")]
    hidden: u32,

    /// Number of layers (for decode workload)
    #[arg(long, default_value = "32")]
    layers: usize,

    /// Batch size
    #[arg(long, default_value = "1")]
    batch: usize,

    /// M dimension (for gemv-micro)
    #[arg(long, default_value = "64")]
    m: u32,

    /// K dimension (for gemv-micro)
    #[arg(long, default_value = "4096")]
    k: u32,

    /// Sequence length (for prefill)
    #[arg(long, default_value = "2048")]
    seq_len: usize,

    /// Data type (fp32 or fp16)
    #[arg(long, default_value = "fp32")]
    dtype: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize tracing with custom format
    let filter = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false) // Don't show module target
        .init();

    info!("[ORCHESTRATOR:INIT] Starting orchestration");
    info!(
        "[ORCHESTRATOR:INIT] Workload: {}, Count: {}, Mode: {}",
        args.workload, args.count, args.mode
    );
    info!(
        "[ORCHESTRATOR:INIT] Data type: {}",
        args.dtype.to_uppercase()
    );

    // Create results directory
    fs::create_dir_all(&args.out_dir).context("Failed to create results directory")?;

    // Save configuration to config.json
    let config_path = Path::new(&args.out_dir).join("config.json");
    let config_json = serde_json::to_string_pretty(&args).context("Failed to serialize config")?;
    fs::write(&config_path, config_json)
        .with_context(|| format!("Failed to write config to {:?}", config_path))?;

    // Generate timestamp-based ready file
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis();
    let ready_file = format!("/tmp/cklo_ready_{}", timestamp);
    let done_file = format!("/tmp/cklo_done_{}", timestamp);

    // Start timing
    let start_time = Instant::now();

    // Start producer first (it creates the queue)
    let mut producer = start_producer(&args)?;
    info!(
        "[ORCHESTRATOR:START] Started producer (PID: {:?})",
        producer.id()
    );

    // Give producer a moment to create the queue
    thread::sleep(Duration::from_millis(50));

    // Now start consumer
    let mut consumer = start_consumer(&args, &ready_file, &done_file)?;
    info!(
        "[ORCHESTRATOR:START] Started consumer (PID: {:?})",
        consumer.id()
    );

    // Wait for consumer to be ready
    wait_for_file(&ready_file, Duration::from_secs(5))
        .context("Consumer failed to signal ready")?;
    info!("[ORCHESTRATOR:SYNC] Consumer ready and processing");

    // Wait for producer to complete
    let producer_status = producer.wait().context("Failed waiting for producer")?;

    if !producer_status.success() {
        error!(
            "[ORCHESTRATOR:ERROR] Producer failed with status: {:?}",
            producer_status
        );
    } else {
        info!("[ORCHESTRATOR:SYNC] Producer completed successfully");
    }

    // Wait for consumer to complete (it should detect stop flag)
    wait_for_file(&done_file, Duration::from_secs(10))
        .context("Consumer failed to signal completion")?;

    let consumer_status = consumer.wait().context("Failed waiting for consumer")?;

    if !consumer_status.success() {
        error!(
            "[ORCHESTRATOR:ERROR] Consumer failed with status: {:?}",
            consumer_status
        );
    } else {
        info!("[ORCHESTRATOR:SYNC] Consumer completed successfully");
    }

    // Clean up signal files
    let _ = fs::remove_file(&ready_file);
    let _ = fs::remove_file(&done_file);

    let elapsed = start_time.elapsed();
    info!(
        "[ORCHESTRATOR:DONE] Orchestrator complete. Total time: {:.2} ms",
        elapsed.as_secs_f64() * 1000.0
    );

    // Aggregate results
    aggregate_results(&args.out_dir)?;

    Ok(())
}

fn start_consumer(args: &Args, ready_file: &str, done_file: &str) -> Result<Child> {
    let mut cmd = Command::new("./target/release/consumer");

    cmd.arg("--shm-name")
        .arg(&args.shm_name)
        .arg("--mode")
        .arg(&args.mode)
        .arg("--max-batch")
        .arg(args.max_batch.to_string())
        .arg("--idle-us")
        .arg(args.idle_us.to_string())
        .arg("--gpu-id")
        .arg(args.gpu_id.to_string())
        .arg("--out")
        .arg(format!("{}/consumer.json", args.out_dir))
        .arg("--ready-file")
        .arg(ready_file)
        .arg("--done-file")
        .arg(done_file);

    if args.cuda_events {
        cmd.arg("--events");
    }

    cmd.env("CUDA_VISIBLE_DEVICES", args.gpu_id.to_string())
        .env("CUDA_DEVICE_MAX_CONNECTIONS", "1")
        .stdout(Stdio::inherit()) // Show output for debugging
        .stderr(Stdio::inherit()) // Show errors for debugging
        .spawn()
        .context("Failed to start consumer")
}

fn start_producer(args: &Args) -> Result<Child> {
    let mut cmd = Command::new("./target/release/producer");

    cmd.arg("--shm-name")
        .arg(&args.shm_name)
        .arg("--capacity")
        .arg(args.capacity.to_string())
        .arg("--workload")
        .arg(&args.workload)
        .arg("--count")
        .arg(args.count.to_string())
        .arg("--dtype")
        .arg(&args.dtype)
        .arg("--out")
        .arg(format!("{}/producer.json", args.out_dir))
        .arg("--keep-shm"); // Tell producer to keep SHM alive

    // Add workload-specific args
    match args.workload.as_str() {
        "decode" => {
            cmd.arg("--hidden")
                .arg(args.hidden.to_string())
                .arg("--layers")
                .arg(args.layers.to_string())
                .arg("--batch")
                .arg(args.batch.to_string());
        }
        "prefill" => {
            cmd.arg("--hidden")
                .arg(args.hidden.to_string())
                .arg("--layers")
                .arg(args.layers.to_string())
                .arg("--batch")
                .arg(args.batch.to_string())
                .arg("--seq-len")
                .arg(args.seq_len.to_string());
        }
        "gemv-micro" => {
            cmd.arg("--m")
                .arg(args.m.to_string())
                .arg("--k")
                .arg(args.k.to_string());
        }
        _ => {}
    }

    cmd.stdout(Stdio::inherit()) // Show output for debugging
        .stderr(Stdio::inherit()) // Show errors for debugging
        .spawn()
        .context("Failed to start producer")
}

fn wait_for_file(path: &str, timeout: Duration) -> Result<()> {
    let start = Instant::now();

    loop {
        if Path::new(path).exists() {
            debug!("[ORCHESTRATOR:SYNC] Found signal file: {}", path);
            return Ok(());
        }

        if start.elapsed() > timeout {
            return Err(anyhow::anyhow!("Timeout waiting for file: {}", path));
        }

        thread::sleep(Duration::from_millis(10));
    }
}

fn aggregate_results(out_dir: &str) -> Result<()> {
    let producer_file = format!("{}/producer.json", out_dir);
    let consumer_file = format!("{}/consumer.json", out_dir);

    if Path::new(&producer_file).exists() && Path::new(&consumer_file).exists() {
        info!("[ORCHESTRATOR:RESULTS] Results aggregated in: {}", out_dir);

        // Read and display key metrics
        if let Ok(consumer_data) = fs::read_to_string(&consumer_file) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&consumer_data) {
                if let Some(consumer) = json.get("consumer") {
                    if let Some(reduction) = consumer.get("launch_reduction_percent") {
                        info!("[ORCHESTRATOR:RESULTS] Launch reduction: {}%", reduction);
                    }
                    if let Some(launches) = consumer.get("total_launches") {
                        info!("[ORCHESTRATOR:RESULTS] Total launches: {}", launches);
                    }
                }
            }
        }
    }

    Ok(())
}
