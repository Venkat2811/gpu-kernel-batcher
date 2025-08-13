//! # GPU GEMM Execution Engine
//!
//! This module provides the `GemmExecutor`, the core GPU execution engine for this
//! project. It is responsible for initializing the CUDA device, managing GPU memory
//! buffers, and executing `GemmCommand`s using the cuBLASLt library.
//!
//! It supports two execution modes:
//! 1.  **Immediate (`execute_single`)**: Launches one CUDA kernel for every command received.
//!     This is inefficient for many small GEMMs due to high kernel launch overhead.
//! 2.  **Batched (`execute_batched`)**: Collects commands of the same shape and executes
//!     them in a single, highly efficient `cublasGemmStridedBatched` call. This is
//!     the central optimization demonstrated by the project.

use anyhow::{Context, Result};
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::result::event;
use cudarc::driver::{sys, CudaDevice, CudaSlice, CudaStream};
use half::f16;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::model::{GemmCommand, ShapeKey};

/// Manages a CUDA device and provides methods for executing GEMM operations.
///
/// An executor holds the CUDA device context, a cuBLAS handle, and a cache of
/// pre-allocated GPU memory buffers (`GemmBuffers`) for various tensor shapes.
/// This prevents repeated, expensive device memory allocations.
pub struct GemmExecutor {
    device: Arc<CudaDevice>,
    stream: CudaStream,
    blas: Arc<CudaBlas>,

    // Pre-allocated buffers per shape
    buffers: HashMap<ShapeKey, GemmBuffers>,

    // Timing configuration and events
    /// Whether CUDA event timing is enabled
    pub timing_enabled: bool,
    start_event: Option<sys::CUevent>,
    stop_event: Option<sys::CUevent>,

    // Statistics
    /// Total number of CUDA kernels launched
    pub total_kernels: u64,
    /// Total number of GEMM operations executed
    pub total_operations: u64,
    /// Wall-clock time measured with CPU timers (milliseconds)
    pub wall_time_ms: f64,
    /// GPU execution time measured with CUDA events (milliseconds)
    pub gpu_time_ms: f64,
}

/// Pre-allocated GPU buffers for a specific shape
enum GemmBuffers {
    Fp32 {
        // Single GEMM buffers (reused for batched via striding)
        a: CudaSlice<f32>,
        b: CudaSlice<f32>,
        c: CudaSlice<f32>,
        // Cached batch buffers (allocated on first use, reused)
        batch_buffers: Option<BatchBuffersFp32>,
    },
    Fp16 {
        // Single GEMM buffers (reused for batched via striding)
        a: CudaSlice<f16>,
        b: CudaSlice<f16>,
        c: CudaSlice<f16>,
        // Cached batch buffers (allocated on first use, reused)
        batch_buffers: Option<BatchBuffersFp16>,
    },
}

struct BatchBuffersFp32 {
    a: CudaSlice<f32>,
    b: CudaSlice<f32>,
    c: CudaSlice<f32>,
    capacity: usize,
}

struct BatchBuffersFp16 {
    a: CudaSlice<f16>,
    b: CudaSlice<f16>,
    c: CudaSlice<f16>,
    capacity: usize,
}

impl GemmExecutor {
    /// Creates a new `GemmExecutor` on the specified GPU device.
    pub fn new(gpu_id: usize) -> Result<Self> {
        Self::new_with_timing(gpu_id, false)
    }

    /// Create new GEMM executor with timing
    pub fn new_with_timing(gpu_id: usize, enable_timing: bool) -> Result<Self> {
        // Initialize CUDA device
        let device = CudaDevice::new(gpu_id).context("Failed to initialize CUDA device")?;

        // Create stream for async execution
        let stream = device
            .fork_default_stream()
            .context("Failed to create CUDA stream")?;

        // Create cuBLAS handle
        let blas = CudaBlas::new(device.clone()).context("Failed to create cuBLAS handle")?;

        // Create CUDA events for timing if enabled
        let (start_event, stop_event) = if enable_timing {
            let start = event::create(sys::CUevent_flags_enum::CU_EVENT_DEFAULT)
                .context("Failed to create start event")?;
            let stop = event::create(sys::CUevent_flags_enum::CU_EVENT_DEFAULT)
                .context("Failed to create stop event")?;
            (Some(start), Some(stop))
        } else {
            (None, None)
        };

        tracing::info!(
            "Initialized CUDA device {} (timing: {})",
            gpu_id,
            enable_timing
        );

        Ok(Self {
            device,
            stream,
            blas: Arc::new(blas),
            buffers: HashMap::new(),
            timing_enabled: enable_timing,
            start_event,
            stop_event,
            total_kernels: 0,
            total_operations: 0,
            wall_time_ms: 0.0,
            gpu_time_ms: 0.0,
        })
    }

    /// Executes a single `GemmCommand` by launching a dedicated CUDA kernel.
    ///
    /// This method is straightforward but inefficient for workloads with many small
    /// operations due to the overhead of launching each kernel individually.
    pub fn execute_single(&mut self, cmd: &GemmCommand) -> Result<()> {
        let shape = cmd.shape_key();

        // Setup dimensions
        let m = cmd.m as usize;
        let n = cmd.n as usize;
        let k = cmd.k as usize;

        // Get or allocate buffers for this shape
        self.get_or_create_buffers(shape, 1)?;

        // Determine transpose operations
        let transa = if cmd.trans_a != 0 {
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T
        } else {
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N
        };

        let transb = if cmd.trans_b != 0 {
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T
        } else {
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N
        };

        // Compute leading dimensions
        let lda = if cmd.trans_a != 0 { k } else { m } as i32;
        let ldb = if cmd.trans_b != 0 { n } else { k } as i32;
        let ldc = m as i32;

        // Execute GEMM on GPU
        // Track wall time
        let wall_start = Instant::now();

        // Get buffers as mutable to allow GEMM to write to C
        let buffers = self.buffers.get_mut(&shape).unwrap();
        let blas = self.blas.clone();

        if cmd.is_fp16() {
            // FP16 GEMM
            match buffers {
                GemmBuffers::Fp16 { a, b, c, .. } => {
                    let cfg = GemmConfig {
                        transa,
                        transb,
                        m: m as i32,
                        n: n as i32,
                        k: k as i32,
                        alpha: f16::from_f32(cmd.alpha),
                        beta: f16::from_f32(cmd.beta),
                        lda,
                        ldb,
                        ldc,
                    };
                    unsafe {
                        blas.gemm(cfg, a, b, c)?;
                    }
                }
                _ => return Err(anyhow::anyhow!("Buffer type mismatch for FP16 GEMM")),
            }
        } else {
            // FP32 GEMM
            match buffers {
                GemmBuffers::Fp32 { a, b, c, .. } => {
                    let cfg = GemmConfig {
                        transa,
                        transb,
                        m: m as i32,
                        n: n as i32,
                        k: k as i32,
                        alpha: cmd.alpha,
                        beta: cmd.beta,
                        lda,
                        ldb,
                        ldc,
                    };
                    unsafe {
                        blas.gemm(cfg, a, b, c)?;
                    }
                }
                _ => return Err(anyhow::anyhow!("Buffer type mismatch for FP32 GEMM")),
            }
        }

        // Track wall time
        let wall_elapsed = wall_start.elapsed();
        self.wall_time_ms += wall_elapsed.as_secs_f64() * 1000.0;

        self.total_kernels += 1;
        self.total_operations += 1;

        Ok(())
    }

    /// Executes a slice of `GemmCommand`s in a single, batched kernel launch.
    ///
    /// This is the core optimization. It uses `cublasGemmStridedBatched` to perform
    /// all GEMMs in the slice with one API call, dramatically reducing CPU and driver
    /// overhead. All commands in the slice **must** have the same `ShapeKey`.
    pub fn execute_batched(&mut self, cmds: &[GemmCommand]) -> Result<()> {
        if cmds.is_empty() {
            return Ok(());
        }

        let batch_size = cmds.len();
        let cmd = &cmds[0]; // All commands have same shape
        let shape = cmd.shape_key();

        // Setup dimensions (same for all in batch)
        let m = cmd.m as usize;
        let n = cmd.n as usize;
        let k = cmd.k as usize;

        // Determine transpose operations
        let transa = if cmd.trans_a != 0 {
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T
        } else {
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N
        };

        let transb = if cmd.trans_b != 0 {
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T
        } else {
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N
        };

        // Compute leading dimensions
        let lda = if cmd.trans_a != 0 { k } else { m } as i32;
        let ldb = if cmd.trans_b != 0 { n } else { k } as i32;
        let ldc = m as i32;

        // Check if all operations are identical (for stride-0 optimization)
        let all_identical = cmds.windows(2).all(|w| {
            w[0].m == w[1].m
                && w[0].n == w[1].n
                && w[0].k == w[1].k
                && w[0].alpha == w[1].alpha
                && w[0].beta == w[1].beta
                && w[0].trans_a == w[1].trans_a
                && w[0].trans_b == w[1].trans_b
        });

        // Use stride-0 for identical operations (e.g., GEMV-micro)
        let (stride_a, stride_b, stride_c) = if all_identical {
            // Stride-0 means reuse the same buffer for all operations
            tracing::debug!(
                "Using stride-0 optimization for {} identical operations",
                batch_size
            );
            (0i64, 0i64, 0i64)
        } else {
            // Normal strided batching for different data
            ((m * k) as i64, (k * n) as i64, (m * n) as i64)
        };

        // Configure strided-batched GEMM
        let cfg = StridedBatchedConfig {
            gemm: GemmConfig {
                transa,
                transb,
                m: m as i32,
                n: n as i32,
                k: k as i32,
                alpha: cmd.alpha,
                beta: cmd.beta,
                lda,
                ldb,
                ldc,
            },
            stride_a,
            stride_b,
            stride_c,
            batch_size: batch_size as i32,
        };

        // Execute batched GEMM on GPU
        // Track wall time
        let wall_start = Instant::now();

        let blas = self.blas.clone();

        // For stride-0 (identical ops), use single buffers; otherwise use batch buffers
        if cmd.is_fp16() {
            // FP16 batched GEMM
            let cfg_fp16 = StridedBatchedConfig {
                gemm: GemmConfig {
                    transa,
                    transb,
                    m: m as i32,
                    n: n as i32,
                    k: k as i32,
                    alpha: f16::from_f32(cmd.alpha),
                    beta: f16::from_f32(cmd.beta),
                    lda,
                    ldb,
                    ldc,
                },
                stride_a,
                stride_b,
                stride_c,
                batch_size: batch_size as i32,
            };

            if all_identical {
                self.get_or_create_buffers(shape, 1)?;
                let buffers = self.buffers.get_mut(&shape).unwrap();
                match buffers {
                    GemmBuffers::Fp16 { a, b, c, .. } => unsafe {
                        blas.gemm_strided_batched(cfg_fp16, a, b, c)?;
                    },
                    _ => {
                        return Err(anyhow::anyhow!(
                            "Buffer type mismatch for FP16 batched GEMM"
                        ))
                    }
                }
            } else {
                self.get_or_create_batch_buffers(shape, batch_size)?;
                let buffers = self.buffers.get_mut(&shape).unwrap();
                match buffers {
                    GemmBuffers::Fp16 {
                        batch_buffers: Some(batch_bufs),
                        ..
                    } => unsafe {
                        blas.gemm_strided_batched(
                            cfg_fp16,
                            &batch_bufs.a,
                            &batch_bufs.b,
                            &mut batch_bufs.c,
                        )?;
                    },
                    _ => {
                        return Err(anyhow::anyhow!(
                            "Buffer type mismatch for FP16 batched GEMM"
                        ))
                    }
                }
            }
        } else {
            // FP32 batched GEMM
            if all_identical {
                self.get_or_create_buffers(shape, 1)?;
                let buffers = self.buffers.get_mut(&shape).unwrap();
                match buffers {
                    GemmBuffers::Fp32 { a, b, c, .. } => unsafe {
                        blas.gemm_strided_batched(cfg, a, b, c)?;
                    },
                    _ => {
                        return Err(anyhow::anyhow!(
                            "Buffer type mismatch for FP32 batched GEMM"
                        ))
                    }
                }
            } else {
                self.get_or_create_batch_buffers(shape, batch_size)?;
                let buffers = self.buffers.get_mut(&shape).unwrap();
                match buffers {
                    GemmBuffers::Fp32 {
                        batch_buffers: Some(batch_bufs),
                        ..
                    } => unsafe {
                        blas.gemm_strided_batched(
                            cfg,
                            &batch_bufs.a,
                            &batch_bufs.b,
                            &mut batch_bufs.c,
                        )?;
                    },
                    _ => {
                        return Err(anyhow::anyhow!(
                            "Buffer type mismatch for FP32 batched GEMM"
                        ))
                    }
                }
            }
        }

        // Track wall time
        let wall_elapsed = wall_start.elapsed();
        self.wall_time_ms += wall_elapsed.as_secs_f64() * 1000.0;

        self.total_kernels += 1; // One kernel for entire batch
        self.total_operations += batch_size as u64;

        Ok(())
    }

    /// Blocks the CPU thread until all previously submitted work on the CUDA stream
    /// has completed. This is essential for accurate wall-clock timing.
    pub fn synchronize(&self) -> Result<()> {
        self.device.synchronize()?;
        Ok(())
    }

    /// Records the start event for GPU timing.
    pub fn record_start(&self) -> Result<()> {
        if let Some(start) = self.start_event {
            unsafe {
                event::record(start, self.stream.stream).context("Failed to record start event")?
            }
        }
        Ok(())
    }

    /// Records the stop event for GPU timing.
    pub fn record_stop(&self) -> Result<()> {
        if let Some(stop) = self.stop_event {
            unsafe {
                event::record(stop, self.stream.stream).context("Failed to record stop event")?
            }
        }
        Ok(())
    }

    /// Calculates the elapsed time between start and stop events.
    pub fn compute_gpu_time(&mut self) -> Result<()> {
        if let (Some(start), Some(stop)) = (self.start_event, self.stop_event) {
            // Synchronize on the stop event before querying time
            self.synchronize()?;

            let time_ms = unsafe {
                event::elapsed(start, stop).context("Failed to get elapsed time from events")?
            };
            self.gpu_time_ms = time_ms as f64;
        }
        Ok(())
    }

    /// Get collected GPU time
    pub fn get_gpu_time_ms(&self) -> f64 {
        self.gpu_time_ms
    }

    /// Warms up the GPU by allocating buffers for a set of unique shapes and
    /// executing a small operation. This helps avoid including one-time
    /// initialization costs in benchmark measurements.
    pub fn warmup(&mut self, unique_shapes: &[ShapeKey]) -> Result<()> {
        // Phase 1: Driver/library initialization with tiny GEMM
        let dummy_cmd = GemmCommand::new(8, 8, 8);
        self.execute_single(&dummy_cmd)?;
        self.synchronize()?;

        // Phase 2: Pre-allocate buffers for all shapes
        for &shape in unique_shapes {
            let cmd = GemmCommand::new(shape.m, shape.n, shape.k);
            self.execute_single(&cmd)?;
        }
        self.synchronize()?;

        // Reset statistics after warmup
        self.total_kernels = 0;
        self.total_operations = 0;
        self.wall_time_ms = 0.0;
        self.gpu_time_ms = 0.0;

        Ok(())
    }

    /// Get or create batch buffers for a shape
    fn get_or_create_batch_buffers(
        &mut self,
        shape: ShapeKey,
        batch_size: usize,
    ) -> Result<&mut GemmBuffers> {
        let m = shape.m as usize;
        let n = shape.n as usize;
        let k = shape.k as usize;
        let is_fp16 = shape.dtype == 1;

        // Ensure single buffers exist
        self.get_or_create_buffers(shape, 1)?;

        // Get buffers and check if we need batch allocation
        let buffers = self.buffers.get_mut(&shape).unwrap();

        // Allocate or grow batch buffers if needed
        if is_fp16 {
            match buffers {
                GemmBuffers::Fp16 { batch_buffers, .. } => {
                    match batch_buffers {
                        None => {
                            // First batch allocation for this shape
                            let a = self.device.alloc_zeros::<f16>(m * k * batch_size)?;
                            let b = self.device.alloc_zeros::<f16>(k * n * batch_size)?;
                            let c = self.device.alloc_zeros::<f16>(m * n * batch_size)?;

                            *batch_buffers = Some(BatchBuffersFp16 {
                                a,
                                b,
                                c,
                                capacity: batch_size,
                            });
                        }
                        Some(batch) if batch.capacity < batch_size => {
                            // Need larger buffers
                            let a = self.device.alloc_zeros::<f16>(m * k * batch_size)?;
                            let b = self.device.alloc_zeros::<f16>(k * n * batch_size)?;
                            let c = self.device.alloc_zeros::<f16>(m * n * batch_size)?;

                            *batch = BatchBuffersFp16 {
                                a,
                                b,
                                c,
                                capacity: batch_size,
                            };
                        }
                        _ => {} // Existing buffers are large enough
                    }
                }
                _ => return Err(anyhow::anyhow!("Buffer type mismatch")),
            }
        } else {
            match buffers {
                GemmBuffers::Fp32 { batch_buffers, .. } => {
                    match batch_buffers {
                        None => {
                            // First batch allocation for this shape
                            let a = self.device.alloc_zeros::<f32>(m * k * batch_size)?;
                            let b = self.device.alloc_zeros::<f32>(k * n * batch_size)?;
                            let c = self.device.alloc_zeros::<f32>(m * n * batch_size)?;

                            *batch_buffers = Some(BatchBuffersFp32 {
                                a,
                                b,
                                c,
                                capacity: batch_size,
                            });
                        }
                        Some(batch) if batch.capacity < batch_size => {
                            // Need larger buffers
                            let a = self.device.alloc_zeros::<f32>(m * k * batch_size)?;
                            let b = self.device.alloc_zeros::<f32>(k * n * batch_size)?;
                            let c = self.device.alloc_zeros::<f32>(m * n * batch_size)?;

                            *batch = BatchBuffersFp32 {
                                a,
                                b,
                                c,
                                capacity: batch_size,
                            };
                        }
                        _ => {} // Existing buffers are large enough
                    }
                }
                _ => return Err(anyhow::anyhow!("Buffer type mismatch")),
            }
        }

        Ok(buffers)
    }

    /// Get or create GPU buffers for a shape
    fn get_or_create_buffers(
        &mut self,
        shape: ShapeKey,
        _batch_size: usize,
    ) -> Result<&mut GemmBuffers> {
        let m = shape.m as usize;
        let n = shape.n as usize;
        let k = shape.k as usize;
        let is_fp16 = shape.dtype == 1;

        // Create entry if not exists
        if !self.buffers.contains_key(&shape) {
            if is_fp16 {
                // Allocate FP16 buffers
                let a = self.device.alloc_zeros::<f16>(m * k)?;
                let b = self.device.alloc_zeros::<f16>(k * n)?;
                let c = self.device.alloc_zeros::<f16>(m * n)?;

                self.buffers.insert(
                    shape,
                    GemmBuffers::Fp16 {
                        a,
                        b,
                        c,
                        batch_buffers: None,
                    },
                );
            } else {
                // Allocate FP32 buffers
                let a = self.device.alloc_zeros::<f32>(m * k)?;
                let b = self.device.alloc_zeros::<f32>(k * n)?;
                let c = self.device.alloc_zeros::<f32>(m * n)?;

                self.buffers.insert(
                    shape,
                    GemmBuffers::Fp32 {
                        a,
                        b,
                        c,
                        batch_buffers: None,
                    },
                );
            }
        }

        // Return existing buffers - batched mode will allocate its own if needed
        Ok(self.buffers.get_mut(&shape).unwrap())
    }

    /// Get statistics
    pub fn get_stats(&self) -> (u64, u64, f64) {
        (self.total_kernels, self.total_operations, self.gpu_time_ms)
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.total_kernels = 0;
        self.total_operations = 0;
        self.wall_time_ms = 0.0;
        self.gpu_time_ms = 0.0;
    }
}

impl Drop for GemmExecutor {
    fn drop(&mut self) {
        // Clean up CUDA events if they were created
        if let Some(event) = self.start_event.take() {
            unsafe {
                event::destroy(event).ok();
            }
        }
        if let Some(event) = self.stop_event.take() {
            unsafe {
                event::destroy(event).ok();
            }
        }
    }
}
