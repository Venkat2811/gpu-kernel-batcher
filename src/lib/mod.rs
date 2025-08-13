#![allow(missing_docs)] // Temporarily allow for rapid development
//! # CUDA Kernel Launch Optimizer
//!
//! This crate provides the core logic for a multi-process system designed to
//! demonstrate and benchmark GPU kernel launch overhead reduction through batching.
//! It simulates a real-world LLM inference engine's architecture, where a
//! CPU-based process generates work and a GPU-based process executes it.
//!
//! ## Core Modules
//!
//! - `ipc`: A lock-free, single-producer, single-consumer (SPSC) queue over shared memory.
//! - `model`: Data structures for defining and simulating realistic LLM workloads.
//! - `gemm`: The GPU execution engine, responsible for running GEMM operations.
//! - `workload`: Logic for generating different sequences of GEMM commands.
//! - `profiling`: NVTX range annotations for profiling with NVIDIA Nsight.

pub mod gemm; // CUDA GEMM execution
pub mod ipc;
pub mod model;
pub mod workload; // Workload generators
                  // pub mod profiling; // NVTX profiling support - removed
                  // pub mod batch;  // Will be added in M3
                  // pub mod timing;  // Will be added in M2

pub use gemm::GemmExecutor;
pub use ipc::{QueueError, ShmQueue};
pub use model::{GemmCommand, ShapeKey};
pub use workload::*;

// Re-export common types
pub use anyhow::{Error, Result};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
