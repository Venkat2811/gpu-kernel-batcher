//! # Workload and Command Data Models
//!
//! This module defines the core data structures used throughout the application.
//! These structures are designed to be Plain Old Data (POD) types, ensuring they
//! can be safely and efficiently sent over the shared memory IPC channel.
//!
//! - `GemmCommand`: The fundamental unit of work, representing a single GEMM operation.
//! - `ShapeKey`: A key used to bucket commands for batched execution.
//! - `ModelConfig`: Defines the parameters for a simulated transformer model to generate realistic workloads.

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

/// Represents a single General Matrix Multiplication (GEMM) operation.
///
/// This struct is the primary message passed from the producer to the consumer.
/// It is marked `#[repr(C)]` and implements `bytemuck::Pod` to ensure it has a
/// stable memory layout and can be safely transferred across process boundaries
/// via shared memory without serialization/deserialization overhead.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Serialize, Deserialize)]
pub struct GemmCommand {
    /// Matrix dimension M
    pub m: u32,
    /// Matrix dimension N
    pub n: u32,
    /// Matrix dimension K
    pub k: u32,
    /// Alpha scaling factor
    pub alpha: f32,
    /// Beta scaling factor
    pub beta: f32,
    /// Transpose A flag (0 or 1)
    pub trans_a: u8,
    /// Transpose B flag (0 or 1)
    pub trans_b: u8,
    /// Data type (0 = FP32, 1 = FP16)
    pub dtype: u8,
    /// Padding for alignment
    pub _pad: [u8; 1],
    /// Batch ID for correlation
    pub batch_id: u64,
}

impl GemmCommand {
    /// Create a new GEMM command
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        Self {
            m,
            n,
            k,
            alpha: 1.0,
            beta: 0.0,
            trans_a: 0,
            trans_b: 0,
            dtype: 0, // Default to FP32
            _pad: [0; 1],
            batch_id: 0,
        }
    }

    /// Create with batch ID
    pub fn with_batch_id(mut self, batch_id: u64) -> Self {
        self.batch_id = batch_id;
        self
    }

    /// Set transpose flags
    pub fn with_transpose(mut self, trans_a: bool, trans_b: bool) -> Self {
        self.trans_a = trans_a as u8;
        self.trans_b = trans_b as u8;
        self
    }

    /// Set alpha and beta
    pub fn with_scaling(mut self, alpha: f32, beta: f32) -> Self {
        self.alpha = alpha;
        self.beta = beta;
        self
    }

    /// Set data type (FP16)
    pub fn with_fp16(mut self) -> Self {
        self.dtype = 1;
        self
    }

    /// Check if this is FP16
    pub fn is_fp16(&self) -> bool {
        self.dtype == 1
    }

    /// Get the shape key for bucketing
    pub fn shape_key(&self) -> ShapeKey {
        ShapeKey {
            m: self.m,
            n: self.n,
            k: self.k,
            trans_a: self.trans_a,
            trans_b: self.trans_b,
            dtype: self.dtype,
        }
    }

    /// Calculate memory requirement for this GEMM
    pub fn memory_bytes(&self) -> usize {
        let a_size = (self.m * self.k) as usize;
        let b_size = (self.k * self.n) as usize;
        let c_size = (self.m * self.n) as usize;
        (a_size + b_size + c_size) * std::mem::size_of::<f32>()
    }

    /// Calculate FLOPs for this GEMM
    pub fn flops(&self) -> u64 {
        // 2 * M * N * K for standard GEMM
        2 * (self.m as u64) * (self.n as u64) * (self.k as u64)
    }
}

/// A key representing the unique shape and data type of a `GemmCommand`.
///
/// This is used by the `GemmExecutor` to group commands into batches. Only
/// commands with the exact same `ShapeKey` can be executed together in a single
/// strided-batched CUDA kernel launch.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ShapeKey {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub trans_a: u8,
    pub trans_b: u8,
    pub dtype: u8,
}

impl ShapeKey {
    /// Create a new shape key
    pub fn new(m: u32, n: u32, k: u32, trans_a: bool, trans_b: bool) -> Self {
        Self {
            m,
            n,
            k,
            trans_a: trans_a as u8,
            trans_b: trans_b as u8,
            dtype: 0, // Default to FP32
        }
    }
}

/// Special command types
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ControlCommand {
    /// Command type (0 = normal, 1 = stop, 2 = flush)
    pub cmd_type: u32,
    /// Padding for alignment
    pub _pad1: u32,
    /// Optional payload
    pub payload: u64,
    /// Padding
    pub _pad2: [u8; 16],
}

impl ControlCommand {
    /// Create a stop command
    pub fn stop() -> Self {
        Self {
            cmd_type: 1,
            _pad1: 0,
            payload: 0,
            _pad2: [0; 16],
        }
    }

    /// Create a flush command
    pub fn flush() -> Self {
        Self {
            cmd_type: 2,
            _pad1: 0,
            payload: 0,
            _pad2: [0; 16],
        }
    }
}

/// Statistics for producer/consumer
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueueStats {
    /// Total commands enqueued
    pub total_enqueued: u64,
    /// Total commands dequeued
    pub total_dequeued: u64,
    /// Number of full events
    pub queue_full_count: u64,
    /// Number of empty events
    pub queue_empty_count: u64,
    /// Wall time in milliseconds
    pub wall_time_ms: f64,
    /// GPU time in milliseconds (if available)
    pub gpu_time_ms: Option<f64>,
}

/// Defines the configuration for a benchmark run, specifying the type of
/// workload and its parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadConfig {
    /// Workload type
    pub workload_type: WorkloadType,
    /// Number of operations
    pub count: usize,
    /// Hidden dimension (for decode workload)
    pub hidden_dim: Option<u32>,
    /// Number of layers
    pub layers: Option<usize>,
    /// Batch size
    pub batch_size: Option<usize>,
    /// Sequence length
    pub seq_length: Option<usize>,
}

/// Enumerates the different types of workload simulations available.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WorkloadType {
    /// Educational micro benchmark
    Micro,
    /// Realistic decode workload
    Decode,
    /// Prefill workload (optional)
    Prefill,
    /// Single shape repeated
    Single,
}

/// Defines the architectural parameters of a transformer model.
///
/// This configuration is used by the `Decode` workload generator to create a
/// sequence of `GemmCommand`s that simulates the Q, K, and V projections in a
/// real LLM inference scenario.
#[derive(Debug, Clone, Copy)]
pub struct ModelConfig {
    /// Hidden dimension size
    pub hidden: u32,
    /// FFN expansion dimension (typically 4x hidden)
    pub ffn: u32,
    /// QKV projection size (typically 1.5x hidden for Q,K,V)
    pub qkv_proj: u32,
    /// Number of transformer layers
    pub layers: usize,
    /// Batch size
    pub batch: usize,
}

impl ModelConfig {
    /// Create a new model configuration
    pub fn new(hidden: u32, layers: usize, batch: usize) -> Self {
        Self {
            hidden,
            ffn: hidden * 4,            // Standard 4x FFN expansion
            qkv_proj: (hidden * 3) / 2, // 1.5x for Q, K, V projections
            layers,
            batch,
        }
    }

    /// Create with custom FFN expansion ratio
    pub fn with_ffn_ratio(mut self, ratio: u32) -> Self {
        self.ffn = self.hidden * ratio;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm_command_size() {
        // Ensure GemmCommand is exactly 32 bytes for cache alignment
        assert_eq!(std::mem::size_of::<GemmCommand>(), 32);
    }

    #[test]
    fn test_gemm_command_pod() {
        // Verify POD properties
        let cmd = GemmCommand::new(64, 256, 128);
        let bytes = bytemuck::bytes_of(&cmd);
        assert_eq!(bytes.len(), 32);

        // Round trip through bytes
        let cmd2: &GemmCommand = bytemuck::from_bytes(bytes);
        assert_eq!(cmd.m, cmd2.m);
        assert_eq!(cmd.n, cmd2.n);
        assert_eq!(cmd.k, cmd2.k);
    }

    #[test]
    fn test_shape_key() {
        let cmd1 = GemmCommand::new(64, 256, 128);
        let cmd2 = GemmCommand::new(64, 256, 128);
        let cmd3 = GemmCommand::new(32, 256, 128);

        assert_eq!(cmd1.shape_key(), cmd2.shape_key());
        assert_ne!(cmd1.shape_key(), cmd3.shape_key());
    }

    #[test]
    fn test_flops_calculation() {
        let cmd = GemmCommand::new(64, 256, 128);
        let expected_flops = 2 * 64 * 256 * 128;
        assert_eq!(cmd.flops(), expected_flops);
    }
}
