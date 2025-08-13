//! # LLM Inference Workload Generators
//!
//! This module provides functions to generate sequences of `GemmCommand`s that
//! simulate different phases of LLM inference. These workloads are used to
//! benchmark the performance of immediate vs. batched kernel execution.
//!
//! ## Key Workloads
//!
//! - **Prefill (`generate_prefill`)**: Simulates the initial processing of a user's
//!   prompt. This phase is characterized by a few, very large GEMM operations.
//!   It is typically **compute-bound**, meaning the GPU is the bottleneck, and
//!   kernel launch overhead is less of a factor.
//!
//! - **Decode (`generate_decode`)**: Simulates the autoregressive generation of
//!   tokens, one after another. This phase involves many small GEMM operations
//!   (effectively GEMVs). It is **launch-bound**, meaning performance is limited
//!   by the overhead of launching many kernels, making it the primary target for
//!   the strided-batching optimization.

use crate::model::{GemmCommand, ModelConfig, ShapeKey};

/// Generates a **prefill** workload, simulating the processing of a prompt.
///
/// This creates a sequence of large GEMM operations, which is characteristic of
/// the compute-bound prefill phase in LLM inference.
pub fn generate_prefill(config: &ModelConfig, seq_len: usize) -> Vec<GemmCommand> {
    let mut commands = Vec::new();
    let mut batch_id = 0u64;

    // For prefill, M is the total number of tokens in the prompt batch
    let m = (config.batch * seq_len) as u32;

    // Process through each layer sequentially (like real inference)
    for _layer in 0..config.layers {
        // Layer processing order matches real transformers:

        // 1. QKV projection: M x hidden -> M x qkv_proj
        commands.push(GemmCommand::new(m, config.qkv_proj, config.hidden).with_batch_id(batch_id));
        batch_id += 1;

        // 2. Attention output projection: M x hidden -> M x hidden
        commands.push(GemmCommand::new(m, config.hidden, config.hidden).with_batch_id(batch_id));
        batch_id += 1;

        // 3. FFN up projection: M x hidden -> M x ffn
        commands.push(GemmCommand::new(m, config.ffn, config.hidden).with_batch_id(batch_id));
        batch_id += 1;

        // 4. FFN down projection: M x ffn -> M x hidden
        commands.push(GemmCommand::new(m, config.hidden, config.ffn).with_batch_id(batch_id));
        batch_id += 1;
    }

    commands
}

/// Generates a **decode** workload, simulating autoregressive token generation.
///
/// This creates a long sequence of small GEMMs, which is characteristic of the
/// launch-bound decode phase. This workload is where strided batching provides
/// the most significant performance improvement.
pub fn generate_decode(config: &ModelConfig, token_count: usize) -> Vec<GemmCommand> {
    let mut commands = Vec::new();
    let mut batch_id = 0u64;

    // For decode, M is just the batch size (generating one token at a time)
    let m = config.batch as u32;

    // Simulate generating `token_count` tokens sequentially
    for _token_idx in 0..token_count {
        // For each token, process through ALL layers sequentially
        // This mimics real autoregressive generation
        for _layer in 0..config.layers {
            // Same 4 operations per layer, in order:

            // 1. QKV projection: batch x hidden -> batch x qkv_proj
            commands
                .push(GemmCommand::new(m, config.qkv_proj, config.hidden).with_batch_id(batch_id));
            batch_id += 1;

            // 2. Attention output: batch x hidden -> batch x hidden
            commands
                .push(GemmCommand::new(m, config.hidden, config.hidden).with_batch_id(batch_id));
            batch_id += 1;

            // 3. FFN up: batch x hidden -> batch x ffn
            commands.push(GemmCommand::new(m, config.ffn, config.hidden).with_batch_id(batch_id));
            batch_id += 1;

            // 4. FFN down: batch x ffn -> batch x hidden
            commands.push(GemmCommand::new(m, config.hidden, config.ffn).with_batch_id(batch_id));
            batch_id += 1;
        }
    }

    commands
}

/// Generates a synthetic **GEMV-micro** workload to isolate kernel launch overhead.
///
/// This workload consists of GEMMs where N=1 (making them matrix-vector multiplies),
/// which have a very low arithmetic intensity. This makes the benchmark almost purely
/// a measure of kernel launch efficiency.
pub fn generate_gemv_micro(m: u32, k: u32, count: usize) -> Vec<GemmCommand> {
    let mut commands = Vec::new();

    // Use N=1 to create GEMV (degenerate GEMM)
    // This creates very short kernels where launch overhead dominates
    let n = 1;

    for i in 0..count {
        commands.push(GemmCommand::new(m, n, k).with_batch_id(i as u64));
    }

    commands
}

/// Generate micro workload (original test workload)
pub fn generate_micro(count: usize) -> Vec<GemmCommand> {
    let mut commands = Vec::new();

    // Original micro workload: small matrices
    let m = 64;
    let n = 256;
    let k = 128;

    for i in 0..count {
        commands.push(GemmCommand::new(m, n, k).with_batch_id(i as u64));
    }

    commands
}

/// Generate single shape workload (for testing)
pub fn generate_single(m: u32, n: u32, k: u32, count: usize) -> Vec<GemmCommand> {
    let mut commands = Vec::new();

    for i in 0..count {
        commands.push(GemmCommand::new(m, n, k).with_batch_id(i as u64));
    }

    commands
}

/// Returns the unique `ShapeKey`s that will be generated by `generate_decode`.
///
/// This is a helper used to warm up the `GemmExecutor` by pre-allocating the
/// necessary GPU buffers before the benchmark run.
pub fn get_decode_shapes(config: &ModelConfig) -> Vec<ShapeKey> {
    let m = config.batch as u32;

    vec![
        // QKV projection shape
        ShapeKey {
            m,
            n: config.qkv_proj,
            k: config.hidden,
            trans_a: 0,
            trans_b: 0,
            dtype: 0,
        },
        // Attention output shape
        ShapeKey {
            m,
            n: config.hidden,
            k: config.hidden,
            trans_a: 0,
            trans_b: 0,
            dtype: 0,
        },
        // FFN up shape
        ShapeKey {
            m,
            n: config.ffn,
            k: config.hidden,
            trans_a: 0,
            trans_b: 0,
            dtype: 0,
        },
        // FFN down shape
        ShapeKey {
            m,
            n: config.hidden,
            k: config.ffn,
            trans_a: 0,
            trans_b: 0,
            dtype: 0,
        },
    ]
}

/// Returns the unique `ShapeKey`s that will be generated by `generate_prefill`.
///
/// This is a helper used to warm up the `GemmExecutor` by pre-allocating the
/// necessary GPU buffers before the benchmark run.
pub fn get_prefill_shapes(config: &ModelConfig, seq_len: usize) -> Vec<ShapeKey> {
    let m = (config.batch * seq_len) as u32;

    vec![
        // QKV projection shape
        ShapeKey {
            m,
            n: config.qkv_proj,
            k: config.hidden,
            trans_a: 0,
            trans_b: 0,
            dtype: 0,
        },
        // Attention output shape
        ShapeKey {
            m,
            n: config.hidden,
            k: config.hidden,
            trans_a: 0,
            trans_b: 0,
            dtype: 0,
        },
        // FFN up shape
        ShapeKey {
            m,
            n: config.ffn,
            k: config.hidden,
            trans_a: 0,
            trans_b: 0,
            dtype: 0,
        },
        // FFN down shape
        ShapeKey {
            m,
            n: config.hidden,
            k: config.ffn,
            trans_a: 0,
            trans_b: 0,
            dtype: 0,
        },
    ]
}

/// Get expected shapes for GEMV-micro workload (for warmup)  
pub fn get_gemv_micro_shapes(m: u32, k: u32) -> Vec<ShapeKey> {
    vec![ShapeKey {
        m,
        n: 1,
        k,
        trans_a: 0,
        trans_b: 0,
        dtype: 0,
    }]
}

// Compatibility functions for old API
pub fn generate_prefill_compat(
    batch: usize,
    seq_len: usize,
    hidden: u32,
    layers: usize,
) -> Vec<GemmCommand> {
    let config = ModelConfig::new(hidden, layers, batch);
    generate_prefill(&config, seq_len)
}

pub fn generate_decode_with_count(
    batch: usize,
    hidden: u32,
    layers: usize,
    target_count: Option<usize>,
) -> Vec<GemmCommand> {
    let config = ModelConfig::new(hidden, layers, batch);
    let count = target_count.unwrap_or(layers * 4); // Default: one token per layer
    generate_decode(&config, count)
}
