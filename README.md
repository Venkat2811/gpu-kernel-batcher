# CUDA Kernel Launch Optimizer: Technical Deep Dive

This document provides a technical overview of the CUDA Kernel Launch Optimizer project, explaining the problem it solves, the architecture, and the core optimization techniques demonstrated.

Built with assistance from Claude and Gemini 2.5 Pro.

# The Scale of the Problem: Kernel Launches in LLMs

In LLaMA-70B:

Each of the model's 80 transformer layers requires about 15 individual kernel launches per token generated (~8 for the attention block and ~7 for the feed-forward MLP). This results in approximately **1,200 launches per token**.

Generating a standard 2048-token response therefore triggers over **2.5 million separate kernel calls**, highlighting why reducing launch overhead is critical for performance.

All Claude code users are familiar with **"You are absolutely right."**

While Claude model architecture is not publicly known, we can guesstimate kernel launches. We have 5 tokens.

At anthropic scale, 10M req/day out of which 2M responses/day, has these 5 tokens, then 2M * 5 = 10M tokens/day.

Kernel launches per day = 10M * 1200 = 12B

Overhead per kernel launch = 5us

Total overhead = 12B * 5us = 60,000s = 1,000min = ~16.7hr

This is just for the first 5 tokens.


Today's SOTA inference engines and kernels have several optimizations to reduce this, so the above numbers are not true anymore:
- CUDA graphs
- Kernel fusion
- Persistent kernels
- Megakernels
- KV cache optimizations - paging, radix attention, flash attention, etc.,


## 1. System Architecture

### Project Architecture

This project uses a decoupled two-process architecture to simulate the relationship between an application generating GPU work and a dedicated worker executing it. This isolates the dispatch mechanism for clear analysis.

```
+--------------------------+      +--------------------------+      +--------------------------+
|     Producer Process     |      |      Shared Memory       |      |     Consumer Process     |
+--------------------------+      +--------------------------+      +--------------------------+
             |                                 ^                                 |
             |                                 |                                 |
             v                                 |                                 v
+--------------------------+                   |                      +--------------------------+
|  Workload Generation     |                   |                      |   Queue Pop Operation    |
|  - Decode, Prefill, etc. |                   |                      +--------------------------+
+--------------------------+                   |                                 |
             |                                 |                                 |
             v                                 |                                 v
+--------------------------+                   |                      +--------------------------+
|  Queue Push Operation    |-------------------'                      |      Batching Logic      |
+--------------------------+                                          |  - Shape Bucketing       |
                                                                      |  - Idle Timeout Flush    |
                                                                      +--------------------------+
                                                                                 |
                                                                                 |
                                                                                 v
                                                                    .------------------------.
                                                                    |  Execution Strategy    |
                                                                    '-----------.------------'
                                                                                |
                                                                    .-----------'-----------.
                                                                    |                       |
                                                                    v                       v
                                                         +---------------------+   +---------------------+
                                                         |    Baseline Mode    |   |     Batched Mode    |
                                                         | (1 Launch per Cmd)  |   | (N Cmds per Launch) |
                                                         +---------------------+   +---------------------+
                                                                    |                       |
                                                                    '----------.------------'
                                                                               |
                                                                               v
                                                                    +---------------------+ 
                                                                    |         GPU         |
                                                                    |      (cuBLASLt)     |
                                                                    +---------------------+
```

### Context: Modern LLM Inference Engine Architecture

This project's architecture is a simplified model of a modern LLM inference engine. The `Producer` mimics the `Inference Engine (CPU Process)`, and the `Consumer` mimics the `GPU Worker`. This allows us to study the critical dispatch loop in isolation.

```
+--------------------------------+
|      User Requests             |
+--------------------------------+
              |
              v
+--------------------------------+
|   API Server (HTTP/gRPC)       |
+--------------------------------+
              |
              | (Generation Tasks)
              v
+--------------------------------+
| Inference Engine (CPU Process) |
|   - Frontend DSL Parser        |
|   - Scheduler/Continuous Batcher|
|   - Manages KV Cache Page Tables|
+--------------------------------+
              |
              | (Micro-batch + Page Tables)
              v
+--------------------------------+      +--------------------------------+
|   GPU Worker (CUDA Kernels)    |----->|         GPU Memory             |
|   - Executes PagedAttention    |      |   - Model Weights              |
|   - Executes standard GEMM     |      |   - KV Cache (in non-contiguous pages) |
+--------------------------------+      +--------------------------------+
              ^
              | (Logits)
              '----------------------.
                                     |
                                     v
+--------------------------------+
|  Constrained Decoding & Sampling |
+--------------------------------+
              |
              v
+--------------------------------+
|      User Responses            |
+--------------------------------+
```

## 2. Modeling a Real Transformer Layer

A complete transformer layer involves multiple steps. However, its performance is overwhelmingly dominated by the General Matrix-Matrix Multiplication (GEMM) operations.

```
Input (from previous layer)
           |
           v
+--------------------------+
| Multi-Head Attention     |
|  - Q, K, V Projections   | <--- GEMM (We model this)
|  - Scaled Dot-Product    | <--- GEMM (We model this)
|  - Softmax               | <--- Element-wise, cheap
|  - Output Projection     | <--- GEMM (We model this)
+--------------------------+
           |
           v
+--------------------------+
| Add & LayerNorm          | <--- Element-wise, cheap
+--------------------------+
           |
           v
+--------------------------+
| Feed-Forward Network     |
|  - Up-Projection         | <--- GEMM (We model this)
|  - Activation (GeLU/SiLU)| <--- Element-wise, cheap
|  - Down-Projection       | <--- GEMM (We model this)
+--------------------------+
           |
           v
+--------------------------+
| Add & LayerNorm          | <--- Element-wise, cheap
+--------------------------+
           |
           v
Output (to next layer)
```

To understand the optimization, think about two kinds of batching:

1.  **The Carpool (Standard Batching)**: A single command can already represent a "batch" of users. Think of it as a carpool taking multiple people at once. This is standard practice.

2.  **The Convoy (Our Optimization)**: This is our project's key trick. When we see multiple, identical carpools (i.e., commands with the same matrix shape) going to the same destination, we group them into a single, highly-efficient convoy—one big `cuBLASLt` strided-batched launch. This is where we save on overhead.

These workloads **intentionally simulate only the GEMM steps** to focus on this second level of batching, which is the primary source of performance improvement.

## 3. Workload Simulation

We simulate three key workloads to represent the different phases of LLM inference.

### Prefill Phase: Processing the Prompt

This is a single, large matrix multiplication to process the user's prompt. It is **compute-bound**, so batching offers little gain.

```
  [   Prompt      ]   x   [  Model   ]  =  [  Logits  ]
  [   Tokens      ]       [  Weights ]     [ for next ]
  [               ]       [          ]     [   token  ]
  [               ]       [          ]
  [               ]
  (Large M x K)         (Large K x N)    (Large M x N)
```

### Decode Phase: The Generation Loop

This is a loop of small matrix-vector multiplications to generate tokens one by one. This process is **launch-bound**, making it a perfect candidate for batching.

```
            +-----------------+
            |   Input Token   |
            | (from previous  |
            |      step)      |
            +-------+---------+
                    |
                    v
+---------------------------------------+
|                 GEMV                  |
|   [1 Token] x [Model Weights]         |
|   (1 x K)   x   (K x N)               |
+-------------------v-------------------+
                    |
          +---------+---------+
          |      Logits       |
          | (Scores for all   |
          | possible tokens)  |
          +---------+---------+
                    |
                    v
          +---------+---------+
          |     Sampling      |
          | (CPU picks best   |
          |   or creative)    |
          +---------+---------+
                    |
+-------------------'
|
|         +---------+---------+
'-------->|    New Token      | (Becomes the next Input Token)
          +-------------------+

```

Here's how one step in the loop works:

1.  **GEMV**: A single token (represented as a vector) is multiplied by the large model weights. This is the `(1 x K) * (K x N)` operation that the project focuses on. The `GemmCommand` in this project is a perfect simulation of one of these Q, K, or V projection steps for a batch of users.
2.  **Logits**: The result is a vector of raw scores called **logits**. This vector is very large, with one score for every possible token in the model's vocabulary (e.g., ~32,000 scores).
3.  **Sampling**: The logits are used to select the *next* token. This is a multi-step process:
    *   A `softmax` function converts the raw logit scores into a probability distribution.
    *   A **sampling** algorithm (e.g., top-k, top-p, or simple greedy argmax) chooses a single token ID from this distribution.

This chosen token then becomes the input for the very next iteration of the loop.

#### The CPU/GPU Partnership in Sampling

Looking into popular inference engines like **vLLM** and **SGLang** reveals a hybrid approach:

*   **CPU Orchestration**: The high-level sampling logic (e.g., applying temperature, choosing top-k vs top-p) is controlled by Python code on the CPU.
*   **GPU Execution**: The actual computation of running the sampling algorithm on the massive logits tensor is often offloaded to a highly optimized CUDA kernel (e.g., from the **FlashInfer** library) for maximum performance.

This entire loop of tiny, sequential operations is what makes the decode phase bottlenecked by kernel launch overhead. As the vLLM team notes, *"the performance bottleneck of vLLM is mainly caused by the CPU overhead that blocks the GPU execution."* This is precisely the problem the project's batching mechanism solves.

### GEMV-micro: The Ultimate Test

This synthetic workload is a pure General Matrix-Vector multiplication (GEMV), designed to be **extremely launch-bound** and show the maximum possible benefit of the optimizer.

```
  [          ]
  [  Matrix  ]   x   [Vector]  =  [Vector]
  [          ]
  [          ]
  (e.g., 256x256)    (256x1)        (256x1)
```

## 4. The Optimization: Strided Batching

### The Problem: Naive Execution

Without batching, each command results in a separate, expensive kernel launch. The overhead of telling the GPU to start work dominates the actual computation time for small tasks.

```
Incoming Commands (all have the same shape):
  - Command 1
  - Command 2
  - Command 3

Execution Flow:
+-----------+     +-----------+     +-----------+
| Process   | --> | Launch    | --> | GPU does  |
| Command 1 |     | Kernel #1 |     | work for #1|
+-----------+     +-----------+     +-----------+

+-----------+     +-----------+     +-----------+
| Process   | --> | Launch    | --> | GPU does  |
| Command 2 |     | Kernel #2 |     | work for #2|
+-----------+     +-----------+     +-----------+

+-----------+     +-----------+     +-----------+
| Process   | --> | Launch    | --> | GPU does  |
| Command 3 |     | Kernel #3 |     | work for #3|
+-----------+     +-----------+     +-----------+

Result: 3 separate, expensive kernel launches.
```

### The Solution: How Strided Batching Works

With strided batching, we combine the data from multiple commands into large, continuous GPU buffers. We then launch a **single kernel** and tell it the **stride**, which is the distance in memory from the start of one problem's data to the start of the next.

```
Incoming Commands (all have the same shape):
  - Command 1
  - Command 2
  - Command 3

      |
      v
Batching Logic: Combine all inputs into single GPU buffers.

GPU Memory Layout:

Buffer for A matrices:
[----------A1----------|----------A2----------|----------A3----------]
^
|
'-----------------------'
      stride_A (the size of one A matrix)


Buffer for B matrices:
[----------B1----------|----------B2----------|----------B3----------]
^
|
'-----------------------'
      stride_B (the size of one B matrix)

      |
      v
Single Kernel Launch:
cuBLASLt is told:
  1. Batch Size: 3
  2. Start of Buffer A, B, C
  3. Stride for A, B, C

Result: ONE single, highly efficient kernel launch that does the work of three.

This technique is critical in real-world transformer inference. During the decode phase, where matrices are small and numerous, batching requests from multiple concurrent users into a single strided-batch call is the key to maximizing GPU utilization and achieving high throughput.
```

## Results
RTX 3060 12GB VRAM

```bash
 USE_CUDA_EVENTS=1 make benchmark

📊 Analyzing latest benchmark results...
Using latest run: 20250813_150938

🚀 CUDA Kernel Launch Optimizer - Results Analysis
================================================================================

✅ GEMV-micro (FP32) - GPU-timed
================================================================================
Parameters: M=64, K=8192
+-----------------+------------------+-----------------+-----------------+---------------+
| Configuration   | Wall Time (ms)   | GPU Time (ms)   | Launches        | Mode          |
+=================+==================+=================+=================+===============+
| Baseline        | 15807.6          | 15575.6         | 2,000,000       | none          |
+-----------------+------------------+-----------------+-----------------+---------------+
| Optimized       | 3482.2           | 3251.1          | 3,907           | shape-batched |
+-----------------+------------------+-----------------+-----------------+---------------+
| **Speedup**     | +78.0%           | +79.1%          | 99.8% reduction |               |
+-----------------+------------------+-----------------+-----------------+---------------+


✅ GEMV-micro (FP16) - GPU-timed
================================================================================
Parameters: M=64, K=8192
+-----------------+------------------+-----------------+-----------------+---------------+
| Configuration   | Wall Time (ms)   | GPU Time (ms)   | Launches        | Mode          |
+=================+==================+=================+=================+===============+
| Baseline        | 17803.2          | 17573.7         | 2,000,000       | none          |
+-----------------+------------------+-----------------+-----------------+---------------+
| Optimized       | 5559.4           | 5333.9          | 3,907           | shape-batched |
+-----------------+------------------+-----------------+-----------------+---------------+
| **Speedup**     | +68.8%           | +69.6%          | 99.8% reduction |               |
+-----------------+------------------+-----------------+-----------------+---------------+


✅ Decode (FP32) - GPU-timed
================================================================================
Parameters: Hidden=1024, Layers=8, Batch=1
+-----------------+------------------+-----------------+-----------------+---------------+
| Configuration   | Wall Time (ms)   | GPU Time (ms)   | Launches        | Mode          |
+=================+==================+=================+=================+===============+
| Baseline        | 13831.7          | 13600.8         | 400,000         | none          |
+-----------------+------------------+-----------------+-----------------+---------------+
| Optimized       | 12282.7          | 12084.9         | 25,000          | shape-batched |
+-----------------+------------------+-----------------+-----------------+---------------+
| **Speedup**     | +11.2%           | +11.1%          | 93.8% reduction |               |
+-----------------+------------------+-----------------+-----------------+---------------+


✅ Decode (FP16) - GPU-timed
================================================================================
Parameters: Hidden=1024, Layers=8, Batch=1
+-----------------+------------------+-----------------+-----------------+---------------+
| Configuration   | Wall Time (ms)   | GPU Time (ms)   | Launches        | Mode          |
+=================+==================+=================+=================+===============+
| Baseline        | 8564.6           | 8347.7          | 400,000         | none          |
+-----------------+------------------+-----------------+-----------------+---------------+
| Optimized       | 6665.3           | 6469.7          | 25,000          | shape-batched |
+-----------------+------------------+-----------------+-----------------+---------------+
| **Speedup**     | +22.2%           | +22.5%          | 93.8% reduction |               |
+-----------------+------------------+-----------------+-----------------+---------------+


❌ Prefill (FP32) - GPU-timed
================================================================================
Parameters: Hidden=1024, Layers=64, Batch=1, SeqLen=256
+-----------------+------------------+-----------------+-----------------+---------------+
| Configuration   | Wall Time (ms)   | GPU Time (ms)   | Launches        | Mode          |
+=================+==================+=================+=================+===============+
| Baseline        | 239.3            | 51.0            | 256             | none          |
+-----------------+------------------+-----------------+-----------------+---------------+
| Optimized       | 273.6            | 44.1            | 32              | shape-batched |
+-----------------+------------------+-----------------+-----------------+---------------+
| **Speedup**     | -14.3%           | +13.6%          | 87.5% reduction |               |
+-----------------+------------------+-----------------+-----------------+---------------+


✅ Prefill (FP16) - GPU-timed
================================================================================
Parameters: Hidden=1024, Layers=64, Batch=1, SeqLen=256
+-----------------+------------------+-----------------+-----------------+---------------+
| Configuration   | Wall Time (ms)   | GPU Time (ms)   | Launches        | Mode          |
+=================+==================+=================+=================+===============+
| Baseline        | 352.4            | 127.1           | 256             | none          |
+-----------------+------------------+-----------------+-----------------+---------------+
| Optimized       | 349.0            | 121.9           | 32              | shape-batched |
+-----------------+------------------+-----------------+-----------------+---------------+
| **Speedup**     | +1.0%            | +4.1%           | 87.5% reduction |               |
+-----------------+------------------+-----------------+-----------------+---------------+
```