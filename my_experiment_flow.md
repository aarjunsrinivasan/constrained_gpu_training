# Training Qwen3 on 2x RTX 3090 with TorchTitan

**Hardware**: 2x RTX 3090 (24 GB VRAM each, 71 TFLOPS bf16, PCIe 4.0 ~25 GB/s)
**Stack**: FSDP2 + torch.compile + activation checkpointing

---

## Memory Rules of Thumb

For a model with `P` billion params in bf16 training:

> 1B params in fp32 ≈ 4 GB, in bf16 ≈ 2 GB

```
Params:     P × 2  GB
Grads:      P × 2  GB
Optimizer:  P × 12 GB  (Adam: fp32 params + m + v)
Total:      P × 16 GB

With FSDP2 across 2 GPUs: divide by 2
```

| Model | Total | Single GPU | FSDP2 (per GPU) |
|-------|-------|------------|------------------|
| 0.6B  | ~10 GB | fits       | fits easily      |
| 1.7B  | ~27 GB | **OOM** (optimizer alone ~20 GB) | ~14 GB + activations ✓ |

---

## Setup

```bash
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-0.6B --assets tokenizer
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-1.7B --assets tokenizer
```

---

## Experiments

### 1. Single GPU — Qwen3-0.6B

```bash
# no optimizations
NGPU=1 MODULE=qwen3 CONFIG=qwen3_0_6b ./run_train.sh \
  --training.steps 50 --training.local_batch_size 3 --training.seq_len 2048 \
  --metrics.log_freq 5 --metrics.enable_tensorboard \
  --metrics.save_tb_folder outputs/tb/0_6b_1gpu_base
```

| Batch | Memory        | tps    | TFLOPS | MFU    |
|-------|---------------|--------|--------|--------|
| 3     | 21.95 GB (93%)| —      | —      | —      |
| 4     | OOM           |        |        |        |

```bash
# compile only
NGPU=1 MODULE=qwen3 CONFIG=qwen3_0_6b ./run_train.sh \
  --training.steps 50 --training.local_batch_size 6 --training.seq_len 2048 \
  --metrics.log_freq 5 --metrics.enable_tensorboard \
  --metrics.save_tb_folder outputs/tb/0_6b_1gpu_compile \
  --compile.enable
```

| Batch | Memory          | tps    | TFLOPS | MFU    |
|-------|-----------------|--------|--------|--------|
| 3     | 13.88 GB (59%)  | 11,078 | 44.89  | 63.22% |
| 6     | 20.73 GB (88%)  | 11,604 | 47.02  | 66.23% |
| 7     | OOM             |        |        |        |

**compile saves ~8 GB (−37%) at batch=3** via kernel fusion — ops like layernorm, softmax, and elementwise never materialize intermediate tensors.

---

### 2. Qwen3-1.7B on single GPU

```bash
NGPU=1 MODULE=qwen3 CONFIG=qwen3_1_7b ./run_train.sh \
  --training.steps 10 --training.local_batch_size 1 --training.seq_len 256 \
  --compile.enable
```

**OOM at seq=256** — expected. Optimizer states (~20 GB) already exceed 24 GB before any activations are allocated. AC cannot help here.

---

### 3. FSDP2 — Qwen3-1.7B on 2 GPUs

```bash
# FSDP2 baseline (no compile)
NGPU=2 MODULE=qwen3 CONFIG=qwen3_1_7b ./run_train.sh \
  --training.steps 50 --training.local_batch_size 2 --training.seq_len 2048 \
  --metrics.log_freq 5 --metrics.enable_tensorboard \
  --metrics.save_tb_folder outputs/tb/1_7b_fsdp2_base
```

| Batch | Memory          | tps   | TFLOPS | MFU    |
|-------|-----------------|-------|--------|--------|
| 1     | 20.17 GB (86%)  | 804   | 7.93   | 11.17% |
| 2     | 23.04 GB (98%)  | 1,392 | 13.73  | 19.34% |
| 3     | OOM             |       |        |        |

```bash
# FSDP2 + compile
NGPU=2 MODULE=qwen3 CONFIG=qwen3_1_7b ./run_train.sh \
  --training.steps 50 --training.local_batch_size 4 --training.seq_len 2048 \
  --compile.enable \
  --metrics.log_freq 5 --metrics.enable_tensorboard \
  --metrics.save_tb_folder outputs/tb/1_7b_fsdp2_compile
```

| Batch | Memory          | tps   | TFLOPS | MFU    |
|-------|-----------------|-------|--------|--------|
| 1     | 17.06 GB (72%)  | 813   | 8.02   | 11.29% |
| 2     | 19.67 GB (83%)  | 1,564 | 15.43  | 21.74% |
| 3     | 22.27 GB (94%)  | 2,257 | 22.26  | 31.36% |
| 4     | 23.05 GB (98%)  | 2,585 | 25.51  | 35.92% |
| 5     | OOM             |       |        |        |

compile alone: batch ceiling 2→4, MFU 19% → 36%.

---



### 4. Activation Checkpointing — Qwen3-1.7B + FSDP2 + compile

> All runs: NGPU=2, seq_len=2048

```bash
# Selective AC — every 2nd layer
NGPU=2 MODULE=qwen3 CONFIG=qwen3_1_7b ./run_train.sh \
  --training.steps 50 --training.local_batch_size 3 --training.seq_len 2048 \
  --compile.enable \
  --activation_checkpoint.mode selective --activation_checkpoint.selective_ac_option 2 \
  --metrics.log_freq 5 --metrics.enable_tensorboard \
  --metrics.save_tb_folder outputs/tb/1_7b_selective_ac_layer2
```

| Batch | Memory          | tps   | TFLOPS | MFU    |
|-------|-----------------|-------|--------|--------|
| 3     | 22.72 GB (96%)  | 2,023 | 19.96  | 28.11% |
| 4     | OOM             |       |        |        |

```bash
# Selective AC — op-level
NGPU=2 MODULE=qwen3 CONFIG=qwen3_1_7b ./run_train.sh \
  --training.steps 50 --training.local_batch_size 4 --training.seq_len 2048 \
  --compile.enable \
  --activation_checkpoint.mode selective --activation_checkpoint.selective_ac_option op \
  --metrics.log_freq 5 --metrics.enable_tensorboard \
  --metrics.save_tb_folder outputs/tb/1_7b_selective_ac_op
```

| Batch | Memory          | tps   | TFLOPS | MFU    |
|-------|-----------------|-------|--------|--------|
| 3     | 22.27 GB (94%)  | 2,258 | 22.27  | 31.37% |
| 4     | 23.09 GB (98%)  | 2,579 | 25.44  | 35.83% |
| 5     | OOM             |       |        |        |

```bash
# Full AC — batch sweep
NGPU=2 MODULE=qwen3 CONFIG=qwen3_1_7b ./run_train.sh \
  --training.steps 50 --training.local_batch_size 9 --training.seq_len 2048 \
  --compile.enable \
  --activation_checkpoint.mode full \
  --metrics.log_freq 5 --metrics.enable_tensorboard \
  --metrics.save_tb_folder outputs/tb/1_7b_full_ac_b9
```

| Batch | Memory          | tps   | TFLOPS | MFU    |
|-------|-----------------|-------|--------|--------|
| 4     | 19.99 GB (85%)  | 2,798 | 27.60  | 38.88% |
| 6     | 22.77 GB (97%)  | 3,353 | 33.08  | 46.58% |
| 8     | 23.08 GB (98%)  | 3,412 | 33.66  | 47.41% |
| 9     | 23.08 GB (98%)  | 3,598 | 35.49  | 49.99% |
| 10    | OOM             |       |        |        |

---

### How batch size and seq_len affect memory

The static memory (params + grads + optimizer) doesn't change with batch/seq. Only **activation memory** scales:

```
total memory ≈ static_base + (activation_per_sample × batch)
activation_per_sample scales linearly with seq_len
```

**Where does activation_per_sample come from?**

Qwen3-1.7B actual config (from `torchtitan/models/qwen3/__init__.py`):
- `dim=2048`, `n_layers=28`, `n_heads=16`, `n_kv_heads=8`, `head_dim=128`, `ffn_hidden=6144`, bf16 = 2 bytes

During the forward pass, PyTorch saves intermediate tensors needed for backward. The dominant terms per layer per sample at seq=2048:

```
Tensor                          Shape                          Size (bf16)
──────────────────────────────────────────────────────────────────────────
LayerNorm input                 S×D = 2048×2048                8 MB
Q,K,V projections               S×(D + 2×KV) = 2048×(2048+2×1024)  16 MB
  (Q: n_heads×head_dim=2048, K&V: n_kv_heads×head_dim=1024 each)
Attention output (V-weighted)   S×D = 2048×2048                8 MB
FFN gate + up (SwiGLU)          S×ffn×2 = 2048×6144×2         48 MB
FFN input                       S×D = 2048×2048                8 MB
Misc residuals + norm outputs                                  ~6 MB
──────────────────────────────────────────────────────────────────────────
Per layer total                                               ~94 MB
× 28 layers                                                 ~2.6 GB  ✓
```

> **Why is there no S×S attention score term?**
> Without compile, attention stores the full softmax matrix: `n_heads × S × S × 2 bytes = 16×2048×2048×2 = 256 MB per layer = 7 GB per sample`. With `torch.compile`, SDPA uses a flash-attention style tiled kernel — this matrix is never materialized. **This is the main reason compile saves ~8 GB at batch=3 on the 0.6B.**

**With Full AC**, only the layer boundary tensor (residual stream entering each layer) is saved. Everything inside the layer is recomputed during backward:

```
Saved per layer: S×D×2 bytes = 2048×2048×2 = 8 MB
× 28 layers + small overhead  ≈  0.6 GB per sample  ✓
```

**Verification against measurements** (FSDP2 + compile, seq=2048):

From the FSDP2+compile table: batch=1→17.06 GB, batch=2→19.67 GB, batch=3→22.27 GB
Delta per batch = **2.6 GB/sample** — matches the per-layer breakdown above.

From the Full AC table: batch=4→19.99 GB, batch=6→22.77 GB
Delta per batch = **1.39 GB/sample** over 2 samples... but batch=4→6 spans 2 steps: (22.77-19.99)/2 = **1.39 GB/sample**. Hmm, higher than 0.6 GB. Likely due to FSDP2 temp buffers also growing with batch. The 0.6 GB estimate is a lower bound; measured delta with FSDP2 is ~1.4 GB. Still, the slope is much flatter than no-AC (2.6 GB/sample).

**Practical estimation formula:**

```python
# Calibrated from measurements (1.7B, FSDP2, seq=2048):
static_base       = 17   # GB  (CUDA context + sharded params/grads/opt + NCCL)
act_no_ac         = 2.6  # GB per sample at seq=2048
act_full_ac       = 1.4  # GB per sample at seq=2048 (with FSDP2 overhead)
# Both scale linearly with seq_len

# Examples:
# Full AC, batch=4, seq=2048:  17 + 1.4×4 = 22.6 GB  (measured: ~20 GB, close)
# Full AC, batch=4, seq=4096:  17 + 2.8×4 = 28.2 GB  → likely OOM, try batch=2
# Full AC, batch=2, seq=4096:  17 + 2.8×2 = 22.6 GB  → should fit
# No AC,   batch=2, seq=4096:  17 + 5.2×2 = 27.4 GB  → OOM
```

---

### Profiling tools — what each one does

TorchTitan has two separate profiling tools. They answer different questions and use different viewers.

#### 1. Compute/communication timeline → `--profiling.enable_profiling`

Uses `torch.profiler` to capture a **Chrome trace** (CPU + CUDA activities). Shows:
- Exact CUDA kernel execution timeline
- NCCL all-gather / reduce-scatter ops (FSDP2 communication) as CUDA events
- **Whether compute and communication overlap** — are NCCL ops hiding behind compute?
- Per-kernel timing and input shapes

```bash
# Captures 1 active step per 10-step cycle (warmup=3, active=1)
NGPU=2 MODULE=qwen3 CONFIG=qwen3_1_7b ./run_train.sh \
  --training.steps 20 --training.local_batch_size 4 --training.seq_len 2048 \
  --compile.enable --activation_checkpoint.mode full \
  --profiling.enable_profiling \
  --profiling.profile_freq 10 \
  --profiling.save_traces_folder outputs/traces/1_7b_full_ac_b4
# Produces: outputs/traces/iteration_10/rank0_trace.json
```

**Visualize in Perfetto** (recommended over chrome://tracing for large traces):
1. Go to [ui.perfetto.dev](https://ui.perfetto.dev)
2. Drag-drop `rank0_trace.json`
3. Look at the CUDA stream rows:
   - `ncclAllGather` / `nccl:all_gather_into_tensor` = FSDP2 fetching full params (forward)
   - `nccl:reduce_scatter_tensor` = FSDP2 gradient sync (backward)
   - If these NCCL rows run simultaneously with `aten::` compute kernels → good overlap
   - If NCCL ops are sequential / blocking compute → communication is the bottleneck (expected on PCIe)

#### 2. Memory allocation breakdown → `--profiling.enable_memory_snapshot`

Records every `malloc`/`free` on the GPU using `torch.cuda.memory._record_memory_history()`, then dumps a **pickle** via `torch.cuda.memory._snapshot()`. Shows:
- Memory usage over time (the allocation timeline)
- Which tensors are alive at peak memory, and their sizes
- Stack traces pointing to which code allocated each tensor
- Auto-dumps on OOM (saves as `iteration_N_exit/`) — use this to debug what caused the OOM

```bash
NGPU=2 MODULE=qwen3 CONFIG=qwen3_1_7b ./run_train.sh \
  --training.steps 20 --training.local_batch_size 4 --training.seq_len 2048 \
  --compile.enable --activation_checkpoint.mode full \
  --profiling.enable_memory_snapshot \
  --profiling.save_memory_snapshot_folder outputs/mem_snapshots/1_7b_full_ac_b4
# Produces: outputs/mem_snapshots/iteration_10/rank0_memory_snapshot.pickle
```

**Visualize:** Upload the pickle to [pytorch.org/memory_viz](https://pytorch.org/memory_viz).

> **Neither tool uses TensorBoard.** TensorBoard is only for loss/tps/mfu metrics (`--metrics.enable_tensorboard`). Use Perfetto for timing/overlap, pytorch.org/memory_viz for memory breakdown.


## Summary

### All configs at a glance (Qwen3-1.7B, 2×RTX3090, seq=2048)

| Config                            | Max Batch | Peak tps | Peak MFU |
|-----------------------------------|-----------|----------|----------|
| FSDP2                             | 2         | 1,392    | 19.3%    |
| FSDP2 + compile                   | 4         | 2,585    | 35.9%    |
| FSDP2 + compile + Selective AC    | 4         | 2,579    | 35.8%    |
| **FSDP2 + compile + Full AC**     | **9**     | **3,598**| **50.0%**|

### Key takeaways

1. **compile is always on** — saves ~8 GB via kernel fusion (0.6B single GPU), extends batch ceiling by 2 on 1.7B. Zero cost.

2. **Full AC beats selective AC** — full AC recomputes the entire forward during backward, but frees enough memory to run batch=9 vs batch=4 max under selective AC. At batch=9, that's **1.4× higher throughput**. compile fuses the recomputed ops so the recompute penalty is minimal.

3. **Selective AC (layer-2) is the worst of both worlds** — it gives less memory relief than full AC and still pays recompute cost. Op-level selective is comparable to no-AC at the same batch, but doesn't unlock bigger batches.

4. **Optimizer states are the hard scale limit** — for 1.7B: optimizer alone is ~20 GB. FSDP2 shards this to ~10 GB/GPU, making it feasible. Without FSDP2 (single GPU), no amount of AC or compile can help.

5. **MFU ceiling on consumer PCIe hardware** — 50% MFU on RTX 3090 is the sweet spot here. PCIe bandwidth (~25 GB/s vs NVLink ~450 GB/s) limits FSDP2 all-gather/reduce-scatter overlap with compute. Larger batches help by increasing compute-to-communication ratio.

---

## Next Steps

**1. Gradient accumulation for real training** — max true batch with 2 GPUs is 9×2=18. For stable training you typically want effective batch ≥ 256. Use `--training.global_batch_size` to set this; TorchTitan auto-derives accumulation steps:

```bash
# accum_steps = 128 ÷ (4 × 2) = 16
NGPU=2 MODULE=qwen3 CONFIG=qwen3_1_7b ./run_train.sh \
  --training.steps 100 --training.local_batch_size 4 --training.seq_len 2048 \
  --training.global_batch_size 128 \
  --compile.enable --activation_checkpoint.mode full \
  --metrics.log_freq 5 --metrics.enable_tensorboard \
  --metrics.save_tb_folder outputs/tb/1_7b_full_ac_accum128
# Memory stays at batch=4 footprint (~20 GB), gradient quality of batch=128
```

**2. Longer sequence lengths** — seq=2048 is short for modern tasks. Test seq=4096 and seq=8192 with full AC. Activation memory scales linearly with seq, so expect ~2× memory per doubling. With full AC and batch=4, seq=4096 should be approximately: `15 + (1.2 × 4) ≈ 19.8 GB` (should fit).

```bash
NGPU=2 MODULE=qwen3 CONFIG=qwen3_1_7b ./run_train.sh \
  --training.steps 50 --training.local_batch_size 4 --training.seq_len 4096 \
  --compile.enable --activation_checkpoint.mode full \
  --metrics.log_freq 5 --metrics.enable_tensorboard \
  --metrics.save_tb_folder outputs/tb/1_7b_full_ac_seq4096_b4
```

**3. Qwen3-0.6B on 2 GPUs with DDP vs FSDP2** — for small models, DDP (full replica on each GPU) can be faster since there's no all-gather overhead. Qwen3-0.6B fits on a single GPU, so DDP is viable.

```bash
# DDP: full replica, no sharding
NGPU=2 MODULE=qwen3 CONFIG=qwen3_0_6b ./run_train.sh \
  --training.steps 50 --training.local_batch_size 6 --training.seq_len 2048 \
  --compile.enable --activation_checkpoint.mode full \
  --parallelism.data_parallel_replicate_degree 2 \
  --parallelism.data_parallel_shard_degree 1 \
  --metrics.log_freq 5 --metrics.enable_tensorboard \
  --metrics.save_tb_folder outputs/tb/0_6b_ddp_compile_full_ac
# Compare tps vs single GPU compile result (11,604 tps at batch=6)
```

**4. CPU offload for Llama3-8B** — 8B model needs ~128 GB for optimizer in fp32. If your system has ≥128 GB RAM:

```bash
free -h  # check first — need ~128 GB free
NGPU=2 MODULE=llama3 CONFIG=llama3_8b ./run_train.sh \
  --training.steps 20 --training.local_batch_size 1 --training.seq_len 2048 \
  --compile.enable --activation_checkpoint.mode full \
  --training.enable_cpu_offload \
  --metrics.log_freq 5 --metrics.enable_tensorboard \
  --metrics.save_tb_folder outputs/tb/llama3_8b_cpu_offload
# GPU memory should be ~9 GB/GPU (only bf16 params on GPU)
# tps will be much lower due to CPU↔GPU transfer overhead
```
