# llama.cpp MTP Benchmark — Qwen3.6-27B-MTP on DGX Spark (GB10)

## Setup

- **Host**: sparky2 (`admin@sparky2`)
- **Model**: `Qwen3.6-27B-MTP-Q8_0.gguf` (Unsloth quantization with native draft heads)
- **Model path**: `~/models/Qwen3.6-27B-MTP-Q8_0.gguf` (28GB)
- **Binary**: `~/llama.cpp/build/bin/llama-server` (local ARM64/CUDA 13 build)
- **CUDA libs**: `/usr/local/cuda-13.0/targets/sbsa-linux/lib/`
- **GPU**: NVIDIA GB10 (124GB, CUDA 13.0.3)
- **Spec type**: `draft-mtp` (native MTP heads, NOT ngram-cache)
- **Port**: 8090

## Launch Command

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib:${LD_LIBRARY_PATH}
~/llama.cpp/build/bin/llama-server \
  -m ~/models/Qwen3.6-27B-MTP-Q8_0.gguf \
  -ngl 999 \
  -c 32000 \
  --host 0.0.0.0 \
  --port 8090 \
  --jinja \
  --temp 0.6 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --spec-draft-p-min 0.75
```

## Key Notes

- Running as **direct host binary** (not Docker, not sparkrun). Docker GPU passthrough failed due to missing NVIDIA runtime config on sparky2. sparkrun v0.2.35 has a hardcoded `hf_hub_download()` bug that blocks local model paths.
- MTP flags `--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.75` are required to utilize the native draft heads baked into the model. Using `ngram-cache` bypasses the native heads entirely.
- Port 8090 was used because 8080 had stale llama-server processes.

## Benchmark Results

### Run: Qwen3.6-27B-MTP-Q8_0, draft-mtp spec

| Metric | Value |
|--------|-------|
| Throughput | **14.13 tok/s** |
| Draft Acceptance Rate | **95.0%** |
| Total Generation Time | 3.35s |
| Completion Tokens | 2,048 |
| Prompt Tokens | 44 |
| Draft Tokens Generated | 1,184 |
| Draft Tokens Accepted | 1,125 |
| Graphs Reused | 457 |

### Comparison: draft-mtp vs ngram-cache

| Spec Type | Throughput | Acceptance Rate |
|-----------|-----------|-----------------|
| `ngram-cache` | ~7.8 tok/s | ~30.8% |
| `draft-mtp` | **14.13 tok/s** | **95.0%** |

The native MTP speculative decoding delivers ~80% higher throughput and ~3x higher acceptance rate compared to ngram-cache.

### MTP Initialization Log

```
common_speculative_impl_draft_mtp: adding speculative implementation 'draft-mtp'
- n_max=2, n_min=0, p_min=0.75, n_embd=5120, backend_sampling=1
- gpu_layers=-1, cache_k=f16, cache_v=f16, ctx_tgt=yes, ctx_dft=yes, devices=[default]
speculative decoding context initialized
```

## Troubleshooting

- **Port conflicts**: Check with `ss -tlnp | grep 8090` and `kill -9 $(pgrep -f llama-server)`
- **Docker GPU failure**: `/etc/docker/daemon.json` missing NVIDIA runtime — not writable without root. Use direct host binary instead.
- **sparkrun local path bug**: `download_model()` unconditionally calls `hf_hub_download()` even for local paths. No fix in v0.2.35/v0.2.36. Bypass by using direct binary.
- **SSH timeouts**: Use `ConnectTimeout=5` and avoid pipes in remote commands on sparky2.

## Files

- Model: `~/models/Qwen3.6-27B-MTP-Q8_0.gguf`
- Binary: `~/llama.cpp/build/bin/llama-server`
- Log: `/tmp/llama-mtp-27b.log`
