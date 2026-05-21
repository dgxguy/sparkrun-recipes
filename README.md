# Qwen3.6-27B MTP Optimized

Quantized Qwen3.6-27B with native MTP heads for maximum throughput on DGX Spark (Blackwell/GB10). Optimized with `draft-mtp` speculation and Blackwell-specific tuning.

**Performance:** ~14-16 tok/s decode, 92%+ MTP acceptance rate
**Quantization:** Q8_0 (Unsloth export)
**Speculation:** draft-mtp with n-max 2
