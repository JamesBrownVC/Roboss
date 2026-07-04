# Feasibility judge env (Qwen-VL)

Isolated env for VLM-based pre-analysis QA. Real mode requires vLLM + Qwen2.5-VL.

```bash
micromamba create -n v2r-feasibility python=3.11
micromamba run -n v2r-feasibility pip install vllm  # operator installs pinned version
```

Fallback: set `V2R_JUDGE_API` to an OpenAI-compatible endpoint for structured JSON judge output.
