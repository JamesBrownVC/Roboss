# Semantics stage environment (Stage G)

**Tool:** Qwen2.5-VL (7B+) via vLLM — temperature 0, JSON-schema outputs

**Pinned weights:** `Qwen/Qwen2.5-VL-7B-Instruct` (record sha256 in manifest)

Subtask segmentation: changepoints from hand aperture + contact transitions, then VLM skill label from `config/verbs.yaml`.

## Install

```bash
micromamba create -n semantics python=3.10 -y
micromamba activate semantics
pip install vllm transformers qwen-vl-utils
# huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
```

## Invocation

```bash
micromamba run -n semantics python envs/semantics/tool_entry.py \
  --workspace workspaces/{episode_id} \
  --verbs config/verbs.yaml \
  --temperature 0
```

## Switch synthetic → real

```yaml
stages:
  semantics: {enabled: true, mode: real, env: semantics}
```

Requires CUDA Linux with sufficient VRAM for 7B VLM.
