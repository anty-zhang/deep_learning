

```bash

python3 -m vllm.entrypoints.openai.api_server --served-model-name qwen2-72b-int4 --model /models/Qwen/Qwen2___5-14B-Instruct-GPTQ-Int4 --tensor-parallel-size 2 --port 6001 --max_model_len 20000 --gpu-memory-utilization 0.92;sleep 100000h
```