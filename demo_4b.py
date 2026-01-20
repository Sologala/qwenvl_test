VLLM_USE_MODELSCOPE=true vllm serve "/home/model/Qwen3-VL-8B-Instruct-4bit-GPTQ" \
    --host 0.0.0.0 \
    --port 3001 \
    --gpu-memory-utilization 0.6 \
    --served-model-name "Qwen3-VL-8B-Instruct-4bit-GPTQ" \
    --quantization "gptq_marlin" \
    --dtype "float16" \
    --max-model-len 32768 \
    --limit-mm-per-prompt '{"image":1,"video":0}'
