# set -x

export base_dir=$(pwd)

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export CUDA_VISIBLE_DEVICES=1

export method=dualrag
# export method=nativerag
export retrieve_mode=local

python3 -m examples.dualrag.run_rag_flow

pkill -9 -f 'ray::WorkerDict'
