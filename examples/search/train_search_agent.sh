# set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

# RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")

export base_dir=$(pwd)

export SWANLAB_API_KEY=ashYc7XpX4pwEzLFrftzx
# export SWANLAB_API_HOST=https://api.bandw.top
export WANDB_API_KEY=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
export WANDB_BASE_URL=https://api.bandw.top

export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=5,6
GPU_LIST=${CUDA_VISIBLE_DEVICES//,/ }
n_gpus=$(echo $GPU_LIST | wc -w)

export search_url="127.0.0.1"

model_path="Qwen/Qwen3-1.7B"
max_model_len=$((1024 * 32))
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 30))
sp=1
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp))
n=8

export method=search
export retrieve_mode=local
adv=rloo
timestamp=$(date +%Y%m%d_%H%M%S)
export log_dir="${base_dir}/outputs/$(date +%Y-%m-%d/%H-%M-%S)"
experiment_name=${method}-${retrieve_mode}-1.7b-asearcher-${adv}-${timestamp}

# Run the training script with the specified configuration
# actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \

python3 -m examples.search.train_search_agent \
    algorithm.adv_estimator=${adv} \
    data.train_batch_size=128 \
    data.val_batch_size=$((512 * 3)) \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=${n} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb','swanlab'] \
    trainer.project_name='DualRAG' \
    trainer.experiment_name=${experiment_name} \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=${n_gpus}\
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    rllm.agent.max_steps=10 \
    trainer.total_epochs=500

pkill -9 -f 'ray::WorkerDict'
chmod -R 777 ./outputs
