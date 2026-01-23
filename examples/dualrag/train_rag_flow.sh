# set -x

export base_dir=$(pwd)

export SWANLAB_API_KEY=ashYc7XpX4pwEzLFrftzx
# export SWANLAB_API_HOST=https://api.bandw.top
export WANDB_API_KEY=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
export WANDB_BASE_URL=https://api.bandw.top

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

export CUDA_VISIBLE_DEVICES=0,1
GPU_LIST=${CUDA_VISIBLE_DEVICES//,/ }
n_gpus=$(echo $GPU_LIST | wc -w)

export search_url="127.0.0.1"

model_path="Qwen/Qwen3-0.6B"
max_model_len=$((1024 * 18))
max_prompt_length=$((1024 * 16))
max_response_length=$((1024 * 2))
sp=1
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp))
n=8

export method=dualrag
export retrieve_mode=local
adv=rloo
timestamp=$(date +%Y%m%d_%H%M%S)
export log_dir="${base_dir}/outputs/$(date +%Y-%m-%d/%H-%M-%S)"
experiment_name=${method}-${retrieve_mode}-${adv}-${timestamp}

python3 -m examples.dualrag.train_rag_flow \
    algorithm.adv_estimator=${adv} \
    data.train_batch_size=128 \
    data.val_batch_size=1024 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${n} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    rllm.compact_filtering.enable=False \
    rllm.compact_filtering.mask_max_prompt_length_exceeded=True \
    rllm.compact_filtering.mask_max_response_length_exceeded=True \
    rllm.compact_filtering.mask_max_turns_exceeded=False \
    rllm.compact_filtering.mask_timeout=True \
    rllm.rejection_sample.enable=False \
    rllm.rejection_sample.multiplier=1.0 \
    rllm.stepwise_advantage.enable=True \
    rllm.stepwise_advantage.mode=per_step \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb','swanlab'] \
    trainer.project_name='DualRAG' \
    trainer.experiment_name=${experiment_name} \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=${n_gpus} \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=100 \
    rllm.workflow.use_workflow=True

pkill -9 -f 'ray::WorkerDict'
chmod -R 777 ./outputs
