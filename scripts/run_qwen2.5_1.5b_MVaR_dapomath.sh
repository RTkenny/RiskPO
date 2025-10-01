set -x

dapo_train_path=$HOME/dataset/dapo_aime2024/dapo-math-17k.parquet
dapo_test_path=$HOME/dataset/dapo_aime2024/aime-2024.parquet

train_files="['$dapo_train_path']"
test_files="['$dapo_test_path']"

max_prompt_length=$((1024*1))
max_response_length=$((1024*3))
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 1))
overlong_penalty_factor=1.0

sp_size=4
train_batch_size=512
ppo_mini_batch_size=128
ppo_micro_batch_size_per_gpu=64
rollout_n=10
ref_log_prob_micro_batch_size_per_gpu=16
use_dynamic_bsz=False
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=4

use_kl_loss=False
kl_loss_coef=0.001
kl_loss_type=low_var_kl

n_gpus_per_node=8

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo_bundle_RVaR_quantile_tracking \
    algorithm.quantile_down=0.2 \
    algorithm.quantile_up=0.8 \
    algorithm.bundle_size=5 \
    algorithm.lr_q=0.1 \
    algorithm.credit_assign_mode='sum-mean' \
    algorithm.use_q_track_mode='track' \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_mixing_risk_measure=True \
    algorithm.w_mix=1.5 \
    algorithm.use_mean_as_baseline=False \
    algorithm.natural_baseline_base=True \
    algorithm.natural_baseline_adv=False \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ref_log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ref_log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='verl_grpo_RVaR_example_dapomath' \
    trainer.experiment_name='qwen2_5_1.5b_bundle_MVaR_mix_1.5' \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_training_steps=500 \
    trainer.total_epochs=30 $@