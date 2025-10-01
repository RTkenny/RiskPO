set -x

# (0.1,0.8) (0.3,0.8) (0.4,0.8) (0.2,0.6) (0.2,0.7) (0.2,0.9) | (0.2,0.8)

# 1 1.1 1.5 1.6 2.0

gsm8k_train_path=$HOME/autodl-tmp/dataset/gsm8k/train.parquet
gsm8k_test_path=$HOME/autodl-tmp/dataset/gsm8k/test.parquet
math_train_path=$HOME/autodl-tmp/dataset/math/train.parquet
math_test_path=$HOME/autodl-tmp/dataset/math/test.parquet

train_files="['$gsm8k_train_path','$math_train_path']"
test_files="['$gsm8k_test_path','$math_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo_bundle_RVaR_quantile_tracking \
    algorithm.quantile_down=0.2 \
    algorithm.quantile_up=0.9 \
    algorithm.bundle_size=5 \
    algorithm.lr_q=0.1 \
    algorithm.credit_assign_mode='std' \
    algorithm.use_q_track_mode='track' \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_mixing_risk_measure=True \
    algorithm.natural_baseline_base=True \
    algorithm.natural_baseline_adv=True \
    algorithm.w_mix=1.5 \
    algorithm.use_mean_as_baseline=False \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='verl_grpo_RVaR_example_math' \
    trainer.experiment_name='qwen2_5_1.5b_bundle_1.5_MVaR_0.2-0.9_credit_std_bundle_2_quantile_tracking_track' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_training_steps=200 \
    trainer.total_epochs=15 $@