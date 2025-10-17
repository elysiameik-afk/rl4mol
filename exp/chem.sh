#!/bin/bash
set -x

# ===================================================================
# Chemistry Molecule Generation RL Training Script
# ===================================================================
# This script is adapted from gspo.sh for EGFR inhibitor molecule 
# generation using reinforcement learning.
#
# Key modifications:
# - Data paths: chemistry molecule dataset
# - Model path: chemistry SFT model
# - Reward manager: chem (chemistry-specific reward function)
#
# Algorithm and other parameters remain the same as gspo.sh
# ===================================================================

# Activate environment (uncomment if needed)
# eval "$(conda shell.bash hook)"
# conda deactivate
# conda activate verl

# Set environment variables
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HYDRA_FULL_ERROR=1

# ===================================================================
# CONFIGURATION - MODIFY THESE PATHS AS NEEDED
# ===================================================================

# Data paths (modify to your actual data directory)
TRAIN_DATA="/root/autodl-tmp/verl97/verl/data/chem_rl/train.parquet"
VAL_DATA="/root/autodl-tmp/verl97/verl/data/chem_rl/val.parquet"

# Model path (modify to your chemistry SFT model path)
MODEL_PATH="/root/autodl-tmp/rlmodels"

# Checkpoint directory (modify to your desired checkpoint location)
CKPT_DIR="/root/autodl-tmp/verl97/verl/ckpts/chem/gspo_1"

# Project name for logging (modify as needed)
PROJECT_NAME="Chemistry-EGFR-RL"
EXPERIMENT_NAME="gspo_1"

# Response length configuration
MAX_RESPONSE_LENGTH=512

# ===================================================================
# PLIC-p Framework Configuration
# ===================================================================
# This script uses the PLIC-p framework with configurable power parameter p:
# - p=0.0: Original GSPO with geometric mean (default)
# - p=1.0: Arithmetic mean
# - p=2.0: Quadratic mean (RMS)
# - p=-1.0: Harmonic mean
#
# To change the power parameter, modify the plic_p value below:
# actor_rollout_ref.actor.policy_loss.plic_p=1.0
# ===================================================================

# ===================================================================
# KL Divergence Configuration (IMPORTANT for diversity)
# ===================================================================
# KL divergence constraint prevents the model from deviating too much
# from the SFT model, which helps maintain output diversity and prevents
# mode collapse (repeatedly generating the same high-scoring molecule).
#
# - use_kl_loss=True: Enable KL divergence constraint
# - kl_loss_coef: Strength of the constraint
#   - 0.05-0.1: Light constraint (recommended starting point)
#   - 0.1-0.2: Medium constraint (good balance)
#   - 0.2-0.5: Strong constraint (if severe mode collapse)
#
# Current setting: 0.1 (recommended default)
# If still seeing repetitive outputs, increase to 0.15 or 0.2
# If reward improvement is too slow, decrease to 0.05
# ===================================================================

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss.loss_mode=plic_p \
    actor_rollout_ref.actor.policy_loss.plic_p=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.0003 \
    actor_rollout_ref.actor.clip_ratio_high=0.0004 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.1 \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=16 \
    data.val_batch_size=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.shuffle=true \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.entropy_checkpointing=true \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-mean" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    reward_model.reward_manager=chem \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=false \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=500 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${CKPT_DIR} \
    trainer.critic_warmup=0 \
    trainer.save_freq=16 \
    trainer.test_freq=1 \
    trainer.total_epochs=4 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.logger='["console","wandb"]' \
    trainer.val_before_train=false \
    trainer.resume_mode=auto \
    trainer.log_val_generations=2 \
    $@

