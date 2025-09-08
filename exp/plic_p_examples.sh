#!/bin/bash
# ===================================================================
# PLIC-p Framework Examples
# ===================================================================
# This file contains example configurations for different PLIC-p values.
# Copy the desired configuration to your training script.

# ===================================================================
# Example 1: Original GSPO (Geometric Mean) - p=0
# ===================================================================
echo "Example 1: Original GSPO (p=0)"
echo "actor_rollout_ref.actor.policy_loss.loss_mode=plic_p \\"
echo "actor_rollout_ref.actor.policy_loss.plic_p=0.0 \\"
echo ""

# ===================================================================
# Example 2: Arithmetic Mean - p=1
# ===================================================================
echo "Example 2: Arithmetic Mean (p=1)"
echo "actor_rollout_ref.actor.policy_loss.loss_mode=plic_p \\"
echo "actor_rollout_ref.actor.policy_loss.plic_p=1.0 \\"
echo ""

# ===================================================================
# Example 3: Quadratic Mean (RMS) - p=2
# ===================================================================
echo "Example 3: Quadratic Mean (p=2)"
echo "actor_rollout_ref.actor.policy_loss.loss_mode=plic_p \\"
echo "actor_rollout_ref.actor.policy_loss.plic_p=2.0 \\"
echo ""

# ===================================================================
# Example 4: Harmonic Mean - p=-1
# ===================================================================
echo "Example 4: Harmonic Mean (p=-1)"
echo "actor_rollout_ref.actor.policy_loss.loss_mode=plic_p \\"
echo "actor_rollout_ref.actor.policy_loss.plic_p=-1.0 \\"
echo ""

# ===================================================================
# Example 5: Fractional Power - p=0.5
# ===================================================================
echo "Example 5: Fractional Power (p=0.5)"
echo "actor_rollout_ref.actor.policy_loss.loss_mode=plic_p \\"
echo "actor_rollout_ref.actor.policy_loss.plic_p=0.5 \\"
echo ""

# ===================================================================
# Comparison with other loss modes
# ===================================================================
echo "Comparison with other loss modes:"
echo ""
echo "# Original GSPO with A-RSIC:"
echo "actor_rollout_ref.actor.policy_loss.loss_mode=gspo \\"
echo ""
echo "# Pure A-RSIC:"
echo "actor_rollout_ref.actor.policy_loss.loss_mode=arsic \\"
echo ""
echo "# Vanilla PPO:"
echo "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \\"
