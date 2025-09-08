#!/usr/bin/env python3
"""
Test script for PLIC-p framework implementation.
This script verifies that the PLIC-p framework works correctly and 
that p=0 case matches the original GSPO implementation.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional

# Import the necessary modules
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from verl.workers.config.actor import ActorConfig, PolicyLossConfig
from verl.trainer.ppo.core_algosori import compute_policy_loss_gspo, compute_policy_loss_plic_p


def create_test_data(batch_size=4, seq_len=8, vocab_size=1000):
    """Create test data for policy loss computation."""
    torch.manual_seed(42)
    
    # Create log probabilities
    old_log_prob = torch.randn(batch_size, seq_len) * 0.1
    log_prob = old_log_prob + torch.randn(batch_size, seq_len) * 0.05
    
    # Create advantages (token-level)
    advantages = torch.randn(batch_size, seq_len) * 0.5
    
    # Create response mask (some tokens are valid, some are padding)
    response_mask = torch.ones(batch_size, seq_len)
    for i in range(batch_size):
        # Randomly mask some tokens at the end
        mask_start = np.random.randint(seq_len//2, seq_len)
        response_mask[i, mask_start:] = 0
    
    return old_log_prob, log_prob, advantages, response_mask


def create_test_config(plic_p=0.0):
    """Create test configuration."""
    policy_loss_config = PolicyLossConfig(
        loss_mode="plic_p",
        plic_p=plic_p
    )
    
    config = ActorConfig(
        strategy="fsdp",
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        policy_loss=policy_loss_config
    )
    
    return config


def test_plic_p_p0_vs_gspo():
    """Test that PLIC-p with p=0 matches original GSPO."""
    print("Testing PLIC-p with p=0 vs original GSPO...")
    
    # Create test data
    old_log_prob, log_prob, advantages, response_mask = create_test_data()
    
    # Create configs
    gspo_config = ActorConfig(
        strategy="fsdp",
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        policy_loss=PolicyLossConfig(loss_mode="gspo")
    )
    
    plic_p_config = create_test_config(plic_p=0.0)
    
    # Compute losses
    gspo_loss, gspo_clipfrac, gspo_kl, gspo_clipfrac_lower = compute_policy_loss_gspo(
        old_log_prob, log_prob, advantages, response_mask, config=gspo_config
    )
    
    plic_p_loss, plic_p_clipfrac, plic_p_kl, plic_p_clipfrac_lower = compute_policy_loss_plic_p(
        old_log_prob, log_prob, advantages, response_mask, config=plic_p_config
    )
    
    # Compare results
    print(f"GSPO loss: {gspo_loss.item():.6f}")
    print(f"PLIC-p (p=0) loss: {plic_p_loss.item():.6f}")
    print(f"Difference: {abs(gspo_loss.item() - plic_p_loss.item()):.8f}")
    
    # Check if they are close (allowing for small numerical differences)
    assert torch.allclose(gspo_loss, plic_p_loss, atol=1e-5), "PLIC-p with p=0 should match GSPO"
    print("✓ PLIC-p with p=0 matches original GSPO!")


def test_plic_p_different_p_values():
    """Test PLIC-p with different p values."""
    print("\nTesting PLIC-p with different p values...")
    
    # Create test data
    old_log_prob, log_prob, advantages, response_mask = create_test_data()
    
    p_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    losses = []
    
    for p in p_values:
        config = create_test_config(plic_p=p)
        loss, _, _, _ = compute_policy_loss_plic_p(
            old_log_prob, log_prob, advantages, response_mask, config=config
        )
        losses.append(loss.item())
        print(f"p={p}: loss={loss.item():.6f}")
    
    # Check that different p values give different results (except p=0 should be special)
    for i in range(1, len(losses)):
        for j in range(i+1, len(losses)):
            if p_values[i] != p_values[j]:
                assert abs(losses[i] - losses[j]) > 1e-6, f"Different p values should give different losses: p={p_values[i]} vs p={p_values[j]}"
    
    print("✓ Different p values produce different losses!")


def test_gradient_flow():
    """Test that gradients flow correctly through the PLIC-p loss."""
    print("\nTesting gradient flow...")
    
    # Create test data with requires_grad
    old_log_prob, log_prob, advantages, response_mask = create_test_data()
    log_prob.requires_grad_(True)
    
    config = create_test_config(plic_p=1.0)
    
    # Compute loss
    loss, _, _, _ = compute_policy_loss_plic_p(
        old_log_prob, log_prob, advantages, response_mask, config=config
    )
    
    # Backpropagate
    loss.backward()
    
    # Check that gradients exist and are non-zero
    assert log_prob.grad is not None, "Gradients should exist"
    assert torch.any(log_prob.grad != 0), "Gradients should be non-zero"
    
    print(f"✓ Gradients computed successfully! Max gradient: {log_prob.grad.abs().max().item():.6f}")


if __name__ == "__main__":
    print("Testing PLIC-p Framework Implementation")
    print("=" * 50)
    
    try:
        test_plic_p_p0_vs_gspo()
        test_plic_p_different_p_values()
        test_gradient_flow()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! PLIC-p framework is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
