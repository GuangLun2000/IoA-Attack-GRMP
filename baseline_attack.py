# baseline_attack.py
# Baseline attack methods for federated learning model poisoning attacks
# This module implements Fang Attack (USENIX Security '20) as a baseline comparison

import torch
import torch.nn as nn
from typing import List, Optional


def fang_attack_generate_malicious_update(
    client_id: int,
    benign_updates: List[torch.Tensor],
    other_attacker_updates: List[torch.Tensor],
    global_model_params: torch.Tensor,
    dist_bound: Optional[float],
    device: torch.device,
    stop_threshold: float = 1.0e-5
) -> torch.Tensor:
    """
    Fang Attack (USENIX Security '20) - Generate malicious update.
    
    Core Logic (preserved from original):
    1. Direction Estimation: Use attackers' before-attack updates (treated as benign) to estimate direction
    2. Binary Search: Find λ value that satisfies distance constraint (adapted from Krum selection)
    3. Generate Malicious Update: malicious = base - λ × direction
    
    Adaptation for FL+LLM:
    - Target changed from "selected by Krum" to "satisfies distance constraint"
    - Adapted for torch.Tensor and LoRA parameter format
    - Integrated with existing multi-attacker coordination mechanism
    
    Args:
        client_id: Client ID for logging
        benign_updates: List of benign client updates
        other_attacker_updates: List of other attackers' updates (for direction estimation)
        global_model_params: Global model parameters (base for attack)
        dist_bound: Distance constraint threshold (None = no constraint)
        device: Device to perform computation on
        stop_threshold: Binary search stopping threshold (default: 1e-5)
    
    Returns:
        Malicious update tensor (on CPU)
    
    Reference:
        Fang, M., Cao, X., Jia, J., & Gong, N. (2020). Local Model Poisoning Attacks to 
        Byzantine-Robust Federated Learning. USENIX Security Symposium.
    """
    # ============================================================
    # STEP 1: Get "before-attack updates" (treated as benign)
    # ============================================================
    # Priority: Use other attackers' updates if available, else use benign updates
    if len(other_attacker_updates) > 0:
        before_attack_updates = other_attacker_updates
        print(f"    [Fang Attack {client_id}] Using {len(before_attack_updates)} other attackers' updates for direction estimation")
    elif len(benign_updates) > 0:
        before_attack_updates = benign_updates
        print(f"    [Fang Attack {client_id}] Using {len(before_attack_updates)} benign updates for direction estimation")
    else:
        print(f"    [Fang Attack {client_id}] No updates available, return zero update")
        # Return zero update with same shape as global_model_params
        return torch.zeros_like(global_model_params).cpu()
        
    # ============================================================
    # STEP 2: Direction Estimation (CORE LOGIC - PRESERVED)
    # ============================================================
    # Convert to torch.Tensor and move to device
    before_attack_stack = torch.stack([
        u.to(device) if isinstance(u, torch.Tensor) else torch.tensor(u, device=device, dtype=torch.float32)
        for u in before_attack_updates
    ])  # (num_updates, param_dim)
    
    # Compute mean update
    mean_update = before_attack_stack.mean(dim=0)  # (param_dim,)
    
    # Estimate direction using sign function (CORE LOGIC - PRESERVED)
    est_direction = torch.sign(mean_update)  # (param_dim,)
    
    # ============================================================
    # STEP 3: Get Base Model (global model parameters)
    # ============================================================
    if global_model_params is None:
        raise ValueError(f"[Fang Attack {client_id}] Global model params not provided.")
    
    perturbation_base = global_model_params.to(device)
    
    # Ensure dimension match
    if perturbation_base.shape != est_direction.shape:
        raise ValueError(
            f"[Fang Attack {client_id}] Dimension mismatch: "
            f"base={perturbation_base.shape}, direction={est_direction.shape}"
        )
    
    # ============================================================
    # STEP 4: Binary Search for λ (ADAPTED: target changed from Krum selection to distance constraint)
    # ============================================================
    lambda_value = 1.0  # Initial λ
    dist_bound_val = None
    
    # Get distance constraint
    if dist_bound is not None:
        if isinstance(dist_bound, torch.Tensor):
            dist_bound_val = dist_bound.item()
        else:
            dist_bound_val = float(dist_bound)
    
    # Binary search loop (CORE LOGIC - PRESERVED, but target adapted)
    iteration = 0
    max_iterations = 50  # Prevent infinite loop
    while iteration < max_iterations:
        # Generate malicious update: base - λ × direction (CORE FORMULA - PRESERVED)
        malicious_update = perturbation_base - lambda_value * est_direction
        
        # Check distance constraint (ADAPTED: replaces Krum selection check)
        if dist_bound_val is not None:
            dist = torch.norm(malicious_update - perturbation_base)
            # If satisfies distance constraint, stop
            if dist <= dist_bound_val:
                print(f"    [Fang Attack {client_id}] Found λ={lambda_value:.6f}, dist={dist:.6f} <= {dist_bound_val:.6f} (iterations={iteration})")
                break
        else:
            # If no distance constraint, use fixed λ (simplified version)
            print(f"    [Fang Attack {client_id}] No dist_bound, using λ={lambda_value:.6f}")
            break
        
        # If λ too small, stop
        if lambda_value <= stop_threshold:
            print(f"    [Fang Attack {client_id}] λ too small ({lambda_value:.6f}), stopping")
            break
        
        # Binary search: reduce λ by half (CORE LOGIC - PRESERVED)
        lambda_value *= 0.5
        iteration += 1
    
    if iteration >= max_iterations:
        print(f"    [Fang Attack {client_id}] WARNING: Binary search reached max iterations, using λ={lambda_value:.6f}")
    
    # ============================================================
    # STEP 5: Generate Final Malicious Update
    # ============================================================
    final_malicious_update = perturbation_base - lambda_value * est_direction
    
    # Ensure distance constraint is satisfied (if specified)
    if dist_bound_val is not None:
        dist = torch.norm(final_malicious_update - perturbation_base)
        if dist > dist_bound_val:
            # Scale to satisfy constraint
            scale = dist_bound_val / (dist + 1e-10)  # Add small epsilon to avoid division by zero
            final_malicious_update = perturbation_base + scale * (final_malicious_update - perturbation_base)
            final_dist = torch.norm(final_malicious_update - perturbation_base)
            print(f"    [Fang Attack {client_id}] Scaled update to satisfy constraint: dist={final_dist:.6f}")
    
    # Move back to CPU (consistent with codebase style)
    final_malicious_update = final_malicious_update.cpu()
    
    return final_malicious_update
