# attack_baseline_sign_flipping.py
# Sign-Flipping Attack Implementation - Model Poisoning Baseline
#
# This module implements the Sign-flipping attack as a Model Poisoning baseline.
# It is completely isolated from the GRMP and ALIE implementations.

import torch
import numpy as np
from typing import List, Optional

# Import base Client class
from client import Client


class SignFlippingAttackerClient(Client):
    """
    Sign-Flipping Attack - Classic Byzantine FL Baseline
    
    Malicious update = -scale * mean(benign_updates).
    
    Simple heuristic: flip the sign of the mean benign update (optionally scaled).
    Commonly used as a baseline in Byzantine-robust federated learning (e.g., Krum, Trimmed Mean).
    
    This attack is:
    - Data-agnostic: Does not use local training data
    - Simple: No optimization, only mean and sign flip
    - Strong in many settings: Pushes aggregate in opposite direction
    """
    
    def __init__(self, client_id: int, model, data_manager,
                 data_indices, lr, local_epochs, alpha,
                 sign_flip_scale: float = 1.0,
                 attack_start_round: Optional[int] = None,
                 claimed_data_size: float = 1.0,
                 grad_clip_norm: float = 1.0):
        """
        Initialize Sign-Flipping attacker client.
        
        Args:
            client_id: Unique identifier for the client
            model: The neural network model (will be deep copied)
            data_manager: DataManager instance (not used for attack, kept for interface)
            data_indices: List of data indices assigned (not used for training)
            lr: Learning rate (not used, required for Client base class)
            local_epochs: Number of local epochs (not used, required for Client base class)
            alpha: Proximal coefficient (not used, required for Client base class)
            sign_flip_scale: Scale factor for flipped update; malicious = -scale * mean(benign). Default 1.0.
            attack_start_round: Round to start attack (None = start immediately)
            claimed_data_size: Data size to claim for weighted aggregation
            grad_clip_norm: Not used, kept for interface compatibility
        """
        super().__init__(client_id, model, data_loader=None, lr=lr, local_epochs=local_epochs, alpha=alpha)
        
        self.is_attacker = True
        self.attack_method = "SignFlipping"
        
        self.sign_flip_scale = sign_flip_scale
        self.attack_start_round = attack_start_round
        self.claimed_data_size = claimed_data_size
        self.grad_clip_norm = grad_clip_norm
        
        self.benign_updates = []
        self.benign_update_client_ids = []
        
        self.data_indices = data_indices or []
        
        self._flat_numel = int(self.model.get_flat_params().numel())
        self.use_lora = hasattr(self.model, 'use_lora') and self.model.use_lora
    
    def receive_benign_updates(self, updates: List[torch.Tensor],
                               client_ids: Optional[List[int]] = None):
        """Store benign updates from server."""
        self.benign_updates = [u.detach().clone().cpu() for u in updates]
        if client_ids is not None:
            self.benign_update_client_ids = client_ids.copy()
        else:
            self.benign_update_client_ids = list(range(len(updates)))
    
    def local_train(self, epochs=None) -> torch.Tensor:
        """Data-agnostic: return zero update with correct dimension."""
        return torch.zeros(self.model.get_flat_params().numel())
    
    def prepare_for_round(self, round_num: int):
        """Prepare for a new round."""
        self.set_round(round_num)
    
    def camouflage_update(self, poisoned_update: torch.Tensor) -> torch.Tensor:
        """
        Generate Sign-Flipping attack update: malicious = -scale * mean(benign_updates).
        
        Args:
            poisoned_update: Zero update from local_train (unused)
        
        Returns:
            Malicious update tensor: -sign_flip_scale * mean(benign_updates)
        """
        if self.attack_start_round is not None:
            if self.current_round < self.attack_start_round:
                return poisoned_update
        
        if not self.benign_updates:
            print(f"    [SignFlipping Attacker {self.client_id}] No benign updates, return zero update")
            return poisoned_update
        
        for idx, update in enumerate(self.benign_updates):
            if int(update.numel()) != self._flat_numel:
                raise RuntimeError(
                    f"[SignFlipping Attacker {self.client_id}] Benign update dimension mismatch: "
                    f"update[{idx}] has {update.numel()} params, expected {self._flat_numel} "
                    f"(LoRA mode: {self.use_lora})."
                )
        
        benign_np = [u.cpu().numpy().flatten() for u in self.benign_updates]
        mean = np.mean(np.array(benign_np), axis=0)
        malicious_np = -self.sign_flip_scale * mean
        malicious_update = torch.from_numpy(malicious_np).float()
        
        if int(malicious_update.numel()) != self._flat_numel:
            raise RuntimeError(
                f"[SignFlipping Attacker {self.client_id}] Attack vector dimension mismatch: "
                f"generated {malicious_update.numel()} params, expected {self._flat_numel}."
            )
        
        lora_info = "LoRA" if self.use_lora else "Full"
        mean_norm = float(np.linalg.norm(mean))
        mal_norm = float(torch.norm(malicious_update).item())
        print(f"    [SignFlipping Attacker {self.client_id}] Generated attack ({lora_info}): "
              f"scale={self.sign_flip_scale}, mean_norm={mean_norm:.4f}, malicious_norm={mal_norm:.4f}, "
              f"num_benign={len(self.benign_updates)}")
        
        return malicious_update
    
    def receive_attacker_updates(self, updates: List[torch.Tensor],
                                  client_ids: List[int],
                                  data_sizes: Optional[dict] = None):
        """Interface compatibility; Sign-Flipping does not use other attackers' updates."""
        pass
    
    def set_global_model_params(self, global_params: torch.Tensor):
        """Interface compatibility; not used."""
        pass
    
    def set_constraint_params(self, dist_bound: Optional[float] = None,
                              sim_center: Optional[float] = None,
                              total_data_size: Optional[float] = None,
                              benign_data_sizes: Optional[dict] = None):
        """Interface compatibility; not used."""
        pass
    
    def set_lagrangian_params(self, **kwargs):
        """Interface compatibility; not used."""
        pass
