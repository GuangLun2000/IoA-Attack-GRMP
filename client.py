# client.py
# Provides the Client class for federated learning clients, including benign and attacker clients.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
from models import VGAE
from torch.nn.utils import stateless

# Client class for federated learning
class Client:

    def __init__(self, client_id: int, model: nn.Module, data_loader, lr, local_epochs, alpha):
        """
        Initialize a federated learning client.
        
        Args:
            client_id: Unique identifier for the client
            model: The neural network model (will be deep copied)
            data_loader: DataLoader for local training data
            lr: Learning rate for local training (must be provided, no default)
            local_epochs: Number of local training epochs per round (must be provided, no default)
            alpha: Proximal regularization coefficient α ∈ [0,1] from paper formula (1) (must be provided, no default)
        
        Note: All parameters must be explicitly provided. Default values are removed to prevent
        inconsistencies with config settings. See main.py for proper usage.
        
        Memory optimization: Model is kept in CPU by default to save GPU memory.
        It will be moved to GPU only during training (for benign clients) or proxy loss calculation (for attackers).
        """
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        # Keep model in CPU initially to save GPU memory
        # Will be moved to GPU only when needed (training or proxy loss calculation)
        self.data_loader = data_loader
        self.lr = lr
        self.local_epochs = local_epochs
        self.alpha = alpha  # Regularization coefficient α ∈ [0,1] from paper formula (1)
        # CRITICAL: Use explicit cuda:0 instead of 'cuda' to ensure device consistency
        # This prevents issues where 'cuda' and 'cuda:0' are treated as different devices
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        # Do NOT move model to GPU here - will be moved on-demand
        self.optimizer = None  # Will be created when needed
        self.current_round = 0
        self.is_attacker = False
        self._model_on_gpu = False  # Track if model is currently on GPU

    def reset_optimizer(self):
        """Reset the optimizer. Only valid when model is on GPU."""
        if self._model_on_gpu:
            # Only optimize trainable parameters (important for LoRA)
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(trainable_params, lr=self.lr)
        else:
            self.optimizer = None

    def set_round(self, round_num: int):
        """Set the current training round."""
        self.current_round = round_num

    def get_model_update(self, initial_params: torch.Tensor) -> torch.Tensor:
        """
        Calculate the model update (Current - Initial).
        
        Args:
            initial_params: Initial model parameters (flattened)
            
        Returns:
            Model update tensor (flattened, on CPU)
        
        Note: Works on both CPU and GPU models. Returns CPU tensor to save GPU memory.
        """
        current_params = self.model.get_flat_params()
        # Ensure both tensors are on the same device before subtraction
        # initial_params is on CPU, so move current_params to CPU
        if current_params.device.type == 'cuda':
            current_params = current_params.cpu()
        # Ensure initial_params is also on CPU (should already be, but double-check)
        if initial_params.device.type == 'cuda':
            initial_params = initial_params.cpu()
        update = current_params - initial_params
        return update

    def local_train(self, epochs=None) -> torch.Tensor:
        """Base local training method (to be overridden)."""
        raise NotImplementedError


# BenignClient class for benign clients
class BenignClient(Client):

    def __init__(self, client_id: int, model: nn.Module, data_loader, lr, local_epochs, alpha,
                 data_indices=None, grad_clip_norm=1.0):
        super().__init__(client_id, model, data_loader, lr, local_epochs, alpha)
        # Track assigned data indices for proper aggregation weighting
        self.data_indices = data_indices or []
        self.grad_clip_norm = grad_clip_norm

    def prepare_for_round(self, round_num: int):
        """Benign clients do not require special preparation."""
        self.set_round(round_num)

    def local_train(self, epochs=None) -> torch.Tensor:
        """Perform local training - includes proximal regularization."""
        if epochs is None:
            epochs = self.local_epochs
            
        # Move model to GPU for training
        if not self._model_on_gpu:
            self.model.to(self.device)
            self._model_on_gpu = True
            # Create optimizer when model is on GPU
            # Only optimize trainable parameters (important for LoRA)
            if self.optimizer is None:
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                self.optimizer = optim.Adam(trainable_params, lr=self.lr)
            
        self.model.train()
        # Get initial params and move to CPU to save GPU memory
        initial_params = self.model.get_flat_params().clone().cpu()
        
        # Proximal regularization coefficient (paper formula (1): α ∈ [0,1])
        mu = self.alpha

        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            pbar = tqdm(self.data_loader,
                    desc=f'Client {self.client_id} - Epoch {epoch + 1}/{epochs}',
                    leave=False)
            
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                # NewsClassifierModel returns logits directly
                logits = outputs
                
                ce_loss = nn.CrossEntropyLoss()(logits, labels)
                
                # Add proximal regularization term
                # Move initial_params to GPU temporarily for computation
                current_params = self.model.get_flat_params()
                initial_params_gpu = initial_params.to(self.device)
                proximal_term = mu * torch.norm(current_params - initial_params_gpu) ** 2
                initial_params_gpu = None  # Release GPU reference
                
                loss = ce_loss + proximal_term
                
                self.optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate update (will be on CPU)
        update = self.get_model_update(initial_params)
        
        # Move model back to CPU to free GPU memory
        self.model.cpu()
        self._model_on_gpu = False
        # Delete optimizer to free its GPU memory (Adam states)
        del self.optimizer
        self.optimizer = None
        torch.cuda.empty_cache()  # Clear CUDA cache
        
        return update

    def receive_benign_updates(self, updates: List[torch.Tensor]):
        # Benign clients do not use this method
        pass


# AttackerClient class for clients that perform attacks
class AttackerClient(Client):

    def __init__(self, client_id: int, model: nn.Module, data_manager,
                 data_indices, lr, local_epochs, alpha,
                 dim_reduction_size=10000,
                 vgae_epochs=20, vgae_lr=0.01, graph_threshold=0.5,
                 proxy_step=0.1,
                 claimed_data_size=1.0,
                 proxy_sample_size=512,
                 proxy_max_batches_opt=2,
                 proxy_max_batches_eval=4,
                 vgae_hidden_dim=32,
                 vgae_latent_dim=16,
                 vgae_dropout=0.0,
                 proxy_steps=20,
                 gsp_perturbation_scale=0.01,
                 opt_init_perturbation_scale=0.001,
                 grad_clip_norm=1.0):
        """
        Initialize an attacker client with VGAE-based camouflage capabilities.
        
        Args:
            client_id: Unique identifier for the client
            model: The neural network model (will be deep copied)
            data_manager: DataManager instance for managing attacker data
            data_indices: List of data indices assigned to this client
            lr: Learning rate for local training (must be provided, no default)
            local_epochs: Number of local training epochs per round (must be provided, no default)
            alpha: Proximal regularization coefficient α ∈ [0,1] (must be provided, no default)
            dim_reduction_size: Dimensionality for feature reduction (default: 10000)
            vgae_epochs: Number of epochs for VGAE training (default: 20)
            vgae_lr: Learning rate for VGAE optimizer (default: 0.01)
            graph_threshold: Threshold for graph adjacency matrix binarization (default: 0.5)
            proxy_step: Step size for gradient-free ascent toward global-loss proxy (default: 0.1)
            claimed_data_size: Reported data size D'_j(t) for weighted aggregation (default: 1.0)
            proxy_sample_size: Number of samples in proxy dataset for F(w'_g) estimation (default: 512)
            proxy_max_batches_opt: Max batches for proxy loss in optimization loop (default: 2)
            proxy_max_batches_eval: Max batches for proxy loss in final evaluation (default: 4)
            vgae_hidden_dim: VGAE hidden layer dimension (default: 32, per paper)
            vgae_latent_dim: VGAE latent space dimension (default: 16, per paper)
            vgae_dropout: VGAE dropout rate (default: 0.0)
            proxy_steps: Number of optimization steps for attack objective (default: 20)
            gsp_perturbation_scale: Perturbation scale for GSP attack diversity (default: 0.01)
            opt_init_perturbation_scale: Perturbation scale for optimization initialization (default: 0.001)
            grad_clip_norm: Gradient clipping norm for training stability (default: 1.0)
        
        Note: lr, local_epochs, and alpha must be explicitly provided to ensure consistency
        with config settings. Other parameters have defaults but should be set via config in main.py.
        """
        self.data_manager = data_manager
        self.data_indices = data_indices
        
        # Store parameters first (before using them)
        self.dim_reduction_size = dim_reduction_size
        self.vgae_epochs = vgae_epochs
        self.vgae_lr = vgae_lr
        self.graph_threshold = graph_threshold
        self.proxy_step = proxy_step
        self.claimed_data_size = claimed_data_size  # For weighted aggregation (paper: D'(t))
        self.proxy_sample_size = proxy_sample_size
        self.proxy_max_batches_opt = proxy_max_batches_opt
        self.proxy_max_batches_eval = proxy_max_batches_eval
        self.vgae_hidden_dim = vgae_hidden_dim
        self.vgae_latent_dim = vgae_latent_dim
        self.vgae_dropout = vgae_dropout
        self.proxy_steps = proxy_steps
        self.gsp_perturbation_scale = gsp_perturbation_scale
        self.opt_init_perturbation_scale = opt_init_perturbation_scale
        self.grad_clip_norm = grad_clip_norm

        dummy_loader = data_manager.get_empty_loader()
        super().__init__(client_id, model, dummy_loader, lr, local_epochs, alpha)
        self.is_attacker = True

        # VGAE components
        self.vgae = None
        self.vgae_optimizer = None
        self.benign_updates = []
        self.benign_update_client_ids = []  # Track client_id for each benign update to enable weighted average calculation
        self.feature_indices = None
        
        # Data-agnostic attack: no local data usage
        self.original_business_loader = None
        self.proxy_loader = data_manager.get_proxy_eval_loader(sample_size=self.proxy_sample_size)
        
        # Formula 4 constraints parameters
        self.d_T = None  # Distance threshold for constraint (4b): d(w'_j(t), w'_g(t)) ≤ d_T
        self.gamma = None  # Upper bound for constraint (4c): Σ β'_{i,j}(t) d(w_i(t), w̄_i(t)) ≤ Γ
        self.global_model_params = None  # Store global model params for constraint (4b) (will be on GPU when needed)
        # Paper Formula (2): w'_g(t) = Σ_{i=1}^I (D_i(t)/D(t)) β'_{i,j}(t) w_i(t) + (D'_j(t)/D(t)) w'_j(t)
        self.total_data_size = None  # D(t): Total data size for aggregation weight calculation
        self.benign_data_sizes = {}  # {client_id: D_i(t)}: Data sizes for each benign client
        
        # Lagrangian dual variables (λ(t) and ρ(t) from paper)
        # Initialized in set_lagrangian_params
        self.lambda_dt = None  # λ(t): Lagrangian multiplier for constraint (4b)
        self.rho_dt = None     # ρ(t): Lagrangian multiplier for constraint (4c)
        self.use_lagrangian_dual = False  # Whether to use Lagrangian Dual mechanism
        self.lambda_lr = 0.01  # Learning rate for λ(t) update
        self.rho_lr = 0.01     # Learning rate for ρ(t) update
        # Save initial values for reset in prepare_for_round (Modification 1 and 2)
        self.lambda_init_value = None  # Save initial λ value for reset in prepare_for_round
        self.rho_init_value = None     # Save initial ρ value for reset in prepare_for_round
        self.enable_final_projection = True  # Whether to apply final projection after optimization (only for Lagrangian mode)
        
        # Track violation history for adaptive λ initialization (Optimization)
        self.last_violation = None  # Last round's constraint violation value (distance, not violation amount)
        
        # Get model parameter count (works on CPU model)
        self._flat_numel = int(self.model.get_flat_params().numel())  # Convert to Python int
        
        # Validate and adjust dim_reduction_size for LoRA mode
        # In LoRA mode, if dim_reduction_size > actual LoRA params, use all LoRA params
        # Rationale: When LoRA params are already small, using all of them is more reasonable
        # than further reducing, as it preserves information and the computation is still feasible.
        use_lora = hasattr(self.model, 'use_lora') and self.model.use_lora
        if use_lora:
            actual_lora_params = self._flat_numel
            if dim_reduction_size > actual_lora_params:
                # Auto-adjust: use all LoRA params (no further reduction needed)
                # When LoRA params are already small, using all of them is reasonable
                # and preserves more information for VGAE training
                print(f"    [Attacker {self.client_id}] Info: dim_reduction_size ({dim_reduction_size}) > LoRA params ({actual_lora_params})")
                print(f"    [Attacker {self.client_id}] Auto-adjusting dim_reduction_size to {actual_lora_params} (using all LoRA params)")
                self.dim_reduction_size = actual_lora_params
            elif dim_reduction_size == actual_lora_params:
                # Use all parameters (no reduction), which is fine
                pass
            else:
                # dim_reduction_size < actual_lora_params, which is the normal case (with reduction)
                pass

    def prepare_for_round(self, round_num: int):
        """
        Prepare for a new training round.
        
        Modification 1: Reset λ and ρ to initial values at the start of each round
        to prevent numerical instability from cross-round accumulation.
        """
        self.set_round(round_num)
        # Data-agnostic attacker keeps an empty loader
        self.data_loader = self.data_manager.get_empty_loader()

        # Modification 1: Reset Lagrangian multipliers (with adaptive initialization based on history)
        # Reason: Prevent λ and ρ from accumulating across rounds, which causes numerical instability and optimization imbalance
        # Optimization: Use adaptive λ initialization based on previous round's violation to provide better starting point
        if self.use_lagrangian_dual and self.lambda_init_value is not None and self.rho_init_value is not None:
            # Adaptive λ initialization: if last round had large violation, use larger initial λ
            if self.last_violation is not None and self.d_T is not None and self.last_violation > self.d_T * 1.5:
                # Estimate required λ based on violation magnitude and optimization steps
                # Target: λ should be large enough to suppress violation in proxy_steps iterations
                # Rough estimate: λ_needed ≈ violation / (lambda_lr * proxy_steps)
                estimated_lambda = self.last_violation / (self.lambda_lr * self.proxy_steps)
                # Use conservative estimate (50% of calculated value) to avoid over-penalization
                adaptive_lambda_init = max(self.lambda_init_value, estimated_lambda * 0.5)
                # Cap at reasonable maximum (10× initial value) to prevent excessive values
                adaptive_lambda_init = min(adaptive_lambda_init, self.lambda_init_value * 10.0)
                self.lambda_dt = torch.tensor(adaptive_lambda_init, requires_grad=False)
                if adaptive_lambda_init > self.lambda_init_value * 1.5:
                    print(f"    [Attacker {self.client_id}] Adaptive λ init: {adaptive_lambda_init:.4f} "
                          f"(based on last violation: {self.last_violation:.4f})")
            else:
                self.lambda_dt = torch.tensor(self.lambda_init_value, requires_grad=False)
            self.rho_dt = torch.tensor(self.rho_init_value, requires_grad=False)

    def receive_benign_updates(self, updates: List[torch.Tensor], client_ids: Optional[List[int]] = None):
        """
        Receive updates from benign clients.
        
        Args:
            updates: List of benign client updates
            client_ids: Optional list of client IDs corresponding to each update.
                       If None, indices will be used as client IDs (fallback for backward compatibility)
        """
        # Store detached copies on CPU to save GPU memory
        # Updates will be moved to GPU only when needed for VGAE processing
        self.benign_updates = [u.detach().clone().cpu() for u in updates]
        # Store corresponding client IDs for weighted average calculation in constraint (4c)
        if client_ids is not None:
            self.benign_update_client_ids = client_ids.copy()
        else:
            # Fallback: use indices as client IDs (for backward compatibility)
            # Note: This may not be accurate, but allows code to work without server changes
            self.benign_update_client_ids = list(range(len(updates)))

    def _select_benign_subset(self) -> List[torch.Tensor]:
        """
        Select a subset of benign updates (β selection) using 0-1 Knapsack optimization.
        
        Paper formulation (Equation 9):
        β'_{i,j}(t)^* = argmin_{β'_{i,j}(t)} Σ_{i=1}^I β'_{i,j}(t) d(w_i(t), w̄_i(t))
        s.t. Σ_{i=1}^I β'_{i,j}(t) d(w_i(t), w̄_i(t)) ≤ Γ
        β'_{i,j}(t) ∈ {0,1}
        
        This is a 0-1 Knapsack problem: minimize sum of selected distances
        subject to sum ≤ capacity (Γ).
        
        Note: Since we want to minimize the sum and the constraint is also on the sum,
        the optimal solution is to select as many items as possible while staying within capacity.
        We use a greedy approach to find an approximate optimal selection.
        
        Returns:
            List of selected benign updates (on CPU to save GPU memory)
        """
        if not self.benign_updates:
            return []
        
        # Compute distances from weighted mean for all benign updates
        # Paper definition: w̄_i(t) = Σ_{i=1}^I (D_i(t)/D(t)) w_i(t) (weighted mean, not simple mean)
        # Move to GPU only for computation, then back to CPU
        benign_updates_gpu = [u.to(self.device) for u in self.benign_updates]
        benign_stack = torch.stack([u.detach() for u in benign_updates_gpu])
        
        # Compute weighted mean: w̄_i(t) = Σ (D_i/D) w_i(t)
        if self.total_data_size is not None and len(self.benign_data_sizes) > 0 and len(self.benign_update_client_ids) > 0:
            D_total = float(self.total_data_size)
            benign_mean = torch.zeros_like(benign_stack[0])
            for idx, benign_update in enumerate(self.benign_updates):
                if idx < len(self.benign_update_client_ids):
                    client_id = self.benign_update_client_ids[idx]
                    D_i = self.benign_data_sizes.get(client_id, 1.0)
                    weight = D_i / D_total
                else:
                    # Fallback: use equal weight if client_id not available
                    weight = 1.0 / len(self.benign_updates)
                benign_mean = benign_mean + weight * benign_update.to(self.device)
        else:
            # Fallback: use simple mean if data sizes not available
            benign_mean = benign_stack.mean(dim=0)
        
        distances = torch.norm(benign_stack - benign_mean, dim=1).cpu().numpy()
        # Clean up GPU references immediately
        del benign_updates_gpu, benign_stack, benign_mean
        torch.cuda.empty_cache()
        
        # If gamma is not set, use all updates
        if self.gamma is None:
            return self.benign_updates
        
        # Convert gamma to capacity
        capacity = float(self.gamma)
        n = len(distances)
        
        # Handle edge cases
        if n == 0:
            return []
        if capacity <= 0:
            # If capacity is 0 or negative, select item with minimum distance
            min_idx = np.argmin(distances)
            return [self.benign_updates[min_idx]]
        
        # For 0-1 Knapsack with minimization objective:
        # We want to minimize sum of selected distances, subject to sum ≤ capacity
        # This is equivalent to: maximize number of items selected (or maximize sum of unselected)
        # while keeping sum of selected ≤ capacity
        
        # Greedy approach: sort by distance and select items until capacity is reached
        # This gives a good approximation and is efficient
        sorted_indices = np.argsort(distances)  # Sort by distance (ascending)
        
        selected_indices = []
        total_dist = 0.0
        
        for idx in sorted_indices:
            d = distances[idx]
            if total_dist + d <= capacity:
                selected_indices.append(idx)
                total_dist += d
            else:
                break
        
        # If no items selected (all distances > capacity), select the one with minimum distance
        if not selected_indices:
            min_idx = np.argmin(distances)
            selected_indices = [min_idx]
        
        # Return selected updates
        selected = [self.benign_updates[i] for i in sorted(selected_indices)]
        
        return selected
    
    def _get_selected_benign_indices(self) -> List[int]:
        """
        Get indices of selected benign updates (β selection).
        This is a helper method to avoid tensor comparison issues.
        """
        if not self.benign_updates:
            return []
        
        # Compute distances from weighted mean for all benign updates
        # Paper definition: w̄_i(t) = Σ_{i=1}^I (D_i(t)/D(t)) w_i(t) (weighted mean, not simple mean)
        # Move to GPU only for computation, then back to CPU
        benign_updates_gpu = [u.to(self.device) for u in self.benign_updates]
        benign_stack = torch.stack([u.detach() for u in benign_updates_gpu])
        
        # Compute weighted mean: w̄_i(t) = Σ (D_i/D) w_i(t)
        if self.total_data_size is not None and len(self.benign_data_sizes) > 0 and len(self.benign_update_client_ids) > 0:
            D_total = float(self.total_data_size)
            benign_mean = torch.zeros_like(benign_stack[0])
            for idx, benign_update in enumerate(self.benign_updates):
                if idx < len(self.benign_update_client_ids):
                    client_id = self.benign_update_client_ids[idx]
                    D_i = self.benign_data_sizes.get(client_id, 1.0)
                    weight = D_i / D_total
                else:
                    # Fallback: use equal weight if client_id not available
                    weight = 1.0 / len(self.benign_updates)
                benign_mean = benign_mean + weight * benign_update.to(self.device)
        else:
            # Fallback: use simple mean if data sizes not available
            benign_mean = benign_stack.mean(dim=0)
        
        distances = torch.norm(benign_stack - benign_mean, dim=1).cpu().numpy()
        # Clean up GPU references immediately
        del benign_updates_gpu, benign_stack, benign_mean
        torch.cuda.empty_cache()
        
        # If gamma is not set, use all updates
        if self.gamma is None:
            return list(range(len(self.benign_updates)))
        
        # Convert gamma to capacity
        capacity = float(self.gamma)
        n = len(distances)
        
        # Handle edge cases
        if n == 0:
            return []
        if capacity <= 0:
            # If capacity is 0 or negative, select item with minimum distance
            min_idx = np.argmin(distances)
            return [int(min_idx)]
        
        # Greedy approach: sort by distance and select items until capacity is reached
        sorted_indices = np.argsort(distances)  # Sort by distance (ascending)
        
        selected_indices = []
        total_dist = 0.0
        
        for idx in sorted_indices:
            d = distances[idx]
            if total_dist + d <= capacity:
                selected_indices.append(int(idx))
                total_dist += d
            else:
                break
        
        # If no items selected (all distances > capacity), select the one with minimum distance
        if not selected_indices:
            min_idx = np.argmin(distances)
            selected_indices = [int(min_idx)]
        
        return sorted(selected_indices)

    def local_train(self, epochs=None) -> torch.Tensor:
        """
        Attacker does not perform local training (data-agnostic attack).
        
        Attackers are not assigned local data, so they return zero update.
        The actual attack is generated in camouflage_update using VGAE+GSP.
        
        Returns:
            Zero update tensor on CPU (to save GPU memory)
        """
        # Attackers don't have local data, return zero update
        # Model is on CPU, so initial_params is on CPU
        initial_params = self.model.get_flat_params().clone()
        return torch.zeros_like(initial_params)  # Already on CPU

    def _get_reduced_features(self, updates: List[torch.Tensor], fix_indices=True) -> torch.Tensor:
        """
        Helper function to reduce dimensionality of updates.
        Randomly selects indices to slice the high-dimensional vector.
        
        Args:
            updates: List of update tensors to reduce
            fix_indices: If True, reuse existing feature_indices; if False, generate new ones
            
        Returns:
            Stacked reduced features tensor of shape (I, M) where I=num_updates, M=dim_reduction_size
        """
        stacked_updates = torch.stack(updates)
        # Ensure stacked_updates has valid shape
        if len(stacked_updates.shape) < 2:
            raise ValueError(f"[Attacker {self.client_id}] stacked_updates must be 2D, got shape {stacked_updates.shape}")
        shape_dim = stacked_updates.shape[1]
        if shape_dim is None:
            raise ValueError(f"[Attacker {self.client_id}] stacked_updates.shape[1] is None")
        try:
            total_dim = int(shape_dim)  # Convert to Python int
        except (TypeError, ValueError) as e:
            raise ValueError(f"[Attacker {self.client_id}] Cannot convert shape[1]={shape_dim} to int: {e}")
        
        # If update dimension is smaller than reduction target, skip reduction
        if total_dim <= self.dim_reduction_size:
            return stacked_updates
            
        # Fix feature indices at the start of each attack round to ensure training consistency within the round
        if self.feature_indices is None or not fix_indices:
            # Randomly select indices, but use client_id to ensure different attackers get different indices
            # This ensures diversity among multiple attackers
            import hashlib
            # Use client_id and total_dim to create a unique seed for each attacker
            # Ensure client_id and total_dim are valid integers
            if self.client_id is None:
                raise ValueError(f"client_id is None for attacker")
            if total_dim is None or total_dim <= 0:
                raise ValueError(f"total_dim is None or invalid: {total_dim}")
            seed_str = f"{self.client_id}_{total_dim}"
            seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16) % (2**31)
            # Use numpy random with client_id-based seed (more reliable than torch random state)
            np_rng = np.random.RandomState(seed)
            indices = np_rng.permutation(total_dim)[:self.dim_reduction_size]
            self.feature_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
            
        # Select features
        reduced_features = torch.index_select(stacked_updates, 1, self.feature_indices)
        return reduced_features

    def _flat_to_param_dict(self, flat_params: torch.Tensor, skip_dim_check: bool = False) -> Dict[str, torch.Tensor]:
        """
        Convert flat tensor to param dict for stateless.functional_call.
        
        In LoRA mode, only sets LoRA parameters (trainable parameters).
        In full fine-tuning mode, sets all parameters.
        
        Important: Handles PEFT model parameter name compatibility.
        PEFT models have nested structure (base_model.model.*), and stateless.functional_call
        may need specific parameter name formats.
        
        Args:
            flat_params: Flattened parameter tensor (LoRA params in LoRA mode, all params in full mode)
            skip_dim_check: If True, skip dimension check (for performance in loops)
        
        Returns:
            Dictionary mapping parameter names to tensors, compatible with stateless.functional_call
        """
        param_dict = {}
        offset = 0
        flat_params = flat_params.view(-1)  # Ensure 1D (O(1), just view change)
        total_numel = int(flat_params.numel())  # Convert to Python int
        
        # Check if model is in LoRA mode
        use_lora = hasattr(self.model, 'use_lora') and self.model.use_lora
        
        # Build a mapping from parameter objects to their names
        # This is more efficient than searching each time
        param_to_name = {}
        for name, param in self.model.named_parameters():
            # In LoRA mode, only track trainable parameters
            if use_lora:
                if param.requires_grad:
                    param_to_name[param] = name
            else:
                # Full fine-tuning: track all parameters
                param_to_name[param] = name
        
        # Iterate through parameters in the same order as get_flat_params
        for param in self.model.parameters():
            # In LoRA mode, skip non-trainable parameters
            if use_lora and not param.requires_grad:
                continue
            
            # Get parameter name from pre-built mapping
            param_name = param_to_name.get(param)
            if param_name is None:
                # Parameter not in mapping (shouldn't happen, but handle gracefully)
                continue
            
            numel = int(param.numel())  # Convert to Python int
            if not skip_dim_check and offset + numel > total_numel:
                # Dimension mismatch: return empty dict to avoid errors
                print(f"    [Attacker {self.client_id}] Param dict dimension mismatch: offset {offset} + numel {numel} > total {total_numel}")
                return {}
            
            # For PEFT models, stateless.functional_call expects parameter names
            # that match the actual model structure. The names from named_parameters()
            # should already be correct, but we verify compatibility.
            param_value = flat_params[offset:offset + numel].view_as(param)
            
            # Ensure param_value is on the same device as param
            # This is important when model is on GPU but flat_params might be on different device
            if param_value.device != param.device:
                param_value = param_value.to(param.device)
            
            # Handle PEFT model parameter names (base_model.model.* format)
            # stateless.functional_call should work with the names as-is from named_parameters()
            # But if we're working with a PEFT-wrapped model, ensure the name is correct
            if use_lora and hasattr(self.model, 'model') and hasattr(self.model.model, 'base_model'):
                # This is a PEFT model - parameter names should already include base_model.model prefix
                # from named_parameters(), so use as-is
                param_dict[param_name] = param_value
            else:
                # Standard model or direct PEFT model access
                param_dict[param_name] = param_value
            
            offset += numel
        
        # Verify we used all parameters
        if not skip_dim_check and offset != total_numel:
            print(f"    [Attacker {self.client_id}] Param dict size mismatch: used {offset} params, provided {total_numel}")
            # This could indicate a serious problem - log warning but continue
        
        return param_dict

    def _device_matches(self, device1, device2):
        """
        Check if two devices are the same, handling 'cuda' vs 'cuda:0' equivalence.
        
        Args:
            device1: First device
            device2: Second device
        
        Returns:
            True if devices are the same, False otherwise
        """
        # Convert to string and normalize
        d1_str = str(device1)
        d2_str = str(device2)
        
        # Normalize 'cuda' to 'cuda:0'
        if d1_str == 'cuda':
            d1_str = 'cuda:0'
        if d2_str == 'cuda':
            d2_str = 'cuda:0'
        
        return d1_str == d2_str

    def _ensure_model_on_device(self, module, device):
        """
        Recursively ensure ALL parameters and buffers of a module are on the specified device.
        This is critical for PEFT models with nested structures.
        
        Args:
            module: The module to move
            device: Target device (will be normalized to 'cuda:0' if it's 'cuda')
        """
        # Normalize device: always use 'cuda:0' instead of 'cuda' for consistency
        device_str = str(device)
        if device_str == 'cuda':
            target_device = torch.device('cuda:0')
        elif device_str.startswith('cuda'):
            target_device = torch.device(device_str if ':' in device_str else 'cuda:0')
        else:
            target_device = device
        
        # Use named_parameters to get all parameters including nested ones
        for name, param in module.named_parameters(recurse=False):
            if not self._device_matches(param.device, target_device):
                # Force move by creating new tensor on target device
                with torch.no_grad():
                    param.data = param.data.to(target_device, non_blocking=True)
        
        for name, buffer in module.named_buffers(recurse=False):
            if not self._device_matches(buffer.device, target_device):
                # Force move buffer
                buffer.data = buffer.data.to(target_device, non_blocking=True)
        
        # Recursively process all child modules
        for child in module.children():
            self._ensure_model_on_device(child, target_device)

    def _proxy_global_loss(self, malicious_update: torch.Tensor, max_batches: int = 1, 
                           skip_dim_check: bool = False, keep_model_on_gpu: bool = False) -> torch.Tensor:
        """
        Differentiable proxy for F(w'_g): cross-entropy on a small clean subset,
        using stateless.functional_call with (w_g + malicious_update).
        
        Args:
            malicious_update: Update vector to evaluate (can be on CPU or GPU)
            max_batches: Maximum number of batches to process
            skip_dim_check: If True, skip dimension check (for performance in loops)
            keep_model_on_gpu: If True, model will NOT be moved back to CPU after computation.
                              This is critical when the returned loss will be used for backward pass.
        
        Note: 
            - If keep_model_on_gpu=False: Model will be temporarily moved to GPU, then moved back to CPU.
            - If keep_model_on_gpu=True: Model stays on GPU (important for backward pass in optimization loops).
        """
        if self.global_model_params is None or self.proxy_loader is None:
            return torch.tensor(0.0, device=self.device)

        # Normalize device: always use 'cuda:0' for consistency
        target_device = torch.device('cuda:0') if self.device.type == 'cuda' else self.device
        
        # Ensure malicious_update is on GPU for computation
        if malicious_update.device.type != 'cuda':
            malicious_update = malicious_update.to(target_device)

        # Ensure shapes match: flatten to 1D and check dimension
        malicious_update = malicious_update.view(-1)  # Flatten to 1D (O(1), just view change)
        if not skip_dim_check and int(malicious_update.numel()) != self._flat_numel:
            # Dimension mismatch: return zero loss to avoid errors
            print(f"    [Attacker {self.client_id}] Proxy loss dimension mismatch: got {malicious_update.numel()}, expected {self._flat_numel}")
            return torch.tensor(0.0, device=self.device)

        # Move model to GPU temporarily for proxy loss calculation
        # Normalize device: always use 'cuda:0' for consistency
        target_device = torch.device('cuda:0') if self.device.type == 'cuda' else self.device
        
        model_was_on_cpu = not self._model_on_gpu
        if model_was_on_cpu:
            # Move entire model to device (including all parameters and buffers)
            # For PEFT models, this should move base model, LoRA parameters, and all buffers
            self.model.to(target_device)
            
            # CRITICAL FIX for PEFT models: Recursively ensure ALL parameters and buffers are on GPU
            # This is necessary because PEFT models have nested structures that .to() might not handle correctly
            self._ensure_model_on_device(self.model, target_device)
            
            # Double-check: Verify ALL parameters and buffers are actually on GPU
            # Use normalized device: always use 'cuda:0' for consistency
            target_device = torch.device('cuda:0') if self.device.type == 'cuda' else self.device
            for name, param in self.model.named_parameters():
                if not self._device_matches(param.device, target_device):
                    print(f"    [Attacker {self.client_id}] ERROR: Parameter {name} on {param.device}, moving to {target_device}")
                    with torch.no_grad():
                        param.data = param.data.to(target_device, non_blocking=True)
            for name, buffer in self.model.named_buffers():
                if not self._device_matches(buffer.device, target_device):
                    print(f"    [Attacker {self.client_id}] ERROR: Buffer {name} on {buffer.device}, moving to {target_device}")
                    buffer.data = buffer.data.to(target_device, non_blocking=True)
            
            self._model_on_gpu = True

        try:
            # CRITICAL: Ensure global_model_params and malicious_update are on the same device
            # Both should be on target_device for proper computation
            if not self._device_matches(self.global_model_params.device, target_device):
                global_params_gpu = self.global_model_params.to(target_device)
            else:
                global_params_gpu = self.global_model_params
            
            candidate_params = global_params_gpu + malicious_update
            
            # Skip dimension check if already validated (performance optimization)
            param_dict = self._flat_to_param_dict(candidate_params, skip_dim_check=skip_dim_check)

            # CRITICAL FIX: Ensure all parameters in param_dict are on the correct device
            # Normalize device: always use 'cuda:0' for consistency  
            # Note: target_device already defined earlier, but redefining here for clarity
            target_device = torch.device('cuda:0') if self.device.type == 'cuda' else self.device
            for name, value in param_dict.items():
                if not self._device_matches(value.device, target_device):
                    param_dict[name] = value.to(target_device, non_blocking=True)
            
            # EXTRA SAFETY: Before using stateless.functional_call, verify model is completely on GPU
            # Normalize device: always use 'cuda:0' for consistency
            target_device = torch.device('cuda:0') if self.device.type == 'cuda' else self.device
            # Check a sample of parameters to ensure the model is really on GPU
            try:
                sample_param = next(iter(self.model.parameters()))
                if not self._device_matches(sample_param.device, target_device):
                    print(f"    [Attacker {self.client_id}] CRITICAL: Model not fully on {target_device}, forcing move")
                    self.model.to(target_device)
                    self._ensure_model_on_device(self.model, target_device)
            except StopIteration:
                pass

            total_loss = 0.0
            batches = 0
            
            # Normalize device once for this batch loop
            target_device = torch.device('cuda:0') if self.device.type == 'cuda' else self.device

            for batch in self.proxy_loader:
                input_ids = batch['input_ids'].to(target_device)
                attention_mask = batch['attention_mask'].to(target_device)
                labels = batch['labels'].to(target_device)

                # CRITICAL FIX: For LoRA/PEFT models, stateless.functional_call has device issues
                # Use direct parameter setting method instead, which gives us full control over devices
                use_lora = hasattr(self.model, 'use_lora') and self.model.use_lora
                
                if use_lora:
                    # For LoRA models, use fallback method directly to avoid stateless.functional_call device issues
                    logits = None
                    original_params = {}
                    try:
                        # CRITICAL: First ensure ALL parameters (including base model) are on target device
                        # This must be done BEFORE setting LoRA parameters
                        self.model.to(target_device)
                        self._ensure_model_on_device(self.model, target_device)
                        
                        # Verify all parameters are on correct device before setting new values
                        for name, param in self.model.named_parameters():
                            if not self._device_matches(param.device, target_device):
                                with torch.no_grad():
                                    param.data = param.data.to(target_device, non_blocking=True)
                        
                        # Verify all buffers are on correct device
                        for name, buffer in self.model.named_buffers():
                            if not self._device_matches(buffer.device, target_device):
                                buffer.data = buffer.data.to(target_device, non_blocking=True)
                        
                        # Now save original parameters and set new values (only LoRA params)
                        # CRITICAL: Use no_grad() for parameter setting to avoid tracking gradients
                        # We only need gradients for the forward pass, not for data copying
                        with torch.no_grad():
                            for name, param in self.model.named_parameters():
                                if name in param_dict:
                                    # Save original parameter value
                                    original_params[name] = param.data.clone()
                                    # Get new parameter value from param_dict
                                    new_value = param_dict[name]
                                    # Ensure new_value is on target_device
                                    if not self._device_matches(new_value.device, target_device):
                                        new_value = new_value.to(target_device, non_blocking=True)
                                    # Ensure data type matches
                                    if new_value.dtype != param.dtype:
                                        new_value = new_value.to(dtype=param.dtype)
                                    # Copy the value
                                    param.data.copy_(new_value)
                        
                        # Final check: ensure all parameters are still on correct device after setting
                        # CRITICAL: This must be done before forward pass to avoid device mismatch in backward
                        # Use multiple methods to ensure everything is on GPU
                        self.model.to(target_device)  # Force move model to GPU
                        self._ensure_model_on_device(self.model, target_device)  # Recursive check
                        
                        # One final verification before forward pass
                        # Check ALL parameters (including base model params in LoRA mode)
                        # This is critical for LoRA models where base model params might not be moved correctly
                        for name, param in self.model.named_parameters():
                            if not self._device_matches(param.device, target_device):
                                print(f"    [Attacker {self.client_id}] FINAL PRE-FORWARD: Parameter {name} on {param.device}, fixing...")
                                with torch.no_grad():
                                    param.data = param.data.to(target_device, non_blocking=True)
                        for name, buffer in self.model.named_buffers():
                            if not self._device_matches(buffer.device, target_device):
                                print(f"    [Attacker {self.client_id}] FINAL PRE-FORWARD: Buffer {name} on {buffer.device}, fixing...")
                                buffer.data = buffer.data.to(target_device, non_blocking=True)
                        
                        # One more recursive check after fixing individual parameters
                        self._ensure_model_on_device(self.model, target_device)
                        
                        # Run forward pass
                        # CRITICAL: All parameters and buffers MUST be on target_device at this point
                        # The computation graph created here will be used for backward, so device consistency is essential
                        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        
                        # Restore original parameters
                        # CRITICAL: Use no_grad() for parameter restoration to avoid tracking gradients
                        with torch.no_grad():
                            for name, param in self.model.named_parameters():
                                if name in original_params:
                                    restored_value = original_params[name]
                                    if not self._device_matches(restored_value.device, target_device):
                                        restored_value = restored_value.to(target_device, non_blocking=True)
                                    param.data.copy_(restored_value)
                    except Exception as fallback_error:
                        print(f"    [Attacker {self.client_id}] LoRA fallback method failed: {fallback_error}")
                        # Restore parameters even if forward failed
                        # CRITICAL: Use no_grad() for parameter restoration
                        with torch.no_grad():
                            for name, param in self.model.named_parameters():
                                if name in original_params:
                                    restored_value = original_params[name]
                                    if not self._device_matches(restored_value.device, target_device):
                                        restored_value = restored_value.to(target_device, non_blocking=True)
                                    param.data.copy_(restored_value)
                        raise
                
                else:
                    # For full fine-tuning, try stateless.functional_call first
                    try:
                        # Final verification before calling stateless.functional_call
                        self._ensure_model_on_device(self.model, target_device)
                        self.model.to(target_device)

                        logits = stateless.functional_call(
                            self.model,
                            param_dict,
                            args=(),
                            kwargs={'input_ids': input_ids, 'attention_mask': attention_mask}
                        )
                    except (RuntimeError, KeyError) as e:
                        # If stateless.functional_call fails (e.g., parameter name mismatch in PEFT),
                        # try using the model directly with temporarily set parameters
                        # This is a fallback for PEFT model compatibility
                        print(f"    [Attacker {self.client_id}] Warning: stateless.functional_call failed: {e}")
                        print(f"    [Attacker {self.client_id}] Attempting fallback method...")
                        
                        # Fallback: temporarily set parameters, run forward, then restore
                        original_params = {}
                        try:
                            # Use normalized device for consistency: always use 'cuda:0'
                            target_device = torch.device('cuda:0') if self.device.type == 'cuda' else self.device
                            
                            # First, ensure entire model is on correct device (defensive check)
                            # This is critical for PEFT models where base model params might not be properly moved
                            self.model.to(target_device)
                            
                            # CRITICAL: Recursively ensure ALL parameters and buffers are on GPU
                            # This is essential for PEFT models with nested structures
                            self._ensure_model_on_device(self.model, target_device)
                            
                            # Ensure model is on target device first
                            self.model.to(target_device)
                            
                            # Verify ALL parameters are on GPU before proceeding
                            for name, param in self.model.named_parameters():
                                if not self._device_matches(param.device, target_device):
                                    print(f"    [Attacker {self.client_id}] CRITICAL in fallback: Parameter {name} on {param.device}, forcing to {target_device}")
                                    with torch.no_grad():
                                        param.data = param.data.to(target_device, non_blocking=True)
                            
                            # Verify ALL buffers are on GPU
                            for name, buffer in self.model.named_buffers():
                                if not self._device_matches(buffer.device, target_device):
                                    print(f"    [Attacker {self.client_id}] CRITICAL in fallback: Buffer {name} on {buffer.device}, forcing to {target_device}")
                                    buffer.data = buffer.data.to(target_device, non_blocking=True)
                            
                            # One more recursive check
                            self._ensure_model_on_device(self.model, target_device)
                            
                            # Save original parameters and set new values
                            # CRITICAL: Use no_grad() for parameter setting to avoid tracking gradients
                            # We only need gradients for the forward pass, not for data copying
                            with torch.no_grad():
                                for name, param in self.model.named_parameters():
                                    if name in param_dict:
                                        # Save original parameter value (on current device)
                                        original_params[name] = param.data.clone()
                                        # Get new parameter value from param_dict
                                        new_value = param_dict[name]
                                        # Ensure new_value and param are both on target_device
                                        # Use normalized device matching to handle 'cuda' vs 'cuda:0'
                                        if not self._device_matches(new_value.device, target_device):
                                            new_value = new_value.to(target_device, non_blocking=True)
                                        # Ensure param is also on target_device
                                        if not self._device_matches(param.device, target_device):
                                            param.data = param.data.to(target_device, non_blocking=True)
                                        # Ensure data type matches
                                        if new_value.dtype != param.dtype:
                                            new_value = new_value.to(dtype=param.dtype)
                                        # Copy the value
                                        param.data.copy_(new_value)
                            
                            # Final verification: ensure all parameters and buffers are on correct device
                            # This is especially important for PEFT models with nested structures
                            # Double-check with recursive function (use normalized device)
                            target_device = torch.device('cuda:0') if self.device.type == 'cuda' else self.device
                            self._ensure_model_on_device(self.model, target_device)
                            
                            # One more explicit check before forward pass
                            # Use normalized device: always 'cuda:0' for consistency
                            target_device = torch.device('cuda:0') if self.device.type == 'cuda' else self.device
                            for name, param in self.model.named_parameters():
                                if not self._device_matches(param.device, target_device):
                                    print(f"    [Attacker {self.client_id}] FINAL CHECK FAILED: Parameter {name} on {param.device}, should be on {target_device}")
                                    # Try one more time to fix it
                                    with torch.no_grad():
                                        param.data = param.data.to(target_device, non_blocking=True)
                            
                            # Final check for buffers
                            for name, buffer in self.model.named_buffers():
                                if not self._device_matches(buffer.device, target_device):
                                    print(f"    [Attacker {self.client_id}] FINAL CHECK FAILED: Buffer {name} on {buffer.device}, should be on {target_device}")
                                    buffer.data = buffer.data.to(target_device, non_blocking=True)
                            
                            # Run forward pass
                            # NewsClassifierModel.forward() returns logits directly
                            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                            
                            # Restore original parameters
                            # CRITICAL: Use no_grad() for parameter restoration to avoid tracking gradients
                            with torch.no_grad():
                                for name, param in self.model.named_parameters():
                                    if name in original_params:
                                        # Ensure restored value is on target_device
                                        restored_value = original_params[name]
                                        if not self._device_matches(restored_value.device, target_device):
                                            restored_value = restored_value.to(target_device, non_blocking=True)
                                        # Ensure param is also on target_device
                                        if not self._device_matches(param.device, target_device):
                                            param.data = param.data.to(target_device, non_blocking=True)
                                        param.data.copy_(restored_value)
                            
                            # Final check after restoration: ensure all parameters still on correct device
                            self._ensure_model_on_device(self.model, target_device)
                        except Exception as fallback_error:
                            print(f"    [Attacker {self.client_id}] Fallback method also failed: {fallback_error}")
                            # Restore parameters even if forward failed
                            # CRITICAL: Use no_grad() for parameter restoration
                            with torch.no_grad():
                                for name, param in self.model.named_parameters():
                                    if name in original_params:
                                        restored_value = original_params[name]
                                        if not self._device_matches(restored_value.device, target_device):
                                            restored_value = restored_value.to(target_device, non_blocking=True)
                                        if not self._device_matches(param.device, target_device):
                                            param.data = param.data.to(target_device, non_blocking=True)
                                        param.data.copy_(restored_value)
                            # Return zero loss as last resort
                            return torch.tensor(0.0, device=target_device)

                ce_loss = F.cross_entropy(logits, labels)
                total_loss = total_loss + ce_loss
                batches += 1
                if batches >= max_batches:
                    break

            if batches == 0:
                result = torch.tensor(0.0, device=self.device)
            else:
                result = total_loss / batches
        except Exception as e:
            print(f"    [Attacker {self.client_id}] Error in proxy loss computation: {e}")
            result = torch.tensor(0.0, device=self.device)
        finally:
            # CRITICAL: Only move model back to CPU if keep_model_on_gpu=False
            # If keep_model_on_gpu=True, the loss will be used for backward pass,
            # and moving the model would break the computation graph!
            if not keep_model_on_gpu and model_was_on_cpu:
                # Before moving back, ensure all gradients are computed if needed
                # Actually, if keep_model_on_gpu is False, we assume backward is not needed
                # So it's safe to move back
                self.model.cpu()
                self._model_on_gpu = False

        return result

    def _construct_graph(self, reduced_features: torch.Tensor) -> torch.Tensor:
        """
        Construct graph according to the paper (Section III).
        
        Paper formulation:
        - Feature matrix F(t) = [w_1(t), ..., w_i(t)]^T ∈ R^{I×M}
        - Adjacency matrix A(t) ∈ R^{M×M} (NOT I×I!)
        - δ_{m,m'} = cosine similarity between w_m(t) and w_{m'}(t)
        - w_m(t) ∈ R^{I×1} is the m-th COLUMN of F(t)
        
        So we need to compute similarity between COLUMNS (parameter dimensions),
        not ROWS (clients).
        
        Args:
            reduced_features: Feature matrix F(t) ∈ R^{I×M} where I=num_clients, M=feature_dim
            
        Returns:
            Adjacency matrix A(t) ∈ R^{M×M} with binary edges based on cosine similarity threshold
        """
        # reduced_features shape: (I, M) where I=num_clients, M=feature_dim
        # We need to compute similarity between columns (parameter dimensions)
        # Transpose to get (M, I), then compute similarity
        
        # F^T shape: (M, I) - each row is a parameter dimension across all clients
        features_transposed = reduced_features.t()  # (M, I)
        
        # Normalize for cosine similarity (along dim=1, i.e., across clients)
        norm_features = F.normalize(features_transposed, p=2, dim=1)  # (M, I)
        
        # Compute adjacency matrix A ∈ R^{M×M}
        # A[m, m'] = cosine_sim(w_m, w_m') where w_m is m-th column of F
        similarity_matrix = torch.mm(norm_features, norm_features.t())  # (M, M)
        
        # Remove self-loops
        adj_matrix = similarity_matrix.clone()
        adj_matrix.fill_diagonal_(0)
        
        # Threshold for binarization (paper doesn't specify, but common practice)
        # Ensure graph_threshold is a Python float (not tensor)
        threshold = float(self.graph_threshold) if isinstance(self.graph_threshold, (int, float)) else 0.5
        adj_matrix = (adj_matrix > threshold).float()
        
        return adj_matrix

    def _train_vgae(self, adj_matrix: torch.Tensor, feature_matrix: torch.Tensor, epochs=None) -> torch.Tensor:
        """
        Train the VGAE model according to the paper.
        
        Paper formulation:
        - Input: A ∈ R^{M×M} (adjacency), F ∈ R^{I×M} (features)
        - For VGAE, we use F^T ∈ R^{M×I} as node features
        - Each node represents a parameter dimension
        - VGAE learns to reconstruct A
        
        Args:
            adj_matrix: Adjacency matrix A ∈ R^{M×M}
            feature_matrix: Feature matrix F ∈ R^{I×M}
            epochs: Number of training epochs (default: self.vgae_epochs)
            
        Returns:
            Reconstructed adjacency matrix Â ∈ R^{M×M} (detached)
        """
        if epochs is None:
            epochs = self.vgae_epochs
        
        # adj_matrix shape: (M, M) - from _construct_graph
        # feature_matrix shape: (I, M) - original features
        # For VGAE input, we use F^T as node features: (M, I)
        node_features = feature_matrix.t()  # (M, I)
        
        input_dim = int(node_features.shape[1])  # I (number of clients) - Convert to Python int
        num_nodes = int(node_features.shape[0])  # M (feature dimension) - Convert to Python int
        
        # Initialize VGAE if needed
        # Paper: input_dim = I (number of clients/benign models)
        vgae_input_dim = int(self.vgae.gc1.weight.shape[0]) if self.vgae is not None else None
        if self.vgae is None or vgae_input_dim != input_dim:
            # Use client_id-based seed for VGAE initialization to ensure diversity among attackers
            # This ensures different attackers have different VGAE initial weights, leading to different attack patterns
            import hashlib
            seed_str = f"vgae_{self.client_id}_{input_dim}_{self.vgae_hidden_dim}_{self.vgae_latent_dim}"
            vgae_seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16) % (2**31)
            
            # Save current random state
            rng_state_before = torch.get_rng_state()
            np_rng_state_before = np.random.get_state()
            
            # Set seed for VGAE initialization
            torch.manual_seed(vgae_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(vgae_seed)
            
            # Use configured VGAE architecture parameters (per paper: hidden1_dim=32, hidden2_dim=16)
            self.vgae = VGAE(input_dim=input_dim, hidden_dim=self.vgae_hidden_dim, 
                            latent_dim=self.vgae_latent_dim, dropout=self.vgae_dropout).to(self.device)
            self.vgae_optimizer = optim.Adam(self.vgae.parameters(), lr=self.vgae_lr)
            
            # Restore random state to avoid affecting other random number generation
            torch.set_rng_state(rng_state_before)
            np.random.set_state(np_rng_state_before)

        self.vgae.train()
        
        for _ in range(epochs):
            self.vgae_optimizer.zero_grad()
            
            # Forward pass: VGAE takes (node_features, adj_matrix)
            # node_features: (M, I), adj_matrix: (M, M)
            adj_recon, mu, logvar = self.vgae(node_features, adj_matrix)
            
            # Loss calculation
            loss = self.vgae.loss_function(adj_recon, adj_matrix, mu, logvar)
            
            loss.backward()
            self.vgae_optimizer.step()
        
        return adj_recon.detach()  # Return reconstructed adjacency for GSP

    def set_global_model_params(self, global_params: torch.Tensor):
        """Set global model parameters for constraint (4b) calculation."""
        # Store on GPU but only when needed (will be moved in _proxy_global_loss)
        # Keep on same device as provided to avoid unnecessary transfers
        self.global_model_params = global_params.clone().detach().to(self.device)
    
    def set_constraint_params(self, d_T: float = None, gamma: float = None, 
                              total_data_size: float = None, benign_data_sizes: dict = None):
        """
        Set constraint parameters for Formula 4.
        
        Args:
            d_T: Distance threshold for constraint (4b): d(w'_j(t), w'_g(t)) ≤ d_T
            gamma: Upper bound for constraint (4c): Σ β'_{i,j}(t) d(w_i(t), w̄_i(t)) ≤ Γ
            total_data_size: D(t) - Total data size for aggregation weight calculation (Paper Formula (2))
            benign_data_sizes: Dict {client_id: D_i(t)} - Data sizes for each benign client (Paper Formula (2))
        """
        self.d_T = d_T  # Constraint (4b): d(w'_j(t), w'_g(t)) ≤ d_T
        self.gamma = gamma  # Constraint (4c): Σ β'_{i,j}(t) d(w_i(t), w̄_i(t)) ≤ Γ
        self.total_data_size = total_data_size  # D(t) for weight calculation
        if benign_data_sizes is not None:
            self.benign_data_sizes = benign_data_sizes  # {client_id: D_i(t)}
    
    def set_lagrangian_params(self, use_lagrangian_dual: bool = False,
                              lambda_init: float = 0.1,
                              rho_init: float = 0.1,
                              lambda_lr: float = 0.01,
                              rho_lr: float = 0.01,
                              enable_final_projection: bool = True):
        """
        Set Lagrangian Dual parameters (initialized according to paper Algorithm 1)
        
        Paper reference: Section 3, Algorithm 1
        - Lagrangian function: eq:lagrangian
        - Optimization subproblem: eq:wprime_sub
        - Initialization: λ(1)≥0, ρ(1)≥0
        
        Args:
            use_lagrangian_dual: Whether to use Lagrangian Dual mechanism
            lambda_init: Initial λ(1) value (≥0, per paper Algorithm 1)
            rho_init: Initial ρ(1) value (≥0, per paper Algorithm 1)
            lambda_lr: Learning rate for λ(t) update (subgradient step size)
            rho_lr: Learning rate for ρ(t) update (subgradient step size)
            enable_final_projection: Whether to apply final projection after optimization (only for Lagrangian mode)
                                   If False, completely relies on Lagrangian mechanism (no final projection)
        
        Modification 2: Save initial values for reset in prepare_for_round
        """
        self.use_lagrangian_dual = use_lagrangian_dual
        if use_lagrangian_dual:
            # Paper: λ(1)≥0, ρ(1)≥0
            # Modification 2: Save initial values for reset each round
            self.lambda_init_value = max(0.0, lambda_init)
            self.rho_init_value = max(0.0, rho_init)
            self.lambda_dt = torch.tensor(self.lambda_init_value, requires_grad=False)
            self.rho_dt = torch.tensor(self.rho_init_value, requires_grad=False)
            self.lambda_lr = lambda_lr
            self.rho_lr = rho_lr
            self.enable_final_projection = enable_final_projection
        else:
            # Hard constraint projection mode
            self.lambda_dt = None
            self.rho_dt = None
            self.lambda_init_value = None
            self.rho_init_value = None
            self.enable_final_projection = True  # Default to True for hard constraint mode (always applies projection)

    def _compute_global_loss(self, malicious_update: torch.Tensor, 
                            selected_benign: List[torch.Tensor],
                            beta_selection: List[int]) -> torch.Tensor:
        """
        Compute global loss F(w'_g(t)) according to paper Equation (3).
        
        Paper Formula (3):
        F(w'_g(t)) = Σ_{i=1}^I (D_i(t)/D(t)) β'_{i,j}(t) F(w_i(t)) 
                    + (D'_j(t)/D(t)) F'(w'_j(t))
        
        Args:
            malicious_update: w'_j(t) - malicious update
            selected_benign: List of selected benign updates (based on β selection)
            beta_selection: List of indices indicating which benign updates are selected
        
        Returns:
            Global loss F(w'_g(t)) to be maximized
        """
        if self.global_model_params is None or self.proxy_loader is None:
            # Fallback: return zero if proxy not available
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Check if we have total_data_size for full formula calculation
        if self.total_data_size is None or len(self.benign_data_sizes) == 0:
            # Fallback: use proxy loss only (old behavior)
            proxy_loss = self._proxy_global_loss(malicious_update, max_batches=self.proxy_max_batches_opt, skip_dim_check=False)
            return proxy_loss
        
        # Paper Formula (3): Full calculation with weights
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        D_total = float(self.total_data_size)
        D_attacker = float(self.claimed_data_size)
        
        # First term: Σ_{i=1}^I (D_i(t)/D(t)) β'_{i,j}(t) F(w_i(t))
        # Note: F(w_i(t)) is approximated by the loss on proxy dataset with benign update w_i(t)
        # This is computationally expensive, so we approximate by using the average loss
        if len(selected_benign) > 0 and len(beta_selection) > 0:
            # Compute average loss over selected benign updates
            benign_loss_sum = torch.tensor(0.0, device=self.device, requires_grad=True)
            benign_weight_sum = 0.0
            
            # Use exact weights D_i(t)/D(t) for each selected benign client according to paper Formula (3)
            for idx, benign_update in enumerate(selected_benign):
                # We approximate F(w_i(t)) by computing loss on proxy dataset with benign_update
                benign_loss = self._proxy_global_loss(benign_update, max_batches=1, skip_dim_check=False)
                
                # Get exact weight D_i(t)/D(t) for this selected benign client
                if idx < len(beta_selection) and len(self.benign_update_client_ids) > 0:
                    # beta_selection[idx] gives the original index in self.benign_updates
                    original_idx = beta_selection[idx]
                    if original_idx < len(self.benign_update_client_ids):
                        client_id = self.benign_update_client_ids[original_idx]
                        # Get exact data size D_i(t) for this client
                        D_i = self.benign_data_sizes.get(client_id, 1.0)
                        benign_weight = D_i / D_total
                    else:
                        # Fallback: use average weight if index out of range
                        if len(self.benign_data_sizes) > 0:
                            avg_benign_data_size = sum(self.benign_data_sizes.values()) / len(self.benign_data_sizes)
                            benign_weight = avg_benign_data_size / D_total
                        else:
                            benign_weight = 1.0 / len(selected_benign) if len(selected_benign) > 0 else 0.0
                else:
                    # Fallback: use average weight if beta_selection or client_ids not available
                    if len(self.benign_data_sizes) > 0:
                        avg_benign_data_size = sum(self.benign_data_sizes.values()) / len(self.benign_data_sizes)
                        benign_weight = avg_benign_data_size / D_total
                    else:
                        benign_weight = 1.0 / len(selected_benign) if len(selected_benign) > 0 else 0.0
                
                # Add weighted benign loss using exact weight: (D_i(t)/D(t)) * F(w_i(t))
                benign_loss_sum = benign_loss_sum + benign_weight * benign_loss
                benign_weight_sum += benign_weight
            
            # Sum all weighted benign losses: Σ (D_i(t)/D(t)) * β'_{i,j}(t) * F(w_i(t))
            if benign_weight_sum > 0:
                total_loss = total_loss + benign_loss_sum
        
        # Second term: (D'_j(t)/D(t)) F'(w'_j(t))
        attacker_weight = D_attacker / D_total
        attacker_loss = self._proxy_global_loss(malicious_update, max_batches=self.proxy_max_batches_opt, skip_dim_check=False)
        total_loss = total_loss + attacker_weight * attacker_loss
        
        return total_loss
    
    def _compute_real_distance_to_global(self, malicious_update: torch.Tensor,
                                         selected_benign: List[torch.Tensor],
                                         beta_selection: List[int]) -> torch.Tensor:
        """
        Compute real Euclidean distance d(w'_j(t), w'_g(t)) according to paper Constraint (4b).
        
        Paper Formula (2):
        w'_g(t) = Σ_{i=1}^I (D_i(t)/D(t)) β'_{i,j}(t) w_i(t) + (D'_j(t)/D(t)) w'_j(t)
        
        Paper Constraint (4b):
        d(w'_j(t), w'_g(t)) = ||w'_j(t) - w'_g(t)|| ≤ d_T
        
        Args:
            malicious_update: w'_j(t) - malicious update
            selected_benign: List of selected benign updates w_i(t) (based on β selection)
            beta_selection: List of indices indicating which benign updates are selected
        
        Returns:
            Real Euclidean distance ||w'_j(t) - w'_g(t)||
        """
        if self.total_data_size is None or self.global_model_params is None:
            # Fallback: use approximation ||w'_j|| if information not available
            return torch.norm(malicious_update.view(-1))
        
        D_total = float(self.total_data_size)
        D_attacker = float(self.claimed_data_size)
        
        # Ensure malicious_update is on correct device
        target_device = malicious_update.device if isinstance(malicious_update, torch.Tensor) else self.device
        malicious_flat = malicious_update.view(-1).to(target_device)
        
        # Compute w'_g(t) according to paper Formula (2):
        # w'_g(t) = Σ_{i=1}^I (D_i(t)/D(t)) β'_{i,j}(t) w_i(t) + (D'_j(t)/D(t)) w'_j(t)
        # Note: w_i(t) and w'_j(t) are complete model parameters (not updates)
        # In code, malicious_update is the update: malicious_update = w'_j(t) - w_g(t-1)
        # So: w'_j(t) = w_g(t-1) + malicious_update
        
        # Get global model from previous round: w_g(t-1)
        w_g_prev = self.global_model_params.view(-1).to(target_device)
        
        # Reconstruct complete malicious model: w'_j(t) = w_g(t-1) + malicious_update
        w_j_malicious = w_g_prev + malicious_flat
        
        # Compute w'_g(t) = Σ (D_i/D) * w_i + (D'_j/D) * w'_j
        # First term: weighted sum of selected benign models w_i(t)
        # Note: selected_benign are updates, so w_i(t) = w_g_prev + benign_update
        w_g_weighted_sum = torch.zeros_like(w_g_prev)
        
        if len(selected_benign) > 0:
            # Weight: D_i(t)/D(t) for each selected benign client
            # Use exact weight for each selected benign client according to paper Formula (2)
            # Get client_id for each selected_benign using beta_selection indices
            for idx, benign_update in enumerate(selected_benign):
                benign_flat = benign_update.view(-1).to(target_device)
                # Reconstruct complete benign model: w_i(t) = w_g_prev + benign_update
                w_i_benign = w_g_prev + benign_flat
                
                # Get exact weight D_i(t)/D(t) for this selected benign client
                if idx < len(beta_selection) and len(self.benign_update_client_ids) > 0:
                    # beta_selection[idx] gives the original index in self.benign_updates
                    original_idx = beta_selection[idx]
                    if original_idx < len(self.benign_update_client_ids):
                        client_id = self.benign_update_client_ids[original_idx]
                        # Get exact data size D_i(t) for this client
                        D_i = self.benign_data_sizes.get(client_id, 1.0)
                        benign_weight = D_i / D_total
                    else:
                        # Fallback: use average weight if index out of range
                        if len(self.benign_data_sizes) > 0:
                            avg_benign_data_size = sum(self.benign_data_sizes.values()) / len(self.benign_data_sizes)
                            benign_weight = avg_benign_data_size / D_total
                        else:
                            benign_weight = 1.0 / len(selected_benign) if len(selected_benign) > 0 else 0.0
                else:
                    # Fallback: use average weight if beta_selection or client_ids not available
                    if len(self.benign_data_sizes) > 0:
                        avg_benign_data_size = sum(self.benign_data_sizes.values()) / len(self.benign_data_sizes)
                        benign_weight = avg_benign_data_size / D_total
                    else:
                        benign_weight = 1.0 / len(selected_benign) if len(selected_benign) > 0 else 0.0
                
                # Add weighted benign model using exact weight
                w_g_weighted_sum = w_g_weighted_sum + benign_weight * w_i_benign
        
        # Second term: (D'_j(t)/D(t)) w'_j(t)
        attacker_weight = D_attacker / D_total
        w_g_weighted_sum = w_g_weighted_sum + attacker_weight * w_j_malicious
        
        # w'_g(t) = weighted sum
        w_g_contaminated = w_g_weighted_sum
        
        # Compute distance: ||w'_j(t) - w'_g(t)||
        # Paper Constraint (4b): d(w'_j, w'_g) = ||w'_j - w'_g||
        distance = torch.norm(w_j_malicious - w_g_contaminated)
        
        return distance

    def _gsp_generate_malicious(self, feature_matrix: torch.Tensor, 
                                  adj_orig: torch.Tensor, adj_recon: torch.Tensor,
                                  poisoned_update: torch.Tensor) -> torch.Tensor:
        """
        Graph Signal Processing (GSP) module according to the paper (Section III).
        
        Paper formulation:
        1. L = diag(A·1) - A                 (Laplacian of original graph)
        2. L = B Λ B^T                       (SVD decomposition)
        3. S = F · B                         (GFT coefficient matrix)
        4. L̂ = diag(Â·1) - Â                 (Laplacian of reconstructed graph)
        5. L̂ = B̂ Λ̂ B̂^T                       (SVD decomposition)
        6. F̂ = S · B̂^T                       (Reconstructed feature matrix)
        7. w'_j(t) selected from F̂           (Malicious model)
        
        Args:
            feature_matrix: F ∈ R^{I×M} - benign model features (reduced dimension)
            adj_orig: A ∈ R^{M×M} - original adjacency matrix
            adj_recon: Â ∈ R^{M×M} - reconstructed adjacency matrix from VGAE
            poisoned_update: Zero update (attackers don't train, unused parameter for compatibility)
            
        Returns:
            Malicious update generated using GSP (reduced dimension M, or None if failed)
        """
        # Ensure feature_matrix is valid and has correct shape
        if feature_matrix is None or not isinstance(feature_matrix, torch.Tensor):
            raise ValueError(f"[Attacker {self.client_id}] feature_matrix is None or invalid")
        if len(feature_matrix.shape) < 2:
            raise ValueError(f"[Attacker {self.client_id}] feature_matrix must be 2D, got shape {feature_matrix.shape}")
        # Get shape dimension - ensure it's a valid integer
        shape_dim = feature_matrix.shape[1]
        if shape_dim is None:
            raise ValueError(f"[Attacker {self.client_id}] feature_matrix.shape[1] is None")
        try:
            M = int(shape_dim)  # Reduced dimension - Convert to Python int
        except (TypeError, ValueError) as e:
            raise ValueError(f"[Attacker {self.client_id}] Cannot convert shape[1]={shape_dim} to int: {e}")
        if M <= 0:
            raise ValueError(f"[Attacker {self.client_id}] Invalid M dimension: {M}")
        
        # Step 1: Compute Laplacian of original graph
        # L = diag(A·1) - A
        degree_orig = adj_orig.sum(dim=1)
        L_orig = torch.diag(degree_orig) - adj_orig  # (M, M)
        
        # Step 2: SVD of original Laplacian
        # L = B Λ B^T
        try:
            U_orig, S_orig, Vh_orig = torch.linalg.svd(L_orig, full_matrices=True)
            B_orig = U_orig  # GFT basis (M, M)
        except Exception as e:
            # Fallback if SVD fails: return zeros in reduced dimension
            print(f"    [Attacker {self.client_id}] SVD failed: {e}, using zero fallback")
            return torch.zeros(M, device=feature_matrix.device, dtype=feature_matrix.dtype)
        
        # Step 3: Compute GFT coefficient matrix
        # S = F · B where F ∈ R^{I×M}, B ∈ R^{M×M}
        S = torch.mm(feature_matrix, B_orig)  # (I, M)
        
        # Step 4: Compute Laplacian of reconstructed graph
        # L̂ = diag(Â·1) - Â
        degree_recon = adj_recon.sum(dim=1)
        L_recon = torch.diag(degree_recon) - adj_recon  # (M, M)
        
        # Step 5: SVD of reconstructed Laplacian
        try:
            U_recon, S_recon, Vh_recon = torch.linalg.svd(L_recon, full_matrices=True)
            B_recon = U_recon  # New GFT basis (M, M)
        except Exception as e:
            # Fallback if SVD fails: return zeros in reduced dimension
            print(f"    [Attacker {self.client_id}] SVD of recon failed: {e}, using zero fallback")
            return torch.zeros(M, device=feature_matrix.device, dtype=feature_matrix.dtype)
        
        # Step 6: Generate reconstructed feature matrix
        # F̂ = S · B̂^T where S ∈ R^{I×M}, B̂ ∈ R^{M×M}
        F_recon = torch.mm(S, B_recon.t())  # (I, M)
        
        # Step 7: Generate malicious update
        # Paper: "vectors w'_j(t) in F̂ are selected as malicious local models"
        # According to paper, we select a vector from F̂ as w'_j(t)
        # F̂ shape: (I, M) where I is number of benign updates, M is feature dimension
        
        # Check if F_recon is valid
        F_recon_rows = int(F_recon.shape[0])  # Convert to Python int
        print(f"    [Attacker {self.client_id}] F_recon.shape = ({F_recon_rows}, {F_recon.shape[1]})")
        if F_recon_rows == 0:
            # Empty feature matrix: return zeros
            print(f"    [Attacker {self.client_id}] F_recon is empty, using zero fallback")
            return torch.zeros(M, device=feature_matrix.device, dtype=feature_matrix.dtype)
        
        # Select a vector from F̂ as the malicious update
        # Use client_id to select different vectors for different attackers (for diversity)
        # This ensures different attackers use different malicious models from F̂
        select_idx = int(self.client_id % F_recon_rows)  # Ensure Python int
        gsp_attack = F_recon[select_idx].clone()  # Select one row from F̂ as w'_j(t), clone to avoid view issues
        
        # Add client-specific perturbation to ensure diversity among attackers
        # This is important because:
        # 1. If F_recon has only one row, all attackers select the same row, so perturbation is essential
        # 2. If F_recon has multiple rows, different attackers may select different rows, but the rows
        #    may still be similar due to similar VGAE training. Perturbation adds additional diversity.
        # Use deterministic random number generation based on client_id and select_idx for reproducibility
        import hashlib
        pert_seed_str = f"gsp_pert_{self.client_id}_{select_idx}_{F_recon_rows}"
        pert_seed = int(hashlib.md5(pert_seed_str.encode()).hexdigest()[:8], 16) % (2**31)
        
        # Save current random state
        rng_state_before = torch.get_rng_state()
        
        # Set seed for perturbation generation
        torch.manual_seed(pert_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(pert_seed)
        
        # Generate perturbation with client_id and select_idx dependent scale
        # Scale increases with client_id and select_idx to ensure different perturbations
        perturbation_scale = self.gsp_perturbation_scale * (self.client_id + 1) * (1.0 + 0.1 * select_idx)
        perturbation = torch.randn_like(gsp_attack) * perturbation_scale
        gsp_attack = gsp_attack + perturbation
        
        # Restore random state
        torch.set_rng_state(rng_state_before)
        
        # Ensure gsp_attack is 1D tensor (not scalar)
        gsp_dim_count = int(gsp_attack.dim())  # Convert to Python int
        if gsp_dim_count == 0:
            # Scalar tensor: expand to 1D
            gsp_attack = gsp_attack.unsqueeze(0)
        elif gsp_dim_count > 1:
            # Multi-dimensional: flatten
            gsp_attack = gsp_attack.flatten()
        
        # Final check: ensure it's a 1D tensor with correct size
        if gsp_attack.numel() != M:
            # Size mismatch: create zeros with correct size
            print(f"    [Attacker {self.client_id}] GSP attack size mismatch: got {gsp_attack.numel()}, expected {M}, using zeros")
            gsp_attack = torch.zeros(M, device=feature_matrix.device, dtype=feature_matrix.dtype)
        
        return gsp_attack

    def camouflage_update(self, poisoned_update: torch.Tensor) -> torch.Tensor:
        """
        GRMP Attack using VGAE + GSP (data-agnostic attack).
        
        Attackers are not assigned local data and do not perform local training.
        The attack is generated purely from benign updates using VGAE+GSP.
        
        Paper Algorithm 1:
        1. Calculate A according to cosine similarity (eq. 8)
        2. Train VGAE to maximize L_loss (eq. 12), obtain optimal Â
        3. Use GSP module to obtain F̂, determine w'_j(t) based on F̂
        
        Args:
            poisoned_update: Zero update (attackers don't train, so this is always zero)
        
        Returns:
            Malicious update generated using VGAE+GSP
        """
        if not self.benign_updates:
            print(f"    [Attacker {self.client_id}] No benign updates, return zero update")
            return poisoned_update  # poisoned_update is always zero (attackers don't train)

        # Reset feature indices for this session
        self.feature_indices = None
        
        # ============================================================
        # STEP 1: Prepare feature matrix F ∈ R^{I×M}
        # ============================================================
        selected_benign = self._select_benign_subset()
        if not selected_benign:
            print(f"    [Attacker {self.client_id}] No benign subset selected, return zero update")
            return poisoned_update  # poisoned_update is always zero (attackers don't train)

        # Move selected updates to GPU for processing
        selected_benign_gpu = [u.to(self.device) for u in selected_benign]
        benign_stack = torch.stack([u.detach() for u in selected_benign_gpu])  # (I, full_dim)
        
        # Reduce dimensionality for computational efficiency
        reduced_benign = self._get_reduced_features(selected_benign_gpu, fix_indices=False)  # (I, M)
        # Clean up benign_stack and selected_benign_gpu after feature reduction (keep selected_benign on CPU for later use)
        del benign_stack, selected_benign_gpu
        torch.cuda.empty_cache()
        
        # Ensure reduced_benign has valid shape
        if reduced_benign is None or not isinstance(reduced_benign, torch.Tensor):
            raise ValueError(f"[Attacker {self.client_id}] reduced_benign is None or invalid")
        if len(reduced_benign.shape) < 2:
            raise ValueError(f"[Attacker {self.client_id}] reduced_benign must be 2D, got shape={reduced_benign.shape}")
        try:
            M = int(reduced_benign.shape[1])  # Convert to Python int
            I = int(reduced_benign.shape[0])  # Convert to Python int
        except (TypeError, ValueError) as e:
            raise ValueError(f"[Attacker {self.client_id}] Cannot convert reduced_benign shape to int: {e}, shape={reduced_benign.shape}")
        
        # ============================================================
        # STEP 2: Construct adjacency matrix A ∈ R^{M×M}
        # According to paper eq. (8): δ_{m,m'} = cosine_sim(w_m, w_m')
        # ============================================================
        adj_matrix = self._construct_graph(reduced_benign)  # (M, M)
        
        # ============================================================
        # STEP 3: Train VGAE to learn graph structure
        # Paper: "Train VGAE to maximize L_loss"
        # ============================================================
        adj_recon = self._train_vgae(adj_matrix, reduced_benign)  # Returns Â
        
        # ============================================================
        # STEP 4: GSP module to generate malicious update
        # Paper: "Use GSP module to obtain F̂, determine w'_j(t)"
        # ============================================================
        gsp_attack_reduced = self._gsp_generate_malicious(
            reduced_benign, adj_matrix, adj_recon, poisoned_update
        )
        
        # Clean up VGAE-related GPU tensors
        del reduced_benign, adj_matrix, adj_recon
        torch.cuda.empty_cache()
        
        # ============================================================
        # STEP 5: Expand GSP attack back to full dimension
        # Expand GSP attack from reduced dimension M back to full dimension.
        # Non-selected dimensions remain zero.
        # ============================================================
        # Create malicious_update on CPU to save GPU memory
        # poisoned_update is likely on CPU, but ensure we create on CPU
        if poisoned_update.device.type == 'cuda':
            malicious_update = torch.zeros_like(poisoned_update)
        else:
            malicious_update = torch.zeros_like(poisoned_update, device='cpu')
        total_dim = int(malicious_update.shape[0])  # Convert to Python int
        
        # _gsp_generate_malicious always returns a tensor (never None)
        # But check if it's valid
        if gsp_attack_reduced is not None and isinstance(gsp_attack_reduced, torch.Tensor):
            # Ensure gsp_attack_reduced is 1D tensor
            gsp_dim_count = int(gsp_attack_reduced.dim())  # Convert to Python int
            if gsp_dim_count == 0:
                # Scalar tensor: expand to 1D
                gsp_attack_reduced = gsp_attack_reduced.unsqueeze(0)
            elif gsp_dim_count > 1:
                # Multi-dimensional: flatten
                gsp_attack_reduced = gsp_attack_reduced.flatten()
            
            # Get dimension as Python int (not tensor)
            gsp_dim = int(gsp_attack_reduced.shape[0])
            
            if self.feature_indices is not None:
                # Dimension reduction was applied
                expected_dim = len(self.feature_indices)
                if gsp_dim == expected_dim:
                    # Correct dimension: expand back to full dimension
                    # Ensure gsp_attack_reduced is on CPU to match malicious_update
                    if gsp_attack_reduced.device.type == 'cuda':
                        gsp_attack_reduced = gsp_attack_reduced.cpu()
                    # Ensure feature_indices is on CPU for indexing
                    feature_indices_cpu = self.feature_indices.cpu() if self.feature_indices.device.type == 'cuda' else self.feature_indices
                    malicious_update[feature_indices_cpu] = gsp_attack_reduced
                else:
                    # Dimension mismatch: log warning and use zeros
                    print(f"    [Attacker {self.client_id}] GSP dimension mismatch: got {gsp_dim}, expected {expected_dim}, using zeros")
            else:
                # No dimension reduction: GSP attack should be full dimension
                if gsp_dim == total_dim:
                    # Correct dimension: use directly
                    # Ensure gsp_attack_reduced is on CPU to match malicious_update
                    if gsp_attack_reduced.device.type == 'cuda':
                        malicious_update = gsp_attack_reduced.cpu()
                    else:
                        malicious_update = gsp_attack_reduced
                else:
                    # Dimension mismatch: log warning and use zeros
                    print(f"    [Attacker {self.client_id}] GSP dimension mismatch: got {gsp_dim}, expected {total_dim}, using zeros")
        else:
            # GSP attack is None: malicious_update remains zeros
            print(f"    [Attacker {self.client_id}] GSP attack is None, using zeros")
        
        # ============================================================
        # Pre-projection removed: Let Lagrangian mechanism handle constraint control directly
        # Reason: Pre-projection effect was negated by optimization process (distance grew back to similar values)
        # If Lagrangian mechanism is effective, it should handle initial distance directly
        # If Lagrangian mechanism is ineffective, pre-projection also doesn't help
        # ============================================================
        # ============================================================
        # STEP 5: (Removed - attackers don't perform local training)
        # Attackers are data-agnostic and don't have local data for training.
        # The attack is generated purely from VGAE+GSP using benign updates.
        # ============================================================
        
        # ============================================================
        # STEP 6: Optimize attack objective according to paper Formula 4
        # Paper: max_{w'_j(t), β'_{i,j}(t)} F(w'_g(t))
        # s.t. d(w'_j(t), w'_g(t)) ≤ d_T  (Constraint 4b)
        #      Σ β'_{i,j}(t) d(w_i(t), w̄_i(t)) ≤ Γ  (Constraint 4c)
        # ============================================================
        
        # Get beta selection indices (which benign updates were selected)
        # Use helper method to avoid tensor comparison issues
        beta_selection = self._get_selected_benign_indices()
        
        # ============================================================
        # STEP 7: Optimize w'_j(t) to maximize F(w'_g(t))
        # According to paper Equation 12, we maximize F(w'_g(t)) subject to constraints
        # ============================================================
        proxy_lr = self.proxy_step
        # Add small client-specific perturbation to initial malicious_update to ensure diversity
        # This helps different attackers converge to different local optima
        if self.is_attacker:
            perturbation_scale = self.opt_init_perturbation_scale * (self.client_id + 1)  # Small scale, client-specific
            initial_perturbation = torch.randn_like(malicious_update) * perturbation_scale
            proxy_param = (malicious_update + initial_perturbation).clone().detach().to(self.device)
        else:
            proxy_param = malicious_update.clone().detach().to(self.device)
        proxy_param.requires_grad_(True)
        proxy_opt = optim.Adam([proxy_param], lr=proxy_lr)
        
        # Check dimension once before loop (performance optimization)
        proxy_param_flat = proxy_param.view(-1)
        dim_valid = int(proxy_param_flat.numel()) == self._flat_numel
        
        # CRITICAL: Ensure model is on GPU before optimization loop
        # Model must stay on GPU during the entire optimization loop to maintain computation graph
        target_device = torch.device('cuda:0') if self.device.type == 'cuda' else self.device
        if not self._model_on_gpu:
            self.model.to(target_device)
            self._ensure_model_on_device(self.model, target_device)
            self._model_on_gpu = True
        
        # OPTIMIZATION 2: Cache constraint (4c) value before optimization loop
        # Constraint (4c): Σ β'_{i,j}(t)^* d(w_i(t), w̄_i(t)) is a constant during optimization
        # because selected_benign does not change during the loop
        constraint_c_value_for_update = 0.0  # Used for updating ρ
        constraint_c_term_base = None  # Cached base term for Lagrangian mode (on GPU)
        
        if self.gamma is not None and len(selected_benign) > 0:
            # Compute constraint (4c) value once before loop (constant value)
            # Paper constraint (4c): Σ β'_{i,j}(t) d(w_i(t), w̄_i(t)) ≤ Γ
            # where w̄_i(t) = Σ_{i=1}^I (D_i(t)/D(t)) w_i(t) is the weighted mean of ALL benign clients
            # (not just selected ones)
            
            # Step 1: Compute weighted mean of ALL benign clients: w̄_i(t) = Σ (D_i/D) w_i(t)
            # Note: This uses all benign_updates, not just selected_benign
            if len(self.benign_updates) > 0 and self.total_data_size is not None and len(self.benign_data_sizes) > 0:
                D_total = float(self.total_data_size)
                # Compute weighted sum of all benign updates
                benign_updates_gpu = [u.to(self.device) for u in self.benign_updates]
                benign_stack = torch.stack(benign_updates_gpu)
                weighted_mean = torch.zeros_like(benign_stack[0])
                
                for idx, benign_update in enumerate(self.benign_updates):
                    # Get client_id for this update
                    if idx < len(self.benign_update_client_ids):
                        client_id = self.benign_update_client_ids[idx]
                        # Get data size for this client
                        D_i = self.benign_data_sizes.get(client_id, 1.0)
                        weight = D_i / D_total
                    else:
                        # Fallback: use equal weight if client_id not available
                        weight = 1.0 / len(self.benign_updates)
                    
                    # Add weighted update
                    weighted_mean = weighted_mean + weight * benign_update.to(self.device)
                
                # Clean up temporary GPU references
                del benign_updates_gpu, benign_stack
                
            else:
                # Fallback: use simple mean if data sizes not available
                sel_benign_gpu = [u.to(self.device) for u in selected_benign]
                sel_stack = torch.stack(sel_benign_gpu)
                weighted_mean = sel_stack.mean(dim=0)
                del sel_benign_gpu, sel_stack
            
            # Step 2: Compute distances d(w_i(t), w̄_i(t)) for each selected benign client
            sel_benign_gpu = [u.to(self.device) for u in selected_benign]
            distances = []
            for idx, benign_update in enumerate(selected_benign):
                # Get the index in beta_selection to find original client_id
                if idx < len(beta_selection):
                    original_idx = beta_selection[idx]
                    # Get distance: d(w_i(t), w̄_i(t))
                    dist = torch.norm(benign_update.to(self.device) - weighted_mean)
                    distances.append(dist)
                else:
                    # Fallback: compute distance anyway
                    dist = torch.norm(benign_update.to(self.device) - weighted_mean)
                    distances.append(dist)
            
            # Step 3: Sum distances: Σ β'_{i,j}(t) d(w_i(t), w̄_i(t))
            if len(distances) > 0:
                distances_tensor = torch.stack(distances)
                agg_dist = distances_tensor.sum()  # Σ β'_{i,j}(t)^* d(...)
            else:
                agg_dist = torch.tensor(0.0, device=self.device)
            
            constraint_c_value_for_update = agg_dist.item()
            
            # For Lagrangian mode, cache the base term on GPU to avoid recomputation
            if self.use_lagrangian_dual:
                constraint_c_term_base = agg_dist  # Keep on GPU for reuse in loop
            else:
                # Hard constraint mode: release GPU memory immediately
                del sel_benign_gpu, distances, weighted_mean
                if 'distances_tensor' in locals():
                    del distances_tensor
                torch.cuda.empty_cache()
        
        # OPTIMIZATION 5: Cache Lagrangian multipliers on GPU before loop
        # Ensure multipliers are on correct device to avoid repeated conversions
        if self.use_lagrangian_dual and self.lambda_dt is not None and self.rho_dt is not None:
            if isinstance(self.lambda_dt, torch.Tensor):
                if not self._device_matches(self.lambda_dt.device, target_device):
                    self.lambda_dt = self.lambda_dt.to(target_device)
            else:
                self.lambda_dt = torch.tensor(self.lambda_dt, device=target_device)
            
            if isinstance(self.rho_dt, torch.Tensor):
                if not self._device_matches(self.rho_dt.device, target_device):
                    self.rho_dt = self.rho_dt.to(target_device)
            else:
                self.rho_dt = torch.tensor(self.rho_dt, device=target_device)
        
        for step in range(self.proxy_steps):
            proxy_opt.zero_grad()
            
            # OPTIMIZATION 4: Reduce device check frequency (every 5 steps instead of every step)
            # Model should remain on GPU during optimization loop, so full check is rarely needed
            if step % 5 == 0:  # Check every 5 steps
                # Quick check using a sample parameter
                sample_param = next(iter(self.model.parameters()), None)
                if sample_param is not None and not self._device_matches(sample_param.device, target_device):
                    # If device mismatch detected, perform full check
                    self._ensure_model_on_device(self.model, target_device)
            # Note: Full device check removed from here for performance (checked before loop and every 5 steps)
            
            # ============================================================
            # Compute base objective function F(w'_g(t)) according to paper Formula (3)
            # ============================================================
            # Use complete global loss formula (3) instead of proxy only
            global_loss = self._compute_global_loss(
                proxy_param,
                selected_benign,
                beta_selection
            )
            
            # ============================================================
            # Build optimization objective: choose mechanism based on whether using Lagrangian Dual
            # ============================================================
            
            # OPTIMIZATION 4: Device check before backward (only if needed)
            # Full device check is expensive, but we need it before backward for LoRA models
            # Check every step before backward to ensure correctness
            device_mismatch_before_backward = False
            sample_param_before_backward = next(iter(self.model.parameters()), None)
            if sample_param_before_backward is not None and not self._device_matches(sample_param_before_backward.device, target_device):
                # If mismatch detected, perform full check
                self._ensure_model_on_device(self.model, target_device)
                device_mismatch_before_backward = True
            
            if self.use_lagrangian_dual and self.lambda_dt is not None and self.rho_dt is not None:
                # ========== Use Lagrangian Dual mechanism (paper eq:lagrangian and eq:wprime_sub) ==========
                
                # OPTIMIZATION 5: Use cached multipliers directly (already on correct device)
                lambda_dt_tensor = self.lambda_dt  # Direct use, no conversion needed
                rho_dt_tensor = self.rho_dt  # Direct use, no conversion needed
                
                # Constraint (4b): d(w'_j(t), w'_g(t)) ≤ d_T
                # Paper Constraint (4b): d(w'_j, w'_g) = ||w'_j - w'_g||
                # Use real distance calculation according to paper Formula (2)
                dist_to_global_for_objective = self._compute_real_distance_to_global(
                    proxy_param,
                    selected_benign,
                    beta_selection
                )
                # Lagrangian method: For constraint g(x) ≤ 0, Lagrangian is L = f(x) - λ g(x)
                # For constraint (4b): d(w'_j, w'_g) - d_T ≤ 0, so L = F(w'_g) - λ (d(w'_j, w'_g) - d_T)
                # Converting to minimization: minimize -L = -F(w'_g) + λ (d(w'_j, w'_g) - d_T)
                # Since λ d_T is constant, we use: minimize -F(w'_g) + λ d(w'_j, w'_g)
                constraint_b_term = lambda_dt_tensor * dist_to_global_for_objective
                
                # Constraint (4c): Σ β'_{i,j}(t) d(w_i(t), w̄_i(t)) ≤ Γ
                # For constraint (4c): Σ(...) - Γ ≤ 0, so L = F(w'_g) - ρ (Σ(...) - Γ)
                # Converting to minimization: minimize -L = -F(w'_g) + ρ (Σ(...) - Γ)
                # Since ρ Γ is constant, we use: minimize -F(w'_g) + ρ Σ(...)
                # OPTIMIZATION 2: Use cached constraint (4c) value (computed before loop)
                constraint_c_term = torch.tensor(0.0, device=self.device)
                if constraint_c_term_base is not None:
                    # Use cached base term (already on GPU)
                    constraint_c_term = rho_dt_tensor * constraint_c_term_base
                
                # ============================================================
                # Build Lagrangian objective function (paper formula eq:wprime_sub)
                # ============================================================
                # Paper: maximize F(w'_g(t)) subject to constraints
                # Lagrangian: L = F(w'_g) - λ (d(w'_j, w'_g) - d_T) - ρ (Σ(...) - Γ)
                # Converting to minimization: minimize -L = -F(w'_g) + λ d(w'_j, w'_g) + ρ Σ(...)
                # (constant terms λ d_T and ρ Γ are omitted as they don't affect optimization direction)
                lagrangian_objective = -global_loss + constraint_b_term + constraint_c_term
                
                # ============================================================
                # ============================================================
                # Compute constraint violations (for updating λ and ρ)
                # ============================================================
                constraint_b_violation = F.relu(dist_to_global_for_objective - self.d_T) if self.d_T is not None else torch.tensor(0.0, device=self.device)
            
            constraint_c_violation = torch.tensor(0.0, device=self.device)
            if self.gamma is not None and len(selected_benign) > 0:
                    constraint_c_violation = F.relu(torch.tensor(constraint_c_value_for_update, device=self.device) - self.gamma)
                
            else:
                # ========== Use hard constraint projection mechanism (original logic) ==========
                # Objective: maximize global_loss => minimize -global_loss
                lagrangian_objective = -global_loss
                
                # Compute constraint violations (for logging only)
                # Use real distance calculation according to paper Constraint (4b)
                dist_to_global = self._compute_real_distance_to_global(
                    proxy_param,
                    selected_benign,
                    beta_selection
                ) if self.d_T is not None else torch.tensor(0.0, device=self.device)
                constraint_b_violation = F.relu(dist_to_global - self.d_T) if self.d_T is not None else torch.tensor(0.0, device=self.device)
                
                # OPTIMIZATION 6: Constraint (4c) calculation moved to after loop for hard constraint mode
                # In hard constraint mode, constraint (4c) is only used for logging, not in optimization
                # So we can delay its computation until after the loop
                constraint_c_violation = torch.tensor(0.0, device=self.device)
                # constraint_c_value_for_update is already computed before loop (for Lagrangian mode compatibility)
            
            # CRITICAL: Before backward pass, ensure ALL model parameters and buffers are on GPU
            # This is essential for LoRA models where nested structures might have been affected
            # The computation graph created during forward pass references these parameters
            # If any parameter is on CPU during backward, we'll get a device mismatch error
            # OPTIMIZATION 4: Only perform full check if mismatch was detected by sample check
            if device_mismatch_before_backward:
                for name, param in self.model.named_parameters():
                    if not self._device_matches(param.device, target_device):
                        print(f"    [Attacker {self.client_id}] CRITICAL PRE-BACKWARD: Parameter {name} on {param.device}, forcing to {target_device}")
                        with torch.no_grad():
                            param.data = param.data.to(target_device, non_blocking=True)
                for name, buffer in self.model.named_buffers():
                    if not self._device_matches(buffer.device, target_device):
                        print(f"    [Attacker {self.client_id}] CRITICAL PRE-BACKWARD: Buffer {name} on {buffer.device}, forcing to {target_device}")
                        buffer.data = buffer.data.to(target_device, non_blocking=True)
                print(f"    [Attacker {self.client_id}] WARNING: Fixed device mismatches before backward pass")
                # Final recursive check to ensure everything is consistent
                self._ensure_model_on_device(self.model, target_device)
                # One more model.to() to be absolutely sure
                self.model.to(target_device)
            
            # ============================================================
            # Backpropagation and parameter update
            # ============================================================
            # Backpropagate Lagrangian objective
            lagrangian_objective.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_([proxy_param], max_norm=self.grad_clip_norm)
            
            proxy_opt.step()
            
            # ============================================================
            # Update Lagrangian multipliers (if using Lagrangian mechanism)
            # Use subgradient method according to paper dual problem
            # Paper Algorithm 1 Step 7: Update λ(t) and ρ(t) according to eq:dual
            # ============================================================
            if self.use_lagrangian_dual and self.lambda_dt is not None and self.rho_dt is not None:
                # Dual ascent method: λ(t+1) = λ(t) + α_λ × subgradient
                # subgradient = (constraint value - bound)
                # When constraint is violated (constraint value > bound), subgradient > 0, λ increases to penalize violation
                
                if self.d_T is not None:
                    # Calculate real distance after step() (proxy_param has been updated)
                    # Use real distance calculation according to paper Constraint (4b)
                    current_dist_tensor = self._compute_real_distance_to_global(
                        proxy_param,
                        selected_benign,
                        beta_selection
                    )
                    current_dist = current_dist_tensor.item()
                    lambda_val = self.lambda_dt.item() if isinstance(self.lambda_dt, torch.Tensor) else self.lambda_dt
                    # Subgradient: d(w'_j, w'_g) - d_T
                    # If constraint is violated (d(...) > d_T), subgradient > 0, λ increases
                    subgradient_b = current_dist - self.d_T
                    new_lambda = lambda_val + self.lambda_lr * subgradient_b
                    new_lambda = max(0.0, new_lambda)  # Ensure non-negative
                    # OPTIMIZATION 5: Keep multiplier on same device when updating
                    self.lambda_dt = torch.tensor(new_lambda, device=target_device, requires_grad=False)
                
                if self.gamma is not None and len(selected_benign) > 0:
                    rho_val = self.rho_dt.item() if isinstance(self.rho_dt, torch.Tensor) else self.rho_dt
                    # Subgradient: Σ(...) - Γ
                    # If constraint is violated (Σ(...) > Γ), subgradient > 0, ρ increases
                    subgradient_c = constraint_c_value_for_update - self.gamma
                    new_rho = rho_val + self.rho_lr * subgradient_c
                    new_rho = max(0.0, new_rho)  # Ensure non-negative
                    # OPTIMIZATION 5: Keep multiplier on same device when updating
                    self.rho_dt = torch.tensor(new_rho, device=target_device, requires_grad=False)
            
            # ============================================================
            # Modification 4: Add constraint safeguard mechanism within optimization loop (Lagrangian framework)
            # Under Lagrangian mechanism, add light projection as safeguard to prevent optimization path from deviating too far
            # ============================================================
            if not self.use_lagrangian_dual and self.d_T is not None:
                # Hard constraint projection (original logic, behavior when not using Lagrangian)
                dist_to_global = torch.norm(proxy_param).item()
                if dist_to_global > self.d_T:
                    # Project to constraint set: scale down to satisfy d ≤ d_T
                    proxy_param.data = proxy_param.data * (self.d_T / dist_to_global)
            elif self.use_lagrangian_dual and self.d_T is not None:
                # Modification 4: Constraint safeguard under Lagrangian mechanism
                # Check within optimization loop, if violation exceeds 20%, immediately apply light projection to prevent path deviation
                # This maintains Lagrangian flexibility while ensuring constraints are promptly safeguarded
                # Use real distance calculation according to paper Constraint (4b)
                dist_to_global_for_projection_tensor = self._compute_real_distance_to_global(
                    proxy_param,
                    selected_benign,
                    beta_selection
                )
                dist_to_global_for_projection = dist_to_global_for_projection_tensor.item()
                
                violation_ratio = (dist_to_global_for_projection - self.d_T) / self.d_T if self.d_T > 0 else 0.0
                
                if violation_ratio > 0.20:  # Apply light projection when violation exceeds 20%
                    # Light projection to 1.1 × d_T, leaving 10% margin to allow Lagrangian to continue optimizing
                    target_dist = self.d_T * 1.1
                    scale_factor = target_dist / dist_to_global_for_projection
                    proxy_param.data = proxy_param.data * scale_factor
        
        malicious_update = proxy_param.detach()
        
        # CRITICAL: Now that optimization is complete, move model back to CPU to free GPU memory
        # The computation graph is no longer needed after optimization
        if self._model_on_gpu:
            self.model.cpu()
            self._ensure_model_on_device(self.model, torch.device('cpu'))
            self._model_on_gpu = False
        
        # Clean up optimizer to free GPU memory
        del proxy_opt
        torch.cuda.empty_cache()
        
        # ============================================================
        # STEP 8: Final constraint check (depending on whether Lagrangian is used)
        # ============================================================
        if self.use_lagrangian_dual:
            # Final constraint check: Graded projection strategy (within Lagrangian framework)
            # Use different strategies based on violation degree to balance flexibility and constraint satisfaction
            if self.d_T is not None:
                # Use real distance calculation according to paper Constraint (4b)
                dist_to_global_tensor = self._compute_real_distance_to_global(
                    malicious_update,
                    selected_benign,
                    beta_selection
                )
                dist_to_global = dist_to_global_tensor.item()
                constraint_violation = max(0, dist_to_global - self.d_T)
                violation_ratio = constraint_violation / self.d_T if self.d_T > 0 else 0.0
                
                # Store violation for adaptive initialization in next round (Optimization)
                self.last_violation = dist_to_global
                
                if constraint_violation > 0:
                    lambda_val = self.lambda_dt.item() if isinstance(self.lambda_dt, torch.Tensor) else self.lambda_dt
                    print(f"    [Attacker {self.client_id}] Constraint(4b) violation: {constraint_violation:.6f} "
                          f"(λ={lambda_val:.4f}, violation={violation_ratio*100:.1f}%)")
                    
                    # Check if final projection is enabled
                    if self.enable_final_projection:
                        # Graded projection strategy (improved to allow Lagrangian mechanism to work):
                        # Increased threshold from 30% to 100% to reduce frequent strict projection
                        # This allows Lagrangian multipliers (λ) to effectively control violations
                        if violation_ratio > 1.00:  # Severe violation (>100%)
                            # Strict projection to d_T, completely satisfy constraint
                            # Only applied when violation is extremely severe to preserve optimization stability
                            scale_factor = self.d_T / dist_to_global
                            malicious_update = malicious_update * scale_factor
                            print(f"      Applied strict projection (violation >100%): scaled to d_T")
                        elif violation_ratio > 0.50:  # Moderate violation (50-100%)
                            # Mild projection to 1.1×d_T, allowing 10% margin for Lagrangian flexibility
                            target_dist = self.d_T * 1.1
                            scale_factor = target_dist / dist_to_global
                            malicious_update = malicious_update * scale_factor
                            print(f"      Applied mild projection (violation 50-100%): scaled to 1.1×d_T")
                        else:  # Minor violation (<50%)
                            # Allow minor violation, leverage Lagrangian mechanism flexibility
                            # No projection, controlled by multipliers (λ, ρ)
                            print(f"      Allowed minor violation (violation <50%): kept as is, controlled by Lagrangian multipliers")
                    else:
                        # Final projection is disabled: completely rely on Lagrangian mechanism
                        print(f"      Final projection disabled: relying entirely on Lagrangian mechanism (violation={violation_ratio*100:.1f}%)")
        else:
            # Hard constraint projection mechanism (original logic)
            if self.d_T is not None:
                # Use real distance calculation according to paper Constraint (4b)
                dist_to_global_tensor = self._compute_real_distance_to_global(
                    malicious_update,
                    selected_benign,
                    beta_selection
                )
                dist_to_global = dist_to_global_tensor.item()
            
            if dist_to_global > self.d_T:
                # Hard constraint: project to satisfy d ≤ d_T
                scale_factor = self.d_T / dist_to_global
                malicious_update = malicious_update * scale_factor
                final_norm = torch.norm(malicious_update).item()
                print(f"    [Attacker {self.client_id}] Applied hard constraint projection: "
                      f"scaled from {dist_to_global:.4f} to {final_norm:.4f}")
        
        # Compute final attack objective value for logging
        # Use evaluation max_batches for more accurate final assessment
        final_global_loss = self._proxy_global_loss(malicious_update, max_batches=self.proxy_max_batches_eval, skip_dim_check=False)
        
        # Compute constraint (4c) for logging
        # OPTIMIZATION 2 & 6: For Lagrangian mode, reuse cached value; for hard constraint mode, compute here
        if self.use_lagrangian_dual and constraint_c_value_for_update > 0:
            # Reuse cached value from before loop (already computed)
            constraint_c_value = torch.tensor(constraint_c_value_for_update, device=self.device)
        else:
            # Hard constraint mode: compute here (only needed for logging)
            # Use the same calculation as in Lagrangian mode for consistency
            constraint_c_value = torch.tensor(0.0, device=self.device)
            if self.gamma is not None and len(selected_benign) > 0:
                # Compute constraint (4c) value using weighted mean of ALL benign clients
                # Paper definition: w̄_i(t) = Σ_{i=1}^I (D_i(t)/D(t)) w_i(t) (weighted mean of ALL benign clients)
                if len(self.benign_updates) > 0 and self.total_data_size is not None and len(self.benign_data_sizes) > 0:
                    D_total = float(self.total_data_size)
                    # Compute weighted mean of ALL benign clients
                    benign_updates_gpu = [u.to(self.device) for u in self.benign_updates]
                    benign_stack = torch.stack(benign_updates_gpu)
                    weighted_mean = torch.zeros_like(benign_stack[0])
                    for idx, benign_update in enumerate(self.benign_updates):
                        if idx < len(self.benign_update_client_ids):
                            client_id = self.benign_update_client_ids[idx]
                            D_i = self.benign_data_sizes.get(client_id, 1.0)
                            weight = D_i / D_total
                        else:
                            weight = 1.0 / len(self.benign_updates)
                        weighted_mean = weighted_mean + weight * benign_update.to(self.device)
                    
                    # Compute distances d(w_i(t), w̄_i(t)) for selected benign clients
                    sel_benign_gpu = [u.to(self.device) for u in selected_benign]
                    distances = [torch.norm(benign_update - weighted_mean) for benign_update in sel_benign_gpu]
                    constraint_c_value = torch.stack(distances).sum()
                    # Clean up GPU references
                    del benign_updates_gpu, benign_stack, weighted_mean, sel_benign_gpu, distances
                else:
                    # Fallback: use simple mean if data sizes not available
                    sel_benign_gpu = [u.to(self.device) for u in selected_benign]
                    sel_stack = torch.stack(sel_benign_gpu)
                    sel_mean = sel_stack.mean(dim=0)
                    distances = torch.norm(sel_stack - sel_mean, dim=1)
                    constraint_c_value = distances.sum()
                    # Clean up GPU references
                    del sel_benign_gpu, sel_stack, sel_mean, distances
                torch.cuda.empty_cache()
        
        malicious_norm = torch.norm(malicious_update).item()
        log_msg = f"    [Attacker {self.client_id}] GRMP Attack: " \
                  f"F(w'_g)={final_global_loss.item():.4f}, " \
                  f"||w'_j||={malicious_norm:.4f}, " \
                  f"constraint_c={constraint_c_value.item():.4f}"
        
        # If using the Lagrangian mechanism, display the multiplier value
        if self.use_lagrangian_dual and self.lambda_dt is not None and self.rho_dt is not None:
            lambda_val = self.lambda_dt.item() if isinstance(self.lambda_dt, torch.Tensor) else self.lambda_dt
            rho_val = self.rho_dt.item() if isinstance(self.rho_dt, torch.Tensor) else self.rho_dt
            log_msg += f", λ={lambda_val:.4f}, ρ={rho_val:.4f}"
        
        print(log_msg)
        
        # Move malicious_update to CPU before returning to free GPU memory
        malicious_update_cpu = malicious_update.cpu()
        # Clean up GPU references
        del malicious_update, final_global_loss, constraint_c_value
        torch.cuda.empty_cache()
        
        return malicious_update_cpu.detach()
