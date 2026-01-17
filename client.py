# client.py
# Provides the Client class for federated learning clients, including benign and attacker clients.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
from models import VGAE
from torch.nn.utils import stateless
# ============================================================================
# CRITICAL: Functional call wrapper for LoRA gradient preservation
# ============================================================================
# Purpose: Use torch.func.functional_call to preserve gradients when injecting
#          LoRA parameters into the model forward pass. This ensures the
#          computational graph remains intact from proxy_param to loss.
#
# API Note: torch.func.functional_call (PyTorch 2.0+) takes (params, buffers) tuple
#           stateless.functional_call (PyTorch < 2.0) only takes params dict
# ============================================================================
try:
    from torch.func import functional_call as _torch_func_call
    def functional_call(model, params_buffers, args=(), kwargs=None):
        """
        Wrapper for torch.func.functional_call (PyTorch 2.0+).
        
        Args:
            model: PyTorch model
            params_buffers: Tuple of (params_dict, buffers_dict)
            args: Positional arguments for forward pass
            kwargs: Keyword arguments for forward pass
            
        Returns:
            Model output with preserved gradients
        """
        params, buffers = params_buffers
        return _torch_func_call(model, (params, buffers), args=args, kwargs=kwargs or {})
except ImportError:
    # Fallback for older PyTorch versions (< 2.0)
    from torch.nn.utils.stateless import functional_call as _stateless_call
    def functional_call(model, params_buffers, args=(), kwargs=None):
        """
        Fallback wrapper for torch.nn.utils.stateless.functional_call (PyTorch < 2.0).
        
        Note: stateless.functional_call doesn't support buffers parameter, so we only
              pass params. This is acceptable because buffers are typically constant
              and don't need to be injected for gradient preservation.
        """
        params, buffers = params_buffers
        return _stateless_call(model, params, args=args, kwargs=kwargs or {})

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
                 grad_clip_norm=1.0,
                 early_stop_constraint_stability_steps=3):
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
        self.early_stop_constraint_stability_steps = early_stop_constraint_stability_steps

        dummy_loader = data_manager.get_empty_loader()
        super().__init__(client_id, model, dummy_loader, lr, local_epochs, alpha)
        self.is_attacker = True

        # VGAE components
        self.vgae = None
        self.vgae_optimizer = None
        self.benign_updates = []
        self.benign_update_client_ids = []  # Track client_id for each benign update to enable weighted average calculation
        self.feature_indices = None
        
        # Other attackers' updates (for coordinated optimization)
        self.other_attacker_updates = []
        self.other_attacker_client_ids = []
        self.other_attacker_data_sizes = {}  # {client_id: claimed_data_size}
        
        # Data-agnostic attack: no local data usage
        self.original_business_loader = None
        self.proxy_loader = data_manager.get_proxy_eval_loader(sample_size=self.proxy_sample_size)
        
        # Formula 4 constraints parameters
        self.d_T = None  # Distance threshold for constraint (4b): d(w'_j(t), w'_g(t)) ≤ d_T
        # ===== CONSTRAINT (4c) COMMENTED OUT =====
        # self.gamma = None  # Upper bound for constraint (4c): Σ β'_{i,j}(t) d(w_i(t), w̄_i(t)) ≤ Γ
        self.gamma = None  # Temporarily set to None to disable constraint (4c)
        # ==========================================
        self.global_model_params = None  # Store global model params for constraint (4b) (will be on GPU when needed)
        # Paper Formula (2): w'_g(t) = Σ_{i=1}^I (D_i(t)/D(t)) β'_{i,j}(t) w_i(t) + (D'_j(t)/D(t)) w'_j(t)
        self.total_data_size = None  # D(t): Total data size for aggregation weight calculation
        self.benign_data_sizes = {}  # {client_id: D_i(t)}: Data sizes for each benign client
        
        # Lagrangian dual variables (λ(t) and ρ(t) from paper)
        # Initialized in set_lagrangian_params
        self.lambda_dt = None  # λ(t): Lagrangian multiplier for constraint (4b)
        # ===== CONSTRAINT (4c) COMMENTED OUT =====
        # self.rho_dt = None     # ρ(t): Lagrangian multiplier for constraint (4c)
        self.rho_dt = None     # Temporarily disabled (constraint 4c is commented out)
        # self.rho_lr = 0.01     # Learning rate for ρ(t) update
        self.rho_lr = 0.01     # Temporarily disabled (constraint 4c is commented out)
        # self.rho_init_value = None     # Save initial ρ value for reset in prepare_for_round
        self.rho_init_value = None     # Temporarily disabled (constraint 4c is commented out)
        # ==========================================
        self.use_lagrangian_dual = False  # Whether to use Lagrangian Dual mechanism
        self.lambda_lr = 0.01  # Learning rate for λ(t) update
        # Save initial values for reset in prepare_for_round (Modification 1 and 2)
        self.lambda_init_value = None  # Save initial λ value for reset in prepare_for_round
        self.enable_final_projection = True  # Whether to apply final projection after optimization (only for Lagrangian mode)
        self.enable_light_projection_in_loop = True  # Whether to apply light projection within optimization loop (only for Lagrangian mode)
        
        # Track violation history for adaptive λ initialization (Optimization)
        self.last_violation = None  # Last round's constraint violation value (distance, not violation amount)
        
        # Get model parameter count (works on CPU model)
        self._flat_numel = int(self.model.get_flat_params().numel())  # Convert to Python int
        
        # ===== CRITICAL: LoRA functional_call cache for gradient preservation =====
        # These will be initialized in _init_functional_param_cache() when needed
        self.lora_param_names: List[str] = []  # Ordered list of LoRA parameter names
        self.lora_param_shapes: Dict[str, torch.Size] = {}  # Shape for each LoRA param
        self.lora_param_numels: Dict[str, int] = {}  # Numel for each LoRA param
        self.lora_param_slices: Dict[str, slice] = {}  # Slice in flat tensor for each LoRA param
        self.base_params: Dict[str, torch.Tensor] = {}  # Frozen base parameters (detached)
        self.base_buffers: Dict[str, torch.Tensor] = {}  # Buffers (detached)
        self._functional_cache_initialized = False  # Cache initialization flag
        # ============================================================================
        
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

        # ===== CRITICAL: Reset functional cache for new round =====
        # Model structure may change between rounds, so cache must be reset
        self._functional_cache_initialized = False
        self.lora_param_names = []
        self.lora_param_shapes = {}
        self.lora_param_numels = {}
        self.lora_param_slices = {}
        self.base_params = {}
        self.base_buffers = {}
        # ============================================================

        # Store base d_T if adaptive d_T is enabled (needed for adaptive calculation)
        if self.adaptive_d_T and self.base_d_T is None:
            self.base_d_T = float(self.d_T) if isinstance(self.d_T, torch.Tensor) else self.d_T if self.d_T is not None else 10.0

        # Modification 1: Reset Lagrangian multipliers (with adaptive initialization based on history)
        # Reason: Prevent λ and ρ from accumulating across rounds, which causes numerical instability and optimization imbalance
        # Optimization: Use adaptive λ initialization based on previous round's violation to provide better starting point
        # ===== CONSTRAINT (4c) COMMENTED OUT: Removed rho_init_value check =====
        if self.use_lagrangian_dual and self.lambda_init_value is not None:  # Removed rho_init_value check
            # Adaptive λ initialization: if last round had large violation, use larger initial λ
            d_T_init = float(self.d_T) if isinstance(self.d_T, torch.Tensor) else self.d_T if self.d_T is not None else None
            if self.last_violation is not None and d_T_init is not None and self.last_violation > d_T_init * 1.5:
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
            # ===== CONSTRAINT (4c) COMMENTED OUT =====
            # self.rho_dt = torch.tensor(self.rho_init_value, requires_grad=False)
            # ==========================================

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
    
    def receive_attacker_updates(self, updates: List[torch.Tensor], client_ids: List[int], data_sizes: Dict[int, float] = None):
        """
        Receive updates from other attackers that have already completed optimization.
        These will be used in distance calculation to match Phase 4's reference point.
        
        Args:
            updates: List of attacker update tensors (already optimized)
            client_ids: List of attacker client IDs
            data_sizes: Dictionary mapping client_id to claimed_data_size (optional)
        """
        # Store detached copies on CPU to save GPU memory
        self.other_attacker_updates = [u.detach().clone().cpu() for u in updates]
        self.other_attacker_client_ids = client_ids.copy() if client_ids else []
        
        # Store data sizes for weighted aggregation
        if data_sizes is not None:
            self.other_attacker_data_sizes = data_sizes.copy()
        else:
            # Fallback: use current attacker's claimed size as estimate
            self.other_attacker_data_sizes = {cid: float(self.claimed_data_size) for cid in client_ids}

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
        
        # ===== CONSTRAINT (4c) COMMENTED OUT: Always use all updates =====
        # If gamma is not set, use all updates
        # if self.gamma is None:
        #     return self.benign_updates
        # 
        # # Convert gamma to capacity
        # capacity = float(self.gamma)
        # n = len(distances)
        # 
        # # Handle edge cases
        # if n == 0:
        #     return []
        # if capacity <= 0:
        #     # If capacity is 0 or negative, select item with minimum distance
        #     min_idx = np.argmin(distances)
        #     return [self.benign_updates[min_idx]]
        # 
        # # For 0-1 Knapsack with minimization objective:
        # # We want to minimize sum of selected distances, subject to sum ≤ capacity
        # # This is equivalent to: maximize number of items selected (or maximize sum of unselected)
        # # while keeping sum of selected ≤ capacity
        # 
        # # Greedy approach: sort by distance and select items until capacity is reached
        # # This gives a good approximation and is efficient
        # sorted_indices = np.argsort(distances)  # Sort by distance (ascending)
        # 
        # selected_indices = []
        # total_dist = 0.0
        # 
        # for idx in sorted_indices:
        #     d = distances[idx]
        #     if total_dist + d <= capacity:
        #         selected_indices.append(idx)
        #         total_dist += d
        #     else:
        #         break
        # 
        # # If no items selected (all distances > capacity), select the one with minimum distance
        # if not selected_indices:
        #     min_idx = np.argmin(distances)
        #     selected_indices = [min_idx]
        # 
        # # Return selected updates
        # selected = [self.benign_updates[i] for i in sorted(selected_indices)]
        # 
        # return selected
        # ====================================================================
        # Since constraint (4c) is disabled, always return all benign updates
        return self.benign_updates
    
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
        
        # ===== CONSTRAINT (4c) COMMENTED OUT: Always return all indices =====
        # If gamma is not set, use all updates
        # if self.gamma is None:
        #     return list(range(len(self.benign_updates)))
        # 
        # # Convert gamma to capacity
        # capacity = float(self.gamma)
        # n = len(distances)
        # 
        # # Handle edge cases
        # if n == 0:
        #     return []
        # if capacity <= 0:
        #     # If capacity is 0 or negative, select item with minimum distance
        #     min_idx = np.argmin(distances)
        #     return [int(min_idx)]
        # 
        # # Greedy approach: sort by distance and select items until capacity is reached
        # sorted_indices = np.argsort(distances)  # Sort by distance (ascending)
        # 
        # selected_indices = []
        # total_dist = 0.0
        # 
        # for idx in sorted_indices:
        #     d = distances[idx]
        #     if total_dist + d <= capacity:
        #         selected_indices.append(int(idx))
        #         total_dist += d
        #     else:
        #         break
        # 
        # # If no items selected (all distances > capacity), select the one with minimum distance
        # if not selected_indices:
        #     min_idx = np.argmin(distances)
        #     selected_indices = [int(min_idx)]
        # 
        # return sorted(selected_indices)
        # =====================================================================
        # Since constraint (4c) is disabled, always return all indices
        return list(range(len(self.benign_updates)))

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

    def _init_functional_param_cache(self, target_device: torch.device):
        """
        Initialize cache for functional_call with full parameters (base + LoRA).
        
        CRITICAL: This must be called before using functional_call in LoRA mode.
        Caches LoRA parameter metadata and base parameters/buffers for gradient-preserving forward.
        
        What this function does:
        1. Collects LoRA parameters (trainable) in the same order as get_flat_params()
        2. Caches base parameters (frozen, detached) to avoid repeated lookups
        3. Caches buffers (detached) for functional_call
        4. Verifies dimension consistency (sum(LoRA numel) == _flat_numel)
        5. Verifies parameter name completeness (base + LoRA = all params)
        
        Args:
            target_device: Device to cache parameters on (typically GPU)
        
        Raises:
            AssertionError: If dimension consistency checks fail
            RuntimeError: If parameter name lookup fails
        """
        if self._functional_cache_initialized:
            return  # Already initialized
        
        use_lora = hasattr(self.model, 'use_lora') and self.model.use_lora
        if not use_lora:
            # Full fine-tuning mode doesn't need special caching
            self._functional_cache_initialized = True
            return
        
        # Step 1: Build LoRA parameter metadata (trainable parameters only)
        # ============================================================
        # CRITICAL: Must match get_flat_params() order EXACTLY
        # ============================================================
        # get_flat_params() in models.py uses: 
        #   for param in self.model.parameters():
        #       if param.requires_grad:
        #           lora_params.append(param.data.view(-1))
        #
        # Problem: parameters() and named_parameters() order may differ!
        # Solution: Build a mapping from param object to name, then iterate in parameters() order
        # ============================================================
        self.lora_param_names = []
        self.lora_param_shapes = {}
        self.lora_param_numels = {}
        self.lora_param_slices = {}
        offset = 0
        
        # CRITICAL: Build param -> name mapping first
        # This allows us to iterate in parameters() order (matching get_flat_params())
        # while still getting parameter names (needed for functional_call)
        param_to_name = {param: name for name, param in self.model.named_parameters()}
        
        # CRITICAL: Iterate in parameters() order (SAME as get_flat_params() in models.py)
        # This ensures exact order match, preventing parameter misalignment
        for param in self.model.parameters():
            if param.requires_grad:
                # Get parameter name from mapping
                name = param_to_name[param]
                numel = int(param.numel())
                self.lora_param_names.append(name)
                self.lora_param_shapes[name] = param.shape
                self.lora_param_numels[name] = numel
                self.lora_param_slices[name] = slice(offset, offset + numel)
                offset += numel
        
        # Step 2: Build base parameters dict (frozen parameters, detached)
        self.base_params = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                # Frozen base parameter - detach and move to target device
                with torch.no_grad():
                    base_param = param.data.clone().detach().to(target_device)
                self.base_params[name] = base_param
        
        # Step 3: Build buffers dict (detached)
        self.base_buffers = {}
        for name, buffer in self.model.named_buffers():
            with torch.no_grad():
                base_buffer = buffer.data.clone().detach().to(target_device)
            self.base_buffers[name] = base_buffer
        
        # Step 4: Consistency assertions (CRITICAL)
        total_lora_numel = sum(self.lora_param_numels.values())
        if total_lora_numel != self._flat_numel:
            # Enhanced error message with diagnostic information
            model_params_info = f"Model has {len(list(self.model.named_parameters()))} total params, " \
                              f"{len([p for p in self.model.parameters() if p.requires_grad])} trainable"
            raise RuntimeError(
                f"[Attacker {self.client_id}] LoRA dimension mismatch:\n"
                f"  - Total LoRA numel (from cache): {total_lora_numel}\n"
                f"  - _flat_numel (from get_flat_params): {self._flat_numel}\n"
                f"  - LoRA param names: {self.lora_param_names}\n"
                f"  - {model_params_info}\n"
                f"This indicates parameter order mismatch between get_flat_params() and _init_functional_param_cache()."
            )
        
        all_param_names = set(dict(self.model.named_parameters()).keys())
        expected_param_names = set(self.base_params.keys()) | set(self.lora_param_names)
        if all_param_names != expected_param_names:
            missing_in_cache = all_param_names - expected_param_names
            extra_in_cache = expected_param_names - all_param_names
            raise RuntimeError(
                f"[Attacker {self.client_id}] Parameter name mismatch:\n"
                f"  - Model params: {len(all_param_names)} params\n"
                f"  - Cache params: {len(expected_param_names)} params\n"
                f"  - Missing in cache: {missing_in_cache}\n"
                f"  - Extra in cache: {extra_in_cache}\n"
                f"  - Base params: {set(self.base_params.keys())}\n"
                f"  - LoRA params: {self.lora_param_names}"
            )
        
        self._functional_cache_initialized = True
        print(f"    [Attacker {self.client_id}] Functional cache initialized: "
              f"{len(self.lora_param_names)} LoRA params, {len(self.base_params)} base params, "
              f"{len(self.base_buffers)} buffers")

    def _flat_to_lora_param_dict(self, flat_lora: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Convert flat LoRA tensor to parameter dict for functional_call.
        
        CRITICAL: This preserves gradients by using view/reshape operations only.
        No .data operations that would break the computational graph.
        
        How it works:
        - Uses pre-computed slices (from _init_functional_param_cache) to extract
          each parameter from the flat tensor
        - Uses .view() to reshape without copying (preserves gradients)
        - Order must match get_flat_params() to ensure correctness
        
        Args:
            flat_lora: 1D flat LoRA parameter tensor (requires_grad=True, on GPU)
                      Should have shape (self._flat_numel,)
        
        Returns:
            Dictionary mapping LoRA parameter names to shaped tensors (preserves gradients)
            Each tensor maintains requires_grad=True and gradient flow
        
        Raises:
            RuntimeError: If functional cache not initialized
        """
        if not self._functional_cache_initialized:
            raise RuntimeError(f"[Attacker {self.client_id}] Functional cache not initialized. "
                             f"Call _init_functional_param_cache() first.")
        
        flat_lora = flat_lora.view(-1)  # Ensure 1D
        
        out = {}
        for name in self.lora_param_names:
            sl = self.lora_param_slices[name]
            shape = self.lora_param_shapes[name]
            # CRITICAL: Use view/reshape to preserve gradients, no copy_() or .data assignment
            out[name] = flat_lora[sl].view(shape)
        
        return out

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
            use_lora = hasattr(self.model, 'use_lora') and self.model.use_lora
            msg = (f"    [Attacker {self.client_id}] Proxy loss dimension mismatch: "
                   f"got {malicious_update.numel()}, expected {self._flat_numel}")
            if use_lora:
                raise RuntimeError(msg)
            print(msg)
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
            
            # CRITICAL: Check LoRA mode BEFORE processing to avoid unnecessary work
            use_lora = hasattr(self.model, 'use_lora') and self.model.use_lora
            
            # ===== CRITICAL: Initialize functional cache for LoRA mode =====
            # This must be done before any parameter processing
            if use_lora:
                self._init_functional_param_cache(target_device)
            # ===================================================================
            
            # For full fine-tuning mode, prepare param_dict (LoRA mode doesn't need this)
            param_dict = {}
            if not use_lora:
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
                
                if use_lora:
                    # ============================================================================
                    # CRITICAL: LoRA mode with functional_call (NO FALLBACK)
                    # ============================================================================
                    # This path MUST preserve gradients from proxy_param to loss.
                    # PROHIBITED: NO .data operations, NO copy_(), NO no_grad() write operations
                    #
                    # Flow:
                    #   1. candidate_lora_flat = global_params + malicious_update (LoRA-only flat)
                    #   2. lora_params = _flat_to_lora_param_dict(candidate_lora_flat) [preserves gradients]
                    #   3. full_params = base_params (detached) + lora_params (with gradients)
                    #   4. functional_call(model, (full_params, full_buffers)) [preserves gradients]
                    #   5. loss = F.cross_entropy(logits, labels) [gradient flows back to proxy_param]
                    # ============================================================================
                    
                    # Step 1: Ensure candidate is LoRA-only flat (from global + malicious_update)
                    # candidate_params is already computed as: global_params_gpu + malicious_update
                    # Both global_params_gpu and malicious_update are LoRA-only flat tensors
                    candidate_lora_flat = candidate_params
                    
                    # Step 2: Convert flat LoRA to param dict (preserves gradients via view/reshape)
                    # Uses pre-computed slices from _init_functional_param_cache to extract each parameter
                    # Maintains requires_grad=True throughout, preserving computational graph
                    lora_params = self._flat_to_lora_param_dict(candidate_lora_flat)
                    
                    # Step 3: Construct full_params = base_params (constants) + lora_params (trainable)
                    # CRITICAL: base_params are detached constants (requires_grad=False),
                    #          lora_params maintain gradients (requires_grad=True)
                    full_params = dict(self.base_params)  # Shallow copy of base params (detached)
                    full_params.update(lora_params)  # Add LoRA params (with gradients)
                    
                    # Step 4: Ensure base_buffers are on correct device
                    # Buffers are constants (e.g., batch norm running means), no gradients needed
                    full_buffers = {}
                    for name, buf in self.base_buffers.items():
                        if not self._device_matches(buf.device, target_device):
                            full_buffers[name] = buf.to(target_device)
                        else:
                            full_buffers[name] = buf
                    
                    # Step 5: Use functional_call for forward pass (preserves gradients)
                    # CRITICAL: This is the ONLY valid path - no fallback allowed
                    # functional_call injects parameters without breaking computational graph
                    # If this fails, it indicates a configuration error, not a recoverable issue
                    try:
                        logits = functional_call(
                            self.model,
                            (full_params, full_buffers),
                            args=(),
                            kwargs={'input_ids': input_ids, 'attention_mask': attention_mask}
                        )
                    except (RuntimeError, KeyError, TypeError) as e:
                        # FATAL ERROR: functional_call failure indicates configuration problem
                        # Possible causes:
                        #   1. Parameter names don't match (check _init_functional_param_cache)
                        #   2. Missing parameters/buffers in full_params/full_buffers
                        #   3. Device mismatch between params and model
                        #   4. Model structure incompatibility with functional_call
                        error_msg = (
                            f"[Attacker {self.client_id}] FATAL: functional_call failed in LoRA mode: {e}\n"
                            f"This indicates a configuration error - functional_call MUST work.\n"
                            f"Check: (1) Parameter names match, (2) All params/buffers present, "
                            f"(3) Device consistency, (4) Model structure compatibility."
                        )
                        raise RuntimeError(error_msg) from e
                
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
            # ============================================================
            # CRITICAL: LoRA mode failures must raise, not return 0 loss
            # ============================================================
            # If LoRA mode, functional_call failure indicates FATAL error
            # Must raise to prevent silent optimization failure
            use_lora = hasattr(self.model, 'use_lora') and self.model.use_lora
            if use_lora:
                error_msg = (
                    f"[Attacker {self.client_id}] FATAL: LoRA functional_call failed in _proxy_global_loss: {e}\n"
                    f"LoRA mode requires functional_call to work - this is a configuration error.\n"
                    f"Cannot continue optimization with broken gradient link."
                )
                raise RuntimeError(error_msg) from e
            
            # Non-LoRA mode: Allow fallback (for backward compatibility)
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
        """
        Set global model parameters for constraint (4b) calculation.
        
        CRITICAL: In LoRA mode, converts full-model flat to LoRA-only flat.
        This ensures consistency with proxy_param and malicious_update (both are LoRA-only).
        
        Args:
            global_params: Global model parameters (full-model flat in non-LoRA mode,
                          may be full-model or LoRA-only flat in LoRA mode)
        """
        use_lora = hasattr(self.model, 'use_lora') and self.model.use_lora
        
        if use_lora:
            # LoRA mode: Convert to LoRA-only flat
            # Strategy: If input is full-model flat, extract LoRA params using named_parameters()
            # If input is already LoRA-only flat and matches _flat_numel, use as-is
            
            input_numel = int(global_params.numel())
            
            if input_numel == self._flat_numel:
                # Already LoRA-only flat, use directly
                self.global_model_params = global_params.clone().detach().to(self.device)
            else:
                # Full-model flat: Extract LoRA parameters in same order as get_flat_params()
                # CRITICAL: Must match get_flat_params() order exactly
                trainables = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
                
                # Reconstruct full model params from flat (temporarily)
                # This is needed to extract LoRA params correctly
                # Note: This is a workaround - ideally server should send LoRA-only flat
                full_model_params = {}
                offset = 0
                for name, param in self.model.named_parameters():
                    numel = int(param.numel())
                    full_model_params[name] = global_params[offset:offset + numel].view_as(param.data)
                    offset += numel
                
                # CRITICAL: Verify offset matches input length (catches silent misalignment)
                if offset != input_numel:
                    raise RuntimeError(
                        f"[Attacker {self.client_id}] Full-model global_params length mismatch: "
                        f"consumed {offset}, provided {input_numel}. Check flatten order."
                    )
                
                # Extract LoRA parameters in order
                lora_params_flat = []
                for name, param in trainables:
                    if name in full_model_params:
                        lora_params_flat.append(full_model_params[name].view(-1))
                    else:
                        raise RuntimeError(
                            f"[Attacker {self.client_id}] LoRA parameter {name} not found in full_model_params"
                        )
                
                # Concatenate LoRA params to form LoRA-only flat
                self.global_model_params = torch.cat(lora_params_flat).clone().detach().to(self.device)
                
                # Verify dimension matches
                assert self.global_model_params.numel() == self._flat_numel, \
                    f"[Attacker {self.client_id}] LoRA extraction failed: " \
                    f"expected {self._flat_numel}, got {self.global_model_params.numel()}"
        else:
            # Non-LoRA mode: Use as-is
            self.global_model_params = global_params.clone().detach().to(self.device)
    
    def set_constraint_params(self, d_T: float = None, gamma: float = None, 
                              total_data_size: float = None, benign_data_sizes: dict = None):
        """
        Set constraint parameters for Formula 4.
        
        Args:
            d_T: Distance threshold for constraint (4b): d(w'_j(t), w'_g(t)) ≤ d_T
            # ===== CONSTRAINT (4c) COMMENTED OUT =====
            # gamma: Upper bound for constraint (4c): Σ β'_{i,j}(t) d(w_i(t), w̄_i(t)) ≤ Γ
            gamma: Temporarily disabled (constraint 4c is commented out)
            # ==========================================
            total_data_size: D(t) - Total data size for aggregation weight calculation (Paper Formula (2))
            benign_data_sizes: Dict {client_id: D_i(t)} - Data sizes for each benign client (Paper Formula (2))
        """
        self.d_T = d_T  # Constraint (4b): d(w'_j(t), w'_g(t)) ≤ d_T
        # ===== CONSTRAINT (4c) COMMENTED OUT =====
        # self.gamma = gamma  # Constraint (4c): Σ β'_{i,j}(t) d(w_i(t), w̄_i(t)) ≤ Γ
        self.gamma = gamma  # Temporarily disabled (constraint 4c is commented out)
        # ==========================================
        self.total_data_size = total_data_size  # D(t) for weight calculation
        if benign_data_sizes is not None:
            self.benign_data_sizes = benign_data_sizes  # {client_id: D_i(t)}
    
    def set_lagrangian_params(self, use_lagrangian_dual: bool = False,
                              lambda_init: float = 0.1,
                              # ===== CONSTRAINT (4c) COMMENTED OUT =====
                              # rho_init: float = 0.1,
                              lambda_lr: float = 0.01,
                              # rho_lr: float = 0.01,
                              # ==========================================
                              enable_final_projection: bool = True,
                              enable_light_projection_in_loop: bool = True,
                              adaptive_d_T: bool = False,
                              d_T_multiplier: float = 1.5,
                              d_T_min: float = 8.0):
        """
        Set Lagrangian Dual parameters (initialized according to paper Algorithm 1)
        
        Paper reference: Section 3, Algorithm 1
        - Lagrangian function: eq:lagrangian
        - Optimization subproblem: eq:wprime_sub
        - Initialization: λ(1)≥0, ρ(1)≥0 [ρ disabled due to constraint 4c being commented out]
        
        Args:
            use_lagrangian_dual: Whether to use Lagrangian Dual mechanism
            lambda_init: Initial λ(1) value (≥0, per paper Algorithm 1)
            # ===== CONSTRAINT (4c) COMMENTED OUT =====
            # rho_init: Initial ρ(1) value (≥0, per paper Algorithm 1)
            # ==========================================
            lambda_lr: Learning rate for λ(t) update (subgradient step size)
            # ===== CONSTRAINT (4c) COMMENTED OUT =====
            # rho_lr: Learning rate for ρ(t) update (subgradient step size)
            # ==========================================
            enable_final_projection: Whether to apply final projection after optimization (only for Lagrangian mode)
                                   If False, completely relies on Lagrangian mechanism (no final projection)
            enable_light_projection_in_loop: Whether to apply light projection within optimization loop (only for Lagrangian mode)
                                           If False, no projection within optimization loop, only relies on Lagrangian mechanism
            adaptive_d_T: Whether to use adaptive d_T based on benign client distances (default: False)
            d_T_multiplier: Multiplier for adaptive d_T calculation (default: 1.5)
            d_T_min: Minimum d_T value to prevent too small thresholds (default: 8.0)
        
        Modification 2: Save initial values for reset in prepare_for_round
        """
        self.use_lagrangian_dual = use_lagrangian_dual
        # Store adaptive d_T parameters
        self.adaptive_d_T = adaptive_d_T
        self.d_T_multiplier = d_T_multiplier
        self.d_T_min = d_T_min
        # Store base d_T value (will be set by server, but we keep original for adaptive calculation)
        self.base_d_T = None  # Will be set from self.d_T in prepare_for_round or camouflage_update
        if use_lagrangian_dual:
            # Paper: λ(1)≥0, ρ(1)≥0 [ρ disabled due to constraint 4c being commented out]
            # Modification 2: Save initial values for reset each round
            self.lambda_init_value = max(0.0, lambda_init)
            # ===== CONSTRAINT (4c) COMMENTED OUT =====
            # self.rho_init_value = max(0.0, rho_init)
            self.rho_init_value = None  # Temporarily disabled (constraint 4c is commented out)
            # ==========================================
            self.lambda_dt = torch.tensor(self.lambda_init_value, requires_grad=False)
            # ===== CONSTRAINT (4c) COMMENTED OUT =====
            # self.rho_dt = torch.tensor(self.rho_init_value, requires_grad=False)
            self.rho_dt = None  # Temporarily disabled (constraint 4c is commented out)
            # ==========================================
            self.lambda_lr = lambda_lr
            # ===== CONSTRAINT (4c) COMMENTED OUT =====
            # self.rho_lr = rho_lr
            # self.rho_lr is already set to 0.01 in __init__, but not used since constraint 4c is disabled
            # ==========================================
            self.enable_final_projection = enable_final_projection
            self.enable_light_projection_in_loop = enable_light_projection_in_loop
        else:
            # Hard constraint projection mode
            self.lambda_dt = None
            self.rho_dt = None
            self.lambda_init_value = None
            self.rho_init_value = None
            self.enable_final_projection = True  # Default to True for hard constraint mode (always applies projection)
            self.enable_light_projection_in_loop = False  # Not applicable in hard constraint mode

    def _aggregate_update_no_beta(self, malicious_update: torch.Tensor, 
                                   benign_updates: List[torch.Tensor]) -> Tuple[torch.Tensor, float, List[float]]:
        """
        FedAvg-style aggregated update (NO beta selection).
        
        Aggregated update:
            Δ_g = Σ_i (D_i / D_eff) * Δ_i + (D_att / D_eff) * Δ_att
        
        where D_eff = Σ D_i + D_att (all participants including attacker).
        
        Args:
            malicious_update: Δ_att (attacker's update)
            benign_updates: List of all benign updates Δ_i
        
        Returns:
            aggregated_update: Δ_g (aggregated update)
            w_att: attacker weight (D_att / D_eff)
            w_ben: list of benign weights [D_i / D_eff]
        """
        device = malicious_update.device
        D_att = float(self.claimed_data_size)
        
        # Collect benign data sizes
        D_sum = D_att
        D_list = []
        for idx in range(len(benign_updates)):
            # Get client_id from benign_update_client_ids if available
            if hasattr(self, 'benign_update_client_ids') and idx < len(self.benign_update_client_ids):
                client_id = self.benign_update_client_ids[idx]
            else:
                client_id = idx  # Fallback to index
            
            # Get data size for this client
            if hasattr(self, 'benign_data_sizes') and client_id in self.benign_data_sizes:
                D_i = float(self.benign_data_sizes[client_id])
            else:
                D_i = 1.0  # Fallback
            
            D_list.append(D_i)
            D_sum += D_i
        
        # ===== NEW: Include other attackers' updates for coordinated optimization =====
        other_attacker_weights = []
        other_attacker_updates_list = []
        if hasattr(self, 'other_attacker_updates') and self.other_attacker_updates:
            for idx, cid in enumerate(self.other_attacker_client_ids):
                if idx < len(self.other_attacker_updates):
                    if hasattr(self, 'other_attacker_data_sizes') and cid in self.other_attacker_data_sizes:
                        D_j = float(self.other_attacker_data_sizes[cid])
                    else:
                        # Fallback: use current attacker's claimed size as estimate
                        D_j = float(self.claimed_data_size)
                    
                    other_attacker_weights.append(D_j)
                    other_attacker_updates_list.append(self.other_attacker_updates[idx])
                    D_sum += D_j
        # ==============================================================================
        
        # Compute weights
        denom = D_sum + 1e-12
        w_att = D_att / denom
        w_ben = [D_i / denom for D_i in D_list]
        w_other_att = [D_j / denom for D_j in other_attacker_weights]
        
        # Aggregate updates: Δ_g = Σ w_i * Δ_i + Σ w_j * Δ_j + w_att * Δ_att
        agg = torch.zeros_like(malicious_update, device=device).view(-1)
        
        # Add benign updates
        for w, benign_update in zip(w_ben, benign_updates):
            agg = agg + w * benign_update.to(device).view(-1)
        
        # ===== NEW: Add other attackers' updates =====
        for w, other_attacker_update in zip(w_other_att, other_attacker_updates_list):
            agg = agg + w * other_attacker_update.to(device).view(-1)
        # ==============================================
        
        # Add current attacker's update
        agg = agg + w_att * malicious_update.view(-1)
        
        return agg.view(-1), w_att, w_ben
    
    def _compute_distance_update_space(self, malicious_update: torch.Tensor,
                                        benign_updates: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute distance in UPDATE space: d(Δ_att, Δ_g) = ||Δ_att - Δ_g||.
        
        This is the correct constraint interpretation:
        - w'_j = w_g + Δ_att
        - w'_g = w_g + Δ_g
        => d(w'_j, w'_g) = ||Δ_att - Δ_g||
        
        Args:
            malicious_update: Δ_att (attacker's update)
            benign_updates: List of all benign updates Δ_i
        
        Returns:
            distance: ||Δ_att - Δ_g||
            aggregated_update: Δ_g (for reuse)
        """
        aggregated_update, _, _ = self._aggregate_update_no_beta(malicious_update, benign_updates)
        diff = malicious_update.view(-1) - aggregated_update.view(-1)
        distance = torch.norm(diff)
        return distance, aggregated_update
    
    def _hard_project_update_space(self, malicious_update: torch.Tensor,
                                     benign_updates: List[torch.Tensor],
                                     d_T: float):
        """
        Project Δ_att onto {Δ: ||Δ - Δ_g|| <= d_T} in UPDATE space.
        
        Args:
            malicious_update: Δ_att (attacker's update, modified in-place)
            benign_updates: List of all benign updates Δ_i
            d_T: distance threshold
        """
        with torch.no_grad():
            dist, aggregated_update = self._compute_distance_update_space(malicious_update, benign_updates)
            dist_val = float(dist.item())
            d_T_val = float(d_T) if isinstance(d_T, torch.Tensor) else float(d_T)
            
            if dist_val > d_T_val:
                # Project: Δ_att_new = Δ_g + (Δ_att - Δ_g) * (d_T / ||Δ_att - Δ_g||)
                diff = malicious_update.view(-1) - aggregated_update.view(-1)
                diff = diff * (d_T_val / (dist_val + 1e-12))
                malicious_update.copy_((aggregated_update.view(-1) + diff).view_as(malicious_update))
    
    def _compute_global_loss(self, malicious_update: torch.Tensor, 
                            benign_updates: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute global loss F(w'_g) where w'_g = w_g + Δ_g (aggregated update).
        
        This is the CORRECT interpretation: optimize loss of the aggregated global model,
        not individual client models.
        
        Objective: maximize F(w_g + Δ_g) where Δ_g is the FedAvg aggregated update.
        
        Args:
            malicious_update: Δ_att (attacker's update)
            benign_updates: List of all benign updates Δ_i
        
        Returns:
            Proxy for F(w'_g) to be maximized
        """
        # Compute aggregated update Δ_g
        aggregated_update, _, _ = self._aggregate_update_no_beta(malicious_update, benign_updates)
        
        # Compute loss on aggregated model: F(w_g + Δ_g)
        return self._proxy_global_loss(
            aggregated_update,
            max_batches=self.proxy_max_batches_opt,
            skip_dim_check=True,
            keep_model_on_gpu=True
        )
    
    def _compute_distance_update_space(self, malicious_update: torch.Tensor,
                                        benign_updates: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute distance in UPDATE space: d(Δ_att, Δ_g) = ||Δ_att - Δ_g||.
        
        This is the correct constraint interpretation:
        - w'_j = w_g + Δ_att
        - w'_g = w_g + Δ_g
        => d(w'_j, w'_g) = ||Δ_att - Δ_g||
        
        Args:
            malicious_update: Δ_att (attacker's update)
            benign_updates: List of all benign updates Δ_i
        
        Returns:
            distance: ||Δ_att - Δ_g||
            aggregated_update: Δ_g (for reuse)
        """
        aggregated_update, _, _ = self._aggregate_update_no_beta(malicious_update, benign_updates)
        diff = malicious_update.view(-1) - aggregated_update.view(-1)
        distance = torch.norm(diff)
        return distance, aggregated_update
    
    def _hard_project_update_space(self, malicious_update: torch.Tensor,
                                     benign_updates: List[torch.Tensor],
                                     d_T: float):
        """
        Project Δ_att onto {Δ: ||Δ - Δ_g|| <= d_T} in UPDATE space.
        
        Args:
            malicious_update: Δ_att (attacker's update, modified in-place)
            benign_updates: List of all benign updates Δ_i
            d_T: distance threshold
        """
        with torch.no_grad():
            dist, aggregated_update = self._compute_distance_update_space(malicious_update, benign_updates)
            dist_val = float(dist.item())
            d_T_val = float(d_T) if isinstance(d_T, torch.Tensor) else float(d_T)
            
            if dist_val > d_T_val:
                # Project: Δ_att_new = Δ_g + (Δ_att - Δ_g) * (d_T / ||Δ_att - Δ_g||)
                diff = malicious_update.view(-1) - aggregated_update.view(-1)
                diff = diff * (d_T_val / (dist_val + 1e-12))
                malicious_update.copy_((aggregated_update.view(-1) + diff).view_as(malicious_update))
    
    def _compute_real_distance_to_global(self, malicious_update: torch.Tensor,
                                         benign_updates: List[torch.Tensor],
                                         legacy_param: Any = None) -> torch.Tensor:
        """
        Compute distance in UPDATE space (NEW implementation).
        
        Legacy signature preserved for backward compatibility, but now uses
        _compute_distance_update_space() internally.
        
        Returns:
            distance: ||Δ_att - Δ_g|| in UPDATE space
        """
        # Use new UPDATE space distance computation
        distance, _ = self._compute_distance_update_space(malicious_update, benign_updates)
        return distance
    
    # [DEPRECATED] Old model-space distance and logging functions - not used in optimization
    def _compute_real_distance_to_global_OLD_MODEL_SPACE(self, malicious_update, selected_benign, beta_selection):
        """[DEPRECATED] Old model-space distance. See _compute_distance_update_space() for current implementation."""
        # This function is kept for backward compatibility but should not be used
        # Use _compute_distance_update_space() instead
        dist, _ = self._compute_distance_update_space(malicious_update, self.benign_updates)
        return dist

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
        # NO BETA SELECTION: Use ALL benign updates
        if not self.benign_updates:
            print(f"    [Attacker {self.client_id}] No benign updates available, return zero update")
            return poisoned_update  # poisoned_update is always zero (attackers don't train)

        # Move updates to GPU for processing
        benign_gpu = [u.to(self.device) for u in self.benign_updates]
        benign_stack = torch.stack([u.detach() for u in benign_gpu])  # (I, full_dim)
        
        # Reduce dimensionality for computational efficiency
        reduced_benign = self._get_reduced_features(benign_gpu, fix_indices=False)  # (I, M)
        # Clean up benign_stack and benign_gpu after feature reduction
        del benign_stack, benign_gpu
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
        # STEP 6: Optimize attack objective (NO BETA SELECTION)
        # ============================================================
        # Paper objective: maximize F(w'_g) subject to d(w'_j, w'_g) ≤ d_T
        # 
        # Correct interpretation:
        # - w'_g = w_g + Δ_g where Δ_g is the aggregated update (including attacker)
        # - Optimize loss of the aggregated global model: F(w_g + Δ_g)
        # - Constraint in UPDATE space: ||Δ_att - Δ_g|| ≤ d_T
        # 
        # NO BETA SELECTION: Server aggregates ALL benign clients; attacker does not control participant set.
        # All benign updates are used for aggregation and distance computation.
        # ============================================================
        
        # ============================================================
        # STEP 7: Optimize w'_j(t) to maximize F(w'_g(t))
        # According to paper Equation 12, we maximize F(w'_g(t)) subject to constraints
        # ============================================================
        # CRITICAL: Hard preconditions check before optimization
        # LoRA mode requires strict gradient validation, so we must ensure prerequisites
        if self.global_model_params is None or self.proxy_loader is None:
            error_msg = (
                f"[Attacker {self.client_id}] Missing prerequisites before optimization:\n"
                f"  - global_model_params: {self.global_model_params is None}\n"
                f"  - proxy_loader: {self.proxy_loader is None}\n"
                f"Cannot proceed with LoRA gradient validation requirements."
            )
            print(f"    {error_msg}")
            raise RuntimeError(error_msg)
        
        # Verify global_model_params dimension matches _flat_numel (LoRA mode requirement)
        use_lora = hasattr(self.model, 'use_lora') and self.model.use_lora
        if use_lora:
            global_numel = self.global_model_params.numel()
            if global_numel != self._flat_numel:
                error_msg = (
                    f"[Attacker {self.client_id}] LoRA mode: global_model_params dimension mismatch:\n"
                    f"  - global_model_params.numel(): {global_numel}\n"
                    f"  - _flat_numel: {self._flat_numel}\n"
                    f"  - model.use_lora: {use_lora}\n"
                    f"  - model.get_flat_params().numel(): {self.model.get_flat_params().numel()}\n"
                    f"Check set_global_model_params() conversion."
                )
                print(f"    {error_msg}")
                raise RuntimeError(error_msg)
            else:
                print(f"    [Attacker {self.client_id}] LoRA dimension check passed: "
                      f"global_params={global_numel}, _flat_numel={self._flat_numel}")
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
        
        # ===== CRITICAL: Initialize functional cache for LoRA mode =====
        # This must be done before optimization loop starts
        use_lora = hasattr(self.model, 'use_lora') and self.model.use_lora
        if use_lora:
            self._init_functional_param_cache(target_device)
            # Ensure base_params and base_buffers are on GPU (cache them on GPU)
            # This avoids repeated device transfers during optimization loop
            if not all(p.device.type == target_device.type for p in self.base_params.values()):
                for name in self.base_params:
                    self.base_params[name] = self.base_params[name].to(target_device)
            if not all(b.device.type == target_device.type for b in self.base_buffers.values()):
                for name in self.base_buffers:
                    self.base_buffers[name] = self.base_buffers[name].to(target_device)
        # Store use_lora for use in optimization loop
        self._use_lora_in_optimization = use_lora
        # ===================================================================
        
        # ============================================================
        # Adaptive d_T calculation (if enabled)
        # ============================================================
        if self.adaptive_d_T and len(self.benign_updates) > 0:
            # Compute distances of benign clients to their weighted average
            # This gives us the "normal" distance range for benign clients
            benign_distances = []
            if len(self.benign_updates) >= 2:
                # Compute weighted average of benign updates (excluding attacker)
                device = target_device if hasattr(self, '_model_on_gpu') and self._model_on_gpu else self.device
                benign_updates_gpu = [u.to(device) for u in self.benign_updates]
                
                # Compute weighted average of benign updates
                if hasattr(self, 'benign_data_sizes') and len(self.benign_update_client_ids) == len(self.benign_updates):
                    total_benign_D = sum(
                        float(self.benign_data_sizes.get(cid, 1.0)) 
                        for cid in self.benign_update_client_ids
                    )
                    benign_avg = torch.zeros_like(benign_updates_gpu[0])
                    for idx, benign_update in enumerate(benign_updates_gpu):
                        if idx < len(self.benign_update_client_ids):
                            cid = self.benign_update_client_ids[idx]
                            D_i = float(self.benign_data_sizes.get(cid, 1.0))
                            weight = D_i / (total_benign_D + 1e-12)
                            benign_avg += weight * benign_update
                    else:
                        # Fallback to simple average
                        benign_avg = torch.stack(benign_updates_gpu).mean(dim=0)
                
                # Compute distance of each benign update to the average
                for benign_update in benign_updates_gpu:
                    dist = torch.norm((benign_update - benign_avg).view(-1)).item()
                    benign_distances.append(dist)
                
                if len(benign_distances) > 0:
                    mean_benign_dist = sum(benign_distances) / len(benign_distances)
                    # Store base_d_T if not already set
                    if self.base_d_T is None:
                        self.base_d_T = float(self.d_T) if isinstance(self.d_T, torch.Tensor) else self.d_T if self.d_T is not None else 10.0
                    
                    # Calculate adaptive d_T: max(base_d_T, mean_benign_dist * multiplier, d_T_min)
                    adaptive_d_T_value = max(
                        self.base_d_T,
                        mean_benign_dist * self.d_T_multiplier,
                        self.d_T_min
                    )
                    
                    # Update d_T (preserve type: tensor or float)
                    if isinstance(self.d_T, torch.Tensor):
                        self.d_T = torch.tensor(adaptive_d_T_value, device=self.d_T.device, dtype=self.d_T.dtype)
                    else:
                        self.d_T = adaptive_d_T_value
                    
                    print(f"    [Attacker {self.client_id}] Adaptive d_T: base={self.base_d_T:.2f}, "
                          f"mean_benign_dist={mean_benign_dist:.2f}, "
                          f"adaptive_d_T={adaptive_d_T_value:.2f} (multiplier={self.d_T_multiplier})")
        
        # OPTIMIZATION 2: Cache constraint (4c) value before optimization loop
        # Constraint (4c): DISABLED (commented out in code)
        constraint_c_value_for_update = 0.0  # Used for updating ρ
        constraint_c_term_base = None  # Cached base term for Lagrangian mode (on GPU)
        
        # ===== CONSTRAINT (4c) COMMENTED OUT =====
        # if self.gamma is not None and len(self.benign_updates) > 0:
        #     # Compute constraint (4c) value once before loop (constant value)
        #     # Paper constraint (4c): Σ β'_{i,j}(t) d(w_i(t), w̄_i(t)) ≤ Γ
        #     # where w̄_i(t) = Σ_{i=1}^I (D_i(t)/D(t)) w_i(t) is the weighted mean of ALL benign clients
        #     # (not just selected ones)
        #     ...
        #     constraint_c_value_for_update = agg_dist.item()
        #     
        #     # For Lagrangian mode, cache the base term on GPU to avoid recomputation
        #     if self.use_lagrangian_dual:
        #         constraint_c_term_base = agg_dist  # Keep on GPU for reuse in loop
        #     else:
        #         ...
        # ==========================================
        constraint_c_value_for_update = 0.0  # Dummy value since constraint (4c) is disabled
        constraint_c_term_base = None  # Dummy value since constraint (4c) is disabled
        
        # OPTIMIZATION 5: Cache Lagrangian multipliers on GPU before loop
        # Ensure multipliers are on correct device to avoid repeated conversions
        # ===== CONSTRAINT (4c) COMMENTED OUT: Only check lambda_dt =====
        if self.use_lagrangian_dual and self.lambda_dt is not None:  # Removed rho_dt check
            if isinstance(self.lambda_dt, torch.Tensor):
                if not self._device_matches(self.lambda_dt.device, target_device):
                    self.lambda_dt = self.lambda_dt.to(target_device)
            else:
                self.lambda_dt = torch.tensor(self.lambda_dt, device=target_device)
            
            # ===== CONSTRAINT (4c) COMMENTED OUT =====
            # if isinstance(self.rho_dt, torch.Tensor):
            #     if not self._device_matches(self.rho_dt.device, target_device):
            #         self.rho_dt = self.rho_dt.to(target_device)
            # else:
            #     self.rho_dt = torch.tensor(self.rho_dt, device=target_device)
            # ==========================================
        
        # ============================================================
        # Print initial optimization state
        # ============================================================
        print(f"    [Attacker {self.client_id}] Preparing optimization: "
              f"proxy_param.shape={proxy_param.shape}, proxy_param.numel()={proxy_param.numel()}, "
              f"_flat_numel={self._flat_numel}, use_lora={use_lora}")
        
        if self.d_T is not None:
            try:
                initial_dist_tensor, _ = self._compute_distance_update_space(
                    proxy_param,
                    self.benign_updates
                )
                initial_dist = initial_dist_tensor.item()
                d_T_val = float(self.d_T) if isinstance(self.d_T, torch.Tensor) else self.d_T
                initial_g = initial_dist - d_T_val
                initial_lambda = self.lambda_dt.item() if isinstance(self.lambda_dt, torch.Tensor) else self.lambda_dt if self.lambda_dt is not None else 0.0
                initial_loss = self._compute_global_loss(proxy_param, self.benign_updates).item()
                print(f"    [Attacker {self.client_id}] Starting optimization (UPDATE space): "
                      f"initial_dist={initial_dist:.4f}, d_T={d_T_val:.4f}, g={initial_g:.4f}, "
                      f"lambda={initial_lambda:.4f}, loss={initial_loss:.4f}, steps={self.proxy_steps}")
            except Exception as e:
                print(f"    [Attacker {self.client_id}] ERROR computing initial state: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        # Early stopping variables: track constraint satisfaction stability
        constraint_satisfied_steps = 0
        constraint_stability_steps = self.early_stop_constraint_stability_steps  # Stop after N consecutive steps satisfying constraint
        prev_dist_val = None
        
        for step in range(self.proxy_steps):
            # ============================================================
            # CRITICAL: Zero gradients using set_to_none=True for efficiency
            # ============================================================
            proxy_opt.zero_grad(set_to_none=True)
            
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
            # ============================================================
            # Compute loss of AGGREGATED global model: F(w_g + Δ_g)
            # ============================================================
            global_loss = self._compute_global_loss(
                proxy_param,
                self.benign_updates
            )
            
            # ============================================================
            # Build optimization objective: choose mechanism based on whether using Lagrangian Dual
            # ============================================================
            
            if self.use_lagrangian_dual and self.lambda_dt is not None:  # Removed rho_dt check
                # ========== Use Lagrangian Dual mechanism (paper eq:lagrangian and eq:wprime_sub) ==========
                
                # OPTIMIZATION 5: Use cached multipliers directly (already on correct device)
                lambda_dt_tensor = self.lambda_dt  # Direct use, no conversion needed
                # ===== CONSTRAINT (4c) COMMENTED OUT =====
                # rho_dt_tensor = self.rho_dt  # Direct use, no conversion needed
                # ==========================================
                
                # ============================================================
                # Constraint (4b): d(Δ_att, Δ_g) ≤ d_T in UPDATE space
                # ============================================================
                dist_to_global_for_objective, _ = self._compute_distance_update_space(
                    proxy_param,
                    self.benign_updates
                )
                
                # ============================================================
                # Standard Lagrangian Dual formulation
                # ============================================================
                # Constraint: g(x) = d(w'_j, w'_g) - d_T ≤ 0
                # 
                # Standard Lagrangian: penalty = λ * g
                # - When dist < d_T (satisfied): g < 0, penalty < 0 (negative, "rewards" satisfaction)
                # - When dist > d_T (violated):  g > 0, penalty > 0 (positive, penalizes violation)
                #
                # Why Standard Lagrangian:
                # - Provides directional guidance even when constraint is satisfied (g < 0)
                # - Negative penalty (reward) when satisfied helps prevent deviation from constraint boundary
                # - Prevents optimization from moving too far from constraint when constraint is satisfied
                #
                # Alternative Option: Hinge Penalty (penalty = λ * relu(g))
                #   - When satisfied: penalty = 0 (no constraint), may allow deviation
                #   - When violated: penalty = λ * g > 0 (penalizes violation)
                #
                # Alternative Option B: Augmented Lagrangian (uncomment if preferred)
                #   rho = getattr(self, "lagrangian_rho", 10.0)
                #   penalty = lambda_dt_tensor * g + 0.5 * rho * (F.relu(g) ** 2)
                # ============================================================
                # CRITICAL: Convert d_T to scalar if it's a tensor
                d_T_val = float(self.d_T) if isinstance(self.d_T, torch.Tensor) else self.d_T if self.d_T is not None else None
                g = dist_to_global_for_objective - d_T_val if d_T_val is not None else torch.tensor(0.0, device=self.device)
                
                # Standard Lagrangian: provides directional guidance even when constraint is satisfied
                # penalty = λ * (dist - d_T)
                # When dist < d_T: penalty < 0 (reward, prevents deviation)
                # When dist > d_T: penalty > 0 (penalty, enforces constraint)
                penalty = lambda_dt_tensor * g
                
                # Alternative Option B: Augmented Lagrangian (uncomment if preferred)
                # rho = getattr(self, "lagrangian_rho", 10.0)
                # penalty = lambda_dt_tensor * g + 0.5 * rho * (F.relu(g) ** 2)
                
                constraint_b_term = penalty
                
                # ===== CONSTRAINT (4c) COMMENTED OUT =====
                # Constraint (4c): Σ β'_{i,j}(t) d(w_i(t), w̄_i(t)) ≤ Γ
                # For constraint (4c): Σ(...) - Γ ≤ 0, so L = F(w'_g) - ρ (Σ(...) - Γ)
                # Converting to minimization: minimize -L = -F(w'_g) + ρ (Σ(...) - Γ)
                # Since ρ Γ is constant, we use: minimize -F(w'_g) + ρ Σ(...)
                # OPTIMIZATION 2: Use cached constraint (4c) value (computed before loop)
                # constraint_c_term = torch.tensor(0.0, device=self.device)
                # if constraint_c_term_base is not None:
                #     # Use cached base term (already on GPU)
                #     constraint_c_term = rho_dt_tensor * constraint_c_term_base
                constraint_c_term = torch.tensor(0.0, device=self.device)  # Dummy value (constraint 4c disabled)
                # ==========================================
                
                # ============================================================
                # Build Lagrangian objective function (paper formula eq:wprime_sub)
                # ============================================================
                # Paper: maximize F(w'_g(t)) subject to constraints
                # Standard Lagrangian: L = F(w'_g) - λ (d(w'_j, w'_g) - d_T) - ρ (Σ(...) - Γ)  [ρ term removed]
                # Converting to minimization: minimize -L = -F(w'_g) + λ (d(w'_j, w'_g) - d_T) + ρ Σ(...)  [ρ term removed]
                # Standard form: minimize -F(w'_g) + λ * (dist - d_T)
                # When dist < d_T: penalty is negative (rewards satisfaction)
                # When dist > d_T: penalty is positive (penalizes violation)
                lagrangian_objective = -global_loss + constraint_b_term  # Removed constraint_c_term
                
                # ============================================================
                # ============================================================
                # Compute constraint violations (for updating λ and ρ)
                # ============================================================
                # g is already computed above (dist - d_T)
                constraint_b_violation = F.relu(g) if self.d_T is not None else torch.tensor(0.0, device=self.device)
            
                # ===== CONSTRAINT (4c) COMMENTED OUT =====
                # constraint_c_violation = torch.tensor(0.0, device=self.device)
                # if self.gamma is not None and len(self.benign_updates) > 0:
                #         constraint_c_violation = F.relu(torch.tensor(constraint_c_value_for_update, device=self.device) - self.gamma)
                constraint_c_violation = torch.tensor(0.0, device=self.device)  # Dummy value (constraint 4c disabled)
                # ==========================================
                
            else:
                # ========== Use hard constraint projection mechanism (original logic) ==========
                # Objective: maximize global_loss => minimize -global_loss
                lagrangian_objective = -global_loss
                
                # Compute constraint violations (for logging only)
                # Use real distance calculation according to paper Constraint (4b)
                # ============================================================
                # Constraint (4b): d(Δ_att, Δ_g) ≤ d_T in UPDATE space
                # ============================================================
                d_T_val = float(self.d_T) if isinstance(self.d_T, torch.Tensor) else self.d_T if self.d_T is not None else None
                if d_T_val is not None:
                    dist_to_global, _ = self._compute_distance_update_space(
                    proxy_param,
                        self.benign_updates
                    )
                    constraint_b_violation = F.relu(dist_to_global - d_T_val)
                else:
                    dist_to_global = torch.tensor(0.0, device=self.device)
                    constraint_b_violation = torch.tensor(0.0, device=self.device)
                
                # ===== CONSTRAINT (4c) COMMENTED OUT =====
                # OPTIMIZATION 6: Constraint (4c) calculation moved to after loop for hard constraint mode
                # In hard constraint mode, constraint (4c) is only used for logging, not in optimization
                # So we can delay its computation until after the loop
                # constraint_c_violation = torch.tensor(0.0, device=self.device)
                # constraint_c_value_for_update is already computed before loop (for Lagrangian mode compatibility)
                constraint_c_violation = torch.tensor(0.0, device=self.device)  # Dummy value (constraint 4c disabled)
                # ==========================================
            
            # ============================================================
            # CRITICAL: Compute gradients using torch.autograd.grad (NO backward())
            # ============================================================
            # PROHIBITED: Cannot use backward() twice on the same graph
            # Solution: Use autograd.grad() once, manually set proxy_param.grad
            # If we need to verify gradient link (every 5 steps in LoRA mode), retain graph for second check
            need_check = bool(self._use_lora_in_optimization and (step % 5 == 0))
            
            try:
                grad = torch.autograd.grad(
                    lagrangian_objective,
                    proxy_param,
                    retain_graph=need_check,   # <-- keep graph only when we will do an extra check
                    allow_unused=False,
                    create_graph=False
                )[0]
                proxy_param.grad = grad
            except RuntimeError as e:
                if "second time" in str(e) or "already been freed" in str(e):
                    raise RuntimeError(
                        f"[Attacker {self.client_id}] FATAL: Graph already used - double backward detected at step {step}. "
                        f"Check for multiple backward/grad passes on the same graph."
                    ) from e
                raise
            
            # ============================================================
            # Hard acceptance criteria for LoRA mode
            # ============================================================
            if self._use_lora_in_optimization:
                if proxy_param.grad is None:
                    raise RuntimeError(
                        f"[Attacker {self.client_id}] FATAL at step {step}: proxy_param.grad is None in LoRA mode."
                    )
                grad_norm = proxy_param.grad.norm().item()
                if grad_norm < 1e-8:
                    raise RuntimeError(
                        f"[Attacker {self.client_id}] FATAL at step {step}: Gradient norm too small ({grad_norm:.2e})."
                    )

                # Additional gradient-link verification (only when need_check=True)
                if need_check:
                    g = torch.autograd.grad(
                        global_loss,
                        proxy_param,
                        retain_graph=False,   # <-- free graph after verification
                        allow_unused=False,
                        create_graph=False
                    )[0]
                    if g is None or g.norm().item() < 1e-8:
                        raise RuntimeError(
                            f"[Attacker {self.client_id}] Gradient verification failed at step {step}: "
                            f"g is None or too small."
                        )
            # ===========================================================
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_([proxy_param], max_norm=self.grad_clip_norm)
            
            proxy_opt.step()
            
            # ============================================================
            # Update Lagrangian multipliers (if using Lagrangian mechanism)
            # Use subgradient method according to paper dual problem
            # Paper Algorithm 1 Step 7: Update λ(t) and ρ(t) according to eq:dual
            # ============================================================
            # ===== CONSTRAINT (4c) COMMENTED OUT: Removed rho_dt check =====
            if self.use_lagrangian_dual and self.lambda_dt is not None:  # Removed rho_dt check
                # Dual ascent method: λ(t+1) = λ(t) + α_λ × subgradient
                # subgradient = (constraint value - bound)
                # When constraint is violated (constraint value > bound), subgradient > 0, λ increases to penalize violation
                
                if self.d_T is not None:
                # Calculate distance in UPDATE space after step()
                    current_dist_tensor, _ = self._compute_distance_update_space(
                        proxy_param,
                    self.benign_updates
                    )
                    current_dist = current_dist_tensor.item()
                    lambda_val = self.lambda_dt.item() if isinstance(self.lambda_dt, torch.Tensor) else self.lambda_dt
                    # Standard dual ascent update: λ(t+1) = max(0, λ(t) + α_λ * g)
                    # where g = dist - d_T is the constraint violation
                    # - If constraint is violated (g > 0): λ increases to penalize violation
                    # - If constraint is satisfied (g < 0): λ decreases (but clamped to ≥ 0)
                    #   This allows the system to "relax" when constraint is well-satisfied
                    d_T_val = float(self.d_T) if isinstance(self.d_T, torch.Tensor) else self.d_T
                    g_val = current_dist - d_T_val
                    new_lambda = lambda_val + self.lambda_lr * g_val
                    new_lambda = max(0.0, new_lambda)  # Ensure non-negative
                    # OPTIMIZATION 5: Keep multiplier on same device when updating
                    self.lambda_dt = torch.tensor(new_lambda, device=target_device, requires_grad=False)
                
                    # Logging: Print every step for detailed monitoring
                    # Print more frequently for better visibility (every step or every 5 steps)
                    print_freq = 1 if self.proxy_steps <= 20 else 5  # Print every step if <=20 steps, else every 5 steps
                    if step % print_freq == 0 or step == 0 or step == self.proxy_steps - 1:
                        grad_norm = proxy_param.grad.norm().item() if proxy_param.grad is not None else 0.0
                        print(f"      [Attacker {self.client_id}] Step {step}/{self.proxy_steps-1}: "
                        f"dist={current_dist:.4f}, g={g_val:.4f}, lambda={lambda_val:.4f}, "
                        f"loss={global_loss.item():.4f}, grad_norm={grad_norm:.4f}")
                    
                    # ============================================================
                    # Early stopping: Stop when constraint is satisfied and stable
                    # ============================================================
                    # Strategy: Stop after N consecutive steps satisfying constraint
                    # This prevents premature stopping due to temporary fluctuations
                    # and avoids violating constraint after satisfying it (e.g., Attacker 8)
                    if current_dist <= d_T_val:
                        constraint_satisfied_steps += 1
                        if constraint_satisfied_steps >= constraint_stability_steps:
                            print(f"    [Attacker {self.client_id}] Early stopping: dist={current_dist:.4f} <= d_T={d_T_val:.4f} "
                                  f"for {constraint_satisfied_steps} consecutive steps (step {step}/{self.proxy_steps-1})")
                            break
                    else:
                        constraint_satisfied_steps = 0  # Reset counter when constraint is violated
                    
                    prev_dist_val = current_dist
                
                    # ===== CONSTRAINT (4c) COMMENTED OUT =====
                    # if self.gamma is not None and len(self.benign_updates) > 0:
                #     rho_val = self.rho_dt.item() if isinstance(self.rho_dt, torch.Tensor) else self.rho_dt
                #     # Subgradient: Σ(...) - Γ
                #     # If constraint is violated (Σ(...) > Γ), subgradient > 0, ρ increases
                #     subgradient_c = constraint_c_value_for_update - self.gamma
                #     new_rho = rho_val + self.rho_lr * subgradient_c
                #     new_rho = max(0.0, new_rho)  # Ensure non-negative
                #     # OPTIMIZATION 5: Keep multiplier on same device when updating
                #     self.rho_dt = torch.tensor(new_rho, device=target_device, requires_grad=False)
                # ==========================================
            
            # ============================================================
            # Modification 4: Add constraint safeguard mechanism within optimization loop (Lagrangian framework)
            # Under Lagrangian mechanism, add light projection as safeguard to prevent optimization path from deviating too far
            # ============================================================
            # ============================================================
            # CRITICAL: Projection must use no_grad() + in-place op, NOT .data
            # ============================================================
            if not self.use_lagrangian_dual and self.d_T is not None:
                # ============================================================
                # CRITICAL FIX: Hard constraint projection using REAL distance
                # ============================================================
                # Hard constraint mechanism: Project in UPDATE space
                # ============================================================
                dist_tensor, _ = self._compute_distance_update_space(
                    proxy_param,
                    self.benign_updates
                )
                dist = dist_tensor.item()
                d_T_val = float(self.d_T) if isinstance(self.d_T, torch.Tensor) else self.d_T
                
                if dist > d_T_val:
                    # Hard projection in UPDATE space
                    self._hard_project_update_space(proxy_param, self.benign_updates, d_T_val)
                    
                    new_dist_tensor, _ = self._compute_distance_update_space(
                        proxy_param,
                        self.benign_updates
                    )
                    new_dist = new_dist_tensor.item()
                    print_freq = 1 if self.proxy_steps <= 20 else 5
                    if step % print_freq == 0 or step == 0 or step == self.proxy_steps - 1:
                        d_T_print = float(self.d_T) if isinstance(self.d_T, torch.Tensor) else self.d_T
                        print(f"      [Attacker {self.client_id}] Step {step}/{self.proxy_steps-1}: Hard projection applied (UPDATE space): "
                              f"dist {dist:.4f} -> {new_dist:.4f} (target: {d_T_print:.4f})")
            elif self.use_lagrangian_dual and self.d_T is not None:
                # ============================================================
                # Optional: Light projection safeguard (can be softened/removed with hinge penalty)
                # ============================================================
                # Note: With hinge penalty (only penalizes violations), light projection
                # may be less necessary as the objective already handles violations correctly.
                # Consider removing or softening this projection after verifying standard Lagrangian behavior.
                # ============================================================
                if self.enable_light_projection_in_loop:
                    dist_to_global_for_projection_tensor, _ = self._compute_distance_update_space(
                    proxy_param,
                        self.benign_updates
                )
                dist_to_global_for_projection = dist_to_global_for_projection_tensor.item()
                
                d_T_val = float(self.d_T) if isinstance(self.d_T, torch.Tensor) else self.d_T
                violation_ratio = (dist_to_global_for_projection - d_T_val) / d_T_val if d_T_val > 0 else 0.0
                
                if violation_ratio > 1:
                    # Light projection to 1.5 × d_T, leaving margin to allow Lagrangian to continue optimizing
                    d_T_proj = float(self.d_T) if isinstance(self.d_T, torch.Tensor) else self.d_T
                    target_dist = d_T_proj * 1.5
                    scale_factor = target_dist / dist_to_global_for_projection
                    # CRITICAL: Use no_grad() + in-place op to avoid breaking gradients
                    with torch.no_grad():
                        proxy_param.mul_(scale_factor)
                    
                    # Logging: Print light projection information
                    print_freq = 1 if self.proxy_steps <= 20 else 5
                    if step % print_freq == 0 or step == 0 or step == self.proxy_steps - 1:
                        new_dist_after_proj_tensor, _ = self._compute_distance_update_space(
                            proxy_param,
                            self.benign_updates
                        )
                        new_dist_after_proj = new_dist_after_proj_tensor.item()
                        print(f"      [Attacker {self.client_id}] Step {step}/{self.proxy_steps-1}: Light projection applied: "
                              f"dist {dist_to_global_for_projection:.4f} -> {new_dist_after_proj:.4f} "
                              f"(target: {target_dist:.4f}, violation_ratio={violation_ratio:.2f})")
        
        # ============================================================
        # Print final optimization state
        # ============================================================
        if self.d_T is not None:
            final_dist_tensor, _ = self._compute_distance_update_space(
                proxy_param,
                self.benign_updates
            )
            final_dist = final_dist_tensor.item()
            d_T_val = float(self.d_T) if isinstance(self.d_T, torch.Tensor) else self.d_T
            final_g = final_dist - d_T_val
            final_lambda = self.lambda_dt.item() if isinstance(self.lambda_dt, torch.Tensor) else self.lambda_dt if self.lambda_dt is not None else 0.0
            final_loss = self._compute_global_loss(proxy_param, self.benign_updates).item()
            final_violation = max(0, final_g)
            violation_pct = (final_violation / d_T_val * 100) if d_T_val > 0 else 0.0
            print(f"    [Attacker {self.client_id}] Optimization completed: "
                  f"final_dist={final_dist:.4f}, d_T={d_T_val:.4f}, g={final_g:.4f}, "
                  f"lambda={final_lambda:.4f}, loss={final_loss:.4f}, "
                  f"violation={final_violation:.4f} ({violation_pct:.1f}%)")
        
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
                # Use UPDATE space distance calculation
                dist_to_global_tensor, _ = self._compute_distance_update_space(
                    malicious_update,
                    self.benign_updates
                )
                dist_to_global = dist_to_global_tensor.item()
                d_T_val = float(self.d_T) if isinstance(self.d_T, torch.Tensor) else self.d_T
                constraint_violation = max(0, dist_to_global - d_T_val)
                violation_ratio = constraint_violation / d_T_val if d_T_val > 0 else 0.0
                
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
                            d_T_val = float(self.d_T) if isinstance(self.d_T, torch.Tensor) else self.d_T
                            scale_factor = d_T_val / dist_to_global
                            malicious_update = malicious_update * scale_factor
                            print(f"      Applied strict projection (violation >100%): scaled to d_T")
                        elif violation_ratio > 0.50:  # Moderate violation (50-100%)
                            # Mild projection to 1.1×d_T, allowing 10% margin for Lagrangian flexibility
                            d_T_proj_final = float(self.d_T) if isinstance(self.d_T, torch.Tensor) else self.d_T
                            target_dist = d_T_proj_final * 1.1
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
            # Hard constraint projection mechanism (UPDATE space)
            if self.d_T is not None:
                # Use hard projection in UPDATE space
                self._hard_project_update_space(malicious_update, self.benign_updates, self.d_T)
                dist_to_global_tensor, _ = self._compute_distance_update_space(
                    malicious_update,
                    self.benign_updates
                )
                dist_to_global = dist_to_global_tensor.item()
                d_T_val = float(self.d_T) if isinstance(self.d_T, torch.Tensor) else self.d_T
                print(f"    [Attacker {self.client_id}] Final hard projection (UPDATE space): dist={dist_to_global:.4f}, d_T={d_T_val:.4f}")
                final_norm = torch.norm(malicious_update).item()
                print(f"    [Attacker {self.client_id}] Applied hard constraint projection: "
                      f"scaled from {dist_to_global:.4f} to {final_norm:.4f}")
        
        # Compute final attack objective value for logging
        # Use aggregated update for final loss (consistent with optimization objective)
        final_aggregated_update, _, _ = self._aggregate_update_no_beta(malicious_update, self.benign_updates)
        final_global_loss = self._proxy_global_loss(final_aggregated_update, max_batches=self.proxy_max_batches_eval, skip_dim_check=True)
        
        # Compute constraint (4c) for logging
        # OPTIMIZATION 2 & 6: For Lagrangian mode, reuse cached value; for hard constraint mode, compute here
        if self.use_lagrangian_dual and constraint_c_value_for_update > 0:
            # Reuse cached value from before loop (already computed)
            constraint_c_value = torch.tensor(constraint_c_value_for_update, device=self.device)
        else:
            # Hard constraint mode: compute here (only needed for logging)
            # Use the same calculation as in Lagrangian mode for consistency
            constraint_c_value = torch.tensor(0.0, device=self.device)
            # ===== CONSTRAINT (4c) COMMENTED OUT =====
            # if self.gamma is not None and len(self.benign_updates) > 0:
            if False:  # Temporarily disabled (constraint 4c is commented out, gamma is None)
                # ===== CONSTRAINT (4c) COMMENTED OUT: All computation below is disabled =====
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
                    
                    # Compute distances d(w_i(t), w̄_i(t)) for all benign clients
                    benign_gpu = [u.to(self.device) for u in self.benign_updates]
                    distances = [torch.norm(benign_update - weighted_mean) for benign_update in benign_gpu]
                    constraint_c_value = torch.stack(distances).sum()
                    # Clean up GPU references
                    del benign_updates_gpu, benign_stack, weighted_mean, benign_gpu, distances
                else:
                    # Fallback: use simple mean if data sizes not available
                    benign_gpu = [u.to(self.device) for u in self.benign_updates]
                    benign_stack_c = torch.stack(benign_gpu)
                    benign_mean_c = benign_stack_c.mean(dim=0)
                    distances = torch.norm(benign_stack_c - benign_mean_c, dim=1)
                    constraint_c_value = distances.sum()
                    # Clean up GPU references
                    del benign_gpu, benign_stack_c, benign_mean_c, distances
                torch.cuda.empty_cache()
                # ==========================================
        
        malicious_norm = torch.norm(malicious_update).item()
        log_msg = f"    [Attacker {self.client_id}] GRMP Attack: " \
                  f"F(w'_g)={final_global_loss.item():.4f}, " \
                  f"||w'_j||={malicious_norm:.4f}, " \
                  f"constraint_c={constraint_c_value.item():.4f}"
        
        # If using the Lagrangian mechanism, display the multiplier value
        # ===== CONSTRAINT (4c) COMMENTED OUT: Removed rho_dt check and rho display =====
        if self.use_lagrangian_dual and self.lambda_dt is not None:  # Removed rho_dt check
            lambda_val = self.lambda_dt.item() if isinstance(self.lambda_dt, torch.Tensor) else self.lambda_dt
            # ===== CONSTRAINT (4c) COMMENTED OUT =====
            # rho_val = self.rho_dt.item() if isinstance(self.rho_dt, torch.Tensor) else self.rho_dt
            # log_msg += f", λ={lambda_val:.4f}, ρ={rho_val:.4f}"
            log_msg += f", λ={lambda_val:.4f} [Constraint 4c disabled]"
            # ==========================================
        
        print(log_msg)
        
        # Move malicious_update to CPU before returning to free GPU memory
        malicious_update_cpu = malicious_update.cpu()
        # Clean up GPU references
        del malicious_update, final_global_loss, constraint_c_value
        torch.cuda.empty_cache()
        
        return malicious_update_cpu.detach()