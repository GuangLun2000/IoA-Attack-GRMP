# client.py
# client.py provides the Client class for federated learning clients, including benign and attacker clients.

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
        """
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.data_loader = data_loader
        self.lr = lr
        self.local_epochs = local_epochs
        self.alpha = alpha  # Regularization coefficient α ∈ [0,1] from paper formula (1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.current_round = 0
        self.is_attacker = False

    def reset_optimizer(self):
        """Reset the optimizer."""
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def set_round(self, round_num: int):
        """Set the current training round."""
        self.current_round = round_num

    def get_model_update(self, initial_params: torch.Tensor) -> torch.Tensor:
        """Calculate the model update (Current - Initial)."""
        current_params = self.model.get_flat_params()
        return current_params - initial_params

    def local_train(self, epochs=None) -> torch.Tensor:
        """Base local training method (to be overridden)."""
        raise NotImplementedError


# BenignClient class for benign clients
class BenignClient(Client):

    def __init__(self, client_id: int, model: nn.Module, data_loader, lr, local_epochs, alpha,
                 data_indices=None):
        super().__init__(client_id, model, data_loader, lr, local_epochs, alpha)
        # Track assigned data indices for proper aggregation weighting
        self.data_indices = data_indices or []

    def prepare_for_round(self, round_num: int):
        """Benign clients do not require special preparation."""
        self.set_round(round_num)

    def local_train(self, epochs=None) -> torch.Tensor:
        """Perform local training - includes proximal regularization."""
        if epochs is None:
            epochs = self.local_epochs
            
        self.model.train()
        initial_params = self.model.get_flat_params().clone()
        
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
                current_params = self.model.get_flat_params()
                proximal_term = mu * torch.norm(current_params - initial_params) ** 2
                
                loss = ce_loss + proximal_term
                
                self.optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': loss.item()})
        
        return self.get_model_update(initial_params)

    def receive_benign_updates(self, updates: List[torch.Tensor]):
        # Benign clients do not use this method
        pass


# AttackerClient class for clients that perform attacks
class AttackerClient(Client):

    def __init__(self, client_id: int, model: nn.Module, data_manager,
                 data_indices, lr, local_epochs, alpha,
                 dim_reduction_size=10000, vgae_lambda=0.5,
                 vgae_epochs=20, vgae_lr=0.01, camouflage_steps=30, camouflage_lr=0.1,
                 lambda_proximity=1.0, lambda_aggregation=0.5, graph_threshold=0.5,
                 attack_start_round=10, lambda_attack=2.0, lambda_camouflage=0.3,
                 benign_select_ratio=1.0, dual_lr=0.01,
                 lambda_dual_init=0.0, rho_dual_init=0.0,
                 proxy_step=0.1,
                 claimed_data_size=1.0):
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
            vgae_lambda: Weight for preservation loss in camouflage (default: 0.5)
            vgae_epochs: Number of epochs for VGAE training (default: 20)
            vgae_lr: Learning rate for VGAE optimizer (default: 0.01)
            camouflage_steps: Number of optimization steps for camouflage (default: 30)
            camouflage_lr: Learning rate for camouflage optimization (default: 0.1)
            lambda_proximity: Weight for constraint (4b) proximity loss (default: 1.0)
            lambda_aggregation: Weight for constraint (4c) aggregation loss (default: 0.5)
            graph_threshold: Threshold for graph adjacency matrix binarization (default: 0.5)
            attack_start_round: Round when attack phase starts (default: 10)
            lambda_attack: Weight for attack objective loss (default: 2.0) - CRITICAL for attack effectiveness (ASR = Performance Degradation Rate)
            lambda_camouflage: Weight for camouflage loss (default: 0.3) - Lower to preserve attack
            benign_select_ratio: Ratio of benign updates selected for graph (β subset, default: 1.0)
            dual_lr: Step size for dual variable updates (λ, ρ) in Lagrangian (default: 0.01)
            lambda_dual_init: Initial dual variable λ for constraint (4b) (default: 0.0)
            rho_dual_init: Initial dual variable ρ for constraint (4c) (default: 0.0)
            proxy_step: Step size for gradient-free ascent toward global-loss proxy (default: 0.1)
            claimed_data_size: Reported data size D'_j(t) for weighted aggregation (default: 1.0)
        
        Note: lr, local_epochs, and alpha must be explicitly provided to ensure consistency
        with config settings. Other parameters have defaults but should be set via config in main.py.
        """
        self.data_manager = data_manager
        self.data_indices = data_indices
        
        # Store parameters first (before using them)
        self.attack_start_round = attack_start_round
        self.vgae_lambda = vgae_lambda
        self.dim_reduction_size = dim_reduction_size
        self.vgae_epochs = vgae_epochs
        self.vgae_lr = vgae_lr
        self.camouflage_steps = camouflage_steps
        self.camouflage_lr = camouflage_lr
        self.lambda_proximity = lambda_proximity
        self.lambda_aggregation = lambda_aggregation
        self.graph_threshold = graph_threshold
        self.lambda_attack = lambda_attack  # Weight for attack objective (Formula 4a)
        self.lambda_camouflage = lambda_camouflage  # Weight for camouflage (reduced to preserve attack)
        self.benign_select_ratio = benign_select_ratio  # β selection ratio for benign updates
        self.dual_lr = dual_lr
        self.lambda_dual = lambda_dual_init  # Dual variable for constraint (4b)
        self.rho_dual = rho_dual_init      # Dual variable for constraint (4c)
        self.proxy_step = proxy_step
        self.claimed_data_size = claimed_data_size  # For weighted aggregation (paper: D'(t))

        dummy_loader = data_manager.get_empty_loader()
        super().__init__(client_id, model, dummy_loader, lr, local_epochs, alpha)
        self.is_attacker = True

        # VGAE components
        self.vgae = None
        self.vgae_optimizer = None
        self.benign_updates = []
        self.feature_indices = None
        
        # Data-agnostic attack: no local data usage
        self.original_business_loader = None
        self.proxy_loader = data_manager.get_proxy_eval_loader()
        
        # Formula 4 constraints parameters
        self.d_T = None  # Distance threshold for constraint (4b): d(w'_j(t), w_g(t)) ≤ d_T
        self.gamma = None  # Upper bound for constraint (4c): Σ β'_{i,j}(t) d(w_i(t), w̄_i(t)) ≤ Γ
        self.global_model_params = None  # Store global model params for constraint (4b)
        self._flat_numel = self.model.get_flat_params().numel()

    def prepare_for_round(self, round_num: int):
        """Prepare for a new training round."""
        self.set_round(round_num)
        # Data-agnostic attacker keeps an empty loader
        self.data_loader = self.data_manager.get_empty_loader()

    def receive_benign_updates(self, updates: List[torch.Tensor]):
        """Receive updates from benign clients."""
        # Store detached copies to avoid graph retention issues
        self.benign_updates = [u.detach().clone() for u in updates]

    def _select_benign_subset(self) -> List[torch.Tensor]:
        """
        Select a subset of benign updates (β selection) to build the graph,
        approximating the 0-1 knapsack in the paper. We use a simple heuristic:
        pick the farthest updates from the mean until ratio or gamma budget is met.
        
        Different attackers select different subsets to ensure diversity in attack strategies.
        """
        if not self.benign_updates:
            return []
        target_k = max(1, int(len(self.benign_updates) * self.benign_select_ratio))
        benign_stack = torch.stack(self.benign_updates)
        benign_mean = benign_stack.mean(dim=0)
        distances = torch.norm(benign_stack - benign_mean, dim=1)
        
        # Use client_id to introduce diversity: different attackers use different selection strategies
        # Strategy 0: Select farthest from mean (original strategy)
        # Strategy 1: Select closest to mean
        # Strategy 2: Select medium distance
        # Strategy 3: Random selection with distance weighting
        strategy = self.client_id % 4
        
        if strategy == 0:
            # Original: Sort by distance descending (farthest first)
            sorted_idx = torch.argsort(distances, descending=True)
        elif strategy == 1:
            # Select closest to mean (different perspective)
            sorted_idx = torch.argsort(distances, descending=False)
        elif strategy == 2:
            # Select medium distance: sort by absolute deviation from median distance
            median_dist = torch.median(distances)
            deviations = torch.abs(distances - median_dist)
            sorted_idx = torch.argsort(deviations, descending=False)
        else:  # strategy == 3
            # Random selection with distance weighting: shuffle but prefer diverse updates
            # Use client_id as seed for reproducibility but different per attacker
            rng = torch.Generator(device=self.device)
            rng.manual_seed(self.client_id * 1000 + 42)
            # Create weighted probabilities: prefer updates with different distances
            # Normalize distances to [0, 1] range for weighting
            normalized_dist = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)
            # Mix distance-based and random selection
            weights = 0.5 * normalized_dist + 0.5 * torch.ones_like(normalized_dist)
            probs = weights / weights.sum()
            # Sample indices based on weighted probabilities
            sampled_indices = torch.multinomial(probs, num_samples=min(target_k * 2, len(self.benign_updates)), 
                                               replacement=False, generator=rng)
            sorted_idx = sampled_indices[torch.argsort(distances[sampled_indices], descending=True)]
        
        selected = []
        total_dist = 0.0
        for idx in sorted_idx:
            if len(selected) >= target_k:
                break
            d = distances[idx].item()
            # If gamma is set, enforce cumulative distance budget
            if self.gamma is not None and (total_dist + d) > self.gamma:
                continue
            selected.append(self.benign_updates[idx])
            total_dist += d
        
        if not selected:
            # Fallback: take based on strategy
            if strategy == 3:
                # For random strategy, just take the first sampled
                if len(sorted_idx) > 0:
                    selected = [self.benign_updates[sorted_idx[0]]]
                else:
                    selected = [self.benign_updates[0]]
            else:
                selected = [self.benign_updates[sorted_idx[0]]]
        
        return selected

    def local_train(self, epochs=None) -> torch.Tensor:
        """
        Enhanced attacker strategy: Use proxy data for understanding updates.
        
        Strategy: Attackers use clean proxy data (proxy_loader) to perform light training,
        which helps them understand the update direction and magnitude. This "understanding update"
        will be combined with VGAE+GSP generated attack in camouflage_update.
        
        This maintains data-agnostic nature (using clean proxy data, not attacker's local data)
        while enabling more targeted and effective attacks.
        """
        if epochs is None:
            epochs = self.local_epochs
        
        # Use proxy_loader for understanding updates (clean data, not attacker's local data)
        if self.proxy_loader is None or len(self.proxy_loader.dataset) == 0:
            # Fallback: return zero update if no proxy data available
            initial_params = self.model.get_flat_params().clone()
            return torch.zeros_like(initial_params)
        
        # Perform light training on proxy data to understand update patterns
        self.model.train()
        initial_params = self.model.get_flat_params().clone()
        
        # Use fewer epochs for understanding (not full training)
        understanding_epochs = max(1, epochs // 2)  # Use half epochs for understanding
        
        # Proximal regularization coefficient (same as benign clients)
        mu = self.alpha
        
        for epoch in range(understanding_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch in self.proxy_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                logits = outputs
                
                ce_loss = nn.CrossEntropyLoss()(logits, labels)
                
                # Add proximal regularization term
                current_params = self.model.get_flat_params()
                proximal_term = mu * torch.norm(current_params - initial_params) ** 2
                
                loss = ce_loss + proximal_term
                
                self.optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Limit batches for efficiency (understanding doesn't need full dataset)
                if num_batches >= 5:  # Use only first 5 batches for understanding
                    break
        
        # Get the understanding update (how model changes after training on proxy data)
        understanding_update = self.get_model_update(initial_params)
        
        return understanding_update

    def _get_reduced_features(self, updates: List[torch.Tensor], fix_indices=True) -> torch.Tensor:
        """
        Helper function to reduce dimensionality of updates.
        Randomly selects indices to slice the high-dimensional vector.
        """
        stacked_updates = torch.stack(updates)
        total_dim = stacked_updates.shape[1]
        
        # 如果更新维度小于降维目标，则不降维
        if total_dim <= self.dim_reduction_size:
            return stacked_updates
            
        # 每一轮攻击开始时，固定一组特征索引，保证这一轮内的训练一致性
        if self.feature_indices is None or not fix_indices:
            # Randomly select indices
            self.feature_indices = torch.randperm(total_dim)[:self.dim_reduction_size].to(self.device)
            
        # Select features
        reduced_features = torch.index_select(stacked_updates, 1, self.feature_indices)
        return reduced_features

    def _flat_to_param_dict(self, flat_params: torch.Tensor, skip_dim_check: bool = False) -> Dict[str, torch.Tensor]:
        """
        Convert flat tensor to param dict for stateless.functional_call.
        
        Args:
            flat_params: Flattened parameter tensor
            skip_dim_check: If True, skip dimension check (for performance in loops)
        """
        param_dict = {}
        offset = 0
        flat_params = flat_params.view(-1)  # Ensure 1D (O(1), just view change)
        total_numel = flat_params.numel()
        
        for name, param in self.model.named_parameters():
            numel = param.numel()
            if not skip_dim_check and offset + numel > total_numel:
                # Dimension mismatch: return empty dict to avoid errors
                print(f"    [Attacker {self.client_id}] Param dict dimension mismatch at {name}: offset {offset} + numel {numel} > total {total_numel}")
                return {}
            param_dict[name] = flat_params[offset:offset + numel].view_as(param)
            offset += numel
        return param_dict

    def _proxy_global_loss(self, malicious_update: torch.Tensor, max_batches: int = 1, 
                           skip_dim_check: bool = False) -> torch.Tensor:
        """
        Differentiable proxy for F(w'_g): cross-entropy on a small clean subset,
        using stateless.functional_call with (w_g + malicious_update).
        
        Args:
            malicious_update: Update vector to evaluate
            max_batches: Maximum number of batches to process
            skip_dim_check: If True, skip dimension check (for performance in loops)
        """
        if self.global_model_params is None or self.proxy_loader is None:
            return torch.tensor(0.0, device=self.device)

        # Ensure shapes match: flatten to 1D and check dimension
        malicious_update = malicious_update.view(-1)  # Flatten to 1D (O(1), just view change)
        if not skip_dim_check and malicious_update.numel() != self._flat_numel:
            # Dimension mismatch: return zero loss to avoid errors
            print(f"    [Attacker {self.client_id}] Proxy loss dimension mismatch: got {malicious_update.numel()}, expected {self._flat_numel}")
            return torch.tensor(0.0, device=self.device)

        candidate_params = self.global_model_params + malicious_update
        # Skip dimension check if already validated (performance optimization)
        param_dict = self._flat_to_param_dict(candidate_params, skip_dim_check=skip_dim_check)

        total_loss = 0.0
        batches = 0

        for batch in self.proxy_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            logits = stateless.functional_call(
                self.model,
                param_dict,
                args=(),
                kwargs={'input_ids': input_ids, 'attention_mask': attention_mask}
            )

            ce_loss = F.cross_entropy(logits, labels)
            total_loss = total_loss + ce_loss
            batches += 1
            if batches >= max_batches:
                break

        if batches == 0:
            return torch.tensor(0.0, device=self.device)

        return total_loss / batches

    def _construct_graph(self, reduced_features: torch.Tensor):
        """
        Construct graph according to the paper (Section III).
        
        Paper formulation:
        - Feature matrix F(t) = [w_1(t), ..., w_i(t)]^T ∈ R^{I×M}
        - Adjacency matrix A(t) ∈ R^{M×M} (NOT I×I!)
        - δ_{m,m'} = cosine similarity between w_m(t) and w_{m'}(t)
        - w_m(t) ∈ R^{I×1} is the m-th COLUMN of F(t)
        
        So we need to compute similarity between COLUMNS (parameter dimensions),
        not ROWS (clients).
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
        adj_matrix = (adj_matrix > self.graph_threshold).float()
        
        return adj_matrix

    def _train_vgae(self, adj_matrix: torch.Tensor, feature_matrix: torch.Tensor, epochs=None):
        """
        Train the VGAE model according to the paper.
        
        Paper formulation:
        - Input: A ∈ R^{M×M} (adjacency), F ∈ R^{I×M} (features)
        - For VGAE, we use F^T ∈ R^{M×I} as node features
        - Each node represents a parameter dimension
        - VGAE learns to reconstruct A
        """
        if epochs is None:
            epochs = self.vgae_epochs
        
        # adj_matrix shape: (M, M) - from _construct_graph
        # feature_matrix shape: (I, M) - original features
        # For VGAE input, we use F^T as node features: (M, I)
        node_features = feature_matrix.t()  # (M, I)
        
        input_dim = node_features.shape[1]  # I (number of clients)
        num_nodes = node_features.shape[0]  # M (feature dimension)
        
        # Initialize VGAE if needed
        # Paper: input_dim = I (number of clients/benign models)
        if self.vgae is None or self.vgae.gc1.weight.shape[0] != input_dim:
            # Following paper: hidden1_dim=32, hidden2_dim=16
            hidden_dim = 32
            latent_dim = 16
            self.vgae = VGAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, dropout=0.0).to(self.device)
            self.vgae_optimizer = optim.Adam(self.vgae.parameters(), lr=self.vgae_lr)

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
        self.global_model_params = global_params.clone().detach().to(self.device)
    
    def set_constraint_params(self, d_T: float = None, gamma: float = None):
        """Set constraint parameters for Formula 4."""
        self.d_T = d_T  # Constraint (4b): d(w'_j(t), w_g(t)) ≤ d_T
        self.gamma = gamma  # Constraint (4c): Σ β'_{i,j}(t) d(w_i(t), w̄_i(t)) ≤ Γ

    def _compute_attack_loss(self, malicious_update: torch.Tensor) -> torch.Tensor:
        """
        Compute attack loss using a DIRECT and EFFECTIVE approach.
        
        NEW STRATEGY: Instead of trying to compute model outputs (which has gradient issues),
        we use a direction-based attack loss that encourages the malicious update to:
        1. Be different from benign updates (attack direction)
        2. But not too different (avoid detection)
        
        The key insight: The REAL attack comes from training on poisoned data (in local_train).
        This function just helps PRESERVE that attack while adding camouflage.
        
        Returns a loss that should be MAXIMIZED (caller will negate it).
        """
        if not self.benign_updates:
            # No benign updates to compare, return zero
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Compute benign update statistics
        benign_stack = torch.stack(self.benign_updates)
        benign_mean = benign_stack.mean(dim=0)
        
        # ATTACK STRATEGY: Encourage update to be in a specific "attack direction"
        # The poisoned_update from local_train already contains attack information
        # We want to preserve this attack direction while camouflaging
        
        # Method 1: Maximize distance from benign mean (encourages distinct attack)
        # But this conflicts with camouflage, so we use a softer version
        distance_from_benign = torch.norm(malicious_update - benign_mean)
        
        # Method 2: Maximize update magnitude (stronger attack)
        # Larger updates have more impact on the global model
        update_magnitude = torch.norm(malicious_update)
        
        # Method 3: Encourage alignment with the original poisoned direction
        # This is implicitly handled by loss_preservation in camouflage_update
        
        # Combined attack loss: We want to MAXIMIZE this
        # Higher distance = more distinct from benign (stronger attack)
        # Higher magnitude = larger impact on global model
        attack_loss = 0.5 * distance_from_benign + 0.5 * update_magnitude
        
        return attack_loss

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
            poisoned_update: The poisoned update from local training (full dimension, for fallback only)
            
        Returns:
            Malicious update generated using GSP (reduced dimension M, or None if failed)
        """
        M = feature_matrix.shape[1]  # Reduced dimension
        
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
        # We combine the reconstructed features with an attack direction that degrades performance
        
        # FIX 1: Generate attack direction that deviates from benign mean (not just mimic)
        benign_mean = feature_matrix.mean(dim=0)  # Mean of benign updates
        gsp_recon_mean = F_recon.mean(dim=0)  # Mean of GSP reconstructed features
        
        # Attack direction: combine GSP reconstruction with deviation from benign mean
        # This ensures the attack moves away from benign updates (degrading performance)
        # while still using the graph structure learned by VGAE
        attack_deviation = gsp_recon_mean - benign_mean  # Deviation from benign
        malicious_direction = gsp_recon_mean + 0.5 * attack_deviation  # Bias towards attack
        
        # Scale by positive factor to ensure attack direction (not just random noise)
        # Use client_id for diversity but ensure positive scale for attack
        rng = torch.Generator(device=self.device)
        rng.manual_seed(self.client_id * 1000 + 42)
        # Positive scale range: 0.1 to 0.3 (ensures attack direction, not just noise)
        # Use torch.rand() with generator, then scale to [0.1, 0.3)
        random_scale = (torch.rand(1, device=self.device, generator=rng) * 0.2 + 0.1).item()
        gsp_attack = malicious_direction * random_scale
        
        return gsp_attack

    def camouflage_update(self, poisoned_update: torch.Tensor) -> torch.Tensor:
        """
        Enhanced GRMP Attack using VGAE + GSP + Understanding Update.
        
        Enhanced Strategy:
        1. Understanding Update: Attacker performs light training on proxy data to understand
           update patterns (from local_train, now returns understanding_update instead of zero)
        2. VGAE + GSP Attack: Generate structured attack using graph-based approach
        3. Combination: Combine understanding update (40%) with GSP attack (60%) for enhanced
           effectiveness while maintaining stealth
        
        Paper Algorithm 1 (Base):
        1. Calculate A according to cosine similarity (eq. 8)
        2. Train VGAE to maximize L_loss (eq. 12), obtain optimal Â
        3. Use GSP module to obtain F̂, determine w'_j(t) based on F̂
        
        Enhancement: The understanding_update (from proxy data training) provides learned
        update patterns, which are combined with GSP-generated attack for more targeted
        and effective poisoning while maintaining data-agnostic nature.
        """
        if not self.benign_updates:
            print(f"    [Attacker {self.client_id}] No benign updates, using raw poisoned update")
            return poisoned_update

        # Reset feature indices for this session
        self.feature_indices = None
        
        # ============================================================
        # STEP 1: Prepare feature matrix F ∈ R^{I×M}
        # ============================================================
        selected_benign = self._select_benign_subset()
        if not selected_benign:
            print(f"    [Attacker {self.client_id}] No benign subset selected, fallback to raw poisoned update")
            return poisoned_update

        benign_stack = torch.stack([u.detach() for u in selected_benign])  # (I, full_dim)
        
        # Reduce dimensionality for computational efficiency
        reduced_benign = self._get_reduced_features(selected_benign, fix_indices=False)  # (I, M)
        M = reduced_benign.shape[1]
        I = reduced_benign.shape[0]
        
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
        
        # ============================================================
        # STEP 5: Combine understanding update with GSP attack for enhanced effectiveness
        # Expand GSP attack back to full dimension; non-selected dims remain zero.
        # Then combine with understanding_update (poisoned_update) to leverage learned patterns
        malicious_update = torch.zeros_like(poisoned_update)
        total_dim = malicious_update.shape[0]
        
        if gsp_attack_reduced is not None:
            gsp_dim = gsp_attack_reduced.shape[0]
            
            if self.feature_indices is not None:
                # Dimension reduction was applied
                expected_dim = len(self.feature_indices)
                if gsp_dim == expected_dim:
                    # Correct dimension: expand back to full dimension
                    malicious_update[self.feature_indices] = gsp_attack_reduced
                else:
                    # Dimension mismatch: log warning and use zeros
                    print(f"    [Attacker {self.client_id}] GSP dimension mismatch: got {gsp_dim}, expected {expected_dim}, using zeros")
            else:
                # No dimension reduction: GSP attack should be full dimension
                if gsp_dim == total_dim:
                    # Correct dimension: use directly
                    malicious_update = gsp_attack_reduced
                else:
                    # Dimension mismatch: log warning and use zeros
                    print(f"    [Attacker {self.client_id}] GSP dimension mismatch: got {gsp_dim}, expected {total_dim}, using zeros")
        else:
            # GSP attack is None: malicious_update remains zeros
            print(f"    [Attacker {self.client_id}] GSP attack is None, using zeros")
        
        # Enhanced strategy: Use understanding update to GUIDE attack, not directly mix
        # The understanding_update (poisoned_update) is a "correct" update that improves performance.
        # We use it to understand update patterns, then guide GSP attack to be more effective.
        understanding_update = poisoned_update  # This is now the understanding update from local_train
        
        # Ensure dimension matching
        if understanding_update.shape[0] != malicious_update.shape[0]:
            print(f"    [Attacker {self.client_id}] Dimension mismatch: understanding_update {understanding_update.shape[0]} vs malicious_update {malicious_update.shape[0]}")
            # If mismatch, use malicious_update only (skip understanding guidance)
            understanding_update = torch.zeros_like(malicious_update)
        
        understanding_norm = torch.norm(understanding_update)
        gsp_norm = torch.norm(malicious_update)
        
        if understanding_norm > 1e-8 and gsp_norm > 1e-8:
            # Strategy: Use understanding update to guide attack, not weaken it
            # 1. Use understanding update's NORM for stealth (match the magnitude)
            # 2. Use understanding update's DIRECTION to guide attack (deviate from it for attack)
            
            # Get directions
            understanding_direction = understanding_update / understanding_norm
            gsp_direction = malicious_update / gsp_norm
            
            # Compute how much GSP attack deviates from understanding direction
            # Negative cosine similarity means opposite direction (good for attack)
            cosine_sim = torch.cosine_similarity(
                gsp_direction.unsqueeze(0),
                understanding_direction.unsqueeze(0)
            ).item()
            
            # If GSP attack is too similar to understanding (both improve performance), enhance deviation
            if cosine_sim > 0.3:  # Too similar, need more deviation
                # Create deviation component: perpendicular to understanding direction
                # Project GSP direction onto understanding direction
                projection = torch.dot(gsp_direction, understanding_direction) * understanding_direction
                perpendicular = gsp_direction - projection
                
                # Enhance perpendicular component (deviation from correct direction)
                if torch.norm(perpendicular) > 1e-8:
                    perpendicular = perpendicular / torch.norm(perpendicular)
                    # Add deviation to make attack more effective
                    deviation_component = perpendicular * understanding_norm * 0.4
                    malicious_update = malicious_update + deviation_component
            
            # Match norm to understanding update for stealth (but allow slight increase for attack)
            current_norm = torch.norm(malicious_update)
            if current_norm > 1e-8:
                # Scale to understanding norm (stealth) with slight increase (attack)
                target_norm = understanding_norm * 1.15  # 15% larger for attack, but close for stealth
                scale_factor = target_norm / current_norm
                scale_factor = torch.clamp(torch.tensor(scale_factor), 0.8, 1.3)  # Limit scaling
                malicious_update = malicious_update * scale_factor.item()
        elif understanding_norm > 1e-8:
            # If GSP attack is zero/empty, use understanding update's norm but invert direction
            # This creates an attack that has similar magnitude (stealth) but opposite direction (attack)
            malicious_update = -understanding_update * 0.9  # Invert and slightly scale down
        # else: malicious_update remains as GSP attack (or zeros if both are empty)
        
        # ============================================================
        # STEP 6: Lagrangian-style objective & dual updates (approximation)
        # FIX 4: Add explicit attack direction to ensure performance degradation
        # attack_obj ~ deviation from benign mean (maximize)
        # constraint_b: d(w'_j, w'_g) ≤ d_T  (we approximate dist by ||malicious_update||)
        # constraint_c: sum β d(w_i, \bar w_i) ≤ Γ (use selected benign subset)
        # ============================================================
        benign_norms = torch.stack([torch.norm(u) for u in self.benign_updates])
        benign_mean_full = torch.stack([u.detach() for u in self.benign_updates]).mean(dim=0)
        
        # FIX 4: Add explicit attack direction component
        # Compute gradient direction that increases global loss (if proxy loader available)
        attack_direction = None
        if self.global_model_params is not None and self.proxy_loader is not None:
            # Compute gradient of proxy loss w.r.t. malicious_update to get attack direction
            temp_param = malicious_update.clone().detach().requires_grad_(True)
            # Check dimension once before loop
            if temp_param.view(-1).numel() == self._flat_numel:
                proxy_loss = self._proxy_global_loss(temp_param, max_batches=1, skip_dim_check=True)
                if proxy_loss.requires_grad:
                    proxy_loss.backward()
                    attack_direction = temp_param.grad
                    # Normalize to unit direction
                    if attack_direction is not None and torch.norm(attack_direction) > 1e-8:
                        attack_direction = attack_direction / torch.norm(attack_direction)
        
        # If attack direction available, bias malicious_update towards it
        if attack_direction is not None:
            # Combine GSP attack with explicit attack direction
            # Weight: 70% GSP attack, 30% explicit attack direction
            attack_component = attack_direction * torch.norm(malicious_update) * 0.3
            malicious_update = malicious_update * 0.7 + attack_component
        
        attack_obj = torch.norm(malicious_update - benign_mean_full)
        
        # Constraint (4b): d(w'_j, w'_g) ≤ d_T
        # FIX: Use distance from global model, not just norm of update
        constraint_b = torch.tensor(0.0, device=self.device)
        if self.d_T is not None and self.global_model_params is not None:
            # Compute malicious model: w'_j = w_g + malicious_update
            malicious_model = self.global_model_params + malicious_update
            # Distance from global model: ||w'_j - w'_g|| = ||malicious_update||
            # But we should use the actual distance, which is the norm of the update
            # However, the constraint in the paper is about the update itself
            # So we use: ||w'_j - w'_g|| = ||malicious_update|| ≤ d_T
            dist_global = torch.norm(malicious_update)
            constraint_b = F.relu(dist_global - self.d_T)
        elif self.d_T is not None:
            # Fallback: use norm if global_model_params not available
            dist_global = torch.norm(malicious_update)
            constraint_b = F.relu(dist_global - self.d_T)
        
        # Constraint (4c): Σ β_i d(w_i, w̄_i) ≤ γ
        # FIX: Use all benign updates, not just selected subset
        constraint_c = torch.tensor(0.0, device=self.device)
        if self.gamma is not None and len(self.benign_updates) > 0:
            # Use all benign updates for accurate constraint calculation
            all_benign_stack = torch.stack([u.detach() for u in self.benign_updates])
            benign_mean_all = all_benign_stack.mean(dim=0)
            # Compute distances from mean for all benign updates
            distances = torch.norm(all_benign_stack - benign_mean_all, dim=1)
            # Sum of distances (aggregation distance)
            agg_dist = distances.sum()
            constraint_c = F.relu(agg_dist - self.gamma)
        elif self.gamma is not None and len(selected_benign) > 0:
            # Fallback: use selected subset if no benign_updates available
            sel_stack = torch.stack(selected_benign)
            sel_mean = sel_stack.mean(dim=0)
            distances = torch.norm(sel_stack - sel_mean, dim=1)
            agg_dist = distances.sum()
            constraint_c = F.relu(agg_dist - self.gamma)
        
        # Lagrangian (no backprop here; used for dual updates and logging)
        lagrangian = -attack_obj
        if self.d_T is not None:
            lagrangian = lagrangian + self.lambda_dual * constraint_b
        if self.gamma is not None:
            lagrangian = lagrangian + self.rho_dual * constraint_c
        
        # Dual variable updates (projected, step size dual_lr)
        if self.d_T is not None:
            self.lambda_dual = max(0.0, self.lambda_dual + self.dual_lr * constraint_b.item())
        if self.gamma is not None:
            self.rho_dual = max(0.0, self.rho_dual + self.dual_lr * constraint_c.item())
        
        # ============================================================
        # STEP 7: Proxy ascent toward global loss (gradient-based on proxy norm)
        # FIX 2 & 3: Increase proxy steps and ensure optimization is not completely negated
        # Use a larger optimizer on malicious_update to maximize global loss while respecting constraints
        proxy_steps = 10  # Increased from 3 to 10 for stronger optimization
        proxy_lr = self.proxy_step * 2  # Increased from 0.1 to 0.2 for faster convergence
        proxy_param = malicious_update.clone().detach().to(self.device)
        proxy_param.requires_grad_(True)
        proxy_opt = optim.Adam([proxy_param], lr=proxy_lr)
        
        # Store initial norm to preserve attack direction after optimization
        initial_norm = torch.norm(proxy_param).item()
        
        # Check dimension once before loop (performance optimization)
        proxy_param_flat = proxy_param.view(-1)
        dim_valid = proxy_param_flat.numel() == self._flat_numel
        
        for step in range(proxy_steps):
            proxy_opt.zero_grad()
            # Proxy objective: maximize F(w'_g) via CE loss on clean subset, minus penalties
            # Skip dimension check in loop for performance (already checked above)
            proxy_obj = self._proxy_global_loss(proxy_param, max_batches=1, skip_dim_check=dim_valid)
            proxy_penalty = torch.tensor(0.0, device=self.device)
            
            # Soft constraints: use squared penalty to allow some violation for stronger attack
            if self.d_T is not None:
                # Soft constraint: allow slight violation (up to 1.2 * d_T) for stronger attack
                # Use norm of update (distance from global model)
                soft_d_T = self.d_T * 1.2
                proxy_penalty = proxy_penalty + 0.1 * F.relu(torch.norm(proxy_param) - soft_d_T) ** 2
            if self.gamma is not None and len(self.benign_updates) > 0:
                # Use all benign updates for accurate constraint calculation
                all_benign_stack = torch.stack([u.detach() for u in self.benign_updates])
                benign_mean_all = all_benign_stack.mean(dim=0)
                distances = torch.norm(all_benign_stack - benign_mean_all, dim=1)
                agg_dist = distances.sum()
                soft_gamma = self.gamma * 1.2
                proxy_penalty = proxy_penalty + 0.1 * F.relu(agg_dist - soft_gamma) ** 2
            elif self.gamma is not None and len(selected_benign) > 0:
                # Fallback: use selected subset if no benign_updates available
                sel_stack = torch.stack(selected_benign)
                sel_mean = sel_stack.mean(dim=0)
                distances = torch.norm(sel_stack - sel_mean, dim=1)
                agg_dist = distances.sum()
                soft_gamma = self.gamma * 1.2
                proxy_penalty = proxy_penalty + 0.1 * F.relu(agg_dist - soft_gamma) ** 2
            
            # Maximize (proxy_obj - penalty) => minimize negative
            proxy_loss = -(proxy_obj - proxy_penalty)
            proxy_loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_([proxy_param], max_norm=1.0)
            
            proxy_opt.step()
        
        malicious_update = proxy_param.detach()
        
        # STEP 8: Apply norm matching + (4b) clipping for camouflage
        # FIX 3: Allow larger norm to preserve attack effectiveness
        # ============================================================
        # Calculate target norm: allow larger deviation from benign mean for stronger attack
        benign_mean_norm = benign_norms.mean().item()
        benign_std_norm = benign_norms.std().item()
        
        # Use consistent strategy across all rounds (no hard-coded round-based logic)
        # Allow moderate deviation from benign mean for effective attack while maintaining stealth
        target_norm = benign_mean_norm + 0.5 * benign_std_norm
        
        # If d_T is set, use it as upper bound but allow larger norm if beneficial
        if self.d_T is not None:
            # Allow norm up to d_T, but prefer larger norm if it helps attack
            target_norm = min(target_norm, self.d_T * 1.1)  # Allow 10% over d_T for stronger attack
        
        current_norm = torch.norm(malicious_update)
        if current_norm > 1e-8:
            scale_factor = target_norm / current_norm
            # FIX 2: Preserve attack direction: allow larger scaling to maintain optimization effect
            # Clamp to allow up to 3x scaling (was 2x) to preserve proxy optimization
            scale_factor = torch.clamp(scale_factor, 0.3, 3.0)
            malicious_update = malicious_update * scale_factor
        
        # FIX 3: Apply hard constraint only if significantly over d_T (allow some violation)
        if self.d_T is not None:
            update_norm = torch.norm(malicious_update)
            # Only clip if significantly over d_T (allow 10% violation for attack effectiveness)
            if update_norm > self.d_T * 1.1:
                malicious_update = malicious_update * (self.d_T * 1.1 / update_norm)
        
        print(f"    [Attacker {self.client_id}] GSP Attack: norm={torch.norm(malicious_update):.4f}, "
              f"attack_obj={attack_obj:.4f}, λ={self.lambda_dual:.4f}, ρ={self.rho_dual:.4f}")
        
        return malicious_update.detach()