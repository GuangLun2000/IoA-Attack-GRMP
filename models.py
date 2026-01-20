# models.py
# This module defines the NewsClassifierModel for News classification
# and the VGAE model for GRMP attack.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
from typing import Tuple, Optional

# --- Constants ---
MODEL_NAME = 'distilbert-base-uncased'
NUM_LABELS = 4

# Optional LoRA support
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("  Warning: peft library not available. LoRA support disabled. Install with: pip install peft")


class NewsClassifierModel(nn.Module):
    """
    DistilBERT-based model for news classification.
    Supports both full fine-tuning and LoRA fine-tuning modes.
    Wraps the Hugging Face AutoModelForSequenceClassification.
    
    Args:
        model_name: Pre-trained model name or path
        num_labels: Number of classification labels
        use_lora: If True, use LoRA fine-tuning instead of full fine-tuning
        lora_r: LoRA rank (rank of the low-rank matrices)
        lora_alpha: LoRA alpha (scaling factor, typically 2*r)
        lora_dropout: LoRA dropout rate
        lora_target_modules: List of module names to apply LoRA to
    """

    def __init__(self, model_name: str = MODEL_NAME, num_labels: int = NUM_LABELS,
                 use_lora: bool = False, lora_r: int = 16, lora_alpha: int = 32,
                 lora_dropout: float = 0.1, lora_target_modules: Optional[list] = None):
        super().__init__()
        
        self.use_lora = use_lora
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load base model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        # Verify that the correct model is loaded
        model_type = type(self.model).__name__
        
        # Setup LoRA if requested
        if use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError(
                    "LoRA support requires peft library. Install with: pip install peft"
                )
            
            # Default target modules based on model family
            if lora_target_modules is None:
                model_name_lower = model_name.lower()
                
                # DistilBERT uses these module names for attention layers
                if "distilbert" in model_name_lower:
                    lora_target_modules = ["q_lin", "k_lin", "v_lin", "out_lin"]
                # DeBERTa v2/v3 uses projection module names in attention
                elif "deberta" in model_name_lower:
                    lora_target_modules = ["query_proj", "key_proj", "value_proj", "dense"]
                # BERT/RoBERTa style attention module names
                elif "bert" in model_name_lower or "roberta" in model_name_lower:
                    lora_target_modules = ["query", "key", "value", "dense"]
                else:
                    # Fallback: keep None and let PEFT raise a clearer error if unsupported
                    lora_target_modules = None
            
            # Configure LoRA
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",  # Don't add bias parameters
            )
            
            # Apply LoRA to model
            self.model = get_peft_model(self.model, peft_config)
            
            # Print LoRA statistics
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"  Loaded model: {model_type} (from {model_name}) with LoRA")
            print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of {total_params:,} total)")
        else:
            print(f"  Loaded model: {model_type} (from {model_name}) [Full Fine-tuning]")
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize classifier weights to avoid initial bias."""
        with torch.no_grad():
            # In LoRA mode, model structure may be different (PEFT wrapper)
            # Try to access classifier from base_model if using PEFT
            if self.use_lora and hasattr(self.model, 'base_model'):
                # PEFT model: access through base_model
                base_model = self.model.base_model.model
                if hasattr(base_model, 'classifier'):
                    nn.init.xavier_uniform_(base_model.classifier.weight)
                    nn.init.zeros_(base_model.classifier.bias)
            elif hasattr(self.model, 'classifier'):
                # Standard model: direct access
                nn.init.xavier_uniform_(self.model.classifier.weight)
                nn.init.zeros_(self.model.classifier.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits

    def get_flat_params(self, requires_grad: bool = False) -> torch.Tensor:
        """
        Get model parameters flattened into a single 1D tensor.
        - Full fine-tuning: Returns all parameters
        - LoRA: Returns only LoRA parameters (trainable parameters)
        
        Args:
            requires_grad: If True, preserve gradients (for training). If False, detach (for aggregation).
        
        Useful for Federated Learning aggregation.
        """
        if self.use_lora:
            return self._get_lora_params(requires_grad=requires_grad)
        else:
            return self._get_full_params(requires_grad=requires_grad)
    
    def _get_full_params(self, requires_grad: bool = False) -> torch.Tensor:
        """Get all model parameters (full fine-tuning mode)."""
        # Use self.model.parameters() to access the actual model parameters
        if requires_grad:
            # Preserve gradients for training (e.g., proximal regularization)
            return torch.cat([p.view(-1) for p in self.model.parameters()])
        else:
            # Detach for aggregation/updates
            return torch.cat([p.data.view(-1) for p in self.model.parameters()])
    
    def _get_lora_params(self, requires_grad: bool = False) -> torch.Tensor:
        """Get only LoRA parameters (LoRA fine-tuning mode)."""
        lora_params = []
        # Use self.model.parameters() to access the actual model parameters
        # In LoRA mode, only trainable parameters are LoRA params
        for param in self.model.parameters():
            if param.requires_grad:
                if requires_grad:
                    # Preserve gradients for training (e.g., proximal regularization)
                    lora_params.append(param.view(-1))
                else:
                    # Detach for aggregation/updates
                    lora_params.append(param.data.view(-1))
        
        if not lora_params:
            # Fallback: if no trainable params found, return empty tensor
            # This shouldn't happen, but handle gracefully
            return torch.tensor([], dtype=torch.float32)
        
        return torch.cat(lora_params)

    def set_flat_params(self, flat_params: torch.Tensor):
        """
        Set model parameters from a single flattened 1D tensor.
        - Full fine-tuning: Sets all parameters
        - LoRA: Sets only LoRA parameters (trainable parameters)
        """
        if self.use_lora:
            self._set_lora_params(flat_params)
        else:
            self._set_full_params(flat_params)
    
    def _set_full_params(self, flat_params: torch.Tensor):
        """Set all model parameters (full fine-tuning mode)."""
        offset = 0
        # Use self.model.parameters() to access the actual model parameters
        for param in self.model.parameters():
            numel = param.numel()
            param.data.copy_(
                flat_params[offset:offset + numel].view(param.shape)
            )
            offset += numel
    
    def _set_lora_params(self, flat_params: torch.Tensor):
        """Set only LoRA parameters (LoRA fine-tuning mode)."""
        offset = 0
        # Use self.model.parameters() to maintain consistent order with _get_lora_params
        # Only update trainable parameters (LoRA params)
        for param in self.model.parameters():
            if param.requires_grad:
                numel = param.numel()
                if offset + numel > flat_params.numel():
                    raise ValueError(
                        f"Flat params size mismatch: trying to set {numel} params "
                        f"but only {flat_params.numel() - offset} remaining. "
                        f"Total needed: {offset + numel}, provided: {flat_params.numel()}"
                    )
                # Get the parameter slice
                param_slice = flat_params[offset:offset + numel].view(param.shape)
                # CRITICAL: Ensure param_slice is on the same device as param
                # This prevents device mismatch errors, especially when flat_params is on CPU
                # but param is on GPU (or vice versa)
                if param_slice.device != param.device:
                    param_slice = param_slice.to(param.device)
                # Ensure dtype matches
                if param_slice.dtype != param.dtype:
                    param_slice = param_slice.to(dtype=param.dtype)
                param.data.copy_(param_slice)
                offset += numel
        
        # Verify we used all parameters
        if offset != flat_params.numel():
            raise ValueError(
                f"Flat params size mismatch: used {offset} params "
                f"but {flat_params.numel()} provided. "
                f"Some LoRA parameters may not have been set."
            )


class GraphConvolutionLayer(nn.Module):
    """
    Simple Graph Convolution Layer (GCN).
    Formula: Output = A * X * W + b
    
    Time Complexity: O(|V|·F·F' + |E|·F')
    where |V| = nodes, F = input features, F' = output features, |E| = edges
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Define parameters
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Xavier Uniform."""
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Support = X * W
        support = torch.mm(x, self.weight)
        # Output = Adj * Support + b
        output = torch.mm(adj, support) + self.bias
        return output


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT) with multi-head attention.
    
    Formula: h_i' = ||_{k=1}^K σ(Σ_{j∈N_i} α_{ij}^k W^k h_j)
    where α_{ij} = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))
    
    Time Complexity: O(K·(|V|·F·F' + |E|·F'))
    where K = num_heads, |V| = nodes, F = input features, F' = output features per head, |E| = edges
    
    Compared to GCN:
    - Additional cost: K× for multi-head, attention computation per edge
    - Memory: K× feature storage, attention coefficients per edge
    - Typically 2-8× slower than GCN depending on graph density and K
    """
    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, 
                 dropout: float = 0.1, alpha: float = 0.2, concat: bool = True):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension per head
            num_heads: Number of attention heads (K, default: 4 for efficiency)
            dropout: Dropout rate for attention coefficients
            alpha: Negative slope for LeakyReLU
            concat: If True, concatenate heads; if False, average heads
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        
        # For each head: W^k (feature transformation)
        if concat:
            self.out_dim = num_heads * out_features
        else:
            self.out_dim = out_features
        
        # Weight matrix for each head: W^k (num_heads, in_features, out_features)
        self.W = nn.Parameter(torch.FloatTensor(num_heads, in_features, out_features))
        
        # Attention mechanism: a^k (learnable attention vector for each head)
        self.a = nn.Parameter(torch.FloatTensor(num_heads, 2 * out_features, 1))
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(alpha)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (|V|, in_features)
            adj: Adjacency matrix (|V|, |V|) - binary or weighted
        
        Returns:
            Output features (|V|, out_dim)
        """
        N = x.size(0)  # Number of nodes
        
        # Transform features for each head: Wh for each head
        # x: (N, in_features), W: (num_heads, in_features, out_features)
        # h: (N, num_heads, out_features)
        h = torch.stack([torch.mm(x, self.W[k]) for k in range(self.num_heads)], dim=1)
        
        # Compute attention scores for all pairs and heads
        # For each head k and pair (i, j): a^T [h_i^k || h_j^k]
        # h: (N, num_heads, out_features)
        # We'll compute efficiently using broadcasting
        
        # Prepare for pairwise computation
        # h_i: (N, num_heads, 1, out_features)
        # h_j: (1, num_heads, N, out_features)
        h_i = h.unsqueeze(2)  # (N, num_heads, 1, out_features)
        h_j = h.unsqueeze(0)  # (1, num_heads, N, out_features)
        
        # Broadcast to get all pairs: (N, num_heads, N, out_features)
        h_i_expanded = h_i.expand(N, self.num_heads, N, self.out_features)
        h_j_expanded = h_j.expand(N, self.num_heads, N, self.out_features)
        
        # Concatenate: [h_i || h_j] -> (N, num_heads, N, 2*out_features)
        h_concat = torch.cat([h_i_expanded, h_j_expanded], dim=-1)
        
        # Compute attention scores: a^T [h_i || h_j] for all heads
        # h_concat: (N, num_heads, N, 2*out_features)
        # a: (num_heads, 2*out_features, 1)
        attention_scores = torch.zeros(N, self.num_heads, N, device=x.device, dtype=x.dtype)
        for k in range(self.num_heads):
            # For head k: compute a[k]^T @ h_concat[:, k, :, :]
            # h_concat[:, k, :, :]: (N, N, 2*out_features)
            # a[k]: (2*out_features, 1)
            scores_k = torch.matmul(h_concat[:, k, :, :], self.a[k])  # (N, N, 1)
            attention_scores[:, k, :] = scores_k.squeeze(-1)  # (N, N)
        
        attention_scores = self.leaky_relu(attention_scores)
        
        # Mask out non-adjacent nodes (set to -inf before softmax)
        # adj: (N, N) -> (1, 1, N, N) for broadcasting
        adj_mask = adj.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        attention_scores = attention_scores.unsqueeze(2)  # (N, num_heads, 1, N)
        attention_scores = attention_scores.masked_fill(adj_mask == 0, float('-inf'))
        attention_scores = attention_scores.squeeze(2)  # (N, num_heads, N)
        
        # Apply softmax to get attention coefficients
        attention_coeffs = F.softmax(attention_scores, dim=-1)  # (N, num_heads, N)
        attention_coeffs = self.dropout(attention_coeffs)
        
        # Aggregate neighbors: Σ_j α_{ij}^k * h_j^k for each head k
        # h: (N, num_heads, out_features)
        # attention_coeffs: (N, num_heads, N)
        h_aggregated = torch.zeros(N, self.num_heads, self.out_features, device=x.device, dtype=x.dtype)
        for k in range(self.num_heads):
            # attention_coeffs[:, k, :]: (N, N) - attention from each node to each node for head k
            # h[:, k, :]: (N, out_features) - features for head k
            h_aggregated[:, k, :] = torch.matmul(attention_coeffs[:, k, :], h[:, k, :])  # (N, out_features)
        
        # Concatenate or average heads
        if self.concat:
            # Concatenate: (N, num_heads * out_features)
            output = h_aggregated.contiguous().view(N, self.out_dim)
        else:
            # Average: (N, out_features)
            output = h_aggregated.mean(dim=1)
        
        return output


class VGAE(nn.Module):
    """
    Variational Graph Autoencoder (VGAE) for GRMP attack.
    
    This model learns the relational structure among benign updates (as a graph)
    to generate adversarial gradients that mimic legitimate patterns.
    
    Supports both GCN and GAT architectures:
    - GCN: Faster, simpler, fixed neighbor aggregation
    - GAT: More expressive, learns neighbor importance, but 2-8× slower
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 32, 
                 dropout: float = 0.2, use_gat: bool = False, gat_num_heads: int = 4):
        """
        Args:
            input_dim: Input feature dimension (number of clients I)
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
            dropout: Dropout rate
            use_gat: If True, use GAT layers; if False, use GCN layers (default: False)
            gat_num_heads: Number of attention heads for GAT (only used if use_gat=True)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.use_gat = use_gat
        
        # --- Encoder Layers ---
        if use_gat:
            # GAT architecture: more expressive but slower
            # Note: GAT output dimension = num_heads * out_features (if concat=True)
            # For first layer: input_dim -> hidden_dim (total, will be split across heads)
            # We need to ensure hidden_dim is divisible by num_heads
            gat_hidden_per_head = hidden_dim // gat_num_heads
            if gat_hidden_per_head * gat_num_heads != hidden_dim:
                # Round down to ensure divisibility
                gat_hidden_per_head = hidden_dim // gat_num_heads
                # Adjust hidden_dim to be divisible (may be slightly smaller)
                actual_hidden_dim = gat_hidden_per_head * gat_num_heads
                if actual_hidden_dim != hidden_dim:
                    print(f"  Warning: GAT hidden_dim adjusted from {hidden_dim} to {actual_hidden_dim} "
                          f"to be divisible by {gat_num_heads} heads")
                    hidden_dim = actual_hidden_dim
            
            self.gc1 = GraphAttentionLayer(input_dim, gat_hidden_per_head, 
                                           num_heads=gat_num_heads, dropout=dropout, concat=True)
            # Second layer: hidden_dim (from gc1 output) -> latent_dim (total, will be split across heads)
            gat_latent_per_head = latent_dim // gat_num_heads
            if gat_latent_per_head * gat_num_heads != latent_dim:
                # Round down to ensure divisibility
                gat_latent_per_head = latent_dim // gat_num_heads
                # Adjust latent_dim to be divisible (may be slightly smaller)
                actual_latent_dim = gat_latent_per_head * gat_num_heads
                if actual_latent_dim != latent_dim:
                    print(f"  Warning: GAT latent_dim adjusted from {latent_dim} to {actual_latent_dim} "
                          f"to be divisible by {gat_num_heads} heads")
                    latent_dim = actual_latent_dim
            
            # gc1 outputs: (N, hidden_dim) where hidden_dim = num_heads * gat_hidden_per_head
            # gc2 inputs: (N, hidden_dim)
            self.gc2_mu = GraphAttentionLayer(hidden_dim, gat_latent_per_head,
                                             num_heads=gat_num_heads, dropout=dropout, concat=True)
            self.gc2_logvar = GraphAttentionLayer(hidden_dim, gat_latent_per_head,
                                                  num_heads=gat_num_heads, dropout=dropout, concat=True)
        else:
            # GCN architecture: faster and simpler (default)
            self.gc1 = GraphConvolutionLayer(input_dim, hidden_dim)
            self.gc2_mu = GraphConvolutionLayer(hidden_dim, latent_dim)
            self.gc2_logvar = GraphConvolutionLayer(hidden_dim, latent_dim)

        self.dropout = nn.Dropout(dropout)

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes input features and adjacency matrix into latent distribution parameters."""
        
        if self.use_gat:
            # GAT: No need for symmetric normalization, uses attention mechanism directly
            # But we still add self-loops for attention computation
            adj_with_loop = adj + torch.eye(adj.size(0), device=adj.device)
            
            # Layer 1: GAT + ELU (GAT typically uses ELU) + Dropout
            hidden = self.gc1(x, adj_with_loop)
            hidden = F.elu(hidden)  # GAT typically uses ELU instead of ReLU
            hidden = self.dropout(hidden)
            
            # Layer 2: Output Mean and Log Variance
            mu = self.gc2_mu(hidden, adj_with_loop)
            logvar = self.gc2_logvar(hidden, adj_with_loop)
        else:
            # GCN: Normalize adjacency matrix (symmetric normalization)
            adj_norm = self._normalize_adj(adj)
            
            # Layer 1: GCN + ReLU + Dropout
            hidden = self.gc1(x, adj_norm)
            hidden = F.relu(hidden)
            hidden = self.dropout(hidden)
            
            # Layer 2: Output Mean and Log Variance
            mu = self.gc2_mu(hidden, adj_norm)
            logvar = self.gc2_logvar(hidden, adj_norm)
        
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon
        Allows backpropagation through stochastic nodes.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Inner product decoder: reconstructs the adjacency matrix.
        A_pred = sigmoid(Z * Z^T)
        """
        adj_reconstructed = torch.sigmoid(torch.mm(z, z.t()))
        return adj_reconstructed

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass."""
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        adj_reconstructed = self.decode(z)
        return adj_reconstructed, mu, logvar

    def _normalize_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Symmetrically normalize adjacency matrix: D^(-1/2) * (A + I) * D^(-1/2).
        Implementation handles self-loops by adding Identity matrix.
        """
        # Add self-loops
        adj_with_loop = adj + torch.eye(adj.size(0), device=adj.device)
        
        # Calculate degree matrix D
        d_vec = adj_with_loop.sum(1)
        
        # Calculate D^(-1/2)
        d_inv_sqrt = torch.pow(d_vec, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        
        # A_norm = D^(-1/2) * A * D^(-1/2)
        return torch.mm(torch.mm(d_mat_inv_sqrt, adj_with_loop), d_mat_inv_sqrt)

    def loss_function(self, adj_reconstructed: torch.Tensor, adj_orig: torch.Tensor, 
                     mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Calculates VGAE loss: Reconstruction Loss (Weighted BCE) + KL Divergence.
        """
        n_nodes = adj_orig.size(0)
        
        # Calculate weights for imbalanced classes (edges vs non-edges)
        # Typically graphs are sparse, so we weight positive edges more
        num_edges = adj_orig.sum().item()  # Convert to Python scalar
        num_non_edges = n_nodes * n_nodes - num_edges
        
        # Avoid division by zero
        if num_edges == 0:
            pos_weight = torch.tensor(1.0, device=adj_orig.device)
        else:
            pos_weight = torch.tensor(num_non_edges / num_edges, device=adj_orig.device)
            
        norm = (n_nodes * n_nodes) / (num_non_edges * 2) if num_non_edges > 0 else 1.0

        # 1. Reconstruction Loss (Weighted Binary Cross Entropy)
        bce_loss = norm * F.binary_cross_entropy_with_logits(
            adj_reconstructed, 
            adj_orig, 
            pos_weight=pos_weight
        )

        # 2. KL Divergence (Regularization term)
        # KL(N(mu, sigma) || N(0, 1))
        kl_loss = -0.5 / n_nodes * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )

        # Combine losses (KL term is often weighted less to prevent posterior collapse)
        return bce_loss + 0.1 * kl_loss