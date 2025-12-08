# main.py
# This script sets up and runs a federated learning experiment with a progressive GRMP attack.

import torch
import torch.nn as nn
import numpy as np
import json
import gc
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

# Import our custom modules
from models import NewsClassifierModel
from data_loader import DataManager, NewsDataset
from client import BenignClient, AttackerClient
from server import Server

warnings.filterwarnings('ignore')

# Initialize experiment components
def setup_experiment(config):
    # Set random seeds for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 50)
    print(f"Setting up Experiment: {config['experiment_name']}")
    print("=" * 50)

    # 1. Initialize Data Manager
    # Note: dataset_size_limit=None uses FULL AG News dataset (~120K train, ~7.6K test) per paper
    # Set dataset_size_limit to a positive int (e.g., 30000) only for faster experimentation
    data_manager = DataManager(
        num_clients=config['num_clients'],
        num_attackers=config['num_attackers'], 
        poison_rate=config['poison_rate'],
        test_sample_rate=config.get('test_sample_rate', 1.0),  # 1.0 = test all Business samples
        test_seed=config.get('seed', 42),  # Use same seed for reproducibility
        dataset_size_limit=config.get('dataset_size_limit', None)  # None = full dataset (per paper)
    )

    # 2. Partition data among clients (Non-IID distribution per paper)
    # Per paper: "heterogeneous IoA system" with heterogeneous data distributions
    print("\nPartitioning data (Non-IID distribution)...")
    indices = np.arange(len(data_manager.train_texts))
    labels = np.array(data_manager.train_labels)
    
    # Fixed shuffle for consistent partitioning across runs
    rng = np.random.default_rng(config['seed'])
    
    # Non-IID distribution: Use Dirichlet distribution to create heterogeneous data
    # Each client gets data with different label distributions
    dirichlet_alpha = config.get('dirichlet_alpha', 0.5)  # Lower alpha = more heterogeneous
    num_labels = 4
    client_indices = {i: [] for i in range(config['num_clients'])}
    
    # Partition data by label first
    label_indices = {label: [] for label in range(num_labels)}
    for idx, label in enumerate(labels):
        label_indices[label].append(idx)
    
    # Assign samples to clients using Dirichlet distribution for non-IID
    for label in range(num_labels):
        label_list = np.array(label_indices[label])
        rng.shuffle(label_list)
        
        # Generate proportions for each client using Dirichlet distribution
        # Lower dirichlet_alpha creates more heterogeneous (non-IID) distribution
        proportions = rng.dirichlet([dirichlet_alpha] * config['num_clients'])
        proportions = np.cumsum(proportions)
        proportions[-1] = 1.0  # Ensure last is exactly 1.0
        
        # Assign samples based on proportions
        start_idx = 0
        for client_id in range(config['num_clients']):
            end_idx = int(len(label_list) * proportions[client_id])
            client_indices[client_id].extend(label_list[start_idx:end_idx].tolist())
            start_idx = end_idx
    
    # Shuffle within each client to mix labels (but distribution remains non-IID)
    for client_id in range(config['num_clients']):
        client_list = np.array(client_indices[client_id])
        rng.shuffle(client_list)
        client_indices[client_id] = client_list.tolist()
    
    # Print distribution statistics
    print(f"  Non-IID distribution (Dirichlet alpha={dirichlet_alpha})")
    for client_id in range(min(3, config['num_clients'])):  # Show first 3 clients
        client_labels = [labels[idx] for idx in client_indices[client_id]]
        label_counts = {l: client_labels.count(l) for l in range(num_labels)}
        total = len(client_indices[client_id])
        dist_str = ", ".join([f"Label {l}: {label_counts[l]/total:.1%}" for l in range(num_labels)])
        print(f"    Client {client_id}: {total} samples ({dist_str})")

    # 3. Get global test loaders
    test_loader = data_manager.get_test_loader()
    attack_test_loader = data_manager.get_attack_test_loader()

    # 4. Initialize Global Model
    print("Initializing global model (DistilBERT)...")
    global_model = NewsClassifierModel()

    # 5. Initialize Server
    server = Server(
        global_model=global_model,
        test_loader=test_loader,
        attack_test_loader=attack_test_loader,
        defense_threshold=config['defense_threshold'],
        total_rounds=config['num_rounds'],
        server_lr=config.get('server_lr', 1.0)
    )

    # 6. Create Clients
    print("\nCreating federated learning clients...")
    for client_id in range(config['num_clients']):
        # Determine if benign or attacker
        # Logic: Last 'num_attackers' clients are attackers
        if client_id < (config['num_clients'] - config['num_attackers']):
            # --- Benign Client ---
            client_texts = [data_manager.train_texts[i] for i in client_indices[client_id]]
            client_labels = [data_manager.train_labels[i] for i in client_indices[client_id]]
            
            # Create static dataloader for benign client
            dataset = NewsDataset(client_texts, client_labels, data_manager.tokenizer)
            client_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

            client = BenignClient(
                client_id=client_id,
                model=global_model,
                data_loader=client_loader,
                lr=config['client_lr'],
                local_epochs=config['local_epochs'],
                alpha=config.get('alpha', 0.01)
            )
        else:
            # --- Attacker Client ---
            print(f"  Client {client_id}: ATTACKER (VGAE Enabled)")
            client = AttackerClient(
                client_id=client_id,
                model=global_model,
                data_manager=data_manager,
                data_indices=client_indices[client_id],
                lr=config['client_lr'],
                local_epochs=config['local_epochs'],
                alpha=config.get('alpha', 0.01),
                dim_reduction_size=config.get('dim_reduction_size', 10000),
                vgae_lambda=config.get('vgae_lambda', 0.5),
                vgae_epochs=config.get('vgae_epochs', 20),
                vgae_lr=config.get('vgae_lr', 0.01),
                camouflage_steps=config.get('camouflage_steps', 30),
                camouflage_lr=config.get('camouflage_lr', 0.1),
                lambda_proximity=config.get('lambda_proximity', 1.0),
                lambda_aggregation=config.get('lambda_aggregation', 0.5),
                graph_threshold=config.get('graph_threshold', 0.5),
                attack_start_round=config.get('attack_start_round', 10)
            )

        server.register_client(client)
    
    return server, results_dir

# Run the experiment
def run_experiment(config):
    server, results_dir = setup_experiment(config)

    # Initial evaluation
    print("\nEvaluating initial model...")
    initial_clean, initial_asr = server.evaluate()
    print(f"Initial Performance - Clean: {initial_clean:.4f}, ASR: {initial_asr:.4f}")

    print("\n" + "=" * 50)
    print("Starting Federated Learning Rounds")
    print("=" * 50)

    progressive_metrics = {
        'rounds': [],
        'clean_acc': [],
        'attack_asr': [],
        'defense_threshold': []
    }

    try:
        for round_num in range(config['num_rounds']):
            round_log = server.run_round(round_num)

            # Track metrics
            progressive_metrics['rounds'].append(round_num + 1)
            progressive_metrics['clean_acc'].append(round_log['clean_accuracy'])
            progressive_metrics['attack_asr'].append(round_log['attack_success_rate'])
            progressive_metrics['defense_threshold'].append(round_log['defense']['threshold'])
            
            # Memory cleanup after each round
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\nExperiment failed with error: {e}")
        import traceback
        traceback.print_exc()

    # Save results
    results_data = {
        'config': config,
        'results': server.log_data,
        'progressive_metrics': progressive_metrics
    }

    results_path = results_dir / f"{config['experiment_name']}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    return server.log_data, progressive_metrics

# Simple analysis
def analyze_results(metrics):
    print("\n" + "=" * 50)
    print("Experiment Summary")
    print("=" * 50)
    
    rounds = metrics['rounds']
    if not rounds:
        print("No rounds completed.")
        return

    clean = metrics['clean_acc']
    asr = metrics['attack_asr']

    print(f"Total Rounds: {len(rounds)}")
    print(f"Final Clean Accuracy: {clean[-1]:.4f}")
    print(f"Final Attack Success Rate: {asr[-1]:.4f}")
    print(f"Peak ASR: {max(asr):.4f}")

def main():
    config = {
        # ========== Experiment Configuration ==========
        'experiment_name': 'vgae_grmp_attack',  # Name for result files and logs
        'seed': 42,  # Random seed for reproducibility (int)
        
        # ========== Federated Learning Setup ==========
        'num_clients': 6,  # Total number of federated learning clients (int)
        'num_attackers': 2,  # Number of attacker clients (int, must be < num_clients)
        'num_rounds': 20,  # Total number of federated learning rounds (int)
        
        # ========== Training Hyperparameters ==========
        'client_lr': 2e-5,  # Learning rate for local client training (float)
        'server_lr': 0.8,  # Server learning rate for model aggregation (float, typically 0.5-1.0)
        'batch_size': 16,  # Batch size for local training (int)
        'local_epochs': 5,  # Number of local training epochs per round (int, per paper Section IV)
        'alpha': 0.01,  # Proximal regularization coefficient α ∈ [0,1] from paper formula (1) (float)
        
        # ========== Data Distribution ==========
        'dirichlet_alpha': 0.5,  # Dirichlet distribution parameter for non-IID data partitioning (float, lower = more heterogeneous)
        'test_sample_rate': 1.0,  # Rate of Business samples to test for ASR evaluation (float, 1.0 = all samples)
        'dataset_size_limit': None,  # Limit dataset size for faster experimentation (None = use FULL AG News dataset per paper, int = limit training samples)
                                      # WARNING: Using limit may affect reproducibility. For paper reproduction, use None.
        
        # ========== Attack Configuration ==========
        'poison_rate': 1.0,  # Base poisoning rate for attack phase (float, 0.0-1.0)
        'attack_start_round': 10,  # Round when attack phase starts (int, learning phase before this round)
        
        # ========== Formula 4 Constraint Parameters ==========
        'd_T': 0.5,  # Distance threshold for constraint (4b): d(w'_j(t), w_g(t)) ≤ d_T (float)
        'gamma': 10.0,  # Upper bound for constraint (4c): Σ β'_{i,j}(t) d(w_i(t), w̄_i(t)) ≤ Γ (float)
        
        # ========== VGAE Training Parameters ==========
        'dim_reduction_size': 10000,  # Dimensionality for feature reduction in VGAE (int, adjust based on GPU memory)
        'vgae_epochs': 20,  # Number of epochs for VGAE training per camouflage step (int)
        'vgae_lr': 0.01,  # Learning rate for VGAE optimizer (float)
        'vgae_lambda': 0.5,  # Weight for preservation loss in camouflage optimization (float, balances attack efficacy vs camouflage)
        
        # ========== Camouflage Optimization Parameters ==========
        'camouflage_steps': 30,  # Number of optimization steps for malicious update camouflage (int)
        'camouflage_lr': 0.1,  # Learning rate for camouflage optimization (float)
        'lambda_proximity': 1.0,  # Weight for constraint (4b) proximity loss in camouflage (float)
        'lambda_aggregation': 0.5,  # Weight for constraint (4c) aggregation loss in camouflage (float)
        
        # ========== Graph Construction Parameters ==========
        'graph_threshold': 0.5,  # Threshold for graph adjacency matrix binarization in VGAE (float, 0.0-1.0)
        
        # ========== Defense Mechanism Parameters ==========
        'defense_threshold': 0.10,  # Base threshold for defense mechanism (float, lower = more strict)
        'similarity_alpha': 0.7,  # Weight for pairwise similarities in mixed similarity computation (float, 0.0-1.0)
    }

    print("Running GRMP Attack with VGAE...")
    results, metrics = run_experiment(config)
    analyze_results(metrics)

if __name__ == "__main__":
    main()