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
from typing import Dict

# Import our custom modules
from models import NewsClassifierModel
from data_loader import DataManager, NewsDataset
from client import BenignClient, AttackerClient
from server import Server
from visualization import ExperimentVisualizer

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
        test_sample_rate=config.get('test_sample_rate', 1.0),  # 1.0 = test all Business samples
        test_seed=config.get('seed', 42),  # Use same seed for reproducibility
        dataset_size_limit=config.get('dataset_size_limit', None),  # None = full dataset (per paper)
        batch_size=config['batch_size'],  # Batch size for training data loaders
        test_batch_size=config.get('test_batch_size')  # Batch size for test data loaders
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
        server_lr=config.get('server_lr', 1.0),
        tolerance_factor=config.get('tolerance_factor', 2.0),
        d_T=config.get('d_T', 0.5),
        gamma=config.get('gamma', 10.0),
        similarity_alpha=config.get('similarity_alpha', 0.7)
    )

    # 6. Create Clients
    print("\nCreating federated learning clients...")
    num_attackers = config.get('num_attackers', 0)  # Allow 0 attackers for baseline experiment
    
    for client_id in range(config['num_clients']):
        # Determine if benign or attacker
        # Logic: Last 'num_attackers' clients are attackers
        # If num_attackers=0, all clients are benign (baseline experiment)
        if client_id < (config['num_clients'] - num_attackers):
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
                alpha=config.get('alpha', 0.01),
                data_indices=client_indices[client_id]
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
                attack_start_round=config.get('attack_start_round'),
                lambda_attack=config.get('lambda_attack', 2.0),
                lambda_camouflage=config.get('lambda_camouflage', 0.3),
                benign_select_ratio=config.get('benign_select_ratio', 1.0),
                dual_lr=config.get('dual_lr', 0.01),
                proxy_step=config.get('proxy_step', 0.1),
                claimed_data_size=config.get('attacker_claimed_data_size', 1.0)
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
        'acc_diff': [],
        'rejection_rate': [],
        'agg_update_norm': []
    }

    try:
        for round_num in range(config['num_rounds']):
            round_log = server.run_round(round_num)

            # Track metrics
            progressive_metrics['rounds'].append(round_num + 1)
            progressive_metrics['clean_acc'].append(round_log['clean_accuracy'])
            progressive_metrics['acc_diff'].append(round_log.get('acc_diff', 0.0))
            progressive_metrics['rejection_rate'].append(round_log['defense'].get('rejection_rate', 0.0))
            progressive_metrics['agg_update_norm'].append(round_log['defense'].get('aggregated_update_norm', 0.0))
            
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
        'progressive_metrics': progressive_metrics,
        'local_accuracies': server.history['local_accuracies']  # Include local accuracies
    }

    results_path = results_dir / f"{config['experiment_name']}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    
    # Generate visualizations
    if config.get('generate_plots', True):
        print("\n" + "=" * 60)
        print("Generating Visualization Plots")
        print("=" * 60)
        
        visualizer = ExperimentVisualizer(results_dir=results_dir)
        
        # Extract attacker IDs
        attacker_ids = [client.client_id for client in server.clients 
                       if getattr(client, 'is_attacker', False)]
        
        # Generate all figures
        # Check if baseline experiment results exist for Figure 5
        baseline_path = results_dir / f"{config['experiment_name']}_no_attack_results.json"
        baseline_local_accs = None
        if baseline_path.exists():
            try:
                with open(baseline_path, 'r') as f:
                    baseline_data = json.load(f)
                    baseline_local_accs = baseline_data.get('local_accuracies', None)
                    if baseline_local_accs:
                        print("  ✅ Found baseline experiment data for Figure 5")
            except Exception as e:
                print(f"  ⚠️  Could not load baseline data: {e}")
        
        visualizer.generate_all_figures(
            server_log_data=server.log_data,
            local_accuracies=server.history['local_accuracies'],
            attacker_ids=attacker_ids,
            experiment_name=config['experiment_name'],
            baseline_local_accuracies=baseline_local_accs,
            num_rounds=config.get('num_rounds'),  # Pass num_rounds to ensure alignment
            attack_start_round=config.get('attack_start_round')  # Pass attack_start_round for Figure 5
        )
    
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

def run_no_attack_experiment(config_base: Dict) -> Dict:
    """
    Run a baseline experiment WITHOUT any attackers (for Figure 5 comparison).
    
    Args:
        config_base: Base configuration dictionary
        
    Returns:
        Dictionary with experiment results
    """
    # Create a copy of config with no attackers
    config = config_base.copy()
    config['experiment_name'] = config_base.get('experiment_name', 'baseline') + '_no_attack'
    config['num_attackers'] = 0  # No attackers
    print("\n" + "=" * 60)
    print("Running BASELINE Experiment (NO ATTACK)")
    print("=" * 60)
    print("This experiment will be used for Figure 5 comparison.")
    print("=" * 60)
    
    return run_experiment(config)


def main():
    config = {
        # ========== Experiment Configuration ==========
        'experiment_name': 'vgae_grmp_attack',  # Name for result files and logs
        'seed': 42,  # Random seed for reproducibility (int)
        
        # ========== Federated Learning Setup ==========
        'num_clients': 6,  # Total number of federated learning clients (int)
        'num_attackers': 2,  # Number of attacker clients (int, must be < num_clients)
        'num_rounds': 30,  # Total number of federated learning rounds (int)
        
        # ========== Training Hyperparameters ==========
        'client_lr': 2e-5,  # Learning rate for local client training (float)
        'server_lr': 0.8,  # Server learning rate for model aggregation (float, typically 0.5-1.0)
        'batch_size': 256,  # Batch size for local training (int)
        'test_batch_size': 256,  # Batch size for test/validation data loaders (int)
        
        # 'local_epochs': 5,  # Number of local training epochs per round (int, per paper Section IV)
        'local_epochs': 5,  # Number of local training epochs per round (int, per paper Section IV)
        'alpha': 0.01,  # Proximal regularization coefficient α ∈ [0,1] from paper formula (1) (float)
        
        # ========== Data Distribution ==========
        'dirichlet_alpha': 0.5,  # Dirichlet distribution parameter for non-IID data partitioning (float, lower = more heterogeneous)
        'test_sample_rate': 1.0,  # Rate of Business samples to test for ASR evaluation (float, 1.0 = all samples)
        # 'dataset_size_limit': None,  # Limit dataset size for faster experimentation (None = use FULL AG News dataset per paper, int = limit training samples)
        'dataset_size_limit': 20000,  # Limit dataset size for faster experimentation (None = use FULL AG News dataset per paper, int = limit training samples)

        # ========== Attack Configuration ==========
        'attack_start_round': 0,  # Round when attack phase starts (int, now all rounds use complete poisoning)
        
        # ========== Formula 4 Constraint Parameters ==========
        'd_T': 1.0,  # Distance threshold for constraint (4b): d(w'_j(t), w_g(t)) ≤ d_T (float)
        'gamma': 10.0,  # Upper bound for constraint (4c): Σ β'_{i,j}(t) d(w_i(t), w̄_i(t)) ≤ Γ (float)
        
        # ========== VGAE Training Parameters ==========
        # Reference paper: input_dim=5, hidden1_dim=32, hidden2_dim=16, num_epoch=10, lr=0.01
        'dim_reduction_size': 10000,  # Reduced dimensionality (reference uses ~100 features)
        'vgae_epochs': 30,  # Number of epochs for VGAE training (reference: 10)
        'vgae_lr': 0.01,  # Learning rate for VGAE optimizer (reference: 0.01)
        'vgae_lambda': 2.0,  # Weight for preservation loss - HIGH to preserve poisoned update (float)
        
        # ========== Camouflage Optimization Parameters ==========
        'camouflage_steps': 50,  # Number of optimization steps for malicious update camouflage (int)
        'camouflage_lr': 0.1,  # Learning rate for camouflage optimization (float)
        'lambda_proximity': 0.5,  # Weight for constraint (4b) proximity loss in camouflage (float)
        'lambda_aggregation': 0.5,  # Weight for constraint (4c) aggregation loss in camouflage (float)
        'lambda_attack': 0.5,  # Weight for attack objective (direction-based) - moderate weight (float)
        'lambda_camouflage': 0.1,  # Weight for camouflage loss (VGAE latent space) - low to preserve attack (float)
        'benign_select_ratio': 1.0,  # β subset ratio for attacker graph (data-agnostic attack)
        'dual_lr': 0.01,  # Step size for dual variable updates (λ, ρ) in Lagrangian
        'proxy_step': 0.1,  # Step size for gradient-free ascent toward global-loss proxy
        'attacker_claimed_data_size': 1.0,  # D'(t) claimed by attacker for weighted aggregation
        
        # ========== Graph Construction Parameters ==========
        'graph_threshold': 0.5,  # Threshold for graph adjacency matrix binarization in VGAE (float, 0.0-1.0)
        
        # ========== Defense Mechanism Parameters ==========
        'defense_threshold': 0.05,  # Base threshold for defense mechanism (float, lower = more strict)
        'tolerance_factor': 3.0,  # Tolerance factor for defense mechanism (float, higher = more lenient)
        'similarity_alpha': 0.5,  # Weight for pairwise similarities in mixed similarity computation (float, 0.0-1.0)
        
        # ========== Visualization ==========
        'generate_plots': True,  # Whether to generate visualization plots (bool)
        'run_both_experiments': False,  # Set to True to run baseline + attack (for Figure 5)
        'run_attack_only': False,  # Set to True to only run attack experiment
    }



    # Option 1: Run attack experiment only
    if config.get('run_attack_only', False):
        print("Running GRMP Attack with VGAE...")
        results, metrics = run_experiment(config)
        analyze_results(metrics)
    
    # Option 2: Run both baseline (no attack) and attack experiments
    elif config.get('run_both_experiments', False):
        print("\n" + "=" * 60)
        print("Running COMPLETE Experiment Suite")
        print("=" * 60)
        print("This will run:")
        print("  1. Baseline experiment (no attack) - for Figure 5")
        print("  2. Attack experiment (with GRMP) - for Figures 3, 4, 6")
        print("=" * 60)
        
        # Run baseline (no attack) experiment
        baseline_results, baseline_metrics = run_no_attack_experiment(config)
        
        # Run attack experiment
        print("\n" + "=" * 60)
        print("Now running ATTACK experiment...")
        print("=" * 60)
        attack_results, attack_metrics = run_experiment(config)
        
        # Generate combined visualizations
        if config.get('generate_plots', True):
            from visualization import ExperimentVisualizer
            from pathlib import Path
            
            results_dir = Path("results")
            visualizer = ExperimentVisualizer(results_dir=results_dir)
            
            # Extract attacker IDs from attack experiment
            # (We need to reload the server or pass it differently)
            # For now, we'll use the config to determine attackers
            num_clients = config['num_clients']
            num_attackers = config['num_attackers']
            attacker_ids = list(range(num_clients - num_attackers, num_clients))
            
            print("\n" + "=" * 60)
            print("Generating Combined Visualizations")
            print("=" * 60)
            
            # Load baseline results for Figure 5
            baseline_path = results_dir / f"{config['experiment_name']}_no_attack_results.json"
            if baseline_path.exists():
                with open(baseline_path, 'r') as f:
                    baseline_data = json.load(f)
                    baseline_local_accs = baseline_data.get('local_accuracies', {})
                    
                    # Generate Figure 5 from baseline
                    print("\n📊 Generating Figure 5: Local Accuracy (No Attack)...")
                    baseline_rounds = list(range(1, len(baseline_results) + 1))
                    visualizer.plot_figure5_local_accuracy_no_attack(
                        baseline_local_accs, baseline_rounds,
                        save_path=results_dir / f"{config['experiment_name']}_figure5.png"
                    )
            
            # Generate attack experiment figures (3, 4, 6)
            # Note: We need server object to get local_accuracies, so we'll regenerate
            # For now, load from saved results
            attack_path = results_dir / f"{config['experiment_name']}_results.json"
            if attack_path.exists():
                with open(attack_path, 'r') as f:
                    attack_data = json.load(f)
                    attack_local_accs = attack_data.get('local_accuracies', {})
                    
                    # Generate Figures 3, 4, 6
                    visualizer.generate_all_figures(
                        server_log_data=attack_results,
                        local_accuracies=attack_local_accs,
                        attacker_ids=attacker_ids,
                        experiment_name=config['experiment_name']
                    )
        
        analyze_results(attack_metrics)
    
    # Option 3: Default - run attack experiment only
    else:
        print("Running GRMP Attack with VGAE...")
        results, metrics = run_experiment(config)
        analyze_results(metrics)
        
        # Note: Figure 5 requires a separate baseline experiment
        # Set 'run_both_experiments': True to generate all figures
        if config.get('generate_plots', True):
            print("\n" + "=" * 60)
            print("NOTE: Figure 5 (No Attack) requires a baseline experiment.")
            print("To generate Figure 5, set 'run_both_experiments': True in config")
            print("or run a separate experiment with 'num_attackers': 0")
            print("=" * 60)

if __name__ == "__main__":
    main()