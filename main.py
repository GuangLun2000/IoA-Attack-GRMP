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
        test_seed=config['seed'],
        dataset_size_limit=config['dataset_size_limit'],
        batch_size=config['batch_size'],
        test_batch_size=config['test_batch_size'],
        model_name=config.get('model_name', 'distilbert-base-uncased'),
        max_length=config.get('max_length', 128)
    )

    # 2. Partition data among clients
    # Supports both IID and Non-IID distributions based on config
    data_distribution = config.get('data_distribution', 'non-iid').lower()
    indices = np.arange(len(data_manager.train_texts))
    labels = np.array(data_manager.train_labels)
    num_labels = 4
    num_clients = config['num_clients']
    num_attackers = config.get('num_attackers', 0)
    num_benign = num_clients - num_attackers
    
    # Fixed shuffle for consistent partitioning across runs
    rng = np.random.default_rng(config['seed'])
    
    client_indices = {i: [] for i in range(num_clients)}
    
    if data_distribution == 'iid':
        # ========== IID Distribution: Uniform Random Partition ==========
        # Each client gets approximately equal number of samples with similar label distribution
        print("\nPartitioning data (IID distribution)...")
        
        # Shuffle all indices
        all_indices = indices.copy()
        rng.shuffle(all_indices)
        
        # Calculate samples per client (approximately equal)
        total_samples = len(all_indices)
        base_samples = total_samples // num_clients
        remainder = total_samples % num_clients
        
        # Assign samples to each client
        start_idx = 0
        for client_id in range(num_clients):
            # First 'remainder' clients get one extra sample
            extra = 1 if client_id < remainder else 0
            end_idx = start_idx + base_samples + extra
            client_indices[client_id] = all_indices[start_idx:end_idx].tolist()
            start_idx = end_idx
        
        # Print distribution statistics
        print(f"  IID distribution (uniform random partition)")
        for client_id in range(num_clients):
            client_labels = [labels[idx] for idx in client_indices[client_id]]
            label_counts = {l: client_labels.count(l) for l in range(num_labels)}
            total = len(client_indices[client_id])
            if total > 0:
                dist_str = ", ".join([f"Label {l}: {label_counts[l]/total:.1%}" for l in range(num_labels)])
                client_type = "BENIGN" if client_id < num_benign else "ATTACKER"
                print(f"    Client {client_id} ({client_type}): {total} samples ({dist_str})")
            else:
                client_type = "BENIGN" if client_id < num_benign else "ATTACKER"
                print(f"    Client {client_id} ({client_type}): 0 samples WARNING: No data assigned!")
    
    else:
        # ========== Non-IID Distribution: Dirichlet-based Partition ==========
        # Per paper: "heterogeneous IoA system" with heterogeneous data distributions
        print("\nPartitioning data (Non-IID distribution)...")
        
        # Use Dirichlet distribution to create heterogeneous data
        # Each client gets data with different label distributions
        dirichlet_alpha = config['dirichlet_alpha']
        
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
            proportions = rng.dirichlet([dirichlet_alpha] * num_clients)
            proportions = np.cumsum(proportions)
            proportions[-1] = 1.0  # Ensure last is exactly 1.0
            
            # Assign samples based on proportions
            start_idx = 0
            for client_id in range(num_clients):
                end_idx = int(len(label_list) * proportions[client_id])
                client_indices[client_id].extend(label_list[start_idx:end_idx].tolist())
                start_idx = end_idx
        
        # Shuffle within each client to mix labels (but distribution remains non-IID)
        for client_id in range(num_clients):
            client_list = np.array(client_indices[client_id])
            rng.shuffle(client_list)
            client_indices[client_id] = client_list.tolist()
        
        # Print distribution statistics
        print(f"  Non-IID distribution (Dirichlet alpha={dirichlet_alpha})")
        for client_id in range(num_clients):
            client_labels = [labels[idx] for idx in client_indices[client_id]]
            label_counts = {l: client_labels.count(l) for l in range(num_labels)}
            total = len(client_indices[client_id])
            if total > 0:
                dist_str = ", ".join([f"Label {l}: {label_counts[l]/total:.1%}" for l in range(num_labels)])
                client_type = "BENIGN" if client_id < num_benign else "ATTACKER"
                print(f"    Client {client_id} ({client_type}): {total} samples ({dist_str})")
            else:
                client_type = "BENIGN" if client_id < num_benign else "ATTACKER"
                print(f"    Client {client_id} ({client_type}): 0 samples WARNING: No data assigned!")

    # 3. Get global test loader
    test_loader = data_manager.get_test_loader()

    # 4. Initialize Global Model
    use_lora = config.get('use_lora', False)
    model_name = config.get('model_name', 'distilbert-base-uncased')
    if use_lora:
        print(f"Initializing global model ({model_name}) with LoRA...")
        global_model = NewsClassifierModel(
            model_name=model_name,
            num_labels=config.get('num_labels', 4),
            use_lora=True,
            lora_r=config.get('lora_r', 16),
            lora_alpha=config.get('lora_alpha', 32),
            lora_dropout=config.get('lora_dropout', 0.1),
            lora_target_modules=config.get('lora_target_modules', None)
        )
    else:
        print(f"Initializing global model ({model_name}) [Full Fine-tuning]...")
        global_model = NewsClassifierModel(
            model_name=model_name,
            num_labels=config.get('num_labels', 4),
            use_lora=False
        )

    # 5. Initialize Server
    server = Server(
        global_model=global_model,
        test_loader=test_loader,
        total_rounds=config['num_rounds'],
        server_lr=config['server_lr'],
        dist_bound=config.get('dist_bound', config.get('d_T', 0.5))  # Renamed from d_T
    )
    # Set sim_center from config (cosine similarity center, optional)
    server.sim_center = config.get('sim_center', config.get('sim_T', None))

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
            dataset = NewsDataset(client_texts, client_labels, data_manager.tokenizer, 
                                  max_length=config.get('max_length', 128))
            client_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

            print(f"  Client {client_id}: BENIGN ({len(client_indices[client_id])} samples)")
            
            client = BenignClient(
                client_id=client_id,
                model=global_model,
                data_loader=client_loader,
                lr=config['client_lr'],
                local_epochs=config['local_epochs'],
                alpha=config['alpha'],
                data_indices=client_indices[client_id],
                grad_clip_norm=config['grad_clip_norm']
            )
        else:
            # --- Attacker Client ---
            attack_method = config.get('attack_method', 'GRMP')
            
            # Use actual assigned data size for claimed_data_size (for fair weighted aggregation)
            # Note: Attackers are data-agnostic (don't use data for training), but use assigned
            # data size for aggregation weight to maintain realistic attack scenario
            actual_data_size = len(client_indices[client_id])
            # Allow config override if attacker wants to claim different size (for attack experiments)
            config_claimed = config.get('attacker_claimed_data_size', None)
            if config_claimed is None:
                # Use actual assigned data size (recommended for realistic scenario)
                claimed_data_size = actual_data_size
            else:
                # Override with config value (for attack experiments)
                claimed_data_size = config_claimed
            
            # Create attacker based on attack_method
            if attack_method == 'ALIE':
                # ========== ALIE Attack Client ==========
                from attack_baseline import ALIEAttackerClient
                print(f"  Client {client_id}: ATTACKER (ALIE Attack)")
                print(f"    Claimed data size D'_j(t): {claimed_data_size} (matches assigned data)")
                
                # Get ALIE-specific parameters
                alie_z_max = config.get('alie_z_max', None)
                alie_attack_start_round = config.get('alie_attack_start_round', None)
                
                client = ALIEAttackerClient(
                    client_id=client_id,
                    model=global_model,
                    data_manager=data_manager,
                    data_indices=client_indices[client_id],
                    lr=config['client_lr'],
                    local_epochs=config['local_epochs'],
                    alpha=config['alpha'],
                    num_clients=config['num_clients'],
                    num_attackers=config['num_attackers'],
                    z_max=alie_z_max,
                    attack_start_round=alie_attack_start_round,
                    claimed_data_size=claimed_data_size,
                    grad_clip_norm=config.get('grad_clip_norm', 1.0)
                )
            else:
                # ========== GRMP Attack Client (default) ==========
                print(f"  Client {client_id}: ATTACKER (GRMP Attack - VGAE Enabled)")
                if config_claimed is None:
                    print(f"    Claimed data size D'_j(t): {claimed_data_size} (matches assigned data)")
                else:
                    print(f"    WARNING: Override: Claimed data size D'_j(t): {claimed_data_size} (actual: {actual_data_size})")
                
                client = AttackerClient(
                client_id=client_id,
                model=global_model,
                data_manager=data_manager,
                data_indices=client_indices[client_id],
                lr=config['client_lr'],
                local_epochs=config['local_epochs'],
                alpha=config['alpha'],
                dim_reduction_size=config['dim_reduction_size'],
                vgae_epochs=config['vgae_epochs'],
                vgae_lr=config['vgae_lr'],
                graph_threshold=config['graph_threshold'],
                proxy_step=config['proxy_step'],
                claimed_data_size=claimed_data_size,
                proxy_sample_size=config['proxy_sample_size'],
                proxy_max_batches_opt=config['proxy_max_batches_opt'],
                proxy_max_batches_eval=config['proxy_max_batches_eval'],
                vgae_hidden_dim=config['vgae_hidden_dim'],
                vgae_latent_dim=config['vgae_latent_dim'],
                vgae_dropout=config['vgae_dropout'],
                vgae_kl_weight=config['vgae_kl_weight'],
                proxy_steps=config['proxy_steps'],
                grad_clip_norm=config['grad_clip_norm'],
                early_stop_constraint_stability_steps=config.get('early_stop_constraint_stability_steps', 3)
            )
            
            # Set Lagrangian Dual parameters (if using)
            if config.get('use_lagrangian_dual', False):
                client.set_lagrangian_params(
                    use_lagrangian_dual=config['use_lagrangian_dual'],
                    lambda_dist_init=config.get('lambda_dist_init', config.get('lambda_init', 0.1)),
                    lambda_dist_lr=config.get('lambda_dist_lr', config.get('lambda_lr', 0.01)),
                    use_cosine_similarity_constraint=config.get('use_cosine_similarity_constraint', False),
                    lambda_sim_low_init=config.get('lambda_sim_low_init', config.get('lambda_sim_init', 0.1)),
                    lambda_sim_up_init=config.get('lambda_sim_up_init', config.get('lambda_sim_init', 0.1)),
                    lambda_sim_low_lr=config.get('lambda_sim_low_lr', config.get('lambda_sim_lr', 0.01)),
                    lambda_sim_up_lr=config.get('lambda_sim_up_lr', config.get('lambda_sim_lr', 0.01)),
                    # ========== Augmented Lagrangian (ALM) parameters ==========
                    use_augmented_lagrangian=config.get('use_augmented_lagrangian', False),
                    lambda_update_mode=config.get('lambda_update_mode', 'classic'),
                    rho_dist_init=config.get('rho_dist_init', 1.0),
                    rho_sim_low_init=config.get('rho_sim_low_init', 1.0),
                    rho_sim_up_init=config.get('rho_sim_up_init', 1.0),
                    rho_adaptive=config.get('rho_adaptive', True),
                    rho_theta=config.get('rho_theta', 0.5),
                    rho_increase_factor=config.get('rho_increase_factor', 2.0),
                    rho_min=config.get('rho_min', 1e-3),
                    rho_max=config.get('rho_max', 1e3),
                )
                print(f"    Lagrangian Dual enabled: λ_dist(1)={config.get('lambda_dist_init', config.get('lambda_init', 0.1))}")
            else:
                print(f"    Using hard constraint mechanism (Lagrangian Dual disabled)")

        server.register_client(client)
    
    return server, results_dir

# Run the experiment
def run_experiment(config):
    server, results_dir = setup_experiment(config)

    # Initial evaluation
    print("\nEvaluating initial model...")
    initial_clean = server.evaluate()
    print(f"Initial Performance - Clean Accuracy: {initial_clean:.4f}")

    print("\n" + "=" * 50)
    print("Starting Federated Learning Rounds")
    print("=" * 50)

    progressive_metrics = {
        'rounds': [],
        'clean_acc': [],
        'acc_diff': [],
        'agg_update_norm': []
    }

    try:
        for round_num in range(config['num_rounds']):
            round_log = server.run_round(round_num)

            # Track metrics
            progressive_metrics['rounds'].append(round_num + 1)
            progressive_metrics['clean_acc'].append(round_log['clean_accuracy'])
            progressive_metrics['acc_diff'].append(round_log.get('acc_diff', 0.0))
            progressive_metrics['agg_update_norm'].append(round_log['aggregation'].get('aggregated_update_norm', 0.0))
            
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
    
    # Print detailed statistics for data collection
    attacker_ids = [client.client_id for client in server.clients 
                   if getattr(client, 'is_attacker', False)]
    print_detailed_statistics(server.log_data, progressive_metrics, 
                            server.history['local_accuracies'], attacker_ids, 
                            config['experiment_name'], results_dir)
    
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
        visualizer.generate_all_figures(
            server_log_data=server.log_data,
            local_accuracies=server.history['local_accuracies'],
            attacker_ids=attacker_ids,
            experiment_name=config['experiment_name'],
            num_rounds=config['num_rounds'],
            attack_start_round=config['attack_start_round'],
            num_clients=config['num_clients'],
            num_attackers=config['num_attackers']
        )
    
    return server.log_data, progressive_metrics

# Detailed statistics printing for data collection
def print_detailed_statistics(server_log_data, progressive_metrics, local_accuracies, attacker_ids, 
                             experiment_name='experiment', results_dir=None):
    """
    Print detailed statistics for data collection and multi-run comparison.
    Outputs all key metrics in tabular format for easy copying to Excel/CSV.
    
    Args:
        server_log_data: List of round logs from server
        progressive_metrics: Dictionary with progressive metrics
        local_accuracies: Dictionary with local accuracies per client
        attacker_ids: List of attacker client IDs
        experiment_name: Name of the experiment (for file naming)
        results_dir: Path to results directory (default: Path("results"))
    """
    import csv
    from pathlib import Path
    
    if results_dir is None:
        results_dir = Path("results")
    else:
        results_dir = Path(results_dir)
    
    print("\n" + "=" * 80)
    print("📊 DETAILED EXPERIMENT STATISTICS FOR DATA COLLECTION")
    print("=" * 80)
    
    rounds = progressive_metrics['rounds']
    if not rounds:
        print("⚠️  No rounds completed.")
        return
    
    # Get all client IDs
    all_client_ids = set()
    for log in server_log_data:
        if 'local_accuracies' in log:
            all_client_ids.update(log['local_accuracies'].keys())
        if 'aggregation' in log and 'similarities' in log['aggregation']:
            # Infer client IDs from similarities count (if available)
            similarities = log['aggregation'].get('similarities', [])
            accepted = log['aggregation'].get('accepted_clients', [])
            all_client_ids.update(accepted)
    
    # Also include from local_accuracies history
    if local_accuracies:
        all_client_ids.update(local_accuracies.keys())
    
    all_client_ids = sorted(all_client_ids)
    attacker_ids_set = set(attacker_ids) if attacker_ids else set()
    
    # ========== 1. Global Accuracy Table ==========
    print("\n" + "-" * 80)
    print("1️⃣  GLOBAL ACCURACY (Per Round)")
    print("-" * 80)
    print(f"{'Round':<8} | {'Clean Accuracy':<15} | {'Accuracy Change':<17}")
    print("-" * 80)
    
    clean_acc = progressive_metrics['clean_acc']
    for i, r in enumerate(rounds):
        acc = clean_acc[i] if i < len(clean_acc) else 0.0
        acc_change = (clean_acc[i] - clean_acc[i-1]) if i > 0 else 0.0
        print(f"{r:<8} | {acc:<15.6f} | {acc_change:>+17.6f}")
    
    print("-" * 80)
    if clean_acc:
        print(f"Summary: Initial={clean_acc[0]:.6f}, Final={clean_acc[-1]:.6f}, "
              f"Best={max(clean_acc):.6f}, Change={clean_acc[-1]-clean_acc[0]:+.6f}")
    
    # ========== 2. Cosine Similarity Table ==========
    print("\n" + "-" * 80)
    print("2️⃣  COSINE SIMILARITY (Per Round, Per Client)")
    print("-" * 80)
    
    # Prepare header
    header = "Round | "
    for cid in all_client_ids:
        client_type = "A" if cid in attacker_ids_set else "B"
        header += f"Client{cid}({client_type}) | "
    header += "Mean | Std"
    print(header)
    print("-" * 80)
    
    for log in server_log_data:
        round_num = log['round']
        aggregation = log.get('aggregation', {})
        similarities = aggregation.get('similarities', [])
        accepted = aggregation.get('accepted_clients', [])
        
        # Create similarity map
        all_clients_round = sorted(set(accepted))
        sim_map = {}
        if len(similarities) == len(all_clients_round):
            for idx, cid in enumerate(all_clients_round):
                sim_map[cid] = similarities[idx]
        
        # Print row
        row = f"{round_num:<6} | "
        for cid in all_client_ids:
            sim = sim_map.get(cid, 0.0)
            row += f"{sim:<14.6f} | "
        
        # Calculate mean and std for this round
        sim_values = [sim_map.get(cid, 0.0) for cid in all_client_ids if cid in sim_map]
        mean_sim = np.mean(sim_values) if sim_values else 0.0
        std_sim = np.std(sim_values) if len(sim_values) > 1 else 0.0
        
        row += f"{mean_sim:<6.6f} | {std_sim:.6f}"
        print(row)
    
    print("-" * 80)
    
    # ========== 3. Local Accuracy Table ==========
    print("\n" + "-" * 80)
    print("3️⃣  LOCAL ACCURACY (Per Round, Per Client)")
    print("-" * 80)
    
    # Prepare header
    header = "Round | "
    for cid in all_client_ids:
        client_type = "A" if cid in attacker_ids_set else "B"
        header += f"Client{cid}({client_type}) | "
    header += "Mean | Std"
    print(header)
    print("-" * 80)
    
    for log in server_log_data:
        round_num = log['round']
        local_accs_round = log.get('local_accuracies', {})
        
        # Print row
        row = f"{round_num:<6} | "
        acc_values = []
        for cid in all_client_ids:
            acc = local_accs_round.get(cid, 0.0)
            acc_values.append(acc)
            row += f"{acc:<14.6f} | "
        
        # Calculate mean and std
        mean_acc = np.mean(acc_values) if acc_values else 0.0
        std_acc = np.std(acc_values) if len(acc_values) > 1 else 0.0
        row += f"{mean_acc:<6.6f} | {std_acc:.6f}"
        print(row)
    
    print("-" * 80)
    
    # ========== 4. Save to CSV files for easy import ==========
    print("\n" + "-" * 80)
    print("💾 SAVING DATA TO CSV FILES FOR EASY COLLECTION")
    print("-" * 80)
    
    # Save Global Accuracy
    csv_path1 = results_dir / f"{experiment_name}_global_accuracy.csv"
    with open(csv_path1, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Round', 'Clean_Accuracy', 'Accuracy_Change'])
        for i, r in enumerate(rounds):
            acc = clean_acc[i] if i < len(clean_acc) else 0.0
            acc_change = (clean_acc[i] - clean_acc[i-1]) if i > 0 else 0.0
            writer.writerow([r, f"{acc:.6f}", f"{acc_change:.6f}"])
    print(f"✅ Global Accuracy saved to: {csv_path1}")
    
    # Save Cosine Similarity
    csv_path2 = results_dir / f"{experiment_name}_cosine_similarity.csv"
    with open(csv_path2, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        header = ['Round'] + [f"Client_{cid}_{'A' if cid in attacker_ids_set else 'B'}" 
                                           for cid in all_client_ids] + ['Mean', 'Std']
        writer.writerow(header)
        
        for log in server_log_data:
            round_num = log['round']
            aggregation = log.get('aggregation', {})
            similarities = aggregation.get('similarities', [])
            accepted = aggregation.get('accepted_clients', [])
            
            all_clients_round = sorted(set(accepted))
            sim_map = {}
            if len(similarities) == len(all_clients_round):
                for idx, cid in enumerate(all_clients_round):
                    sim_map[cid] = similarities[idx]
            
            row = [round_num]
            sim_values = []
            for cid in all_client_ids:
                sim = sim_map.get(cid, 0.0)
                sim_values.append(sim)
                row.append(f"{sim:.6f}")
            
            mean_sim = np.mean(sim_values) if sim_values else 0.0
            std_sim = np.std(sim_values) if len(sim_values) > 1 else 0.0
            row.extend([f"{mean_sim:.6f}", f"{std_sim:.6f}"])
            writer.writerow(row)
    print(f"✅ Cosine Similarity saved to: {csv_path2}")
    
    # Save Local Accuracy
    csv_path3 = results_dir / f"{experiment_name}_local_accuracy.csv"
    with open(csv_path3, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        header = ['Round'] + [f"Client_{cid}_{'A' if cid in attacker_ids_set else 'B'}" 
                             for cid in all_client_ids] + ['Mean', 'Std']
        writer.writerow(header)
        
        for log in server_log_data:
            round_num = log['round']
            local_accs_round = log.get('local_accuracies', {})
            
            row = [round_num]
            acc_values = []
            for cid in all_client_ids:
                acc = local_accs_round.get(cid, 0.0)
                acc_values.append(acc)
                row.append(f"{acc:.6f}")
            
            mean_acc = np.mean(acc_values) if acc_values else 0.0
            std_acc = np.std(acc_values) if len(acc_values) > 1 else 0.0
            row.extend([f"{mean_acc:.6f}", f"{std_acc:.6f}"])
            writer.writerow(row)
    print(f"✅ Local Accuracy saved to: {csv_path3}")
    
    print("\n" + "=" * 80)
    print("✅ All statistics printed and saved to CSV files!")
    print("   You can now easily collect data from multiple runs and compare them.")
    print("=" * 80)

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

    print(f"Total Rounds: {len(rounds)}")
    print(f"Final Clean Accuracy: {clean[-1]:.4f}")
    if len(clean) > 1:
        print(f"Best Clean Accuracy: {max(clean):.4f}")
        print(f"Accuracy Change: {clean[-1] - clean[0]:+.4f}")

def main():
    config = {
        # ========== Experiment Configuration ==========
        'experiment_name': 'vgae_grmp_attack',  # Name for result files and logs
        'seed': 42,  # Random seed for reproducibility (int), 42 is the default
        
        # ========== Federated Learning Setup ==========
        'num_clients': 7,  # Total number of federated learning clients (int)
        'num_attackers': 2,  # Number of attacker clients (int, must be < num_clients)
        'num_benign_clients': None,  # Optional: Explicit number of benign clients for baseline experiment
                                    # If None, baseline will use (num_clients - num_attackers) to ensure fair comparison
                                    # If set, baseline experiment will use exactly this many benign clients
        'num_rounds': 50,  # Total number of federated learning rounds (int)
        
        # ========== Training Hyperparameters ==========
        'client_lr': 5e-5,  # Learning rate for local client training (float)
        'server_lr': 1.0,  # Server learning rate for model aggregation (fixed at 1.0)
        'batch_size': 128,  # Batch size for local training (int)
        'test_batch_size': 256,  # Batch size for test/validation data loaders (int)
        'local_epochs': 5,  # Number of local training epochs per round (int, per paper Section IV)
        'alpha': 0.0,  # FedProx proximal coefficient μ: loss += (μ/2)*||w - w_global||². Set 0 for standard FedAvg, >0 to penalize local drift from global model (helps Non-IID stability)
        
        # ========== Data Distribution ==========
        'data_distribution': 'non-iid',  # 'iid' for uniform random, 'non-iid' for Dirichlet-based heterogeneous distribution
        'dirichlet_alpha': 0.3,  # Only used when data_distribution='non-iid'. Lower = more heterogeneous, higher = more balanced
        # 'dataset_size_limit': None,  # Limit dataset size for faster experimentation (None = use FULL AG News dataset per paper, int = limit training samples)
        'dataset_size_limit': 20000,  # Limit dataset size for faster experimentation (None = use FULL AG News dataset per paper, int = limit training samples)
        # 'dataset_size_limit': 10000,  # Limit dataset size for faster experimentation (None = use FULL AG News dataset per paper, int = limit training samples)

        # ========== Training Mode Configuration ==========
        'use_lora': True,  # True for LoRA fine-tuning, False for full fine-tuning
        # LoRA parameters (only used when use_lora=True)
        # NOTE: Lower r values = faster training but potentially less capacity
        # Recommended: r=8 for speed, r=16 for better performance (default)
        'lora_r': 8,  # LoRA rank (controls the rank of low-rank matrices) - REDUCED from 16 to 8 for speed
        'lora_alpha': 16,  # LoRA alpha (scaling factor, typically 2*r) - UPDATED to match r=8
        'lora_dropout': 0.1,  # LoRA dropout rate
        'lora_target_modules': None,  # None = use default for DistilBERT (["q_lin", "k_lin", "v_lin", "out_lin"])
        # Model configuration
        # Supported models:
        #   Encoder-only (BERT-style): 'distilbert-base-uncased', 'bert-base-uncased', 'roberta-base', 'microsoft/deberta-v3-base'
        #   Decoder-only (GPT-style):  'EleutherAI/pythia-160m', 'EleutherAI/pythia-1b', 'facebook/opt-125m', 'gpt2'
        # 'model_name': 'distilbert-base-uncased',  # Hugging Face model name for classification
        'model_name': 'EleutherAI/pythia-160m',  # Alternative: Pythia-160M (Decoder-only, 160M params)
        'num_labels': 4,  # Number of classification labels (AG News: 4, IMDB: 2)
        'max_length': 128,  # Max token length for tokenizer. AG News: 128 (avg ~50 tokens), IMDB: 256-512 (avg ~230 tokens)
        
        # ========== VGAE Training Parameters ==========
        # Reference paper: input_dim=5, hidden1_dim=32, hidden2_dim=16, num_epoch=10, lr=0.01
        # Note: dim_reduction_size should be <= total trainable parameters
        'dim_reduction_size': 100,  # Reduced dimensionality of LLM parameters (auto-adjusted for LoRA if needed)
        'vgae_epochs': 20,  # Number of epochs for VGAE training (reference: 20)
        'vgae_lr': 0.01,  # Learning rate for VGAE optimizer (reference: 0.01)
        'vgae_hidden_dim': 64,  # VGAE hidden layer dimension (per paper: hidden1_dim=32)
        'vgae_latent_dim': 32,  # VGAE latent space dimension (per paper: hidden2_dim=16)
        'vgae_dropout': 0,  # VGAE encoder dropout rate (0=no dropout, higher=more regularization to prevent overfitting)
        'vgae_kl_weight': 0.1,  # KL divergence weight in VGAE loss: L = L_recon + kl_weight * KL(q||p). Higher=stronger latent regularization
        # ========== Graph Construction Parameters ==========
        'graph_threshold': 0.5,  # Cosine similarity threshold for adjacency matrix: A[i,j]=1 if sim(Δ_i,Δ_j)>threshold, else 0. Higher=sparser graph

        # ========== Attack Configuration ==========
        'attack_method': 'GRMP',  # Attack method: 'GRMP' (VGAE-based) or 'ALIE' (statistical baseline)
        'attack_start_round': 0,  # Round when attack phase starts (int, now all rounds use complete poisoning)
        
        # ========== ALIE Attack Parameters (only used when attack_method='ALIE') ==========
        'alie_z_max': None,  # Z-score multiplier for ALIE. None = auto-compute based on num_clients and num_attackers
        'alie_attack_start_round': None,  # Round to start ALIE attack (None = start immediately, overrides attack_start_round)

        # ========== GRMP Attack Optimization Parameters ==========
        'proxy_step': 0.001,  # Step size for gradient-free ascent toward global-loss proxy
        'proxy_steps': 200,  # Number of optimization steps for attack objective (int)
        'grad_clip_norm': 1.0,  # Gradient clipping norm for training stability (float)
        'attacker_claimed_data_size': None,  # None = use actual assigned data size
        'early_stop_constraint_stability_steps': 1,  # Early stopping: stop after N consecutive steps satisfying constraint (int)

        # ========== Formula 4 Constraint Parameters ==========
        'dist_bound': None,  # Distance threshold for constraint (4b): d(w'_j(t), w'_g(t)) ≤ dist_bound (None = use benign max distance)
        'sim_center': None,  # Optional center for similarity bounds (None = use benign min/max)

        # ========== Lagrangian Dual Parameters ==========
        'use_lagrangian_dual': True,  # Whether to use Lagrangian Dual mechanism (bool, True/False)
        # Distance constraint multiplier parameters
        'lambda_dist_init': 0.1,  # Initial λ_dist(t) value for distance constraint: dist(Δ_att, Δ_g) ≤ dist_bound
        'lambda_dist_lr': 0.01,    # Learning rate for λ_dist(t) update (dual ascent step size)
        # ========== Cosine Similarity Constraint Parameters (TWO-SIDED with TWO multipliers) ==========
        'use_cosine_similarity_constraint': False,  # Whether to enable cosine similarity constraints (bool, True/False)
        'lambda_sim_low_init': 0.1,  # Initial λ_sim_low(t) value for lower bound constraint: sim_bound_low <= sim_att
        'lambda_sim_up_init': 0.1,   # Initial λ_sim_up(t) value for upper bound constraint: sim_att <= sim_bound_up
        'lambda_sim_low_lr': 0.1,    # Learning rate for λ_sim_low(t) update
        'lambda_sim_up_lr': 0.1,     # Learning rate for λ_sim_up(t) update

        # ========== Augmented Lagrangian Method (ALM) Parameters ==========
        # Standard ALM adds quadratic penalties: (ρ/2) * g(x)^2 for each inequality constraint g(x) ≤ 0.
        'use_augmented_lagrangian': True,   # Enable Augmented Lagrangian (requires use_lagrangian_dual=True)
        'lambda_update_mode': 'classic',    # Dual variable update: "classic"=λ += lr*g (fixed step), "alm"=λ += ρ*g (penalty-scaled step, standard ALM)
        # Penalty parameters ρ (per-constraint): controls quadratic penalty strength (ρ/2)*max(0,g)^2 in ALM objective
        'rho_dist_init': 1.0,
        'rho_sim_low_init': 1.0,
        'rho_sim_up_init': 1.0,
        # Adaptive ρ update (monotone increase)
        'rho_adaptive': True,
        'rho_theta': 0.5,            # If σ_k > theta * σ_{k-1} then increase ρ
        'rho_increase_factor': 2.0,
        'rho_min': 1e-3,
        'rho_max': 1e4,
        
        # ========== Proxy Loss Estimation Parameters ==========
        'proxy_sample_size': 512,  # Number of samples in proxy dataset for F(w'_g) estimation (int)
                                # Increased from 128 to 512 for better accuracy (4 batches with test_batch_size=128)
        'proxy_max_batches_opt': 1,  # Max batches for proxy loss in optimization loop (int)
                                # Used during gradient-based optimization (20 steps per round)
        'proxy_max_batches_eval': 2,  # Max batches for proxy loss in final evaluation (int)
                                # Used for final attack objective logging (1 call per round)
        
        # ========== Visualization ==========
        'generate_plots': True,  # Whether to generate visualization plots (bool)
    }

    # Run experiment (attack if num_attackers > 0, baseline if num_attackers == 0)
    if config.get('num_attackers', 0) > 0:
        attack_method = config.get('attack_method', 'GRMP')
        if attack_method == 'ALIE':
            print("Running ALIE Attack (Model Poisoning Baseline)...")
        else:
            print("Running GRMP Attack with VGAE...")
    else:
        print("Running Baseline Experiment (No Attack)...")
    
    results, metrics = run_experiment(config)
    analyze_results(metrics)
        

if __name__ == "__main__":
    main()