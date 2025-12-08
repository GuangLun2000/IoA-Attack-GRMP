# colab_setup.py
# Setup script for Google Colab environment
# This file helps set up the environment and run experiments in Colab

import os
import sys
from pathlib import Path

def setup_colab_environment():
    """Setup Colab environment for GRMP experiment"""
    print("🔧 Setting up Colab environment...")
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    print(f"✅ Created results directory: {results_dir}")
    
    # Check GPU
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  No GPU detected. Training will be slower.")
        print("   Go to Runtime → Change runtime type → GPU")
    
    # Check required files
    required_files = ['main.py', 'client.py', 'server.py', 'data_loader.py', 'models.py', 'visualization.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        print("   Please ensure all files are uploaded to Colab.")
        return False
    else:
        print("\n✅ All required files found!")
        return True

def get_quick_test_config():
    """Get quick test configuration for Colab"""
    return {
        'experiment_name': 'colab_quick_test',
        'seed': 42,
        'num_clients': 6,
        'num_attackers': 2,
        'num_rounds': 5,  # Reduced for quick test
        'client_lr': 2e-5,
        'server_lr': 0.8,
        'batch_size': 16,
        'local_epochs': 5,
        'alpha': 0.01,
        'dirichlet_alpha': 0.5,
        'test_sample_rate': 1.0,
        'dataset_size_limit': 10000,  # Limited dataset
        'poison_rate': 1.0,
        'attack_start_round': 3,
        'd_T': 0.5,
        'gamma': 10.0,
        'dim_reduction_size': 5000,  # Reduced for GPU memory
        'vgae_epochs': 10,
        'vgae_lr': 0.01,
        'vgae_lambda': 0.5,
        'camouflage_steps': 20,
        'camouflage_lr': 0.1,
        'lambda_proximity': 1.0,
        'lambda_aggregation': 0.5,
        'graph_threshold': 0.5,
        'defense_threshold': 0.10,
        'similarity_alpha': 0.7,
        'generate_plots': True,
        'run_both_experiments': False,
        'run_attack_only': False,
    }

def get_full_config():
    """Get full experiment configuration"""
    return {
        'experiment_name': 'colab_grmp_attack',
        'seed': 42,
        'num_clients': 6,
        'num_attackers': 2,
        'num_rounds': 20,
        'client_lr': 2e-5,
        'server_lr': 0.8,
        'batch_size': 16,
        'local_epochs': 5,
        'alpha': 0.01,
        'dirichlet_alpha': 0.5,
        'test_sample_rate': 1.0,
        'dataset_size_limit': None,  # Full dataset
        'poison_rate': 1.0,
        'attack_start_round': 10,
        'd_T': 0.5,
        'gamma': 10.0,
        'dim_reduction_size': 10000,
        'vgae_epochs': 20,
        'vgae_lr': 0.01,
        'vgae_lambda': 0.5,
        'camouflage_steps': 30,
        'camouflage_lr': 0.1,
        'lambda_proximity': 1.0,
        'lambda_aggregation': 0.5,
        'graph_threshold': 0.5,
        'defense_threshold': 0.10,
        'similarity_alpha': 0.7,
        'generate_plots': True,
        'run_both_experiments': False,
        'run_attack_only': False,
    }

if __name__ == "__main__":
    if setup_colab_environment():
        print("\n✅ Environment setup complete!")
        print("\nTo run experiment:")
        print("  from main import run_experiment")
        print("  from colab_setup import get_quick_test_config")
        print("  config = get_quick_test_config()")
        print("  results, metrics = run_experiment(config)")
    else:
        print("\n❌ Environment setup failed. Please check missing files.")

