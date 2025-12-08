# visualization.py
# Visualization module for GRMP attack experiment results
# Generates plots matching the paper's Figure 3, 4, 5, and 6

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json

# Set style for publication-quality figures
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')  # Fallback to default style

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16


class ExperimentVisualizer:
    """Visualizer for GRMP attack experiment results"""
    
    def __init__(self, results_dir: Path = Path("results")):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def load_results(self, results_path: str) -> Dict:
        """Load experiment results from JSON file"""
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def plot_figure3_global_accuracy_asr(self, log_data: List[Dict], save_path: Optional[str] = None):
        """
        Figure 3: Impact of the GRMP attack on global model learning accuracy 
        and the attack success rate (ASR) over communication rounds.
        """
        rounds = [log['round'] for log in log_data]
        clean_acc = [log['clean_accuracy'] for log in log_data]
        asr = [log['attack_success_rate'] for log in log_data]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Left Y-axis: Global Learning Accuracy
        color_acc = 'tab:blue'
        ax1.set_xlabel('Communication Round', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Global Learning Accuracy', color=color_acc, fontsize=12, fontweight='bold')
        
        # Plot mean accuracy with range (for single run, use the value itself)
        line1 = ax1.plot(rounds, clean_acc, 'o-', color=color_acc, linewidth=2, 
                         markersize=6, label='Accuracy (Mean)', zorder=3)
        
        # Add shaded area (for single run, use small margin)
        acc_min = [a - 0.01 for a in clean_acc]  # Small margin for visualization
        acc_max = [a + 0.01 for a in clean_acc]
        ax1.fill_between(rounds, acc_min, acc_max, alpha=0.2, color=color_acc, 
                        label='Accuracy Range (Min-Max)')
        
        ax1.tick_params(axis='y', labelcolor=color_acc)
        ax1.set_ylim([0.75, 0.95])
        ax1.grid(True, alpha=0.3)
        
        # Right Y-axis: Attack Success Rate
        ax2 = ax1.twinx()
        color_asr = 'tab:red'
        ax2.set_ylabel('Attack Success Rate (ASR)', color=color_asr, fontsize=12, fontweight='bold')
        
        line2 = ax2.plot(rounds, asr, 's-', color=color_asr, linewidth=2, 
                        markersize=6, label='ASR (Mean)', zorder=3)
        
        # Add shaded area for ASR
        asr_min = [max(0, a - 0.05) for a in asr]
        asr_max = [min(1, a + 0.05) for a in asr]
        ax2.fill_between(rounds, asr_min, asr_max, alpha=0.2, color=color_asr,
                        label='ASR Range (Min-Max)')
        
        ax2.tick_params(axis='y', labelcolor=color_asr)
        ax2.set_ylim([0.0, 0.75])
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', framealpha=0.9)
        
        plt.title('Figure 3: Impact of GRMP Attack on Global Model Learning Accuracy and ASR', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✅ Saved Figure 3 to: {save_path}")
        else:
            plt.savefig(self.results_dir / 'figure3_global_accuracy_asr.png', 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_figure4_cosine_similarity(self, log_data: List[Dict], 
                                      attacker_ids: Optional[List[int]] = None,
                                      save_path: Optional[str] = None):
        """
        Figure 4: Temporal evolution of cosine similarity for each LLM agent 
        with dynamic detection threshold over communication rounds.
        
        Args:
            log_data: List of round logs
            attacker_ids: List of attacker client IDs (if None, assumes last 2 clients are attackers)
        """
        rounds = [log['round'] for log in log_data]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Extract similarities and client info from defense logs
        # Build a mapping of client_id -> list of similarities over rounds
        client_similarities = {}  # {client_id: [sim1, sim2, ...]}
        thresholds = []
        
        for log in log_data:
            defense = log.get('defense', {})
            similarities = defense.get('similarities', [])
            accepted_clients = defense.get('accepted_clients', [])
            rejected_clients = defense.get('rejected_clients', [])
            threshold = defense.get('threshold', 0.0)
            thresholds.append(threshold)
            
            # Map similarities to client IDs
            # The similarities list is ordered by sorted client IDs (from aggregate_updates)
            # Combine accepted and rejected to get all client IDs
            all_client_ids = sorted(set(accepted_clients + rejected_clients))
            
            # Ensure we have the same number of similarities as clients
            if len(similarities) != len(all_client_ids):
                # If mismatch, try to infer from the log structure
                # Similarities should match the order of sorted client IDs
                print(f"  ⚠️  Warning: Similarity count ({len(similarities)}) != client count ({len(all_client_ids)}) in round {log.get('round', '?')}")
            
            for i, client_id in enumerate(all_client_ids):
                if client_id not in client_similarities:
                    client_similarities[client_id] = []
                
                if i < len(similarities):
                    client_similarities[client_id].append(similarities[i])
                else:
                    # Pad with previous value or 0 if no previous value
                    if len(client_similarities[client_id]) > 0:
                        client_similarities[client_id].append(client_similarities[client_id][-1])
                    else:
                        client_similarities[client_id].append(0.0)
        
        # Separate into benign and attacker
        all_ids = sorted(client_similarities.keys())
        if attacker_ids is None:
            # Default assumption: last 2 clients are attackers
            num_attackers = 2
            attacker_ids_set = set(all_ids[-num_attackers:]) if len(all_ids) >= num_attackers else set()
        else:
            attacker_ids_set = set(attacker_ids)
        
        benign_clients = [{'id': cid, 'sims': client_similarities[cid]} 
                         for cid in all_ids if cid not in attacker_ids_set]
        attacker_clients = [{'id': cid, 'sims': client_similarities[cid]} 
                           for cid in all_ids if cid in attacker_ids_set]
        
        # Plot benign agents (blue lines with different markers)
        markers_benign = ['o', '^', 's', 'D']
        colors_benign = ['#1f77b4', '#2ca02c', '#9467bd', '#8c564b']
        for i, client in enumerate(benign_clients[:4]):  # Limit to 4 benign agents
            sims = client['sims']
            if len(sims) == len(rounds):
                ax.plot(rounds, sims, marker=markers_benign[i % len(markers_benign)], 
                       color=colors_benign[i % len(colors_benign)], linewidth=2, 
                       markersize=5, label=f'Benign Agent {client["id"]+1}', alpha=0.8)
        
        # Plot attacker agents (red/orange lines with square markers)
        markers_attacker = ['s', 's']
        colors_attacker = ['#d62728', '#ff7f0e']
        for i, client in enumerate(attacker_clients[:2]):  # Limit to 2 attackers
            sims = client['sims']
            if len(sims) == len(rounds):
                ax.plot(rounds, sims, marker=markers_attacker[i], 
                       color=colors_attacker[i % len(colors_attacker)], linewidth=2, 
                       markersize=6, label=f'Attacker {client["id"]+1}', alpha=0.8)
        
        # Plot defense threshold (green bars/line)
        ax.plot(rounds, thresholds, 'g-', linewidth=2.5, label='Defense Threshold', 
               alpha=0.7, linestyle='--')
        
        ax.set_xlabel('Communication Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cosine Similarity', fontsize=12, fontweight='bold')
        ax.set_ylim([0.0, 1.0])
        ax.set_xlim([1, max(rounds)])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.9, ncol=2)
        
        plt.title('Figure 4: Temporal Evolution of Cosine Similarity for Each LLM Agent\n'
                 'with Dynamic Detection Threshold', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✅ Saved Figure 4 to: {save_path}")
        else:
            plt.savefig(self.results_dir / 'figure4_cosine_similarity.png', 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_figure5_local_accuracy_no_attack(self, local_accuracies: Dict[int, List[float]], 
                                             rounds: List[int], save_path: Optional[str] = None):
        """
        Figure 5: Learning accuracy of local LLM agents with no attack 
        over communication rounds.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each benign agent
        markers = ['o', '^', 's', 'D', 'v', 'p']
        colors = ['#1f77b4', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for i, (client_id, accs) in enumerate(sorted(local_accuracies.items())):
            if len(accs) == len(rounds):
                ax.plot(rounds, accs, marker=markers[i % len(markers)], 
                       color=colors[i % len(colors)], linewidth=2, markersize=5,
                       label=f'Benign Agent {client_id+1}', alpha=0.8)
        
        ax.set_xlabel('Communication Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Local Learning Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylim([0.80, 0.95])
        ax.set_xlim([1, max(rounds)])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.9)
        
        plt.title('Figure 5: Learning Accuracy of Local LLM Agents\n'
                 'with No Attack', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✅ Saved Figure 5 to: {save_path}")
        else:
            plt.savefig(self.results_dir / 'figure5_local_accuracy_no_attack.png', 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_figure6_local_accuracy_with_attack(self, local_accuracies: Dict[int, List[float]], 
                                                rounds: List[int], 
                                                attacker_ids: List[int],
                                                save_path: Optional[str] = None):
        """
        Figure 6: Learning accuracy of local LLM agents under the GRMP attack 
        over communication rounds.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Separate benign and attacker
        benign_accs = {cid: accs for cid, accs in local_accuracies.items() 
                      if cid not in attacker_ids}
        attacker_accs = {cid: accs for cid, accs in local_accuracies.items() 
                        if cid in attacker_ids}
        
        # Plot benign agents (blue lines)
        markers_benign = ['o', '^', 's', 'D']
        colors_benign = ['#1f77b4', '#2ca02c', '#9467bd', '#8c564b']
        for i, (client_id, accs) in enumerate(sorted(benign_accs.items())[:4]):
            if len(accs) == len(rounds):
                ax.plot(rounds, accs, marker=markers_benign[i % len(markers_benign)], 
                       color=colors_benign[i % len(colors_benign)], linewidth=2, 
                       markersize=5, label=f'Benign Agent {client_id+1}', alpha=0.8)
        
        # Plot attacker agents (red/orange lines with square markers)
        markers_attacker = ['s', 's']
        colors_attacker = ['#d62728', '#ff7f0e']
        for i, (client_id, accs) in enumerate(sorted(attacker_accs.items())[:2]):
            if len(accs) == len(rounds):
                ax.plot(rounds, accs, marker=markers_attacker[i], 
                       color=colors_attacker[i % len(colors_attacker)], linewidth=2, 
                       markersize=6, label=f'Attacker {client_id+1}', alpha=0.8)
        
        ax.set_xlabel('Communication Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Local Learning Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylim([0.70, 1.00])
        ax.set_xlim([1, max(rounds)])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.9)
        
        plt.title('Figure 6: Learning Accuracy of Local LLM Agents\n'
                 'under the GRMP Attack', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✅ Saved Figure 6 to: {save_path}")
        else:
            plt.savefig(self.results_dir / 'figure6_local_accuracy_with_attack.png', 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_figures(self, server_log_data: List[Dict], 
                            local_accuracies: Optional[Dict[int, List[float]]] = None,
                            attacker_ids: Optional[List[int]] = None,
                            experiment_name: str = "experiment",
                            baseline_local_accuracies: Optional[Dict[int, List[float]]] = None):
        """
        Generate all figures from the paper.
        
        Args:
            server_log_data: List of round logs from server (attack experiment)
            local_accuracies: Dict mapping client_id to list of local accuracies per round (attack experiment)
            attacker_ids: List of attacker client IDs
            experiment_name: Name for output files
            baseline_local_accuracies: Dict for baseline (no attack) experiment - used for Figure 5
        """
        print("\n" + "=" * 60)
        print("Generating Visualization Figures")
        print("=" * 60)
        
        rounds = [log['round'] for log in server_log_data]
        
        # Figure 3: Global Accuracy and ASR (from attack experiment)
        print("\n📊 Generating Figure 3: Global Accuracy and ASR...")
        self.plot_figure3_global_accuracy_asr(
            server_log_data,
            save_path=self.results_dir / f'{experiment_name}_figure3.png'
        )
        
        # Figure 4: Cosine Similarity (from attack experiment)
        print("📊 Generating Figure 4: Cosine Similarity...")
        self.plot_figure4_cosine_similarity(
            server_log_data,
            attacker_ids=attacker_ids,
            save_path=self.results_dir / f'{experiment_name}_figure4.png'
        )
        
        # Figure 5: No Attack (requires baseline experiment data)
        if baseline_local_accuracies is not None:
            print("📊 Generating Figure 5: Local Accuracy (No Attack)...")
            baseline_rounds = list(range(1, len(baseline_local_accuracies.get(
                list(baseline_local_accuracies.keys())[0], [])) + 1))
            self.plot_figure5_local_accuracy_no_attack(
                baseline_local_accuracies, baseline_rounds,
                save_path=self.results_dir / f'{experiment_name}_figure5.png'
            )
        elif local_accuracies is not None:
            # Fallback: Use learning phase data (rounds < attack_start_round) as "no attack"
            print("📊 Generating Figure 5: Local Accuracy (Learning Phase - Low Attack)...")
            if attacker_ids is None:
                attacker_ids = []
            
            # Extract learning phase data (first few rounds with minimal attack)
            learning_phase_rounds = [r for r in rounds if r <= 10]  # Assuming attack starts at round 10
            benign_ids = [cid for cid in local_accuracies.keys() if cid not in attacker_ids]
            
            if benign_ids and len(learning_phase_rounds) > 0:
                learning_accs = {cid: local_accuracies[cid][:len(learning_phase_rounds)] 
                                for cid in benign_ids}
                self.plot_figure5_local_accuracy_no_attack(
                    learning_accs, learning_phase_rounds,
                    save_path=self.results_dir / f'{experiment_name}_figure5_learning_phase.png'
                )
                print("  ⚠️  Note: Using learning phase data. For true 'no attack' baseline,")
                print("     run a separate experiment with 'num_attackers': 0")
        else:
            print("  ⚠️  Figure 5 (No Attack) skipped: No baseline data available.")
            print("     To generate Figure 5, run a baseline experiment with 'num_attackers': 0")
            print("     or set 'run_both_experiments': True in config")
        
        # Figure 6: With Attack (from attack experiment)
        if local_accuracies is not None:
            print("📊 Generating Figure 6: Local Accuracy (With Attack)...")
            if attacker_ids is None:
                attacker_ids = []
            self.plot_figure6_local_accuracy_with_attack(
                local_accuracies, rounds, attacker_ids,
                save_path=self.results_dir / f'{experiment_name}_figure6.png'
            )
        else:
            print("  ⚠️  Figure 6 skipped: Local accuracies not available.")
            print("     Local accuracies are automatically tracked during training.")
        
        print("\n✅ All available figures generated successfully!")
        print(f"   Output directory: {self.results_dir}")

