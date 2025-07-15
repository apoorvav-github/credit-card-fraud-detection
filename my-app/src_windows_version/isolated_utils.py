"""
Utilities for isolated training analysis and visualization
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

def load_isolated_results(results_folder: str) -> Dict:
    """
    Load results from an isolated training experiment
    """
    all_clients_path = os.path.join(results_folder, "all_clients_metrics.json")
    summary_path = os.path.join(results_folder, "summary_stats.json")
    
    if not os.path.exists(all_clients_path):
        raise FileNotFoundError(f"Results file not found: {all_clients_path}")
    
    with open(all_clients_path, "r") as f:
        all_metrics = json.load(f)
    
    summary_stats = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary_stats = json.load(f)
    
    return all_metrics, summary_stats

def plot_isolated_training_curves(results_folder: str, save_plots: bool = True):
    """
    Plot training curves for isolated training
    """
    all_metrics, summary_stats = load_isolated_results(results_folder)
    
    num_clients = len([k for k in all_metrics.keys() if k.startswith("client_")])
    rounds = len(all_metrics["client_0"]["test_loss"])
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Isolated Training Results", fontsize=16)
    
    # Plot 1: Test Loss for all clients
    ax1 = axes[0, 0]
    for client_id in range(num_clients):
        client_key = f"client_{client_id}"
        if client_key in all_metrics:
            test_losses = all_metrics[client_key]["test_loss"]
            ax1.plot(range(1, rounds + 1), test_losses, 
                    marker='o', label=f"Client {client_id}")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Test Loss")
    ax1.set_title("Test Loss per Client")
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: AUC for all clients
    ax2 = axes[0, 1]
    for client_id in range(num_clients):
        client_key = f"client_{client_id}"
        if client_key in all_metrics:
            aucs = [metrics["auc"] for metrics in all_metrics[client_key]["test_metrics"]]
            ax2.plot(range(1, rounds + 1), aucs, 
                    marker='s', label=f"Client {client_id}")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("AUC")
    ax2.set_title("AUC per Client")
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Accuracy for all clients
    ax3 = axes[1, 0]
    for client_id in range(num_clients):
        client_key = f"client_{client_id}"
        if client_key in all_metrics:
            accuracies = [metrics["accuracy"] for metrics in all_metrics[client_key]["test_metrics"]]
            ax3.plot(range(1, rounds + 1), accuracies, 
                    marker='^', label=f"Client {client_id}")
    ax3.set_xlabel("Round")
    ax3.set_ylabel("Accuracy")
    ax3.set_title("Accuracy per Client")
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Final performance comparison
    ax4 = axes[1, 1]
    if summary_stats:
        final_aucs = summary_stats.get("individual_final_aucs", [])
        final_accuracies = summary_stats.get("individual_final_accuracies", [])
        
        x_pos = np.arange(len(final_aucs))
        width = 0.35
        
        bars1 = ax4.bar(x_pos - width/2, final_aucs, width, label='AUC', alpha=0.8)
        bars2 = ax4.bar(x_pos + width/2, final_accuracies, width, label='Accuracy', alpha=0.8)
        
        ax4.set_xlabel("Client ID")
        ax4.set_ylabel("Score")
        ax4.set_title("Final Performance per Client")
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f"Client {i}" for i in range(len(final_aucs))])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_plots:
        plot_path = os.path.join(results_folder, "isolated_training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
    
    return fig

def compare_isolated_vs_federated(isolated_folder: str, federated_folder: str):
    """
    Compare results between isolated and federated training
    """
    # Load isolated results
    isolated_metrics, isolated_summary = load_isolated_results(isolated_folder)
    
    # Load federated results (assuming similar structure)
    federated_path = os.path.join(federated_folder, "metrics.json")
    if os.path.exists(federated_path):
        with open(federated_path, "r") as f:
            federated_metrics = json.load(f)
    else:
        print(f"Warning: Federated results not found at {federated_path}")
        return
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Isolated vs Federated Learning Comparison", fontsize=16)
    
    # Final performance comparison
    ax1 = axes[0]
    if isolated_summary:
        isolated_auc = isolated_summary.get("final_avg_auc", 0)
        isolated_acc = isolated_summary.get("final_avg_accuracy", 0)
        
        # Extract federated final metrics (this might need adjustment based on your federated output format)
        # Assuming the last round contains the final metrics
        last_round = max(federated_metrics.keys()) if federated_metrics else "1"
        federated_auc = federated_metrics.get(last_round, {}).get("auc", 0)
        federated_acc = federated_metrics.get(last_round, {}).get("accuracy", 0)
        
        methods = ["Isolated", "Federated"]
        auc_scores = [isolated_auc, federated_auc]
        acc_scores = [isolated_acc, federated_acc]
        
        x_pos = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, auc_scores, width, label='AUC', alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, acc_scores, width, label='Accuracy', alpha=0.8)
        
        ax1.set_xlabel("Training Method")
        ax1.set_ylabel("Score")
        ax1.set_title("Final Performance Comparison")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(methods)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
    
    # Variance comparison for isolated training
    ax2 = axes[1]
    if isolated_summary:
        client_aucs = isolated_summary.get("individual_final_aucs", [])
        client_accs = isolated_summary.get("individual_final_accuracies", [])
        
        if client_aucs and client_accs:
            ax2.scatter(client_aucs, client_accs, alpha=0.7, s=100)
            ax2.set_xlabel("Final AUC")
            ax2.set_ylabel("Final Accuracy")
            ax2.set_title("Client Performance Distribution (Isolated)")
            ax2.grid(True, alpha=0.3)
            
            # Add client labels
            for i, (auc, acc) in enumerate(zip(client_aucs, client_accs)):
                ax2.annotate(f"C{i}", (auc, acc), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_path = os.path.join(os.path.dirname(isolated_folder), "isolated_vs_federated_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {comparison_path}")
    
    return fig

def print_isolated_summary(results_folder: str):
    """
    Print a summary of isolated training results
    """
    try:
        all_metrics, summary_stats = load_isolated_results(results_folder)
        
        print("\n" + "="*50)
        print("ISOLATED TRAINING SUMMARY")
        print("="*50)
        
        if summary_stats:
            print(f"Average Final AUC: {summary_stats['final_avg_auc']:.4f} ± {summary_stats['final_std_auc']:.4f}")
            print(f"Average Final Accuracy: {summary_stats['final_avg_accuracy']:.4f} ± {summary_stats['final_std_accuracy']:.4f}")
            print(f"Average Final Loss: {summary_stats['final_avg_loss']:.4f} ± {summary_stats['final_std_loss']:.4f}")
            
            print(f"\nIndividual Client Performance:")
            for i, (auc, acc) in enumerate(zip(summary_stats['individual_final_aucs'], 
                                             summary_stats['individual_final_accuracies'])):
                print(f"  Client {i}: AUC={auc:.4f}, Accuracy={acc:.4f}")
        
        num_clients = len([k for k in all_metrics.keys() if k.startswith("client_")])
        rounds = len(all_metrics["client_0"]["test_loss"]) if "client_0" in all_metrics else 0
        
        print(f"\nExperiment Details:")
        print(f"  Number of clients: {num_clients}")
        print(f"  Number of rounds: {rounds}")
        print(f"  Results folder: {results_folder}")
        
    except Exception as e:
        print(f"Error reading results: {e}")

def convert_to_round_based_format(client_metrics_dict):
    """
    Convert from client-based format to round-based format
    
    Args:
        client_metrics_dict: Dict in format {"client_0": {"test_metrics": [...], "test_loss": [...]}},
                              where "test_metrics" is a list of dictionaries containing "auc" and "accuracy"
    
    Returns:
        Dict in format {"1": {"0": {"loss": ..., "auc": ..., "accuracy": ...}}}
    """
    round_based = {}
    
    # Find the number of rounds (assuming all clients have same number of rounds)
    first_client = list(client_metrics_dict.values())[0]
    num_rounds = len(first_client["test_metrics"])
    
    for round_idx in range(num_rounds):
        round_key = str(round_idx + 1)
        round_based[round_key] = {}
        
        for client_key, client_data in client_metrics_dict.items():
            client_id = client_key.replace("client_", "")
            round_based[round_key][client_id] = {
                "loss": client_data["test_loss"][round_idx],
                "auc": client_data["test_metrics"][round_idx]["auc"],
                "accuracy": client_data["test_metrics"][round_idx]["accuracy"]
            }
    
    return round_based

def convert_to_client_based_format(round_based_dict):
    """
    Convert from round-based format to client-based format
    
    Args:
        round_based_dict: Dict in format {"1": {"0": {"loss": ..., "auc": ..., "accuracy": ...}}}
    
    Returns:
        Dict in format {"client_0": {"test_metrics": [...], "test_loss": [...]}},
                        where "test_metrics" is a list of dictionaries containing "auc" and "accuracy"
    """
    client_based = {}
    
    # Get all client IDs
    first_round = list(round_based_dict.values())[0]
    client_ids = list(first_round.keys())
    
    # Initialize client data
    for client_id in client_ids:
        client_based[f"client_{client_id}"] = {
            "test_loss": [],
            "test_metrics": []
        }
   