"""
Label Propagation for improving predictions using graph structure.

Key insight: 96.2% of test nodes have at least 1 training neighbor.
We can propagate known labels from training nodes to improve test predictions.
"""

import torch
import numpy as np
from torch_geometric.utils import degree, add_self_loops
from typing import Optional
import pandas as pd
from pathlib import Path


def label_propagation(
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    num_iterations: int = 50,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Propagate labels from training nodes to all nodes.
    
    Uses iterative label spreading:
    Y^(t+1) = alpha * A_norm @ Y^(t) + (1 - alpha) * Y^(0)
    
    Where Y^(0) is the initial labels (known for train, 0 for others).
    
    Args:
        edge_index: Graph edges [2, E]
        y: Labels [N, C] (can contain -1 for unknown)
        train_mask: Boolean mask for training nodes
        num_iterations: Number of propagation iterations
        alpha: Propagation weight (higher = more propagation)
        
    Returns:
        Propagated soft labels [N, C]
    """
    num_nodes = y.shape[0]
    num_classes = y.shape[1]
    
    # Initialize: use training labels, 0 for others
    Y = torch.zeros(num_nodes, num_classes)
    Y[train_mask] = y[train_mask].float()
    Y_init = Y.clone()
    
    # Normalize adjacency matrix (symmetric normalization)
    edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    deg = degree(edge_index_with_loops[0], num_nodes=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    # Propagation iterations
    for _ in range(num_iterations):
        # Message passing: aggregate neighbor labels
        Y_new = torch.zeros_like(Y)
        
        src, dst = edge_index_with_loops
        # Normalize by degree
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        
        # Scatter add
        for i in range(edge_index_with_loops.shape[1]):
            Y_new[dst[i]] += norm[i] * Y[src[i]]
        
        # Combine with initial labels
        Y = alpha * Y_new + (1 - alpha) * Y_init
        
        # Keep training labels fixed
        Y[train_mask] = Y_init[train_mask]
    
    return Y


def label_propagation_fast(
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    num_iterations: int = 10,
    alpha: float = 0.8,
) -> torch.Tensor:
    """
    Fast label propagation using sparse matrix operations.
    
    Args:
        edge_index: Graph edges [2, E]
        y: Labels [N, C]
        train_mask: Boolean mask for training nodes
        num_iterations: Number of propagation iterations
        alpha: Propagation weight
        
    Returns:
        Propagated soft labels [N, C]
    """
    from torch_sparse import SparseTensor
    
    num_nodes = y.shape[0]
    num_classes = y.shape[1]
    
    # Add self-loops
    edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    
    # Create sparse adjacency matrix with symmetric normalization
    row, col = edge_index_with_loops
    deg = degree(col, num_nodes=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    
    adj = SparseTensor(row=col, col=row, value=norm, 
                       sparse_sizes=(num_nodes, num_nodes))
    
    # Initialize labels
    Y = torch.zeros(num_nodes, num_classes)
    Y[train_mask] = y[train_mask].float().clamp(0, 1)  # Handle -1
    Y_init = Y.clone()
    
    # Propagation
    for _ in range(num_iterations):
        Y = alpha * adj @ Y + (1 - alpha) * Y_init
        Y[train_mask] = Y_init[train_mask]
    
    return Y


def combine_predictions_with_propagation(
    model_predictions: np.ndarray,
    propagated_labels: np.ndarray,
    weight: float = 0.5,
) -> np.ndarray:
    """
    Combine model predictions with propagated labels.
    
    Args:
        model_predictions: GNN/MLP predictions [N, C]
        propagated_labels: Label propagation results [N, C]
        weight: Weight for model predictions (1-weight for propagation)
        
    Returns:
        Combined predictions [N, C]
    """
    return weight * model_predictions + (1 - weight) * propagated_labels


def apply_label_propagation_to_submission(
    submission_path: str,
    data_dir: str = ".",
    output_path: str = None,
    alpha: float = 0.8,
    num_iterations: int = 20,
    model_weight: float = 0.5,
    init_with_model: bool = False,
) -> pd.DataFrame:
    """
    Apply label propagation to improve a submission.
    
    Args:
        submission_path: Path to model predictions CSV
        data_dir: Directory with data files
        output_path: Path for output (default: adds _lp suffix)
        alpha: Propagation weight (higher = more neighbor influence)
        num_iterations: Number of LP iterations
        model_weight: Weight for original model predictions (only if init_with_model=False)
        init_with_model: If True, initialize test nodes with model predictions inside LP
                        (then alpha controls neighbor vs model anchor, no model_weight needed)
        
    Returns:
        Improved submission DataFrame
    """
    # Load data
    edge_index = torch.load(Path(data_dir) / 'edge_index.pt', weights_only=True)
    train_idx = torch.load(Path(data_dir) / 'train_idx.pt', weights_only=True)
    y = torch.load(Path(data_dir) / 'y.pt', weights_only=True)
    num_nodes = y.shape[0]
    num_classes = y.shape[1]
    
    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    
    # Load submission to get test indices and model predictions
    submission = pd.read_csv(submission_path)
    test_idx = submission['node_id'].values
    label_cols = [c for c in submission.columns if c.startswith('label_')]
    model_preds = submission[label_cols].values
    
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_idx] = True
    
    # Run label propagation
    if init_with_model:
        print(f"Running LP with model-initialized test nodes (alpha={alpha}, iters={num_iterations})...")
        print(f"  Y_test = {alpha:.0%} × neighbors + {1-alpha:.0%} × model_prediction")
    else:
        print(f"Running LP with zero-initialized test nodes (alpha={alpha}, iters={num_iterations})...")
    
    edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    deg = degree(edge_index_with_loops[0], num_nodes=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    # Initialize Y
    Y = torch.zeros(num_nodes, num_classes)
    train_labels = y[train_mask].float().clamp(0, 1)
    Y[train_mask] = train_labels
    
    # If init_with_model, set test nodes to model predictions
    if init_with_model:
        Y[test_idx] = torch.from_numpy(model_preds).float()
    
    Y_init = Y.clone()
    
    row, col = edge_index_with_loops
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    
    for iteration in range(num_iterations):
        Y_new = torch.zeros_like(Y)
        Y_new.index_add_(0, col, norm.unsqueeze(1) * Y[row])
        Y = alpha * Y_new + (1 - alpha) * Y_init
        Y[train_mask] = Y_init[train_mask]  # Keep train labels fixed
        if iteration % 5 == 0:
            print(f"  Iteration {iteration}/{num_iterations}")
    
    propagated = Y.numpy()
    print("Label propagation complete!")
    
    # Get final predictions for test nodes
    if init_with_model:
        # Already combined inside LP, use directly
        final_preds = propagated[test_idx]
    else:
        # Combine LP result with model predictions
        prop_preds = propagated[test_idx]
        final_preds = combine_predictions_with_propagation(model_preds, prop_preds, model_weight)
    
    # Clip to [0, 1]
    final_preds = np.clip(final_preds, 0, 1)
    
    # Create output efficiently
    result = pd.DataFrame({'node_id': test_idx})
    result = pd.concat([result, pd.DataFrame(final_preds, columns=label_cols)], axis=1)
    
    # Save
    if output_path is None:
        stem = Path(submission_path).stem
        if init_with_model:
            output_path = f"{stem}_lp_init_a{int(alpha*100)}.csv"
        else:
            output_path = f"{stem}_lp_a{int(alpha*10)}_mw{int(model_weight*10)}.csv"
    
    result.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply label propagation to submission")
    parser.add_argument("submission", help="Path to submission CSV")
    parser.add_argument("--alpha", type=float, default=0.8, help="Propagation weight")
    parser.add_argument("--iters", type=int, default=10, help="Number of iterations")
    parser.add_argument("--model-weight", type=float, default=0.5, help="Weight for model predictions")
    parser.add_argument("--output", "-o", help="Output path")
    
    args = parser.parse_args()
    
    apply_label_propagation_to_submission(
        args.submission,
        alpha=args.alpha,
        num_iterations=args.iters,
        model_weight=args.model_weight,
        output_path=args.output,
    )

