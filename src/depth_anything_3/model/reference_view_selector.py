# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reference View Selection Strategies

This module provides different strategies for selecting a reference view
from multiple input views in multi-view depth estimation.
"""

import torch
from typing import Literal


RefViewStrategy = Literal["first", "middle", "saddle_balanced", "saddle_sim_range"]


def select_reference_view(
    x: torch.Tensor,
    strategy: RefViewStrategy = "saddle_balanced",
) -> torch.Tensor:
    """
    Select a reference view from multiple views using the specified strategy.
    
    Args:
        x: Input tensor of shape (B, S, N, C) where
           B = batch size
           S = number of views
           N = number of tokens
           C = channel dimension
        strategy: Selection strategy, one of:
            - "first": Always select the first view
            - "middle": Select the middle view
            - "saddle_balanced": Select view with balanced features across multiple metrics
            - "saddle_sim_range": Select view with largest similarity range
    
    Returns:
        b_idx: Tensor of shape (B,) containing the selected view index for each batch
    """
    B, S, N, C = x.shape
    
    # Simple position-based strategies
    if strategy == "first":
        return torch.zeros(B, dtype=torch.long, device=x.device)
    
    elif strategy == "middle":
        return torch.full((B,), x.shape[1] // 2, dtype=torch.long, device=x.device)
    
    # Feature-based strategies require normalized class tokens
    # Extract and normalize class tokens (first token of each view)
    img_class_feat = x[:, :, 0] / x[:, :, 0].norm(dim=-1, keepdim=True)  # B S C
    
    # Dynamic view indices (B, S) via cumsum for ONNX dynamic shape tracing
    view_indices = torch.cumsum(torch.ones_like(x[:, :, 0, 0]), dim=1) - 1.0
    tie_bias = view_indices * 1e-4
    
    if strategy == "saddle_balanced":
        # Select view with balanced features across multiple metrics
        # Compute similarity matrix
        sim = torch.matmul(img_class_feat, img_class_feat.transpose(1, 2))  # B S S
        sim_no_diag = sim - torch.eye(S, device=sim.device).unsqueeze(0)

        S_tensor = torch.tensor(x.shape[1], dtype=torch.float32, device=x.device)
        denom = torch.clamp(S_tensor - 1.0, min=1.0)

        sim_score = sim_no_diag.sum(dim=-1) / denom  # B S
        
        feat_norm = x[:, :, 0].norm(dim=-1)  # B S
        feat_var = img_class_feat.var(dim=-1)  # B S
        
        # Normalize all metrics to [0, 1] with safe diff (no denominator bias)
        def normalize_metric(metric):
            min_val = metric.min(dim=1, keepdim=True).values
            max_val = metric.max(dim=1, keepdim=True).values
            diff = max_val - min_val
            safe_diff = torch.where(diff > 1e-6, diff, torch.ones_like(diff))
            return torch.where(diff > 1e-6, (metric - min_val) / safe_diff, torch.full_like(metric, 0.5))
        
        sim_score_norm = normalize_metric(sim_score)
        norm_norm = normalize_metric(feat_norm)
        var_norm = normalize_metric(feat_var)
        
        # Select view closest to the median (0.5) across all metrics
        # Add dynamic tie-bias (1e-4) to deterministically prefer earlier indices on ties (e.g. S=2)
        balance_score = (
            (sim_score_norm - 0.5).abs() +
            (norm_norm - 0.5).abs() +
            (var_norm - 0.5).abs() +
            tie_bias
        )
        b_idx = balance_score.argmin(dim=1)
        
    elif strategy == "saddle_sim_range":
        # Select view with largest similarity range (max - min)
        sim = torch.matmul(img_class_feat, img_class_feat.transpose(1, 2))  # B S S
        sim_no_diag = sim - torch.eye(S, device=sim.device).unsqueeze(0)
        
        sim_max = sim_no_diag.max(dim=-1).values  # B S
        sim_min = sim_no_diag.min(dim=-1).values  # B S
        
        # Subtract dynamic tie-bias (1e-4) prioritizing earlier view indices on exact ties
        sim_range = (sim_max - sim_min) - tie_bias
        b_idx = sim_range.argmax(dim=1)
    
    else:
        raise ValueError(
            f"Unknown reference view selection strategy: {strategy}. "
            f"Must be one of: 'first', 'middle', 'saddle_balanced', 'saddle_sim_range'"
        )
    
    return b_idx


def reorder_by_reference(
    x: torch.Tensor,
    b_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Reorder views to place the selected reference view first.
    
    Args:
        x: Input tensor of shape (B, S, N, C)
        b_idx: Reference view indices of shape (B,)
    
    Returns:
        Reordered tensor with reference view at position 0
    """
    # Create position indices dynamically: (B, S) with [0, 1, 2, ..., S-1]
    positions = torch.cumsum(torch.ones_like(x[:, :, 0, 0], dtype=torch.long), dim=1) - 1
    
    b_idx_expanded = b_idx.unsqueeze(1)  # B 1
    
    # Create the reordering indices
    reorder_indices = positions.clone()
    reorder_indices = torch.where(
        (positions > 0) & (positions <= b_idx_expanded),
        positions - 1,
        positions
    )
    # Set position 0 to ref_idx
    reorder_indices = torch.where(positions == 0, b_idx_expanded, reorder_indices)
    
    # Gather using dynamic batch indices
    batch_indices = torch.cumsum(torch.ones_like(x[:, 0, 0, 0], dtype=torch.long), dim=0).unsqueeze(1) - 1
    x_reordered = x[batch_indices, reorder_indices]
    
    return x_reordered


def restore_original_order(
    x: torch.Tensor,
    b_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Restore original view order after processing.
    
    Args:
        x: Reordered tensor of shape (B, S, ...)
        b_idx: Original reference view indices of shape (B,)
    
    Returns:
        Tensor with original view order restored
    """
    # Create target position indices dynamically: (B, S) with [0, 1, 2, ..., S-1]
    target_positions = torch.cumsum(torch.ones_like(x[:, :, 0, 0], dtype=torch.long), dim=1) - 1
    
    b_idx_expanded = b_idx.unsqueeze(1)  # B 1
    
    # Create the restore indices
    restore_indices = torch.where(
        target_positions < b_idx_expanded,
        target_positions + 1,  # Positions before ref_idx come from current position + 1
        target_positions        # Positions after ref_idx stay the same
    )
    # Target position = ref_idx comes from current position 0
    restore_indices = torch.where(
        target_positions == b_idx_expanded,
        torch.zeros_like(target_positions),
        restore_indices
    )
    
    # Gather using dynamic batch indices
    batch_indices = torch.cumsum(torch.ones_like(x[:, 0, 0, 0], dtype=torch.long), dim=0).unsqueeze(1) - 1
    x_restored = x[batch_indices, restore_indices]
    
    return x_restored

