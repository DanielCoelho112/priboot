import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import math

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


def update_target_network(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def weights_init_rlad(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf.
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

def get_n_params_network(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def weighted_mae(waypoints, ground_truth_waypoints, start_weight=1.0, end_weight=0.5):
    """
    Computes the weighted mean absolute error (MAE) for predicted waypoints, with temporal weighting
    such that future waypoints have less weight.
    
    Parameters:
    - waypoints: Tensor of predicted waypoints with shape [batch_size, num_waypoints, 2 (for x, y)]
    - ground_truth_waypoints: Tensor of ground truth waypoints with the same shape as `waypoints`
    - start_weight: The weight for the first waypoint (closest to the present time)
    - end_weight: The weight for the last waypoint (furthest into the future)
    
    Returns:
    - Weighted MAE loss
    """
    
    # Number of waypoints
    num_waypoints = waypoints.shape[1]
    
    # Generate temporal weights - linearly decrease from start_weight to end_weight
    weights = torch.linspace(start_weight, end_weight, steps=num_waypoints).to(waypoints.device)
    
    # Reshape weights to match waypoints shape (for broadcasting)
    weights = weights.view(1, num_waypoints, 1)
    
    # Calculate the absolute errors
    abs_errors = torch.abs(waypoints - ground_truth_waypoints)
    
    # Apply weights to the errors
    weighted_errors = abs_errors * weights
    
    # Calculate the weighted mean absolute error
    weighted_mae_loss = weighted_errors.mean()
    
    return weighted_mae_loss

def weighted_mse(waypoints, ground_truth_waypoints, start_weight=1.0, end_weight=0.5):
    """
    Computes the weighted mean squared error (MSE) for predicted waypoints, with temporal weighting
    such that future waypoints have less weight.
    
    Parameters:
    - waypoints: Tensor of predicted waypoints with shape [batch_size, num_waypoints, 2 (for x, y)]
    - ground_truth_waypoints: Tensor of ground truth waypoints with the same shape as `waypoints`
    - start_weight: The weight for the first waypoint (closest to the present time)
    - end_weight: The weight for the last waypoint (furthest into the future)
    
    Returns:
    - Weighted MSE loss
    """
    
    # Number of waypoints
    num_waypoints = waypoints.shape[1]
    
    # Generate temporal weights - linearly decrease from start_weight to end_weight
    weights = torch.linspace(start_weight, end_weight, steps=num_waypoints).to(waypoints.device)
    
    # Reshape weights to match waypoints shape (for broadcasting)
    weights = weights.view(1, num_waypoints, 1)
    
    # Calculate the squared errors
    squared_errors = (waypoints - ground_truth_waypoints) ** 2
    
    # Apply weights to the errors
    weighted_errors = squared_errors * weights
    
    # Calculate the weighted mean squared error
    weighted_mse_loss = weighted_errors.mean()
    
    return weighted_mse_loss