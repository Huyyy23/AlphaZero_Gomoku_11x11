"""AlphaZero ResNet for Gomoku 11x11.

Small ResNet (5 residual blocks, 64 filters) as specified in AGENTS.md.
Input: (batch, 4, 11, 11) feature planes. Output: policy logits (batch, 121),
value in [-1, 1] (batch, 1).

Architecture follows AlphaZero: conv block -> residual tower -> policy head
and value head. Torch-free modules live in gomoku/; this module owns all
torch imports.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Standard residual block: two conv layers with batch norm and skip."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class GomokuNet(nn.Module):
    """ResNet policy and value network for Gomoku."""
    
    def __init__(
        self,
        board_size: int = 11,
        input_channels: int = 4,
        num_blocks: int = 5,
        num_filters: int = 64,
    ):
        super().__init__()
        self.board_size = board_size
        self.input_channels = input_channels
        self.num_blocks = num_blocks
        self.num_filters = num_filters
        
        # Initial convolution
        self.conv_input = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_filters)
        
        # Residual tower
        self.res_blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # Value head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Input block
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual tower
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        
        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v
    
    def predict(self, batch: torch.Tensor) -> tuple:
        """Run inference in no_grad mode, return (policy_probs, value) as numpy arrays."""
        self.eval()
        with torch.no_grad():
            if isinstance(batch, torch.Tensor):
                x = batch
            else:
                x = torch.from_numpy(batch)
            # Ensure correct device
            device = next(self.parameters()).device
            x = x.to(device)
            policy_logits, value = self.forward(x)
            policy_probs = F.softmax(policy_logits, dim=1)
            return policy_probs.cpu().numpy(), value.cpu().numpy()


def get_device() -> torch.device:
    """Auto-select device: cuda if available, else cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_model(
    board_size: int = 11,
    input_channels: int = 4,
    num_blocks: int = 5,
    num_filters: int = 64,
    device: torch.device | str | None = None,
) -> GomokuNet:
    """Create a model and move it to the appropriate device."""
    model = GomokuNet(
        board_size=board_size,
        input_channels=input_channels,
        num_blocks=num_blocks,
        num_filters=num_filters,
    )
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)
    model = model.to(device)
    return model


def save_checkpoint(
    path: str | Path,
    model: GomokuNet,
    optimizer: torch.optim.Optimizer | None = None,
    iteration: int = 0,
    config: dict[str, Any] | None = None,
    **kwargs,
) -> None:
    """Save checkpoint with full state for cross-machine loading.
    
    Saves model architecture config so it can be reconstructed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "board_size": model.board_size,
            "input_channels": model.input_channels,
            "num_blocks": model.num_blocks,
            "num_filters": model.num_filters,
        },
        "iteration": iteration,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if config is not None:
        checkpoint["config"] = config
    # Allow extra metadata
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: GomokuNet | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str | None = None,
) -> tuple[GomokuNet, dict[str, Any]]:
    """Load checkpoint. Can reconstruct model if not provided.
    
    Returns (model, checkpoint_dict). Checkpoint dict contains iteration,
    config, etc. Safe to load on different machines.
    """
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)
    
    path = Path(path)
    # Load with map_location for cross-device/machine compatibility
    checkpoint = torch.load(path, map_location=device)
    
    if model is None and "model_config" in checkpoint:
        cfg = checkpoint["model_config"]
        model = create_model(
            board_size=cfg.get("board_size", 11),
            input_channels=cfg.get("input_channels", 4),
            num_blocks=cfg.get("num_blocks", 5),
            num_filters=cfg.get("num_filters", 64),
            device=device,
        )
    elif model is None:
        raise ValueError("No model provided and checkpoint missing model_config")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return model, checkpoint
