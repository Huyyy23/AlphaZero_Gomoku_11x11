"""Config loader for the Gomoku AlphaZero project.

Reads a YAML file and merges it over ``DEFAULT_CONFIG``. Unknown keys are
rejected so typos fail loudly instead of being silently ignored.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

# Defaults follow AGENTS.md "Default hyperparameters (test run)" plus the
# extra training knobs agreed in plan.md (decision 19).
DEFAULT_CONFIG: dict = {
    # Rule set this config trains/evaluates ("normal" or "renju").
    "rule_set": "normal",
    # Self-play budget: total games = n_iter * games_per_iter (never hardcoded).
    "n_iter": 20,
    "games_per_iter": 50,
    # MCTS
    "n_simulations": 200,
    "c_puct": 1.5,
    "dirichlet_alpha": 0.15,
    "dirichlet_eps": 0.25,
    "temp_moves": 15,
    # Replay buffer / trainer
    "buffer_size": 50000,  # before augmentation
    "train_steps_per_iter": 200,
    "batch_size": 128,
    # Evaluation
    "arena_win_threshold": 0.55,
    # Optimisation (added per plan.md decision 19)
    "learning_rate": 1e-3,
    "optimizer": "adam",
    "weight_decay": 1e-4,
    "value_loss_weight": 1.0,
    "policy_loss_weight": 1.0,
    "lr_schedule": "cosine",
}


def _coerce_types(data: dict) -> dict:
    """Cast values to the type of their default.

    PyYAML parses ``1e-3`` as a string (scientific notation needs a dot in
    YAML 1.1); this restores numeric types based on ``DEFAULT_CONFIG``.
    """
    out = {}
    for key, value in data.items():
        default = DEFAULT_CONFIG.get(key)
        if isinstance(value, str) and isinstance(default, bool):
            value = value.strip().lower() in {"true", "yes", "1"}
        elif isinstance(value, str) and isinstance(default, (int, float)) and not isinstance(default, bool):
            try:
                value = type(default)(value)
            except ValueError:
                pass  # leave as str; unknown-key / type validation handles it
        out[key] = value
    return out


def _load_raw(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping, got {type(data).__name__}")
    return _coerce_types(data)


def load_config(path: str | Path | None = None) -> dict:
    """Load a YAML config merged over ``DEFAULT_CONFIG``.

    Supports an optional ``extends: <name>`` key that pulls in
    ``configs/<name>.yaml`` first (single level of inheritance).

    Args:
        path: Optional path to a YAML file. When ``None``, a deep copy of
            ``DEFAULT_CONFIG`` is returned.

    Returns:
        A dict containing every default key plus any overrides from the file.

    Raises:
        ValueError: If the YAML contains a key not present in ``DEFAULT_CONFIG``.
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is None:
        return cfg

    path = Path(path)
    data = _load_raw(path)

    # Optional single-level inheritance: `extends: normal` -> configs/normal.yaml
    parent_name = data.pop("extends", None)
    if parent_name is not None:
        parent_path = path.parent / f"{parent_name}.yaml"
        if not parent_path.exists():
            parent_path = Path(__file__).resolve().parent / "configs" / f"{parent_name}.yaml"
        if not parent_path.exists():
            raise ValueError(f"Extended config '{parent_name}' not found for {path}")
        cfg.update(_load_raw(parent_path))

    unknown = sorted(set(data) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {unknown}")

    cfg.update(data)
    return cfg


def total_games(cfg: dict) -> int:
    """Derive the self-play game budget from config (no hardcoded totals)."""
    return int(cfg["n_iter"]) * int(cfg["games_per_iter"])
