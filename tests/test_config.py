"""Tests for config.py (Task 2)."""

from pathlib import Path

import pytest

from config import DEFAULT_CONFIG, load_config, total_games

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

REQUIRED_KEYS = [
    "n_iter",
    "games_per_iter",
    "n_simulations",
    "c_puct",
    "dirichlet_alpha",
    "dirichlet_eps",
    "temp_moves",
    "buffer_size",
    "train_steps_per_iter",
    "batch_size",
    "arena_win_threshold",
    "learning_rate",
    "optimizer",
    "weight_decay",
    "value_loss_weight",
    "policy_loss_weight",
    "lr_schedule",
]


def test_defaults_match_agents_md():
    assert DEFAULT_CONFIG["n_iter"] == 20
    assert DEFAULT_CONFIG["games_per_iter"] == 50
    assert DEFAULT_CONFIG["n_simulations"] == 200
    assert DEFAULT_CONFIG["c_puct"] == 1.5
    assert DEFAULT_CONFIG["dirichlet_alpha"] == 0.15
    assert DEFAULT_CONFIG["dirichlet_eps"] == 0.25
    assert DEFAULT_CONFIG["temp_moves"] == 15
    assert DEFAULT_CONFIG["buffer_size"] == 50000
    assert DEFAULT_CONFIG["train_steps_per_iter"] == 200
    assert DEFAULT_CONFIG["batch_size"] == 128
    assert DEFAULT_CONFIG["arena_win_threshold"] == 0.55
    # Extra knobs from plan.md decision 19
    assert DEFAULT_CONFIG["learning_rate"] == 1e-3
    assert DEFAULT_CONFIG["optimizer"] == "adam"
    assert DEFAULT_CONFIG["weight_decay"] == 1e-4
    assert DEFAULT_CONFIG["value_loss_weight"] == 1.0
    assert DEFAULT_CONFIG["policy_loss_weight"] == 1.0
    assert DEFAULT_CONFIG["lr_schedule"] == "cosine"


def test_load_none_returns_copy_of_defaults():
    cfg = load_config(None)
    assert cfg == DEFAULT_CONFIG
    cfg["n_iter"] = 999  # mutation must not leak into DEFAULT_CONFIG
    assert DEFAULT_CONFIG["n_iter"] == 20


def test_normal_yaml_has_all_required_keys_with_expected_values():
    cfg = load_config(CONFIG_DIR / "normal.yaml")
    for key in REQUIRED_KEYS:
        assert key in cfg, f"missing key {key}"
    assert cfg["rule_set"] == "normal"
    assert cfg["n_iter"] == 20
    assert cfg["games_per_iter"] == 50
    assert cfg["learning_rate"] == 1e-3
    assert cfg["lr_schedule"] == "cosine"


def test_renju_yaml_inherits_normal_and_overrides_rule_set():
    normal = load_config(CONFIG_DIR / "normal.yaml")
    renju = load_config(CONFIG_DIR / "renju.yaml")
    assert renju["rule_set"] == "renju"
    for key in REQUIRED_KEYS:
        assert renju[key] == normal[key], f"{key} should be inherited from normal.yaml"


def test_partial_override_fills_missing_keys_from_defaults(tmp_path):
    p = tmp_path / "tiny.yaml"
    p.write_text("n_iter: 3\n")
    cfg = load_config(p)
    assert cfg["n_iter"] == 3
    assert cfg["games_per_iter"] == DEFAULT_CONFIG["games_per_iter"]
    assert cfg["c_puct"] == DEFAULT_CONFIG["c_puct"]


def test_unknown_key_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("not_a_real_key: 1\n")
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(p)


def test_total_games_derived_not_hardcoded():
    assert total_games(load_config(CONFIG_DIR / "normal.yaml")) == 1000
    scaled = dict(DEFAULT_CONFIG, n_iter=24, games_per_iter=50)
    assert total_games(scaled) == 1200
    scaled2 = dict(DEFAULT_CONFIG, n_iter=30, games_per_iter=50)
    assert total_games(scaled2) == 1500
