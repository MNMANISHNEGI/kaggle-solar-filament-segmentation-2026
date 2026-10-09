from pathlib import Path

import pytest

from filament_seg.config import Config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_shipped_configs_load():
    for path in CONFIG_DIR.glob("*.yaml"):
        assert isinstance(Config.from_yaml(path), Config)


def test_yaml_roundtrip(tmp_path):
    cfg = Config(epochs=3, encoder="resnet34")
    cfg.to_yaml(tmp_path / "c.yaml")
    assert Config.from_yaml(tmp_path / "c.yaml") == cfg


def test_unknown_keys_are_rejected(tmp_path):
    (tmp_path / "bad.yaml").write_text("epoch: 3\n")   # typo of 'epochs'
    with pytest.raises(ValueError, match="epoch"):
        Config.from_yaml(tmp_path / "bad.yaml")
