from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_training_does_not_warm_start_from_a_placeholder_by_default():
    config = yaml.safe_load((REPO_ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))

    assert config["training"]["finetune_ckpt_path"] is None


def test_pretraining_defaults_load_all_four_final_datasets():
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        config = compose(config_name="train", overrides=["dataloader=mixture"])

    expected = {
        "agibotworld": "AgiBotWorld-Beta",
        "robocoin": "RoboCOIN",
        "oxe_droid": "Droid",
        "interndata_a1": "InternData-A1",
    }
    assert config.dataloader.type == "mixture"
    assert config.dataloader.weight_strategy == "proportional"
    assert set(config.dataloader.datasets) == set(expected)
    for name, directory in expected.items():
        source = config.dataloader.datasets[name]
        assert source.enabled
        assert source.dataset_dir == f"/mnt/data/wangyuran/pretrain_data_final/{directory}"
        assert source.normalize_mode == "quantile"
        assert source.total_hours is None
        assert not {"trim_csv", "stats_root", "a1_stats_root", "segment_start", "segment_end"}.intersection(source)

    assert not (REPO_ROOT / "configs" / "dataloader" / "pretrain_data").exists()


@pytest.mark.parametrize("name", ["agibotworld", "robocoin", "oxe_droid", "interndata_a1"])
def test_flat_pretraining_configs_compose_as_standalone_dataloaders(name):
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        config = compose(config_name="train", overrides=[f"dataloader={name}"])
    assert config.dataloader.type == name
    assert config.dataloader.num_frames == 33
    assert config.dataloader.video_stride == 4
    assert config.dataloader.normalize_mode == "quantile"


def test_mixture_hydra_source_overrides_remain_independent():
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        config = compose(
            config_name="train",
            overrides=[
                "dataloader=mixture",
                "dataloader.datasets.robocoin.dataset_dir=/tmp/relocated_robocoin",
                "dataloader.datasets.oxe_droid.enabled=false",
                "dataloader.datasets.interndata_a1.total_hours=1.5",
            ],
        )
    assert config.dataloader.datasets.robocoin.dataset_dir == "/tmp/relocated_robocoin"
    assert config.dataloader.datasets.agibotworld.dataset_dir.endswith("/AgiBotWorld-Beta")
    assert not config.dataloader.datasets.oxe_droid.enabled
    assert config.dataloader.datasets.interndata_a1.total_hours == 1.5
    assert config.dataloader.datasets.agibotworld.total_hours is None
