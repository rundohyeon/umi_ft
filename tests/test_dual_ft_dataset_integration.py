from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.nested_zarr import (
    detect_zarr_prefix,
    open_nested_zip_group,
    sha256_file,
)


REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "data" / "dataset_multirate_clean.zarr.zip"
CONFIG = REPO / "diffusion_policy" / "config"


pytestmark = pytest.mark.skipif(not DATASET.is_file(), reason="integration ZIP absent")


@pytest.fixture(scope="module")
def dataset():
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG)):
        cfg = compose(
            config_name="train_diffusion_unet_timm_umi_dual_ft_workspace",
            overrides=["task=umi_dual_ft"],
        )
    value = instantiate(cfg.task.dataset)
    yield value
    value.close()


def test_source_sha_and_nested_read_only_open():
    register_codecs()
    before = sha256_file(DATASET)
    info = detect_zarr_prefix(DATASET)
    assert info.prefix == "dataset_multirate_clean.zarr"
    store, root, prefix = open_nested_zip_group(DATASET)
    try:
        assert prefix == info.prefix
        assert root["data/rgb_0"].shape == (111838, 224, 224, 3)
    finally:
        store.close()
    assert sha256_file(DATASET) == before


def test_jpegxl_decode_matches_zarr_metadata(dataset):
    sample = dataset[0]
    rgb = sample["obs"]["camera0_rgb"]
    assert tuple(rgb.shape) == (2, 3, 224, 224)
    assert rgb.dtype.is_floating_point
    assert float(rgb.min()) >= 0.0
    assert float(rgb.max()) <= 1.0


def test_first_pre_ft_anchor_is_dropped_and_all_selected_ft_is_causal(dataset):
    assert len(dataset.causal_drop_report) == 146
    assert all(row["dropped"] == 1 for row in dataset.causal_drop_report)
    assert sum(row["dropped"] for row in dataset.causal_drop_report) == 146
    sample = dataset[0]
    info = sample["sample_info"]
    assert np.all(info["left_ft_timestamps"].numpy() <= info["anchor_timestamp"].numpy())
    assert np.all(info["right_ft_timestamps"].numpy() <= info["anchor_timestamp"].numpy())


def test_left_right_normalization_is_independent(dataset):
    normalizer = dataset.get_normalizer()
    left = normalizer.params_dict["robot0_ft_left"]
    right = normalizer.params_dict["robot0_ft_right"]
    assert left is not right
    assert left["scale"].shape == right["scale"].shape == (6,)
    assert not np.array_equal(
        left["scale"].detach().numpy(), right["scale"].detach().numpy()
    )


def test_multiworker_loader_uses_process_local_lazy_zip_open(dataset):
    # Open once in the parent deliberately. PID validation in the worker must
    # close the inherited handle and perform a worker-local read-only open.
    _ = dataset[0]
    loader = DataLoader(dataset, batch_size=2, num_workers=2, shuffle=False)
    batch = next(iter(loader))
    assert tuple(batch["obs"]["camera0_rgb"].shape) == (2, 2, 3, 224, 224)
    assert tuple(batch["obs"]["robot0_ft_left"].shape) == (2, 32, 6)
    assert tuple(batch["obs"]["robot0_ft_right"].shape) == (2, 32, 6)
    assert tuple(batch["action"].shape) == (2, 16, 11)
