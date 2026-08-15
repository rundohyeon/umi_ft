from __future__ import annotations

import numpy as np


def prepare_rg2ft_policy_obs(env_obs: dict, shape_meta) -> dict:
    """Add the split RG2-FT channels requested by an RG2 policy checkpoint.

    UmiEnv intentionally keeps the collection/replay representation as one
    ``robot0_ft`` vector ordered as left-six then right-six.  RG2 policy
    checkpoints use two six-dimensional observation keys, so adapt only the
    shallow eval dictionary and leave the environment/recording API intact.
    """
    out = dict(env_obs)
    obs_meta = shape_meta["obs"]
    wants_left = "robot0_ft_left" in obs_meta
    wants_right = "robot0_ft_right" in obs_meta
    if not (wants_left or wants_right):
        return out

    if "robot0_ft" not in out:
        raise KeyError(
            "RG2 checkpoint requires robot0_ft_left/right, but UmiEnv did "
            "not provide the combined robot0_ft observation"
        )
    ft = np.asarray(out["robot0_ft"])
    if ft.ndim < 1 or ft.shape[-1] != 12:
        raise ValueError(
            "robot0_ft must have 12 channels ordered left[6], right[6]; "
            f"got shape {ft.shape}"
        )

    if wants_left:
        out["robot0_ft_left"] = ft[..., :6].copy()
    if wants_right:
        out["robot0_ft_right"] = ft[..., 6:].copy()
    return out
