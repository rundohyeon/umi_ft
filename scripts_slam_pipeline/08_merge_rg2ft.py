# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import pickle
import click
import zarr
import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer
from umi.common.interpolation_util import get_interp1d


# %%
@click.command()
@click.option('-i', '--input', required=True,
    help='Input replay_buffer (zarr or zarr.zip) produced by 07_generate_replay_buffer.py')
@click.option('-p', '--plan', required=True,
    help='dataset_plan.pkl produced by 06_generate_dataset_plan.py (same input as 07)')
@click.option('-r', '--rg2ft_dir', required=True,
    help='Directory containing rg2ft_*.zarr recordings '
         '(saved via OnRobotRGSimpleController t/y commands)')
@click.option('-o', '--output', required=True,
    help='Output replay_buffer zarr.zip with robot0_ft added')
def main(input, plan, rg2ft_dir, output):
    input = pathlib.Path(os.path.expanduser(input)).absolute()
    plan_path = pathlib.Path(os.path.expanduser(plan)).absolute()
    rg2ft_dir = pathlib.Path(os.path.expanduser(rg2ft_dir)).absolute()
    output = pathlib.Path(os.path.expanduser(output)).absolute()

    if output.is_file():
        if click.confirm(f'Output file {output} exists! Overwrite?', abort=True):
            pass

    print(f"Loading replay buffer from {input}")
    replay_buffer = ReplayBuffer.copy_from_path(str(input))

    print(f"Loading dataset plan from {plan_path}")
    plan_episodes = pickle.load(plan_path.open('rb'))
    assert len(plan_episodes) == replay_buffer.n_episodes, \
        f"plan has {len(plan_episodes)} episodes but replay buffer has " \
        f"{replay_buffer.n_episodes}. Are 'plan' and 'input' from the same dataset?"

    # load all rg2ft recordings
    rg2ft_paths = sorted(rg2ft_dir.glob('rg2ft_*.zarr'))
    if len(rg2ft_paths) == 0:
        raise RuntimeError(f"No rg2ft_*.zarr recordings found in {rg2ft_dir}")
    print(f"Found {len(rg2ft_paths)} rg2ft recordings")

    recordings = list()
    for p in rg2ft_paths:
        root = zarr.open(str(p), mode='r')
        timestamp = root['timestamp'][:]
        gripper_ft = root['gripper_ft'][:]
        recordings.append({
            'path': p,
            'timestamp': timestamp,
            'gripper_ft': gripper_ft,
            't_min': timestamp[0],
            't_max': timestamp[-1],
        })

    # match each episode to the rg2ft recording with the largest timestamp overlap,
    # then interpolate the 12-axis F/T onto the episode's per-frame timestamps
    robot0_ft_episodes = list()
    for i, plan_episode in enumerate(plan_episodes):
        episode_timestamps = np.asarray(plan_episode['episode_timestamps'])
        ep_min, ep_max = episode_timestamps[0], episode_timestamps[-1]

        best_rec = None
        best_overlap = 0.0
        for rec in recordings:
            overlap = min(rec['t_max'], ep_max) - max(rec['t_min'], ep_min)
            if overlap > best_overlap:
                best_overlap = overlap
                best_rec = rec

        if best_rec is None:
            print(f"Episode {i}: no overlapping rg2ft recording found, filling zeros")
            robot0_ft = np.zeros((len(episode_timestamps), 12), dtype=np.float32)
        else:
            print(f"Episode {i}: matched {best_rec['path'].name} "
                  f"(overlap={best_overlap:.2f}s)")
            ft_interp = get_interp1d(
                t=best_rec['timestamp'], x=best_rec['gripper_ft'])
            robot0_ft = ft_interp(episode_timestamps).astype(np.float32)

        robot0_ft_episodes.append(robot0_ft)

    robot0_ft_all = np.concatenate(robot0_ft_episodes, axis=0)
    assert robot0_ft_all.shape[0] == replay_buffer.n_steps

    replay_buffer.data['robot0_ft'] = robot0_ft_all

    print(f"Saving merged replay buffer to {output}")
    with zarr.ZipStore(str(output), mode='w') as zip_store:
        replay_buffer.save_to_store(store=zip_store)
    print("Done!")


# %%
if __name__ == "__main__":
    main()
