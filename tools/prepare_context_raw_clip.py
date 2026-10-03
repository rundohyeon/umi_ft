#!/usr/bin/env python
"""Prepare a full raw video and native F/T CSV for labeling without SLAM poses."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import av
import cv2
import numpy as np
import pandas as pd
from diffusion_policy.context.raw_recording import estimate_width_sync
from umi.common.cv_util import get_image_transform, draw_predefined_mask, inpaint_tag


def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    video, csv_path, output = Path(args.video).resolve(), Path(args.csv).resolve(), Path(args.output)
    if output.exists() or output.suffix != '.npz':
        raise ValueError('Choose a new .npz output file')
    # Repository-generated tag detections, not an external downloaded pickle.
    tag_path = video.parent/'tag_detection.pkl'
    tags = pickle.loads(tag_path.read_bytes())
    pairs = [(r['time'], float(r['tag_dict'][1]['tvec'][0]-r['tag_dict'][0]['tvec'][0]))
             for r in tags if 0 in r['tag_dict'] and 1 in r['tag_dict']]
    if len(pairs) < 30:
        raise ValueError('Too few paired gripper tags for synchronization')
    vt, vw = np.array(pairs).T
    csv = pd.read_csv(csv_path)
    times = csv['timestamp'].to_numpy(dtype=float)
    ct = times-times[0]
    width = csv['width_m'].to_numpy(dtype=float)
    sync = estimate_width_sync(vt, vw, ct, width)
    ft_times = ct+sync['csv_relative_to_video_shift_s']
    columns = [f'{axis}_{side}' for side in ('l', 'r') for axis in ('fx','fy','fz','tx','ty','tz')]
    wrench = csv[columns].to_numpy(dtype=float)
    if not np.isfinite(wrench).all():
        raise ValueError('Nonfinite native wrench samples')
    # CSV is already software-tared. Remove only residual bias from an unloaded
    # pre-video interval; never re-add ft_offset.json or rotate sensor axes.
    baseline = (ct <= .5) & (ft_times < 0)
    if baseline.sum() < 30:
        raise ValueError('Insufficient pre-video samples for residual force bias')
    bias = np.median(wrench[baseline], axis=0)
    if np.max(np.linalg.norm((wrench[baseline]-bias)[:, 6:9], axis=1)) > .3:
        raise ValueError('Pre-video baseline is not sufficiently quiet')
    wrench = wrench-bias
    cv2.setNumThreads(1)
    frames, rgb_times = [], []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        resize = get_image_transform((stream.width, stream.height), (224,224))
        first_time = None
        for i, frame in enumerate(container.decode(stream)):
            timestamp = float(frame.pts*stream.time_base)
            if first_time is None:
                first_time = timestamp
            timestamp -= first_time
            if i >= len(tags) or tags[i]['frame_idx'] != i or abs(tags[i]['time']-timestamp) > .0001:
                raise ValueError('Video/tag clocks disagree')
            img = frame.to_ndarray(format='rgb24')
            for tag in tags[i]['tag_dict'].values():
                img = inpaint_tag(img, tag['corners'])
            img = draw_predefined_mask(img, color=(0,0,0), mirror=False, gripper=True, finger=False)
            frames.append(resize(img))
            rgb_times.append(timestamp)
    if len(frames) != len(tags):
        raise ValueError('Video/tag frame counts disagree')
    rgb_times = np.array(rgb_times)
    indices = np.searchsorted(ft_times, rgb_times, side='right')-1
    if np.any(indices < 0):
        raise ValueError('Some RGB frames have no preceding sensor measurement')
    ages = rgb_times-ft_times[indices]
    if ages.max() > .05:
        raise ValueError('Some RGB frames have stale sensor measurements')
    metadata = dict(recording_name=Path(args.video).stem, video=identity(video),
                    csv=identity(csv_path), tag_detections=identity(tag_path),
                    sync=sync, frame_count=len(frames), fps=1/float(np.median(np.diff(rgb_times))),
                    source_video_frame_range=[0,len(frames)],
                    force_bias_removed=True, force_bias_method='median of first 0.5s CSV, entirely before video',
                    force_residual_bias_12d=bias.tolist(), force_axes='native sensor, no coordinate transform',
                    force_input='software-tared CSV, ft_offset.json not re-added',
                    max_causal_sensor_age_s=float(ages.max()),
                    tcp_pose_available=False, gripper_width_source='latest causal CSV measurement',
                    image_processing='UMI tag inpaint, gripper mask, center crop/resize 224x224; mirrors retained',
                    timestamp_convention='video-relative seconds; sensor time = CSV time - CSV first time + estimated shift')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        np.savez_compressed(stream, rgb=np.array(frames), rgb_timestamp_s=rgb_times,
                            wrench_timestamp_s=ft_times, wrench_left=wrench[:,:6], wrench_right=wrench[:,6:],
                            gripper_width_m=width[indices,None], metadata_json=json.dumps(metadata))
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
