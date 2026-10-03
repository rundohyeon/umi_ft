"""Read a prepared RGB/native-force clip for offline labeling without SLAM poses."""
import json
from pathlib import Path
import numpy as np


class RawLabelRecording:
    def __init__(self, path):
        self.dataset_path = str(Path(path).resolve())
        self.force_sidecar_path = None
        with np.load(self.dataset_path, allow_pickle=False) as data:
            self.rgb = data['rgb']
            self.rgb_timestamps = data['rgb_timestamp_s']
            self.ft_left = data['wrench_left']
            self.ft_right = data['wrench_right']
            self.ft_left_timestamps = data['wrench_timestamp_s']
            self.gripper_width = data['gripper_width_m']
            self.metadata = json.loads(str(data['metadata_json']))
        self.ft_right_timestamps = self.ft_left_timestamps
        for times in (self.rgb_timestamps, self.ft_left_timestamps):
            if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError('Recording requires finite, strictly increasing clocks')
        if (self.rgb.dtype != np.uint8 or self.rgb.shape != (len(self.rgb_timestamps), 224, 224, 3)
                or self.gripper_width.shape != (len(self.rgb_timestamps), 1)):
            raise ValueError('Invalid RGB/width recording shapes')
        for wrench in (self.ft_left, self.ft_right):
            if wrench.shape != (len(self.ft_left_timestamps), 6) or not np.isfinite(wrench).all():
                raise ValueError('Invalid native wrench recording')
        if not np.isfinite(self.gripper_width).all():
            raise ValueError('Invalid gripper width')
        self.rgb_episode_ends = np.array([len(self.rgb_timestamps)])
        self.ft_left_episode_ends = self.ft_right_episode_ends = np.array([len(self.ft_left_timestamps)])
        self.ft_bias_removed = bool(self.metadata['force_bias_removed'])
        self.data_keys = {'rgb': 'rgb'}

    def _get_rgb_array(self):
        return self.rgb

    def close(self):
        pass


def estimate_width_sync(video_times, video_widths, csv_times, csv_widths,
                        min_shift=-4., max_shift=2., step=.002):
    """Fit t_video=t_csv_relative+shift using independent gripper motion.

    The tag separation and sensor width can have different scales/zero points.
    Force or task-phase predictions are never used to choose the alignment.
    """
    vt, vw, ct, cw = [np.asarray(x, dtype=float) for x in
                      (video_times, video_widths, csv_times, csv_widths)]
    if (vt.ndim != 1 or ct.ndim != 1 or vt.shape != vw.shape or ct.shape != cw.shape
            or len(vt) < 30 or len(ct) < 30 or any(not np.isfinite(x).all() for x in (vt, vw, ct, cw))
            or np.any(np.diff(vt) <= 0) or np.any(np.diff(ct) <= 0)
            or np.ptp(vw) < .02 or np.ptp(cw) < .02 or not min_shift < max_shift or step <= 0):
        raise ValueError('Insufficient gripper motion or invalid synchronization inputs')
    fits = []
    for shift in np.arange(min_shift, max_shift+step/2, step):
        mask = (vt-shift >= ct[0]) & (vt-shift <= ct[-1])
        if mask.sum() < max(30, .9*len(vt)):
            continue
        observed = np.interp(vt[mask]-shift, ct, cw)
        if np.std(observed) < 1e-5 or np.std(vw[mask]) < 1e-5:
            continue
        fits.append((float(np.corrcoef(observed, vw[mask])[0, 1]), float(shift)))
    if not fits:
        raise ValueError('No overlapping synchronization candidates')
    correlation, shift = max(fits)
    if correlation < .95 or shift <= min_shift+step or shift >= max_shift-step:
        raise ValueError('Gripper synchronization fit is weak or at the search boundary')
    distant = [c for c, s in fits if abs(s-shift) > .2]
    if distant and max(distant) > correlation-.01:
        raise ValueError('Gripper synchronization is ambiguous')
    near = [s for c, s in fits if c >= correlation-.001]
    return dict(method='tag_separation_vs_sensor_width_correlation',
                csv_relative_to_video_shift_s=round(shift, 6), correlation=correlation,
                paired_video_samples=len(vt), search_step_s=step,
                near_optimal_shift_range_s=[round(min(near), 6), round(max(near), 6)],
                status='estimated_from_gripper_motion_not_hardware_verified')
