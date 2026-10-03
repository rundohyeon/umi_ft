import time

import numpy as np
import pytest
import torch
from torch import nn

from models.rgb_force_context_encoder import RGBForceContextEncoder
from umi.real_world.context_sensor_streams import CameraStream, NativeFTStream
from umi.real_world.rgb_force_context import (
    PHASE_NAMES, ObservationUnavailable, RGBForceContextRuntime,
    TrainingImageTransform, select_context_window,
)


def timestamps():
    anchor = 100.505
    return anchor - np.arange(3, -1, -1) / 60, 100 + np.arange(65) / 100


def test_native_selection_ignores_future_samples_and_matches_frame_stride():
    rt, ft = timestamps()
    ri, fi, timing = select_context_window(rt, ft, now=rt[-1])
    np.testing.assert_array_equal(ri, [0, 3])
    np.testing.assert_array_equal(fi, np.arange(10, 51))
    assert timing['ft_age_s'] == pytest.approx(0.005)
    assert timing['ft_history_span_s'] == pytest.approx(0.4)
    assert timing['rgb_frame_index_gap'] == 3
    assert timing['rgb_selection'] == 'fixed_stride'
    assert 'rgb_target_span_s' not in timing
    # Future values/timestamps never change the native past window.
    ft[51:] += 50
    np.testing.assert_array_equal(select_context_window(rt, ft)[1], fi)


def test_target_span_selects_causal_pair_at_slower_processed_rgb_cadence():
    rt = np.asarray([100.0, 100.029, 100.058, 100.087])
    anchor = float(rt[-1])
    ft = anchor - 0.005 - np.arange(50, -1, -1) / 100

    with pytest.raises(ObservationUnavailable, match='rgb_cadence_mismatch'):
        select_context_window(rt, ft)

    ri, fi, timing = select_context_window(
        rt, ft, rgb_target_span_s=0.05
    )
    np.testing.assert_array_equal(ri, [1, 3])
    np.testing.assert_array_equal(fi, np.arange(10, 51))
    assert timing['rgb_pair_span_s'] == pytest.approx(0.058)
    assert timing['rgb_frame_index_gap'] == 2
    assert timing['rgb_selection'] == 'nearest_target_span'
    assert timing['rgb_target_span_s'] == pytest.approx(0.05)


@pytest.mark.parametrize('problem,expected', [
    ('episode', 'warming_up_ft'), ('stale_rgb', 'stale_or_future_rgb'),
    ('future_rgb', 'stale_or_future_rgb'), ('stale_ft', 'stale_ft_at_rgb_anchor'),
    ('gap', 'gap_in_ft_history'), ('slow_rgb', 'rgb_cadence_mismatch'),
    ('duplicate_ft', 'strictly increasing'), ('nonfinite', 'finite'),
    ('slow_ft', 'ft_cadence_mismatch'),
])
def test_invalid_windows_are_not_classified(problem, expected):
    rt, ft = timestamps()
    kwargs = dict(now=rt[-1])
    if problem == 'episode':
        kwargs['episode_start'] = 100.4
    elif problem == 'stale_rgb':
        kwargs['now'] += 1
    elif problem == 'future_rgb':
        kwargs['now'] -= 0.1
    elif problem == 'stale_ft':
        ft = ft[:49]
    elif problem == 'gap':
        ft = np.delete(ft, [25, 26])
    elif problem == 'slow_rgb':
        rt = rt[-1] - np.arange(3, -1, -1) / 30
    elif problem == 'duplicate_ft':
        ft[25] = ft[24]
    elif problem == 'nonfinite':
        ft[25] = np.nan
    elif problem == 'slow_ft':
        ft = rt[-1] - 0.005 - np.arange(45, -1, -1) / 60
    with pytest.raises(ObservationUnavailable, match=expected):
        select_context_window(rt, ft, **kwargs)


class TinyVision(nn.Module):
    num_features = 768
    pretrained_cfg = dict(mean=(0.48, 0.46, 0.4), std=(0.27, 0.26, 0.28))

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 768)

    def forward(self, rgb):
        return self.proj(rgb.mean((-1, -2)))


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    torch.set_num_threads(2)

    def vision(*_, **kwargs):
        assert kwargs['pretrained'] is False  # Even restoration must not request HF weights.
        return TinyVision()

    monkeypatch.setattr('models.rgb_force_context_encoder.timm.create_model', vision)
    model = RGBForceContextEncoder(pretrained=False, ft_feature_mode='raw_mean_delta').eval()
    model.set_force_statistics(torch.arange(36) / 20, torch.arange(36) / 10 + 0.1)
    payload = dict(schema='context_rgb_force_4state_ft_features_v2',
                   state_dict=model.state_dict(), model_config=model.config,
                   manifest=dict(phase_names=list(PHASE_NAMES)), epoch=122,
                   smoke_test=False, training_config=dict(dataset=dict(rgb_stride=3, max_force_age_s=0.012)))
    path = tmp_path / 'best.pt'
    torch.save(payload, path)
    return path, payload, model


def test_checkpoint_predictions_replay_exactly_and_keep_normalization(checkpoint, tmp_path):
    path, _, original = checkpoint
    runtime = RGBForceContextRuntime(path)
    assert runtime.metadata['epoch'] == 123
    assert runtime.required_ft_history == 41
    assert not runtime.model.training
    torch.testing.assert_close(runtime.model.ft_std, original.ft_std)
    rt, ft = timestamps()
    frames = np.full((4, 224, 224, 3), 125, dtype=np.uint8)
    wrench = np.arange(65 * 12, dtype=np.float32).reshape(65, 12) / 100
    obs, timing = runtime.prepare(rt, frames, ft, wrench, now=rt[-1])
    np.testing.assert_array_equal(obs['robot0_ft_left'], wrench[10:51, :6])
    np.testing.assert_array_equal(obs['robot0_ft_right'], wrench[10:51, 6:])
    result = runtime.predict(obs)
    with torch.no_grad():
        expected = original(dict(camera0_rgb=torch.from_numpy(frames[[0, 3]].transpose(0, 3, 1, 2))[None].float() / 255,
                                 robot0_ft_left=torch.from_numpy(wrench[10:51, :6])[None],
                                 robot0_ft_right=torch.from_numpy(wrench[10:51, 6:])[None]))[0]
    np.testing.assert_allclose(result['logits'], expected.numpy(), atol=1e-6)
    assert sum(result['probabilities']) == pytest.approx(1)
    assert result['phase'] == PHASE_NAMES[result['class_id']]
    np.savez_compressed(tmp_path / 'input.npz', **obs, **timing)
    with np.load(tmp_path / 'input.npz', allow_pickle=False) as data:
        replay = runtime.predict({k: data[k] for k in runtime.model.input_keys})
    assert replay['logits'] == result['logits']
    with pytest.raises(ValueError, match='uint8'):
        runtime.predict(dict(obs, camera0_rgb=obs['camera0_rgb'].astype(np.float32) / 255))
    obs['robot0_ft_right'][0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        runtime.predict(obs)


@pytest.mark.parametrize('problem,expected', [
    ('smoke', 'Smoke-test'), ('schema', 'Not an RGBForce'),
    ('class_order', 'classes'), ('normalization', 'normalization'),
])
def test_reject_incompatible_checkpoints(checkpoint, problem, expected):
    path, payload, _ = checkpoint
    if problem == 'smoke':
        payload['smoke_test'] = True
    elif problem == 'schema':
        payload['schema'] = 'other_observer'
    elif problem == 'class_order':
        payload['manifest']['phase_names'][2] = 'finish'
    else:
        payload['state_dict']['ft_std'][0] = 0
    torch.save(payload, path)
    with pytest.raises(ValueError, match=expected):
        RGBForceContextRuntime(path)


def test_preprocessing_matches_training_order_without_mutating_capture():
    from pathlib import Path
    from diffusion_policy.common.cv2_util import get_image_transform
    from umi.common.cv_util import draw_predefined_mask
    root = Path(__file__).resolve().parents[1]
    transform = TrainingImageTransform(root / 'slam_pipeline_latest/calibration/aruco_config.yaml')
    bgr = np.full((480, 640, 3), (20, 70, 120), dtype=np.uint8)
    original = bgr.copy()
    actual = transform(bgr)
    expected = draw_predefined_mask(bgr.copy(), mirror=False, gripper=True, finger=False, use_aa=False)
    expected = get_image_transform((640, 480), (224, 224), bgr_to_rgb=True)(expected)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(bgr, original)
    np.testing.assert_array_equal(actual[20, 112], [120, 70, 20])


class FakeModbus:
    def __init__(self, *_args, **_kwargs):
        self.reads = 0
        self.closed = False

    def connect(self):
        pass

    def write_register(self, *args, **kwargs):
        raise AssertionError('Observer must never write a Modbus register')

    def read_holding_registers(self, address, count):
        assert (address, count) == (257, 26)
        self.reads += 1
        registers = [0] * count
        registers[2], registers[5], registers[11] = (-12) & 0xffff, 25, 34
        return registers

    def close(self):
        self.closed = True


def test_readonly_native_sensor_units_and_software_bias():
    with NativeFTStream('fake', client_factory=FakeModbus, receive_latency=0) as stream:
        bias = stream.calibrate(dict(sample_count=5, timeout_s=1))
        np.testing.assert_allclose(bias['bias_12d'][[0, 3, 6]], [-1.2, 0.25, 3.4])
        ts, values = stream.snapshot()
        assert len(ts) >= 5 and np.all(np.diff(ts) > 0)
        np.testing.assert_allclose(np.stack(values) - bias['bias_12d'], 0, atol=1e-10)
    assert stream.client.closed


def test_sensor_read_failure_is_propagated_and_connection_closed():
    class BrokenModbus(FakeModbus):
        def read_holding_registers(self, *args, **kwargs):
            raise OSError('disconnected')

    with NativeFTStream('fake', client_factory=BrokenModbus) as stream:
        stream._thread.join(timeout=1)
        with pytest.raises(RuntimeError, match='disconnected'):
            stream.snapshot()
    assert stream.client.closed


def test_camera_keeps_capturing_independent_of_inference_and_releases():
    class FakeCapture:
        def __init__(self, *args):
            self.released = False

        def isOpened(self):
            return True

        def set(self, *args):
            return True

        def read(self):
            time.sleep(1 / 60)
            return True, np.zeros((224, 224, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    capture = FakeCapture()
    with CameraStream('fake', resolution=(224, 224), capture_factory=lambda *_: capture) as stream:
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            ts, frames = stream.snapshot()
            if len(ts) >= 4:
                break
            time.sleep(0.01)
        assert len(ts) >= 4 and np.all(np.diff(ts) > 0)
        assert frames[-1].shape == (224, 224, 3)
    assert capture.released


def test_live_loop_logs_valid_and_warmup_records_and_replay_inputs(checkpoint, tmp_path, monkeypatch):
    """Exercise the real entry point with simulated streams and a real tiny model."""
    import argparse
    import json
    from pathlib import Path
    import eval_real_context_rgb_force as cli

    class Stream:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class FT(Stream):
        def calibrate(self, config):
            return dict(bias_12d=np.ones(12))

        def snapshot(self):
            # Fixed 100 Hz grid, including samples later than the camera anchor.
            end = np.floor(time.time() * 100) / 100
            return end - np.arange(99, -1, -1) / 100, list(np.ones((100, 12)))

    class Camera(Stream):
        def snapshot(self):
            return time.time() - 0.125 - np.arange(15, -1, -1) / 60, [np.full((224, 224, 3), 120, np.uint8)] * 16

    monkeypatch.setattr('umi.real_world.context_sensor_streams.NativeFTStream', FT)
    monkeypatch.setattr('umi.real_world.context_sensor_streams.CameraStream', Camera)
    monkeypatch.setattr(cli, 'TrainingImageTransform', lambda _: lambda img: img)
    path, _, _ = checkpoint
    runtime = RGBForceContextRuntime(path)
    output = tmp_path / 'live'
    args = argparse.Namespace(config=Path(cli.ROOT / 'example/eval_context_rgb_force.yaml'),
                              camera='simulated', gripper_ip=None, output=output,
                              checkpoint=path, rate=20, max_seconds=1.3,
                              save_inputs=True, headless=True)
    cli.run_live(args, runtime)
    records = [json.loads(line) for line in (output / 'predictions.jsonl').read_text().splitlines()]
    assert any(not r['valid'] and 'warming_up' in r['status'] for r in records)
    valid = [r for r in records if r['valid']]
    assert valid
    with np.load(output / valid[-1]['input_file'], allow_pickle=False) as data:
        obs = {k: data[k] for k in runtime.model.input_keys}
        assert np.count_nonzero(obs['robot0_ft_left']) == 0  # Startup bias subtracted once.
        assert data['ft_timestamps'][-1] <= data['rgb_timestamps'][-1]
        replay = runtime.predict(obs)
    assert replay['logits'] == valid[-1]['logits']
    metadata = json.loads((output / 'metadata.json').read_text())
    assert metadata['startup_bias']['bias_12d'] == [1] * 12
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['exit_status'] == 'completed'
    assert summary['valid_predictions'] == len(valid)
