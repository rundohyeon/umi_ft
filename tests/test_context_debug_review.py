import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from diffusion_policy.context.debug_review import DebugRecording
from diffusion_policy.context.ft_features import causal_ft_features
from test_context_rgb_force import canonical_files  # Reuse the canonical-data fixture.


def test_review_alignment_causal_features_and_episode_boundaries(canonical_files):
    recording = DebugRecording(**canonical_files)
    ds = recording.dataset
    assert recording.original_class(3) == -1  # Invalid source labels stay unassigned.
    for episode in (0, 1):
        start, end = recording.bounds(episode)
        for frame in range(start, end):
            wrench, age = recording.causal_force(frame)
            np.testing.assert_array_equal(wrench, ds.wrench[ds.force_index[frame]])
            assert age >= 0
        times, raw, mean, delta = recording.force_episode(episode)
        assert times[0] == ds.force_times[ds.force_starts[episode]]
        history = torch.from_numpy(np.pad(raw[:, :6], ((9, 0), (0, 0)), mode='edge'))[None]
        features = causal_ft_features(history, 'raw_mean_delta')[0].numpy()
        np.testing.assert_allclose(mean[:, :6], features[:, 6:12])
        np.testing.assert_allclose(delta[:, :6], features[:, 12:])
        assert not delta[0].any()  # No average/delta history leaks across episodes.
    assert recording.image(0).shape == (224, 224, 3)
    ds.close()


def test_inclusive_edit_save_reload_and_concurrent_save_protection(canonical_files, tmp_path):
    recording = DebugRecording(**canonical_files)
    source_before = Path(canonical_files['label_path']).read_bytes()
    changes = recording.apply({}, 0, 2, 4, 3)
    assert changes == {2: 3, 3: 3, 4: 3}
    with pytest.raises(ValueError, match='범위'):
        recording.apply(changes, 0, 6, 8, 0)
    # Restore an originally invalid frame to unknown; omit the redundant patch.
    assert recording.apply(changes, 0, 3, 3, -1) == {2: 3, 4: 3}
    output = tmp_path / 'review.json'
    digest = recording.save(output, changes, None)
    restored, loaded_digest = recording.load(output)
    assert (restored, loaded_digest) == (changes, digest)
    assert json.loads(output.read_text())['edits'][1]['valid'] is True
    with pytest.raises(ValueError, match='다른 창'):
        recording.save(output, {}, None)
    assert recording.load(output)[0] == changes
    # Paths are informational: the source hash and frame mapping identify data.
    document = json.loads(output.read_text())
    document['source_labels'] = '/old/computer/labels.npz'
    output.write_text(json.dumps(document))
    assert recording.load(output)[0] == changes
    document['edits'][0]['frame'] = 99
    output.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='metadata'):
        recording.load(output)
    assert Path(canonical_files['label_path']).read_bytes() == source_before
    with pytest.raises(ValueError, match='source dataset'):
        recording.save(Path(canonical_files['force_sidecar_path']) / 'review.json', {}, None)
    recording.dataset.close()


def test_debug_ui_frame_step_range_edit_save_resume_and_episode_reset(canonical_files, tmp_path, monkeypatch):
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    source_before = Path(canonical_files['label_path']).read_bytes()
    output = tmp_path / 'review.json'
    argv = ['tools/debug_context_labels.py', '--episode', '1', '--output', str(output)]
    for flag, name in [('dataset', 'dataset_path'), ('force-sidecar', 'force_sidecar_path'), ('labels', 'label_path')]:
        argv.extend(['--' + flag, canonical_files[name]])
    monkeypatch.setattr(sys, 'argv', argv)
    app = AppTest.from_file('tools/debug_context_labels.py').run(timeout=30)
    assert not app.exception
    assert app.session_state.position == 0
    next(b for b in app.button if b.label == '1 ▶').click().run()
    assert not app.exception
    assert app.session_state.position == 1
    app.slider(key='frame_slider').set_value(5).run()
    assert not app.exception
    assert app.session_state.position == 5
    app.number_input(key='frame_number').set_value(4).run()
    assert not app.exception
    assert app.session_state.position == app.slider(key='frame_slider').value == 4
    app.radio(key='scope').set_value('선택 구간')
    app.number_input(key='range_start').set_value(2)
    app.number_input(key='range_end').set_value(4)
    next(b for b in app.button if b.label == '4 · error').click().run()
    assert not app.exception
    assert app.session_state.edits == {2: 3, 3: 3, 4: 3}
    next(b for b in app.button if b.label == '되돌리기 Z').click().run()
    assert app.session_state.edits == {}
    next(b for b in app.button if b.label == '4 · error').click().run()
    next(b for b in app.button if b.label == '저장 S').click().run()
    assert not app.exception
    assert output.exists()
    resumed = AppTest.from_file('tools/debug_context_labels.py').run(timeout=30)
    assert resumed.session_state.edits == {2: 3, 3: 3, 4: 3}
    app.selectbox(key='episode').set_value(1).run()
    assert not app.exception
    assert app.session_state.position == app.session_state.range_start == app.session_state.range_end == 0
    assert Path(canonical_files['label_path']).read_bytes() == source_before
    next(b for b in app.button if b.label == '재생 / 정지').click().run()
    assert app.session_state.playing
    app.session_state.play_clock -= 10
    app.run()
    assert not app.exception
    assert not app.session_state.playing
    assert app.slider(key='frame_slider').value == app.session_state.position == 7
