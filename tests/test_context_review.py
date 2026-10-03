import json
import yaml
import pytest
from pathlib import Path
from diffusion_policy.context.labels import read_rows


def test_gui_segment_save_resume_preserves_automatic_labels(context_base,tmp_path,monkeypatch):
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    base,_=context_base
    output=tmp_path/'labels';output.mkdir()
    automatic=output/'auto_labels.jsonl'
    automatic.write_text(json.dumps(dict(episode_id=0,anchor_index=0,timestamp=float(base.rgb_timestamps[0]),class_id=1,confidence=.9))+'\n')
    original=automatic.read_bytes()
    cfg=yaml.safe_load(Path('context/qwen/config/context_labels.yaml').read_text())
    cfg.update(output_dir=str(output),dataset_overrides=['task=umi_dual_ft',f'task.dataset_path={base.dataset_path}'])
    path=tmp_path/'review.yaml';path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv('CONTEXT_REVIEW_CONFIG',str(path))
    app=AppTest.from_file('tools/review_context_labels.py').run(timeout=30)
    assert not app.exception
    class_buttons=[b.label for b in app.button if ' · class ' in b.label]
    assert class_buttons==[f'{i+1} · class {i}' for i in range(len(cfg['contexts']))]
    app.radio[0].set_value('Segment')
    app.number_input(key='segment_start').set_value(2)
    app.number_input(key='segment_end').set_value(4)
    next(b for b in app.button if b.label=='3 · class 2').click().run()
    assert not app.exception
    next(b for b in app.button if b.label=='S · save').click().run()
    assert not app.exception
    rows=read_rows(output/'reviewed_labels.parquet')
    assert [r['anchor_index'] for r in rows]==[2,3,4]
    assert all(r['reviewed_class_id']==2 and r['is_modified'] for r in rows)
    resumed=AppTest.from_file('tools/review_context_labels.py').run(timeout=30)
    assert len(resumed.session_state.reviews)==3
    next(b for b in resumed.button if b.label.startswith('Mark episode fully')).click().run()
    assert all(r['reviewed_class_id']==2 for r in resumed.session_state.reviews if r['anchor_index'] in [2,3,4])
    assert automatic.read_bytes()==original
