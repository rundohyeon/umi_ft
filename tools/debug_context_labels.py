#!/usr/bin/env python3
"""streamlit run tools/debug_context_labels.py --server.port 8502"""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

from diffusion_policy.context.debug_review import DebugRecording
from diffusion_policy.context.canonical_dataset import PHASE_NAMES


@st.cache_resource
def load_recording(dataset, force, labels):
    return DebugRecording(dataset, force, labels)


def install_keyboard():
    keys = {'ArrowLeft': '◀ 1', 'ArrowRight': '1 ▶', ' ': '재생 / 정지',
            '[': '시작 지정 [', ']': '끝 지정 ]', 's': '저장 S', 'u': '미지정 U', 'z': '되돌리기 Z'}
    keys.update({str(i + 1): f'{i + 1} · {name}' for i, name in enumerate(PHASE_NAMES)})
    components.html('''<script>
    const doc = window.parent.document;
    if (window.parent.contextDebugKeys) doc.removeEventListener('keydown', window.parent.contextDebugKeys);
    window.parent.contextDebugKeys = event => {
      if (event.ctrlKey || event.metaKey || event.altKey ||
          event.target.closest('input,textarea,select,[contenteditable=true],[role=combobox]')) return;
      const label = KEYS[event.key.length === 1 ? event.key.toLowerCase() : event.key];
      const button = Array.from(doc.querySelectorAll('button')).find(b => b.innerText.trim() === label);
      if (button && !button.disabled) { event.preventDefault(); button.click(); }
    };
    doc.addEventListener('keydown', window.parent.contextDebugKeys);
    </script>'''.replace('KEYS', json.dumps(keys)), height=0)


def main():
    parser = argparse.ArgumentParser()
    data = ROOT / 'three_dataset'
    parser.add_argument('--dataset', default=str(data / 'dataset.zarr.zip'))
    parser.add_argument('--force-sidecar', default=str(data / 'dataset_force_sidecar.zarr'))
    parser.add_argument('--labels', default=str(data / 'canonical_context_supervision_v2_4state_284.npz'))
    parser.add_argument('--output', default=str(ROOT / 'outputs/context_label_debug/review.json'))
    parser.add_argument('--episode', type=int, default=194, help='One-based initial episode')
    args = parser.parse_args()
    st.set_page_config(page_title='RGB · F/T label debug', layout='wide')
    st.title('RGB · F/T 프레임 디버거')
    st.caption('학습 RGB 224×224 · 센서 좌표계 F/T · 에피소드 번호는 1부터, 프레임 번호는 0부터')
    try:
        recording = load_recording(args.dataset, args.force_sidecar, args.labels)
    except Exception as exc:
        st.error(str(exc))
        st.stop()
    ds, state = recording.dataset, st.session_state
    output = Path(args.output).expanduser().resolve()
    identity = (str(output), recording.label_digest)
    if state.get('identity') != identity:
        try:
            edits, digest = recording.load(output)
        except Exception as exc:
            st.error(str(exc))
            st.stop()
        state.update(identity=identity, edits=edits, saved_edits=dict(edits), save_digest=digest,
                     undo=[], position=0, playing=False, episode_seen=None, message='')
    episode = st.sidebar.selectbox('에피소드', range(len(ds.rgb_ends)),
        index=min(max(args.episode - 1, 0), len(ds.rgb_ends) - 1),
        format_func=lambda e: f'EP{e+1:03d} · {recording.episode_names[e]}', key='episode')
    start, end = recording.bounds(episode)
    times = ds.rgb_times[start:end]
    if state.episode_seen != episode:
        state.update(position=0, playing=False, episode_seen=episode, range_start=0, range_end=0)

    def reset_play_clock():
        state.play_clock, state.play_time = time.monotonic(), float(times[state.position])

    speed = st.sidebar.select_slider('재생 배속', options=[0.1, 0.25, 0.5, 1.0, 2.0], value=0.5,
                                    on_change=reset_play_clock)
    radius = st.sidebar.slider('그래프 구간 ±초', 0.25, 5.0, 2.0, 0.25)
    sides = st.sidebar.multiselect('표시할 손가락', ['right', 'left'], default=['right'])
    feature_mode = st.sidebar.radio('F/T 그래프', ['원본', '5샘플 평균', '평균 변화량'])
    show_labels = st.sidebar.checkbox('기존 / 수정 라벨 표시', value=True)
    st.sidebar.caption('← / →: 1프레임 · Space: 재생/정지\n\n1–4: 라벨 · U: 미지정 · [: 시작 · ]: 끝 · S: 저장 · Z: 되돌리기')
    st.sidebar.caption(f'수정 저장: {output}')
    st.sidebar.caption('재생은 시간 기준 미리보기입니다. 모든 프레임을 확인하려면 좌우 키 또는 0.1배속을 사용하세요.')
    st.sidebar.caption('원본은 학습용 bias 보정 F/T입니다. 평균은 과거 5샘플, 변화량은 현재 평균 − 5샘플 전 평균입니다. 단위는 N / Nm이며 초당 변화율이 아닙니다.')
    install_keyboard()

    def seek(position):
        state.position = int(np.clip(position, 0, len(times) - 1))
        state.playing = False

    def move(delta):
        seek(state.position + delta)

    def toggle():
        state.playing = not state.playing
        state.play_clock, state.play_time = time.monotonic(), float(times[state.position])

    def mark(key):
        state[key] = state.position
        state.playing = False

    def correct(class_id):
        state.playing = False
        a, b = ((state.position, state.position) if state.scope == '현재 프레임'
                else (state.range_start, state.range_end))
        try:
            updated = recording.apply(state.edits, episode, int(a), int(b), class_id)
            if updated != state.edits:
                state.undo = (state.undo + [dict(state.edits)])[-30:]
                state.edits = updated
            state.message = f'프레임 {a}–{b} 수정됨. 저장 S로 파일에 기록하세요.'
        except ValueError as exc:
            state.message = str(exc)

    def undo():
        state.playing = False
        if state.undo:
            state.edits = state.undo[-1]
            state.undo = state.undo[:-1]

    def save():
        state.playing = False
        try:
            state.save_digest = recording.save(output, state.edits, state.save_digest)
            state.saved_edits = dict(state.edits)
            state.message = f'{len(state.edits)}개 프레임 수정 저장 완료: {output}'
        except ValueError as exc:
            state.message = str(exc)

    controls = st.columns(7)
    controls[0].button('◀ 10', on_click=move, args=(-10,))
    controls[1].button('◀ 1', on_click=move, args=(-1,))
    controls[2].button('재생 / 정지', on_click=toggle)
    controls[3].button('1 ▶', on_click=move, args=(1,))
    controls[4].button('10 ▶', on_click=move, args=(10,))
    controls[5].button('되돌리기 Z', on_click=undo, disabled=not state.undo)
    controls[6].button('저장 S', on_click=save, type='primary')
    edit_columns = st.columns([2, 1, 1, 1, 1])
    edit_columns[0].radio('수정 범위 (양 끝 포함)', ['현재 프레임', '선택 구간'], horizontal=True, key='scope')
    edit_columns[1].button('시작 지정 [', on_click=mark, args=('range_start',))
    edit_columns[2].number_input('시작 프레임', 0, len(times) - 1, key='range_start')
    edit_columns[3].button('끝 지정 ]', on_click=mark, args=('range_end',))
    edit_columns[4].number_input('끝 프레임', 0, len(times) - 1, key='range_end')
    buttons = st.columns(5)
    for i, name in enumerate(PHASE_NAMES):
        buttons[i].button(f'{i+1} · {name}', on_click=correct, args=(i,))
    buttons[4].button('미지정 U', on_click=correct, args=(-1,))
    if state.edits != state.saved_edits:
        st.warning('저장하지 않은 수정이 있습니다. 저장 S를 누르세요.')
    if state.message:
        st.info(state.message)

    @st.fragment(run_every=0.1 if state.playing else None)
    def playback():
        if state.playing:
            target = state.play_time + (time.monotonic() - state.play_clock) * speed
            state.position = min(len(times) - 1, max(0, int(np.searchsorted(times, target, side='right') - 1)))
            if state.position == len(times) - 1:
                state.playing = False
                st.rerun()
        state.frame_slider = state.position
        state.frame_number = state.position

        def slide(key):
            seek(state[key])
            state.seek_requested = True

        seek_columns = st.columns([5, 1])
        seek_columns[0].slider('프레임 이동', 0, max(1, len(times) - 1), key='frame_slider',
                               on_change=slide, args=('frame_slider',), disabled=len(times) == 1)
        seek_columns[1].number_input('프레임 번호', 0, len(times) - 1, key='frame_number',
                                     on_change=slide, args=('frame_number',))
        if state.pop('seek_requested', False):
            st.rerun()
        local = state.position
        frame = start + local
        now, zero = times[local], times[0]
        wrench, age = recording.causal_force(frame)
        left, right = st.columns([1, 2])
        with left:
            st.image(recording.image(frame), width=420)
            st.write(f'**EP{episode+1:03d} · 프레임 {local}/{len(times)-1} · {now-zero:.3f}초**')
            st.caption('재생 중' if state.playing else '정지')
            st.caption(f'전체 프레임 {frame} · RGB timestamp {now:.6f}')
            if show_labels:
                original = recording.original_class(frame)
                edited = state.edits.get(frame, original)
                name = lambda v: PHASE_NAMES[v] if v >= 0 else '미지정 (학습 제외)'
                st.write(f'기존: **{name(original)}** → 검토: **{name(edited)}**')
                source_id = int(ds.label_source[frame])
                st.caption(f'원본 라벨 출처: {ds.source_names[source_id] if 0 <= source_id < len(ds.source_names) else "미지정"}')
            if wrench is None:
                st.warning('이 RGB 프레임 이전에 측정된 F/T가 없습니다.')
            else:
                st.caption(f'수치 표: RGB 이전 최근 F/T · 시차 {age*1000:.2f} ms')
                if age > 0.012:
                    st.warning('F/T 시차가 학습의 유효 기준 12 ms를 초과합니다.')
                table = pd.DataFrame(wrench.reshape(2, 6).T, index=['Fx [N]', 'Fy [N]', 'Fz [N]', 'Tx [Nm]', 'Ty [Nm]', 'Tz [Nm]'], columns=['left', 'right'])
                table.loc['|F| [N]'] = [np.linalg.norm(wrench[:3]), np.linalg.norm(wrench[6:9])]
                st.dataframe(table, width='stretch', column_config={
                    side: st.column_config.NumberColumn(format='%.3f') for side in ('left', 'right')})
        with right:
            ft_times, raw, mean, delta = recording.force_episode(episode)
            values = {'원본': raw, '5샘플 평균': mean, '평균 변화량': delta}[feature_mode]
            prefix = 'Δ' if feature_mode == '평균 변화량' else ''
            select = (ft_times >= now - radius) & (ft_times <= now + radius)
            plot = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=[f'{prefix}Force [N]', f'{prefix}Torque [Nm]'])
            for side in sides:
                offset = 0 if side == 'left' else 6
                for row, axes in ((1, ('Fx', 'Fy', 'Fz')), (2, ('Tx', 'Ty', 'Tz'))):
                    begin = offset + (row - 1) * 3
                    for axis, label in enumerate(axes):
                        plot.add_trace(go.Scatter(x=ft_times[select] - zero, y=values[select, begin + axis],
                            name=f'{side} {prefix}{label}', line=dict(color=['#4c9fff', '#ffac55', '#b88aff'][axis], dash='dot' if side == 'left' else 'solid')), row=row, col=1)
                    plot.add_trace(go.Scatter(x=ft_times[select] - zero, y=np.linalg.norm(values[select, begin:begin+3], axis=1),
                        name=f'{side} |{prefix}{"F" if row == 1 else "T"}|', line=dict(color='#32ccad', dash='dot' if side == 'left' else 'solid')), row=row, col=1)
            plot.add_vline(x=float(now - zero), line_color='#ff4066')
            plot.update_xaxes(range=[max(float(ft_times[0] - zero), float(now - zero - radius)), float(now - zero + radius)])
            plot.update_layout(height=570, title=f'{feature_mode} · 세로선 = 현재 RGB 프레임', margin=dict(l=30, r=15, t=65, b=30))
            st.plotly_chart(plot, use_container_width=True, key='force_plot')
            st.caption('그래프는 앞뒤 구간을 함께 표시합니다. 수치 표는 미래 샘플을 사용하지 않습니다. 범례를 클릭하면 축을 숨길 수 있습니다.')
        if show_labels:
            original = np.where(ds.valid[start:end], ds.targets[start:end], -1)
            edited = np.array([state.edits.get(start + i, int(v)) for i, v in enumerate(original)])
            timeline = go.Figure()
            timeline.add_trace(go.Scatter(x=times - zero, y=original, name='기존', line_shape='hv'))
            if not np.array_equal(original, edited):
                timeline.add_trace(go.Scatter(x=times - zero, y=edited, name='수정', line_shape='hv', line=dict(dash='dot')))
            timeline.add_vline(x=float(now - zero), line_color='#ff4066')
            timeline.update_layout(height=200, margin=dict(l=30, r=15, t=20, b=30), xaxis_title='에피소드 시간 [s]',
                yaxis=dict(tickvals=[-1, 0, 1, 2, 3], ticktext=['미지정'] + list(PHASE_NAMES), range=[-1.3, 3.3]))
            st.plotly_chart(timeline, use_container_width=True, key='label_timeline')

    playback()


if __name__ == '__main__':
    main()
