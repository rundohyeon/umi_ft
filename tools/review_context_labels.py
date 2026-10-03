#!/usr/bin/env python
"""Run: streamlit run tools/review_context_labels.py"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import os
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from diffusion_policy.context.data import load_base, episode_bounds, validate_label_alignment
from diffusion_policy.context.labels import definitions, read_rows, LabelIndex, review_segment, save_review, atomic_json
from diffusion_policy.context.evaluation import segments


@st.cache_resource
def load_recording(config_path):
    cfg,digest=definitions(config_path)
    base,_=load_base(cfg['policy_config'],cfg.get('dataset_overrides',[]))
    return base,cfg,digest


def install_keyboard(num_classes):
    # Browser-only bridge to existing buttons. No network/CDN or keyboard package.
    labels={' ':'Play / pause','ArrowLeft':'Previous frame','ArrowRight':'Next frame',
            'u':'U · unknown','s':'S · save'}
    labels.update({str(i+1):f'{i+1} · class {i}' for i in range(min(num_classes,9))})
    script='''<script>
    const doc = window.parent.document;
    if (window.parent.contextReviewKeys) doc.removeEventListener('keydown',window.parent.contextReviewKeys);
    window.parent.contextReviewKeys = function(event) {
      const target = event.target;
      if (target.closest('input,textarea,select,[contenteditable="true"],[role="textbox"],[role="combobox"]') || event.ctrlKey || event.metaKey || event.altKey) return;
      const labels = __KEY_LABELS__;
      const label = labels[event.key.length === 1 ? event.key.toLowerCase() : event.key];
      if (!label) return;
      const button = Array.from(doc.querySelectorAll('button')).find(b => b.innerText.trim() === label);
      if (button && !button.disabled) { event.preventDefault(); button.click(); }
    };
    doc.addEventListener('keydown',window.parent.contextReviewKeys);
    </script>'''
    components.html(script.replace('__KEY_LABELS__',json.dumps(labels)),height=0)


def main():
    st.set_page_config(page_title='Context label review',layout='wide')
    st.title('Context label review')
    config_path=st.sidebar.text_input('Context configuration',os.environ.get('CONTEXT_REVIEW_CONFIG','context/qwen/config/context_labels.yaml'))
    try: base,cfg,digest=load_recording(config_path)
    except Exception as exc: st.error(str(exc)); return
    num_classes=len(cfg['contexts'])
    output=Path(st.sidebar.text_input('Label directory',cfg['output_dir']))
    try:
        auto=read_rows(output/'auto_labels.jsonl',num_classes)
        validate_label_alignment(base,auto)
    except Exception as exc: st.error(str(exc)); return
    from diffusion_policy.context.data import protect_raw_data
    from diffusion_policy.context.labels import validate_definition_version
    try:
        protect_raw_data(base,output)
        validate_definition_version(auto,digest)
    except ValueError as exc:
        st.error(str(exc));return
    identity=str(output.resolve())+digest
    if st.session_state.get('review_directory') != identity:
        st.session_state.review_directory=identity
        st.session_state.reviews=read_rows(output/'reviewed_labels.parquet',num_classes)
        validate_label_alignment(base,st.session_state.reviews)
        validate_definition_version(st.session_state.reviews,digest)
        progress=output/'review_progress.json'
        st.session_state.progress=json.loads(progress.read_text()) if progress.exists() else {'fully_reviewed_episodes':[]}
        st.session_state.frame=int(st.session_state.progress.get('frame',0))
        st.session_state.last_episode=int(st.session_state.progress.get('episode',0))
        st.session_state.playing=False
        st.session_state.dirty=False
        st.session_state.loaded_review_mtime=(output/'reviewed_labels.parquet').stat().st_mtime_ns if (output/'reviewed_labels.parquet').exists() else None
    index=LabelIndex(auto,num_classes=num_classes)
    episode=st.sidebar.selectbox('Episode',range(len(base.rgb_episode_ends)),index=int(st.session_state.progress.get('episode',0)))
    lo,hi=episode_bounds(base,episode); ts=base.rgb_timestamps[lo:hi]
    if st.session_state.get('last_episode') != episode:
        st.session_state.frame=0; st.session_state.playing=False; st.session_state.last_episode=episode
    st.session_state.frame=min(len(ts)-1,st.session_state.frame)
    camera=st.sidebar.selectbox('Camera',[base.data_keys['rgb']])
    speed=st.sidebar.select_slider('Playback speed',options=[.25,.5,1.,2.,4.],value=1.)
    radius=st.sidebar.slider('Sensor window ± seconds',.5,5.,2.,.5)
    threshold=st.sidebar.slider('Low confidence threshold',0.,1.,.9)
    short_duration=st.sidebar.number_input('Short segment threshold (seconds)',min_value=.01,value=.15)
    st.sidebar.caption(f'Space: play/pause · arrows: frames · 1–{min(num_classes,9)}: classes · U: unknown · S: save. Shortcuts pause while typing.')
    if st.session_state.dirty: st.warning('Unsaved review changes. Press S or Save before closing.')
    def seek(frame):
        st.session_state.frame=int(np.clip(frame,0,len(ts)-1))
        st.session_state.clock=time.monotonic()
        st.session_state.play_timestamp=float(ts[st.session_state.frame])
    def toggle():
        seek(st.session_state.frame);st.session_state.playing=not st.session_state.playing
    def move(delta): seek(st.session_state.frame+delta)
    controls=st.columns(3)
    controls[0].button('Previous frame',on_click=move,args=(-1,))
    controls[1].button('Play / pause',on_click=toggle)
    controls[2].button('Next frame',on_click=move,args=(1,))
    auto_ids=np.array([index.automatic(episode,i)['class_id'] if index.automatic(episode,i) else -1 for i in range(len(ts))])
    confidence=np.array([index.automatic(episode,i)['confidence'] if index.automatic(episode,i) else 0 for i in range(len(ts))])
    changes=np.flatnonzero(auto_ids[1:]!=auto_ids[:-1])+1
    def jump(direction,low=False):
        candidates=np.flatnonzero(confidence<threshold) if low else changes
        candidates=candidates[candidates>st.session_state.frame] if direction>0 else candidates[candidates<st.session_state.frame]
        if len(candidates): seek(candidates[0] if direction>0 else candidates[-1])
    jumps=st.columns(3)
    jumps[0].button('Previous transition',on_click=jump,args=(-1,))
    jumps[1].button('Next transition',on_click=jump,args=(1,))
    jumps[2].button('Next low confidence',on_click=jump,args=(1,True))
    st.radio('Correction scope',['Current frame','Segment'],horizontal=True,key='correction_scope')
    segment_columns=st.columns(4)
    def mark(which): st.session_state[which]=st.session_state.frame
    segment_columns[0].button('Mark segment start',on_click=mark,args=('segment_start',))
    segment_columns[1].button('Mark segment end',on_click=mark,args=('segment_end',))
    for k in ('segment_start','segment_end'):
        st.session_state[k]=min(st.session_state.get(k,0),len(ts)-1)
    segment_columns[2].number_input('Start frame',0,len(ts)-1,key='segment_start')
    segment_columns[3].number_input('End frame',0,len(ts)-1,key='segment_end')
    def correct(label=None,whole=False):
        a,b=(0,len(ts)-1) if whole else ((st.session_state.frame,st.session_state.frame) if st.session_state.correction_scope=='Current frame' else (st.session_state.segment_start,st.session_state.segment_end))
        if a>b:
            st.session_state.review_error='Segment start must precede end';return
        st.session_state.reviews=review_segment(st.session_state.reviews,index,episode,a,b,ts,label,definition_hash=digest)
        st.session_state.dirty=True
        if whole and episode not in st.session_state.progress['fully_reviewed_episodes']:
            st.session_state.progress['fully_reviewed_episodes'].append(episode)
    buttons=st.columns(num_classes+1)
    for i in range(num_classes): buttons[i].button(f'{i+1} · class {i}',help=f"{cfg['contexts'][i]['name']}: {cfg['contexts'][i]['description']}",on_click=correct,args=(i,))
    buttons[num_classes].button('U · unknown',on_click=correct,args=(-1,))
    st.button('Approve automatic label / segment',on_click=correct)
    def approve_episode():
        # Keep existing human corrections; approve only frames without a human decision.
        rows=st.session_state.reviews
        reviewed={(r['episode_id'],r['anchor_index']) for r in rows if r['is_reviewed']}
        for i in range(len(ts)):
            if (episode,i) not in reviewed: rows=review_segment(rows,index,episode,i,i,ts,definition_hash=digest)
        st.session_state.reviews=rows;st.session_state.dirty=True
        if episode not in st.session_state.progress['fully_reviewed_episodes']:
            st.session_state.progress['fully_reviewed_episodes'].append(episode)
    st.button('Mark episode fully reviewed (approve remaining automatic labels)',on_click=approve_episode)
    def save():
        path=output/'reviewed_labels.parquet'
        current=path.stat().st_mtime_ns if path.exists() else None
        if current != st.session_state.loaded_review_mtime:
            st.session_state.review_error='Another session changed the review file. Reload before saving; your edits remain in this session.';return
        save_review(path,st.session_state.reviews)
        st.session_state.loaded_review_mtime=path.stat().st_mtime_ns
        st.session_state.progress.update(episode=episode,frame=st.session_state.frame,definition_hash=digest)
        atomic_json(output/'review_progress.json',st.session_state.progress)
        st.session_state.dirty=False
    st.button('S · save',on_click=save)
    if st.session_state.get('review_error'): st.error(st.session_state.review_error)
    install_keyboard(num_classes)
    names=[cfg['contexts'][i]['name'] for i in range(num_classes)]
    timeline_key=f'timeline_{episode}'
    def slide(): seek(st.session_state[timeline_key])
    st.slider('Seek frame',0,len(ts)-1,key=timeline_key,on_change=slide,
              help='Drag to seek. The current playback position is shown by the live cursor below.')

    @st.fragment(run_every=.1 if st.session_state.playing else None)
    def playback():
        if st.session_state.playing:
            target=st.session_state.play_timestamp+(time.monotonic()-st.session_state.clock)*speed
            st.session_state.frame=min(len(ts)-1,max(0,int(np.searchsorted(ts,target,side='right')-1)))
            if st.session_state.frame==len(ts)-1:
                st.session_state.playing=False
                st.rerun()
        frame=st.session_state.frame
        st.progress(frame/max(1,len(ts)-1),text=f'Frame {frame} of {len(ts)-1} · {"Playing" if st.session_state.playing else "Paused"}')
        now=ts[frame]; current_review=next((r for r in st.session_state.reviews if r['episode_id']==episode and r['anchor_index']==frame),None)
        left,right=st.columns([1,2])
        with left:
            st.image(base._get_rgb_array()[lo+frame],caption=f'{camera} · frame {frame} · timestamp {now:.6f}',width=350)
            st.write(dict(automatic_class=int(auto_ids[frame]),confidence=float(confidence[frame]),reviewed=current_review))
            row=index.automatic(episode,frame)
            if row: st.caption(row.get('reason',''))
        reviewed=np.full(len(ts),-1); modified=np.zeros(len(ts),bool); done=np.zeros(len(ts),bool)
        for row in st.session_state.reviews:
            if row['episode_id']==episode:
                i=row['anchor_index'];reviewed[i]=row['reviewed_class_id'];modified[i]=row['is_modified'];done[i]=row['is_reviewed']
        with right:
            figure=go.Figure()
            figure.add_trace(go.Scatter(x=ts,y=auto_ids,mode='lines+markers',name='Automatic',line_shape='hv',customdata=np.arange(len(ts))))
            selected=np.flatnonzero(done)
            figure.add_trace(go.Scatter(x=ts[selected],y=reviewed[selected],mode='markers',name='Reviewed',
                marker=dict(symbol='diamond',size=9,color=np.where(modified[selected],'#d62728','#2ca02c')),customdata=selected))
            figure.add_trace(go.Scatter(x=ts,y=confidence,name='Auto confidence',yaxis='y2'))
            figure.add_vline(x=float(now))
            figure.update_layout(clickmode='event+select',height=280,yaxis=dict(tickvals=list(range(-1,num_classes)),ticktext=['unknown']+names),
                yaxis2=dict(overlaying='y',side='right',range=[0,1],title='Confidence'),xaxis_title='Dataset timestamp (seconds)')
            event=st.plotly_chart(figure,on_select='rerun',selection_mode='points',key=f'context_timeline_{episode}',use_container_width=True)
            points=event.selection.points
            if points:
                target=int(np.argmin(np.abs(ts-float(points[-1]['x']))))
                signature=(episode,target,points[-1].get('curve_number'))
                if st.session_state.get('last_seek_selection')!=signature:
                    st.session_state.last_seek_selection=signature;seek(target);st.rerun()
        start,end=np.searchsorted(ts,[now-radius,now+radius])
        plot=go.Figure()
        for side in ('left','right'):
            times=getattr(base,f'ft_{side}_timestamps');ends=getattr(base,f'ft_{side}_episode_ends')
            a,b=(0 if episode==0 else int(ends[episode-1])),int(ends[episode])
            indices=np.arange(a,b)[(times[a:b]>=now-radius)&(times[a:b]<=now+radius)]
            values=getattr(base,f'ft_{side}')[indices]
            for j,axis in enumerate(('Fx','Fy','Fz','Tx','Ty','Tz')):
                plot.add_trace(go.Scatter(x=times[indices],y=values[:,j],name=f'{side} {axis}'))
            for label,slice_ in [('force norm',slice(0,3)),('torque norm',slice(3,6))]:
                plot.add_trace(go.Scatter(x=times[indices],y=np.linalg.norm(values[:,slice_],axis=1),name=f'{side} {label}',visible='legendonly'))
        plot.add_vline(x=float(now));plot.update_layout(height=280,title='Native F/T — N and Nm (select traces in legend)')
        st.plotly_chart(plot,use_container_width=True)
        state=go.Figure()
        from scipy.spatial.transform import Rotation
        xyz=base.pose_mats[lo+start:lo+end,:3,3]
        rot=Rotation.from_matrix(base.pose_mats[lo+start:lo+end,:3,:3]).as_rotvec()
        for j,axis in enumerate(('x','y','z')):
            state.add_trace(go.Scatter(x=ts[start:end],y=xyz[:,j],name=f'position {axis}'))
            state.add_trace(go.Scatter(x=ts[start:end],y=rot[:,j],name=f'rotation {axis}',visible='legendonly'))
        state.add_trace(go.Scatter(x=ts[start:end],y=base.gripper_width[lo+start:lo+end,0],name='gripper width'))
        state.add_trace(go.Scatter(x=ts[start:end],y=base.grasp_force[lo+start:lo+end,0],name='grasp force target',visible='legendonly'))
        state.add_vline(x=float(now));state.update_layout(height=260,title='TCP/state and recorded targets — m, rad, N')
        st.plotly_chart(state,use_container_width=True)
        st.caption('UMI action targets are recorded TCP/width/force trajectories. No separate executed-command, joint, or object-state streams exist in this recording.')
    playback()
    with st.expander('Review statistics and quality flags',expanded=False):
        rows=pd.DataFrame(st.session_state.reviews)
        auto_frame=pd.DataFrame(auto)
        if len(auto_frame):
            st.bar_chart(auto_frame.class_id.value_counts().reindex(range(-1,num_classes),fill_value=0))
            hist=np.histogram(auto_frame.confidence,bins=np.linspace(0,1,11))
            st.bar_chart(pd.Series(hist[0],index=hist[1][:-1]))
        if len(rows):
            st.bar_chart(rows.loc[rows.is_reviewed,'reviewed_class_id'].value_counts().reindex(range(-1,num_classes),fill_value=0))
            st.write({'correction_rate':float(rows.is_modified.mean()),'percentage_reviewed':100*int(rows.is_reviewed.sum())/len(base.rgb_timestamps),
                'episodes_with_review':int(rows.episode_id.nunique()),'fully_reviewed_episodes':st.session_state.progress['fully_reviewed_episodes']})
            st.dataframe(rows.groupby('auto_class_id').is_modified.mean().rename('correction_rate_per_class'))
            coverage=rows.groupby('episode_id').is_reviewed.sum()
            lengths=np.diff(np.r_[0,base.rgb_episode_ends])
            st.dataframe(pd.DataFrame({'reviewed_frames':coverage,'coverage':coverage/pd.Series(lengths)}).fillna(0))
        runs=segments([episode]*len(ts),ts,auto_ids)
        st.dataframe(pd.DataFrame(runs))
        st.bar_chart(pd.Series([r['duration_s'] for r in runs]).value_counts(bins=10).rename('segment_duration_count').astype(int).rename_axis('duration_bin').reset_index(drop=True))
        st.write({'context_transition_count':max(0,len(runs)-1),'short_segments':sum(r['duration_s']<short_duration for r in runs),
            'low_confidence_frames':int((confidence<threshold).sum()),
            'neighbor_disagreements':int(((auto_ids[1:-1]!=auto_ids[:-2])&(auto_ids[1:-1]!=auto_ids[2:])).sum()),
            'rapid_oscillations':sum(runs[i]['class_id']==runs[i-2]['class_id'] and runs[i-1]['duration_s']<short_duration for i in range(2,len(runs)))})
        st.caption('Flags are advisory. No labels or samples are automatically removed.')

if __name__=='__main__': main()
