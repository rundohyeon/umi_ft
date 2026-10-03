"""Episode-aware context metrics, calibration and standalone evaluation figures."""
from pathlib import Path
import csv
import numpy as np
from diffusion_policy.context.labels import atomic_json


def classification_metrics(labels, probabilities):
    y, p = np.asarray(labels,dtype=int), np.asarray(probabilities,dtype=float)
    if p.ndim != 2 or p.shape[1] < 2 or y.shape != (len(p),):
        raise ValueError('Expected labels [B] and probabilities [B, num_classes]')
    num_classes = p.shape[1]
    if np.any((y < -1) | (y >= num_classes)):
        raise ValueError('Reference class IDs exceed the model class range')
    valid = y >= 0
    y,p = y[valid],p[valid]
    if not len(y): raise ValueError('No known labels to evaluate')
    pred=p.argmax(1)
    cm=np.zeros((num_classes,num_classes),dtype=int)
    np.add.at(cm,(y,pred),1)
    tp=np.diag(cm)
    precision=np.divide(tp,cm.sum(0),out=np.zeros(num_classes),where=cm.sum(0)>0)
    recall=np.divide(tp,cm.sum(1),out=np.zeros(num_classes),where=cm.sum(1)>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(num_classes),where=precision+recall>0)
    confidence=p.max(1)
    bins=[]
    ece=0.
    for i in range(10):
        selected=(confidence>=i/10)&((confidence<(i+1)/10) if i<9 else (confidence<=1))
        n=int(selected.sum())
        acc=float((pred[selected]==y[selected]).mean()) if n else 0.
        conf=float(confidence[selected].mean()) if n else 0.
        ece+=n/len(y)*abs(acc-conf)
        bins.append(dict(lower=i/10,count=n,accuracy=acc,confidence=conf))
    return dict(sample_count=len(y),accuracy=float((pred==y).mean()),
        balanced_accuracy=float(recall[cm.sum(1)>0].mean()),macro_f1=float(f1.mean()),
        per_class=[dict(class_id=i,precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i]),support=int(cm[i].sum())) for i in range(num_classes)],
        confusion_matrix=cm.tolist(), class_distribution=cm.sum(1).tolist(),
        mean_confidence=float(confidence.mean()),expected_calibration_error=ece,calibration_bins=bins,
        brier_score=float(((p-np.eye(num_classes)[y])**2).sum(1).mean()),
        negative_log_likelihood=float(-np.log(p[np.arange(len(y)),y].clip(1e-9)).mean()))


def segments(episode_ids,timestamps,classes):
    rows=[]
    ep=np.asarray(episode_ids); ts=np.asarray(timestamps); classes=np.asarray(classes)
    for episode in np.unique(ep):
        selected=np.flatnonzero(ep==episode)
        order=selected[np.argsort(ts[selected])]
        t,c=ts[order],classes[order]
        changes=np.r_[0,np.flatnonzero(c[1:]!=c[:-1])+1,len(c)]
        dt=float(np.median(np.diff(t))) if len(t)>1 else 0.
        for a,b in zip(changes[:-1],changes[1:]):
            rows.append(dict(episode_id=int(episode),class_id=int(c[a]),start_timestamp=float(t[a]),
                end_timestamp=float(t[b]) if b<len(t) else float(t[-1]+dt),duration_s=float((t[b] if b<len(t) else t[-1]+dt)-t[a]),
                samples=int(b-a)))
    return rows


def write_evaluation(output, labels, probabilities, episodes, timestamps, names=None, extra=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    y,p=np.asarray(labels),np.asarray(probabilities)
    metrics=classification_metrics(y,p)
    num_classes=p.shape[1]
    names=names or [f'context_{i}' for i in range(num_classes)]
    if len(names) != num_classes:
        raise ValueError('Class names must match the model class count')
    if extra: metrics.update(extra)
    atomic_json(output/'metrics.json',metrics)
    fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(metrics['confusion_matrix']); fig.colorbar(im,ax=ax)
    ax.set(xticks=range(num_classes),yticks=range(num_classes),xticklabels=names,yticklabels=names,xlabel='Predicted',ylabel='Reviewed / selected reference')
    ax.tick_params(axis='x',rotation=45); fig.tight_layout(); fig.savefig(output/'confusion_matrix.png'); plt.close(fig)
    fig,ax=plt.subplots(); ax.hist(p.max(1),bins=np.linspace(0,1,21)); ax.set(xlabel='Confidence',ylabel='Count')
    fig.savefig(output/'confidence_histogram.png'); plt.close(fig)
    fig,ax=plt.subplots(); ax.bar(np.arange(num_classes)-.2,np.bincount(y[y>=0],minlength=num_classes),width=.4,label='Reference')
    ax.bar(np.arange(num_classes)+.2,np.bincount(p.argmax(1),minlength=num_classes),width=.4,label='Predicted'); ax.legend()
    ax.set(xticks=range(num_classes),xticklabels=names); fig.tight_layout(); fig.savefig(output/'class_distribution.png'); plt.close(fig)
    unique=np.unique(episodes)
    fig,axes=plt.subplots(min(6,len(unique)),1,figsize=(12,3*min(6,len(unique))),squeeze=False)
    for ax,ep in zip(axes[:,0],unique[:6]):
        selected=np.flatnonzero(np.asarray(episodes)==ep); order=selected[np.argsort(np.asarray(timestamps)[selected])]
        t=np.asarray(timestamps)[order]; t=t-t[0]
        ax.step(t,y[order],where='post',label='Reference'); ax.step(t,p.argmax(1)[order],where='post',label='Predicted')
        ax.set(title=f'Episode {ep}',xlabel='Seconds',yticks=range(-1,num_classes)); ax.legend()
    fig.tight_layout(); fig.savefig(output/'context_timeline.png'); plt.close(fig)
    rows=segments(episodes,timestamps,p.argmax(1))
    with (output/'transition_statistics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(output/'predictions.npz',labels=y,probabilities=p,episode_ids=episodes,timestamps=timestamps)
    return metrics
