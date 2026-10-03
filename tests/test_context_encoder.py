import torch
import pytest
from models.context_encoder import ContextEncoder,context_loss


@pytest.mark.parametrize('num_classes',[4,5])
def test_shapes_cpu_backward_checkpoint(context_meta,context_obs,tmp_path,num_classes):
    torch.set_num_threads(2)
    model=ContextEncoder(context_meta,num_classes=num_classes)
    logits=model(context_obs)
    assert logits.shape==(2,num_classes)
    assert torch.allclose(logits.softmax(-1).sum(-1),torch.ones(2),atol=1e-6)
    loss=context_loss(logits,torch.tensor([2,-1]));loss.backward()
    assert model.classifier.weight.grad.abs().sum()>0
    model.eval();expected=model(context_obs)
    path=tmp_path/'context.pt';torch.save(model.state_dict(),path)
    restored=ContextEncoder(**model.config).eval();restored.load_state_dict(torch.load(path))
    torch.testing.assert_close(restored(context_obs),expected)


def test_unknown_loss_finite_and_targets_rejected(context_meta,context_obs):
    model=ContextEncoder(context_meta)
    logits=model(context_obs)
    loss=context_loss(logits,torch.tensor([-1,-1]));assert loss.item()==0;loss.backward()
    context_meta['obs']['action']={'shape':[11],'horizon':16}
    with pytest.raises(ValueError,match='never targets'):ContextEncoder(context_meta)


def test_changed_history_rejected(context_meta,context_obs):
    model=ContextEncoder(context_meta)
    context_obs['robot0_ft_left']=context_obs['robot0_ft_left'][:,:31]
    with pytest.raises(ValueError,match='history differs'):model(context_obs)


def test_rgb_force_classifier_is_independent_of_tcp(context_meta,context_obs):
    torch.set_num_threads(2)
    keys=['camera0_rgb','robot0_ft_left','robot0_ft_right']
    model=ContextEncoder(context_meta,num_classes=4,input_keys=keys).eval()
    selected={k:context_obs[k] for k in keys}
    # Statistics and inference both work without any pose observations.
    model.fit_statistics([{'obs':selected}])
    expected=model(selected)
    poisoned=dict(context_obs)
    for key in ('robot0_eef_pos','robot0_eef_rot_axis_angle'):
        poisoned[key]=torch.full_like(poisoned[key],float('nan'))
    torch.testing.assert_close(model(poisoned),expected,rtol=0,atol=0)
    loss=context_loss(expected,torch.tensor([0,3]));loss.backward()
    assert model.classifier.weight.grad.abs().sum()>0
    restored=ContextEncoder(**model.config).eval()
    restored.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(restored(selected),expected,rtol=0,atol=0)


@pytest.mark.parametrize('keys',[[],['missing_camera'],['camera0_rgb','camera0_rgb'],'camera0_rgb'])
def test_invalid_input_selection_rejected(context_meta,keys):
    with pytest.raises(ValueError):
        ContextEncoder(context_meta,input_keys=keys)
