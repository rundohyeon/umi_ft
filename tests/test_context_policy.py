import torch
import pytest
from torch import nn
from diffusers import DDIMScheduler
from diffusion_policy.policy.context_aware_policy import ContextAwarePolicy
from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy
from diffusion_policy.model.common.normalizer import LinearNormalizer

class ObservationFixture(nn.Module):
    def __init__(self):super().__init__();self.project=nn.Linear(3,786)
    def forward(self,obs):return self.project(obs['robot0_eef_pos'][:,-1])
    def output_shape(self):return (1,786)


def policy(meta,**context):
    m=ContextAwarePolicy(shape_meta=meta,noise_scheduler=DDIMScheduler(num_train_timesteps=10),obs_encoder=ObservationFixture(),
        context=dict(encoder=dict(hidden_dim=32,num_heads=4,num_layers=1),**context),
        down_dims=[16,32],diffusion_step_embed_dim=32,kernel_size=3,n_groups=4,num_inference_steps=2)
    normalizer=LinearNormalizer()
    examples={k:torch.randn(4,v['shape'][0]) for k,v in meta['obs'].items()}
    examples['camera0_rgb']=torch.randn(4,1);examples['action']=torch.randn(4,11)
    normalizer.fit(examples);m.set_normalizer(normalizer)
    return m


@pytest.mark.parametrize('num_classes',[4,5])
def test_policy_backward_reload_and_smoothing_reset(context_meta,context_obs,tmp_path,num_classes):
    torch.set_num_threads(2)
    m=policy(context_meta,num_classes=num_classes,freeze_encoder=False,curriculum='full',smoothing_alpha=.5)
    batch=dict(obs=context_obs,action=torch.randn(2,16,11),context_label=torch.tensor([2,-1]),context_weight=torch.tensor([1.,0.]))
    loss=m.compute_loss(batch);loss.backward()
    assert m.context_encoder.classifier.weight.grad.abs().sum()>0
    assert m.context_fusion.mlp[-1].weight.grad.abs().sum()>0
    path=tmp_path/'policy.pt';torch.save(m.state_dict(),path)
    other=policy(context_meta,num_classes=num_classes,freeze_encoder=False,curriculum='full',smoothing_alpha=.5)
    other.load_state_dict(torch.load(path),strict=True)
    other.eval();obs={k:v[:1] for k,v in context_obs.items()}
    with torch.no_grad():
        result=other.predict_action(obs)
        assert result['action'].shape==(1,16,11)
        assert result['context']['raw_probabilities'].shape==(1,num_classes)
        torch.testing.assert_close(result['context']['raw_probabilities'],result['context']['smoothed_probabilities'])
        other.predict_action(obs);other.reset()
        assert other._smoothed is None


def test_old_five_class_checkpoint_is_not_reinterpreted(context_meta,tmp_path):
    old=policy(context_meta)
    config=dict(old.context_encoder.config)
    config.pop('num_classes')
    payload=dict(encoder_config=config,state_dict=old.context_encoder.state_dict(),
        context_definition_hash=None,definitions={},episode_splits={},source_fingerprint='fixture')
    path=tmp_path/'legacy.pt';torch.save(payload,path)
    old.load_context_checkpoint(path)
    new=policy(context_meta,num_classes=4)
    with pytest.raises(ValueError,match='5 classes; policy expects 4'):
        new.load_context_checkpoint(path)


@pytest.mark.parametrize('mode',['hard','oracle'])
def test_four_class_discrete_conditioning(context_meta,context_obs,mode):
    m=policy(context_meta,mode=mode,num_classes=4).eval()
    batch={'context_label':torch.tensor([0,3])}
    conditioning,_=m.encode_condition(m.normalizer.normalize(context_obs),context_obs,batch=batch)
    assert conditioning.shape==(2,786)
    used=m.last_context['used_probabilities']
    assert used.shape==(2,4)
    assert torch.all(used.sum(-1)==1) and torch.all((used==0)|(used==1))
    if mode=='oracle':
        with pytest.raises(ValueError,match='Oracle'):
            m.predict_action(context_obs,context_label=torch.tensor([0,4]))


def test_no_context_matches_original_and_freeze_groups(context_meta,context_obs):
    m=policy(context_meta,mode='none')
    nobs=m.normalizer.normalize(context_obs)
    actual,_=m.encode_condition(nobs,context_obs)
    torch.testing.assert_close(actual,m.obs_encoder(nobs))
    m=policy(context_meta,freeze_encoder=True,curriculum='adapter');m.train()
    assert not any(p.requires_grad for p in m.model.parameters())
    assert not m.context_encoder.training and not m.obs_encoder.training
    grouped=[p for group in m.optimizer_groups(.001) for p in group['params']]
    assert {id(p) for p in grouped}=={id(p) for p in m.parameters() if p.requires_grad}


def test_oracle_is_explicit_and_soft_does_not_use_ground_truth(context_meta,context_obs):
    import pytest
    m=policy(context_meta,mode='soft').eval();nobs=m.normalizer.normalize(context_obs)
    a,_=m.encode_condition(nobs,context_obs,batch={'context_label':torch.tensor([0,0])})
    b,_=m.encode_condition(nobs,context_obs,batch={'context_label':torch.tensor([4,4])})
    torch.testing.assert_close(a,b)
    m.context_mode='oracle'
    with pytest.raises(ValueError,match='Oracle'):m.predict_action(context_obs)
    assert m.predict_action(context_obs,context_label=torch.tensor([1,2]))['action'].shape==(2,16,11)


def test_legacy_checkpoint_and_zero_adapter_predict_identically(context_meta,context_obs):
    m=policy(context_meta,freeze_encoder=True,curriculum='adapter').eval()
    legacy=DiffusionUnetTimmPolicy(shape_meta=context_meta,
        noise_scheduler=DDIMScheduler(num_train_timesteps=10),obs_encoder=ObservationFixture(),
        down_dims=[16,32],diffusion_step_embed_dim=32,kernel_size=3,n_groups=4,num_inference_steps=2)
    legacy.load_state_dict({k:v for k,v in m.state_dict().items() if not k.startswith('context_')},strict=True)
    legacy.eval()
    with torch.no_grad():
        torch.manual_seed(72);original=legacy.predict_action(context_obs)['action_pred']
        torch.manual_seed(72);extended=m.predict_action(context_obs)['action_pred']
    torch.testing.assert_close(original,extended,rtol=0,atol=0)
