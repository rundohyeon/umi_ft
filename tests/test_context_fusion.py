import torch
from models.context_encoder import ContextFusion

def test_identity_initialization_and_training_at_original_condition_dimension():
    module=ContextFusion(786,32)
    robot=torch.randn(2,786);context=torch.randn(2,32)
    result=module(robot,context)
    assert result.shape==robot.shape
    torch.testing.assert_close(result,robot,rtol=0,atol=0)
    (result-1).square().mean().backward()
    assert module.mlp[-1].weight.grad.abs().sum()>0
