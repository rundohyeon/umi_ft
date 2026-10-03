import torch
import pytest
from models.context_encoder import ContextEmbedding

@pytest.mark.parametrize('num_classes',[4,5])
def test_soft_mixture_keeps_all_classes_and_gradients(num_classes):
    embedding=ContextEmbedding(32,num_classes=num_classes)
    probabilities=torch.arange(1,num_classes+1,dtype=torch.float32).unsqueeze(0)
    logits=(probabilities/probabilities.sum()).requires_grad_()
    result=embedding(logits)
    assert result.shape==(1,32)
    torch.testing.assert_close(result,sum(logits[:,i:i+1]*embedding.context_embedding_table[i] for i in range(num_classes)))
    result.square().sum().backward()
    assert torch.all(embedding.context_embedding_table.grad.abs().sum(1)>0)
    assert logits.grad is not None
