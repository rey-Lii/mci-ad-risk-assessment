import torch
from ra_fmlr import LongitudinalTransformer, ModelConfig, select_route


def test_router():
    assert select_route(1) == "snapshot"
    assert select_route(2) == "longitudinal_transformer"


def test_transformer_shape():
    model = LongitudinalTransformer(ModelConfig())
    token = torch.zeros(2, 5, 3, 8)
    mask = torch.ones(2, 5, 3, dtype=torch.bool)
    available = torch.ones(2, 5, dtype=torch.bool)
    context = torch.zeros(2, 6)
    assert model(token, mask, available, context).shape == (2, 4)
