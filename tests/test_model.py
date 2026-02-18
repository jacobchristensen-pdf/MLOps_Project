import torch
import torch.nn as nn
from src.model import build_model

# Tests if the model exists
def test_model_exist():
    model = build_model()
    assert model is not None

# Tests if the model is a pytorch module
def test_model_is_pytorch_module():
    model = build_model()
    assert isinstance(model, nn.Module)

# Tests if the head exists
def test_head_exist():
    model = build_model(num_classes = 2)
    assert isinstance(model.classifier[1], nn.Linear)
    assert model.classifier[1].out_features == 2

# Tests if the weights exist
def test_model_weights_exist():
    model = build_model()
    # Fetch parameters/weights
    params = list(model.parameters())

    # The model must have parameters
    assert len(params) > 0

    # Not all weights should be zero
    total_weight_sum = sum(p.abs().sum().item() for p in params)
    assert total_weight_sum > 0