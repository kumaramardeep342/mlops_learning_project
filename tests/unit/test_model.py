import torch
import pytest
from src.model import get_model

def test_model_initialization():
    """Test if the model initializes without errors."""
    model = get_model()
    assert model is not None, "Model failed to initialize"

def test_model_output_shape():
    """Test if the model returns the correct output shape (batch_size, 10)."""
    model = get_model()
    model.eval()
    
    # Create dummy batch of 2 images: [batch_size, channels, height, width]
    dummy_input = torch.randn(2, 3, 32, 32)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    # CIFAR-10 has 10 classes, so expected shape is (2, 10)
    assert output.shape == (2, 10), f"Expected shape (2, 10), but got {output.shape}"

def test_model_forward_pass_values():
    """Test that model output contains valid numerical values (no NaNs or Infs)."""
    model = get_model()
    model.eval()
    
    dummy_input = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = model(dummy_input)
        
    assert not torch.isnan(output).any(), "Model output contains NaN values"
    assert not torch.isinf(output).any(), "Model output contains Inf values"