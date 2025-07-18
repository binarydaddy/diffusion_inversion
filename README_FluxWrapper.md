# FluxWrapper: Unified Interface for Flux Models

The FluxWrapper provides a unified interface for using Flux diffusion models with both BlackForest Labs and Diffusers backends. This allows you to seamlessly switch between implementations while maintaining the same API.

## Features

- **Unified Interface**: Same API regardless of backend
- **Backend Flexibility**: Easy switching between BlackForest and Diffusers
- **Standard PyTorch Methods**: Full support for `.to()`, `.eval()`, `.train()`, etc.
- **Consistent Output**: `unified_forward()` always returns the same format
- **Backend-Specific Features**: Access to unique features of each backend

## Installation

Make sure you have the required dependencies:

```bash
pip install torch diffusers transformers
```

## Basic Usage

### Quick Start

```python
from src.flux.flux_wrapper import create_flux_wrapper

# Create wrapper with BlackForest backend
wrapper = create_flux_wrapper(
    model_name="flux-dev",
    backend="blackforest",
    device="cuda"
)

# Get model information
info = wrapper.get_model_info()
print(f"Loaded {info['backend']} backend with {info['num_parameters']:,} parameters")
```

### Backend Comparison

```python
# BlackForest backend
wrapper_bf = create_flux_wrapper(
    model_name="flux-dev",
    backend="blackforest",
    device="cuda"
)

# Diffusers backend
wrapper_diff = create_flux_wrapper(
    model_name="flux-dev",
    backend="diffusers",
    device="cuda"
)

# Both use the same interface!
```

## Advanced Usage

### Configuration

```python
from src.flux.flux_wrapper import FluxWrapper, FluxWrapperConfig

config = FluxWrapperConfig(
    model_name="flux-dev",
    backend="blackforest",
    device="cuda",
    torch_dtype=torch.bfloat16,
    hf_download=True
)

wrapper = FluxWrapper(config)
```

### Unified Forward Pass

The `unified_forward()` method provides consistent output regardless of backend:

```python
# Same code works for both backends
output = wrapper.unified_forward(
    img=img_tensor,
    img_ids=img_ids,
    txt=txt_tensor,
    txt_ids=txt_ids,
    timesteps=timesteps,
    y=pooled_embeddings,
    guidance=guidance_tensor
)
# Always returns just the output tensor
```

### Backend-Specific Features

#### BlackForest Backend
```python
# Use the info dictionary for feature injection
info = {
    'feature_path': 'features/',
    'feature': {},
    'inject_step': 5
}

output, updated_info = wrapper.forward(
    img=img_tensor,
    img_ids=img_ids,
    txt=txt_tensor,
    txt_ids=txt_ids,
    timesteps=timesteps,
    y=pooled_embeddings,
    guidance=guidance_tensor,
    info=info
)
```

#### Diffusers Backend
```python
# Use joint attention kwargs for LoRA scaling
joint_attention_kwargs = {'scale': 1.0}

output = wrapper.forward(
    img=img_tensor,
    img_ids=img_ids,
    txt=txt_tensor,
    txt_ids=txt_ids,
    timesteps=timesteps,
    y=pooled_embeddings,
    guidance=guidance_tensor,
    joint_attention_kwargs=joint_attention_kwargs,
    return_dict=False
)
```

## API Reference

### FluxWrapper Class

#### Methods

- `__init__(config: FluxWrapperConfig)`: Initialize wrapper with configuration
- `forward(...)`: Full forward pass with backend-specific parameters
- `unified_forward(...)`: Simplified forward pass with consistent output
- `get_model_info()`: Get model information dictionary
- `to(device_or_dtype)`: Move model to device or change dtype
- `eval()`: Set model to evaluation mode
- `train(mode=True)`: Set model to training mode

#### Properties

- `backend`: Current backend ("blackforest" or "diffusers")
- `device`: Current device
- `params`: Model parameters (unified across backends)
- `in_channels`: Input channels
- `out_channels`: Output channels

### FluxWrapperConfig

Configuration dataclass for FluxWrapper:

```python
@dataclass
class FluxWrapperConfig:
    model_name: str = "flux-dev"          # "flux-dev" or "flux-schnell"
    backend: str = "blackforest"          # "blackforest" or "diffusers"
    device: str = "cuda"                  # Device to load model
    hf_download: bool = True              # Download from HuggingFace if needed
    torch_dtype: torch.dtype = torch.bfloat16  # Model dtype
```

### Convenience Functions

#### create_flux_wrapper()

```python
def create_flux_wrapper(
    model_name: str = "flux-dev",
    backend: str = "blackforest", 
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    hf_download: bool = True
) -> FluxWrapper
```

## Examples

### Example 1: Model Comparison

```python
import torch
from src.flux.flux_wrapper import create_flux_wrapper

# Load both backends
backends = ["blackforest", "diffusers"]
wrappers = {}

for backend in backends:
    wrapper = create_flux_wrapper(
        model_name="flux-dev",
        backend=backend,
        device="cuda"
    )
    wrappers[backend] = wrapper
    
    info = wrapper.get_model_info()
    print(f"{backend}: {info['num_parameters']:,} parameters")

# Use same inputs for both
img = torch.randn(1, 1024, 3072, device="cuda", dtype=torch.bfloat16)
# ... other inputs

# Compare outputs
for name, wrapper in wrappers.items():
    output = wrapper.unified_forward(
        img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
        timesteps=timesteps, y=y, guidance=guidance
    )
    print(f"{name} output shape: {output.shape}")
```

### Example 2: Switching Backends

```python
# Easy backend switching
def load_model(backend="blackforest"):
    return create_flux_wrapper(
        model_name="flux-dev",
        backend=backend,
        device="cuda"
    )

# Switch between backends
model = load_model("blackforest")  # Use BlackForest
# model = load_model("diffusers")  # Use Diffusers

# Same code works for both
output = model.unified_forward(...)
```

## Error Handling

The wrapper includes comprehensive error handling:

```python
try:
    wrapper = create_flux_wrapper(
        model_name="flux-dev",
        backend="blackforest",
        device="cuda"
    )
    output = wrapper.unified_forward(...)
except Exception as e:
    print(f"Error: {e}")
    # Handle error or fallback to different backend
```

## Performance Notes

- **Memory Usage**: Both backends have similar memory requirements
- **Speed**: BlackForest may be slightly faster for custom implementations
- **Compatibility**: Diffusers backend offers better ecosystem integration
- **Features**: BlackForest backend provides more advanced feature injection

## Testing

Run the test script to verify both backends work:

```bash
python test_flux_wrapper.py
```

This will test:
- Model loading for both backends
- Forward pass compatibility
- Memory usage
- Error handling

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Model Download Errors**: Check internet connection and HuggingFace access
3. **Import Errors**: Ensure all dependencies are installed
4. **Backend Differences**: Use `unified_forward()` for consistent behavior

### Debug Information

```python
# Get detailed model information
info = wrapper.get_model_info()
print(f"Backend: {info['backend']}")
print(f"Device: {info['device']}")
print(f"Parameters: {info['num_parameters']:,}")
print(f"Guidance: {info['guidance_embed']}")
```

## Contributing

When adding new features:

1. Ensure compatibility with both backends
2. Add appropriate error handling
3. Update documentation
4. Test with both backends
5. Consider backward compatibility

## License

This wrapper follows the same license as the underlying Flux models:
- BlackForest Labs models: Check individual model licenses
- Diffusers: Apache 2.0 License 