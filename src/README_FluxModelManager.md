# FluxModelManager

A unified manager class for Flux diffusion models that handles model loading, image inversion, and generation with intermediate data collection.

## Overview

The `FluxModelManager` class provides a clean, object-oriented interface for working with Flux diffusion models. It encapsulates all model initialization and provides methods for:

1. **Image Inversion**: Convert images to latent space using DDIM inversion
2. **Image Generation**: Generate images from text prompts (with or without starting latents)
3. **Intermediate Data Collection**: Save all intermediate latents and scores during diffusion processes

## Installation

Ensure you have the required dependencies from the Flux model repository.

## Basic Usage

### Initialize the Manager

```python
from flux_model_manager import FluxModelManager

# Initialize with default settings
manager = FluxModelManager(name="flux-dev", device="cuda", offload=False)

# Initialize with memory optimization
manager = FluxModelManager(name="flux-dev", device="cuda", offload=True)
```

### Image Inversion

```python
# Invert an image to latent space
inverted_latent, intermediate_data = manager.invert_image(
    image_path="path/to/image.jpg",
    source_prompt="a photo of a cat",
    num_steps=50,
    guidance=3.5,
    save_intermediates=True
)

# Save the intermediate data
manager.save_intermediate_data(intermediate_data, "output/inversion_data.pt")
```

### Simple Generation

```python
# Generate an image from a text prompt
result = manager.generate(
    prompt="a beautiful sunset over mountains",
    width=1024,
    height=1024,
    num_steps=50,
    guidance=7.5,
    seed=42,
    save_intermediates=True
)

# Save the results
manager.save_generation_result(result, "output", "sunset_generation")
```

### Inversion-to-Generation (Image Editing)

```python
# First invert the image
inverted_latent, inversion_data = manager.invert_image(
    image_path="input.jpg",
    source_prompt="a photo of a dog",
    num_steps=50,
    guidance=3.5
)

# Then generate with a different prompt
result = manager.generate(
    prompt="a painting of a dog in van gogh style",
    width=inversion_data['metadata']['width'],
    height=inversion_data['metadata']['height'],
    starting_latent=inverted_latent,
    num_steps=50,
    guidance=7.5
)
```

## Class API

### FluxModelManager

#### `__init__(name: str = "flux-dev", device: str = "cuda", offload: bool = False)`
Initialize the model manager.

**Parameters:**
- `name`: Model name ("flux-dev" or "flux-schnell")
- `device`: Device to load models on ("cuda" or "cpu")
- `offload`: Whether to offload models to CPU to save memory

#### `invert_image(...) -> Tuple[torch.Tensor, Dict[str, Any]]`
Perform DDIM inversion on an image.

**Parameters:**
- `image_path`: Path to the input image
- `source_prompt`: Source prompt describing the image
- `num_steps`: Number of inversion steps (default: 50)
- `guidance`: Guidance scale (default: 3.5)
- `seed`: Random seed (optional)
- `order`: Solver order, 1 or 2 (default: 2)
- `save_intermediates`: Whether to save intermediate latents (default: True)
- `feature_path`: Path for feature storage (default: "feature")
- `inject_step`: Step for feature injection (default: 0)

**Returns:**
- Tuple of (inverted_latent, intermediate_data_dict)

#### `generate(...) -> GenerationResult`
Generate an image from a text prompt.

**Parameters:**
- `prompt`: Text prompt for generation
- `width`: Image width in pixels, must be multiple of 16 (default: 1024)
- `height`: Image height in pixels, must be multiple of 16 (default: 1024)
- `num_steps`: Number of denoising steps (default: 50)
- `guidance`: Guidance scale (default: 3.5)
- `seed`: Random seed (optional)
- `order`: Solver order, 1 or 2 (default: 2)
- `save_intermediates`: Whether to save intermediate latents (default: True)
- `starting_latent`: Optional starting latent tensor
- `feature_path`: Path for feature storage (default: "feature")
- `inject_step`: Step for feature injection (default: 0)

**Returns:**
- `GenerationResult` object containing the generated image and intermediate data

#### `save_intermediate_data(data: Dict[str, Any], output_path: str)`
Save intermediate diffusion data to a .pt file.

#### `save_generation_result(result: GenerationResult, output_dir: str, filename_prefix: str)`
Save generation results including image and intermediate data.

### GenerationResult

A dataclass containing generation results:
- `final_image`: PIL Image object
- `final_latent`: Final latent tensor
- `intermediate_latents`: Dictionary of timestep -> latent tensors
- `intermediate_scores`: Dictionary of timestep -> score tensors
- `timesteps`: Timestep schedule used
- `metadata`: Dictionary with generation parameters

## Example Scripts

### Basic Examples

Run the example script to see different use cases:

```bash
# Simple generation
python example_flux_manager.py --example generation

# Image inversion
python example_flux_manager.py --example inversion --image_path input.jpg

# Inversion-to-generation (editing)
python example_flux_manager.py --example inversion_to_generation --image_path input.jpg

# Run all examples
python example_flux_manager.py --example both --image_path input.jpg
```

### Integration with Existing Code

The `inversion_re_with_manager.py` script shows how to integrate FluxModelManager with existing analysis pipelines:

```bash
# Run inversion with the manager
python inversion_re_with_manager.py --run_mode inversion --source_img_dir input.jpg \
    --source_prompt "a photo" --target_prompt "a painting"

# Run analysis on saved data
python inversion_re_with_manager.py --run_mode single_analysis --output_dir output
```

## Memory Optimization

For systems with limited GPU memory, use the `offload` option:

```python
manager = FluxModelManager(name="flux-dev", device="cuda", offload=True)
```

This will:
- Keep models on CPU when not in use
- Move only the active model to GPU during processing
- Automatically manage memory transfers

## Saved Data Format

The manager saves intermediate data in PyTorch .pt files containing:

```python
{
    'latents': Dict[float, torch.Tensor],      # Intermediate latents by timestep
    'scores': Dict[float, torch.Tensor],       # Intermediate scores by timestep  
    'timesteps': torch.Tensor,                 # Timestep schedule
    'final_latent': torch.Tensor,              # Final output latent
    'encoded_image': torch.Tensor,             # Encoded input (for inversion)
    'metadata': {
        'prompt': str,                         # Text prompt used
        'width': int,                          # Image width
        'height': int,                         # Image height
        'num_steps': int,                      # Number of diffusion steps
        'guidance': float,                     # Guidance scale
        'seed': int,                           # Random seed
        'order': int,                          # Solver order
        'model_name': str                      # Model name used
    }
}
```

## Tips and Best Practices

1. **Image Dimensions**: Always ensure images have dimensions that are multiples of 16
2. **Guidance Scale**: Use lower guidance (3-5) for inversion, higher (7-10) for generation
3. **Memory Management**: Use `offload=True` for GPUs with <16GB memory
4. **Intermediate Data**: Set `save_intermediates=False` if you only need final results
5. **Reproducibility**: Always set a seed for reproducible results

## Troubleshooting

**Out of Memory Errors**: Enable offloading with `offload=True`

**Dimension Errors**: Ensure width and height are multiples of 16

**Model Loading Issues**: Check that model name is either "flux-dev" or "flux-schnell" 