import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass

# Import both backends
from flux.model import Flux, FluxParams
from flux.util import load_flow_model, configs
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput


@dataclass
class FluxWrapperConfig:
    """Configuration for the FluxWrapper"""
    model_name: str = "flux-dev"
    backend: str = "blackforest"  # "blackforest" or "diffusers"
    device: str = "cuda"
    hf_download: bool = True
    torch_dtype: torch.dtype = torch.bfloat16
    lora_weights_path: Optional[str] = None


class FluxWrapper(nn.Module):
    """
    Unified wrapper for Flux models that supports both blackforest and diffusers backends.
    Provides a consistent interface regardless of the underlying implementation.
    """
    
    def __init__(self, config: FluxWrapperConfig):
        super().__init__()
        self.config = config
        self.backend = config.backend
        self.device = torch.device(config.device)
        
        # Load the appropriate model
        if self.config.lora_weights_path is not None:
            self.model = self._load_model_with_lora()
        else:
            self.model = self._load_model()
        
        # Store model parameters for reference
        self._setup_params()
    
    def _load_model(self) -> Union[Flux, FluxTransformer2DModel]:
        """Load the model based on the specified backend"""
        print(f"Loading {self.config.model_name} with {self.backend} backend...")
        
        model = load_flow_model(
            name=self.config.model_name,
            device=self.config.device,
            hf_download=self.config.hf_download,
            model_backend=self.backend
        )
        
        return model.to(self.config.torch_dtype)
    
    def _load_model_with_lora(self) -> Union[Flux, FluxTransformer2DModel]:
        """Load the model with lora weights"""
        model = self._load_model()
        model.load_lora_adapter(self.config.lora_weights_path)
        return model.to(self.config.torch_dtype)

    def _setup_params(self):
        """Setup parameters based on the backend"""
        if self.backend == "blackforest":
            self.params = self.model.params
            self.in_channels = self.model.in_channels
            self.out_channels = self.model.out_channels
        else:  # diffusers
            self.in_channels = self.model.config.in_channels
            self.out_channels = self.model.config.out_channels or self.in_channels
            # Create params-like object for consistency
            self.params = type('Params', (), {
                'guidance_embed': self.model.config.guidance_embeds,
                'hidden_size': self.model.config.num_attention_heads * self.model.config.attention_head_dim,
                'num_heads': self.model.config.num_attention_heads,
                'depth': self.model.config.num_layers,
                'depth_single_blocks': self.model.config.num_single_layers,
            })()
    
    def forward(
        self,
        # Common parameters (unified interface)
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        info: Optional[Dict[str, Any]] = None,
        # Additional diffusers-specific parameters
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
        with_lora: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict], Transformer2DModelOutput]:
        """
        Unified forward function that works with both backends.
        
        Args:
            img: Image tensor (latent space)
            img_ids: Image position IDs
            txt: Text embeddings
            txt_ids: Text position IDs
            timesteps: Timestep tensor
            y: Pooled text embeddings
            guidance: Guidance scale tensor (optional)
            info: Additional info dict (blackforest only)
            joint_attention_kwargs: Additional attention kwargs (diffusers only)
            return_dict: Whether to return dict output (diffusers only)
            
        Returns:
            Model output (format depends on backend and return_dict)
        """
        if self.backend == "blackforest":
            return self._forward_blackforest(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=timesteps,
                y=y,
                guidance=guidance,
                info=info,
                **kwargs
            )
        else:  # diffusers
            return self._forward_diffusers(
                hidden_states=img,
                img_ids=img_ids,
                encoder_hidden_states=txt,
                txt_ids=txt_ids,
                timestep=timesteps,
                pooled_projections=y,
                guidance=guidance,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=return_dict,
                with_lora=with_lora,
                **kwargs
            )[0]
    
    def _forward_blackforest(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        """Forward pass for blackforest backend"""
        return self.model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            y=y,
            guidance=guidance,
            info=info,
            **kwargs
        )
    
    def _forward_diffusers(
        self,
        hidden_states: torch.Tensor,
        img_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        txt_ids: torch.Tensor,
        timestep: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        with_lora: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """Forward pass for diffusers backend"""
        if with_lora:
            model = self.model_with_lora
        else:
            model = self.model
        
        return model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=return_dict,
            **kwargs
        )
    
    def unified_forward(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Simplified unified forward that always returns just the output tensor,
        regardless of backend. This is useful for inference pipelines.
        """
        if self.backend == "blackforest":
            output, _ = self._forward_blackforest(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=timesteps,
                y=y,
                guidance=guidance,
                info=info,
                **kwargs
            )
            return output
        else:  # diffusers
            result = self._forward_diffusers(
                hidden_states=img,
                img_ids=img_ids,
                encoder_hidden_states=txt,
                txt_ids=txt_ids,
                timestep=timesteps,
                pooled_projections=y,
                guidance=guidance,
                return_dict=False,
                **kwargs
            )
            return result[0] if isinstance(result, tuple) else result.sample
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "backend": self.backend,
            "model_name": self.config.model_name,
            "device": str(self.device),
            "dtype": str(self.config.torch_dtype),
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "guidance_embed": self.params.guidance_embed,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
        }
    
    def to(self, device_or_dtype):
        """Move model to device or change dtype"""
        self.model = self.model.to(device_or_dtype)
        if isinstance(device_or_dtype, torch.device) or isinstance(device_or_dtype, str):
            self.device = torch.device(device_or_dtype)
        return self
    
    def eval(self):
        """Set model to evaluation mode"""
        self.model.eval()
        return self
    
    def train(self, mode=True):
        """Set model to training mode"""
        self.model.train(mode)
        return self
    
    @property
    def parameters(self):
        """Get model parameters"""
        return self.model.parameters()
    
    def state_dict(self):
        """Get model state dict"""
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict, strict=True):
        """Load model state dict"""
        return self.model.load_state_dict(state_dict, strict)

    def load_lora_weights(self, lora_weights_path: str):
        """Load lora weights"""
        self.model.load_lora_weights(lora_weights_path)


def create_flux_wrapper(
    model_name: str = "flux-dev",
    backend: str = "blackforest",
    lora_weights_path: Optional[str] = None,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    hf_download: bool = True
) -> FluxWrapper:
    """
    Convenience function to create a FluxWrapper instance.
    
    Args:
        model_name: Name of the model ("flux-dev" or "flux-schnell")
        backend: Backend to use ("blackforest" or "diffusers")
        device: Device to load model on
        torch_dtype: Data type for model
        hf_download: Whether to download from HuggingFace if needed
        
    Returns:
        FluxWrapper instance
    """
    config = FluxWrapperConfig(
        model_name=model_name,
        backend=backend,
        lora_weights_path=lora_weights_path,
        device=device,
        torch_dtype=torch_dtype,
        hf_download=hf_download
    )
    
    return FluxWrapper(config)


# Example usage
if __name__ == "__main__":
    # Example 1: Using blackforest backend
    print("=== Example 1: Blackforest Backend ===")
    wrapper_bf = create_flux_wrapper(
        model_name="flux-dev",
        backend="blackforest",
        device="cuda"
    )
    
    print("Model info:", wrapper_bf.get_model_info())
    
    # Example 2: Using diffusers backend
    print("\n=== Example 2: Diffusers Backend ===")
    wrapper_diff = create_flux_wrapper(
        model_name="flux-dev",
        backend="diffusers",
        device="cuda"
    )
    
    print("Model info:", wrapper_diff.get_model_info())
    
    # Example 3: Using both backends with same interface
    print("\n=== Example 3: Unified Interface ===")
    
    # Create dummy inputs (shapes for flux-dev)
    batch_size = 1
    seq_len_img = 1024  # Example sequence length for image
    seq_len_txt = 512   # Example sequence length for text
    hidden_size = 3072  # Flux-dev hidden size
    
    img = torch.randn(batch_size, seq_len_img, hidden_size, device="cuda", dtype=torch.bfloat16)
    img_ids = torch.randn(seq_len_img, 3, device="cuda", dtype=torch.bfloat16)
    txt = torch.randn(batch_size, seq_len_txt, hidden_size, device="cuda", dtype=torch.bfloat16)
    txt_ids = torch.randn(seq_len_txt, 3, device="cuda", dtype=torch.bfloat16)
    timesteps = torch.tensor([0.5], device="cuda", dtype=torch.bfloat16)
    y = torch.randn(batch_size, 768, device="cuda", dtype=torch.bfloat16)  # pooled embeddings
    guidance = torch.tensor([3.5], device="cuda", dtype=torch.bfloat16)
    
    # Both backends can use the same interface
    for name, wrapper in [("Blackforest", wrapper_bf), ("Diffusers", wrapper_diff)]:
        print(f"\n{name} backend:")
        try:
            # Use unified_forward for consistent output
            output = wrapper.unified_forward(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=timesteps,
                y=y,
                guidance=guidance,
                info={} if name == "Blackforest" else None  # info only for blackforest
            )
            print(f"Output shape: {output.shape}")
        except Exception as e:
            print(f"Error with {name}: {e}")
    
    print("\n=== Example 4: Backend-specific features ===")
    
    # Blackforest-specific usage with info
    info = {'feature_path': 'test_features', 'feature': {}, 'inject_step': 0}
    output_bf, updated_info = wrapper_bf.forward(
        img=img,
        img_ids=img_ids,
        txt=txt,
        txt_ids=txt_ids,
        timesteps=timesteps,
        y=y,
        guidance=guidance,
        info=info
    )
    print(f"Blackforest output shape: {output_bf.shape}")
    print(f"Updated info keys: {list(updated_info.keys()) if updated_info else 'None'}")
    
    # Diffusers-specific usage with joint_attention_kwargs
    joint_attention_kwargs = {'scale': 1.0}
    output_diff = wrapper_diff.forward(
        img=img,
        img_ids=img_ids,
        txt=txt,
        txt_ids=txt_ids,
        timesteps=timesteps,
        y=y,
        guidance=guidance,
        joint_attention_kwargs=joint_attention_kwargs,
        return_dict=False
    )
    print(f"Diffusers output shape: {output_diff[0].shape if isinstance(output_diff, tuple) else output_diff.sample.shape}")
