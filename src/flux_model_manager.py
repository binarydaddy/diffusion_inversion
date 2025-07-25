import os
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import json
from einops import rearrange

from flux.sampling import denoise, get_schedule, prepare, unpack, denoise_starting_particular_step, denoise_with_timestep_skipping_lora, denoise_with_straight_timestep_skipping
from flux.util import load_ae, load_clip, load_flow_model, load_t5, configs

from flux.flux_wrapper import create_flux_wrapper           

@dataclass
class GenerationResult:
    """Container for generation results"""
    final_image: Image.Image
    final_latent: torch.Tensor
    initial_latent: torch.Tensor
    intermediate_latents: Dict[float, torch.Tensor]
    intermediate_scores: Dict[float, torch.Tensor]
    timesteps: torch.Tensor
    metadata: Dict[str, Any]


@dataclass
class InversionResult:
    """Container for inversion results"""
    final_latent: torch.Tensor
    final_image: Image.Image
    initial_image: torch.Tensor
    intermediate_latents: Dict[float, torch.Tensor]
    intermediate_scores: Dict[float, torch.Tensor]
    timesteps: torch.Tensor
    metadata: Dict[str, Any]


class FluxModelManager:
    """
    Manager class for Flux diffusion models.
    Handles model loading, image inversion, and generation with intermediate data collection.
    """
    
    def __init__(self, name: str = "flux-dev", device: str = "cuda", offload: bool = False, model_backend: str = "blackforest", time_skipping_lora_path: str = None):
        """
        Initialize the model manager with specified configuration.
        
        Args:
            name: Model name (e.g., "flux-dev", "flux-schnell")
            device: Device to load models on
            offload: Whether to offload models to CPU to save memory
        """
        if name not in configs:
            available = ", ".join(configs.keys())
            raise ValueError(f"Unknown model name: {name}, choose from {available}")
        
        self.name = name
        self.device = device
        self.torch_device = torch.device(device)
        self.offload = offload
        
        # Initialize all models
        self.model_backend = model_backend

        self.time_skipping_lora_path = time_skipping_lora_path
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all required models for diffusion and analysis."""
        print(f"Initializing models: {self.name}")
        
        # Initialize text encoders
        print("Loading T5 text encoder...")
        self.t5 = load_t5(self.torch_device, max_length=256 if self.name == "flux-schnell" else 512)
        
        print("Loading CLIP text encoder...")
        self.clip = load_clip(self.torch_device)
        
        # Initialize main diffusion model
        print("Loading diffusion model...")

        # self.model = load_flow_model(self.name, device="cpu" if self.offload else self.torch_device, model_backend=self.model_backend)
        self.model = create_flux_wrapper(
                model_name="flux-dev",
                backend=self.model_backend,
                device=self.torch_device,
                torch_dtype=torch.bfloat16,
                lora_weights_path=None,
                hf_download=True
            )

        print("Base Model initialized...")
        
        if self.time_skipping_lora_path is not None:

            if os.path.exists(self.time_skipping_lora_path) == False:
                raise ValueError(f"Time skipping LoRA weights path {self.time_skipping_lora_path} does not exist.")

            self.model_with_lora = create_flux_wrapper(
                model_name="flux-dev",
                backend=self.model_backend,
                device=self.torch_device,
                torch_dtype=torch.bfloat16,
                lora_weights_path=self.time_skipping_lora_path,
                hf_download=True
            )
            print("Timestep Skipping Model initialized...")

        # Initialize autoencoder
        print("Loading autoencoder...")
        self.ae = load_ae(self.name, device="cpu" if self.offload else self.torch_device)
        
        # Handle offloading
        if self.offload:
            print("Offloading models to CPU...")
            self.model.cpu()
            if getattr(self, "model_with_lora", None) is not None:
                self.model_with_lora.cpu()
            self.ae.encoder.cpu()
            torch.cuda.empty_cache()
        
        print("Model initialization complete.")
    
    def _encode_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Encode an image to latent space.
        
        Args:
            image: Numpy array of shape (H, W, 3) with values in [0, 255]
            
        Returns:
            Encoded latent tensor
        """
        # Move encoder to device if offloading
        if self.offload:
            self.ae.encoder.to(self.torch_device)
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 127.5 - 1
        image_tensor = image_tensor.unsqueeze(0).to(self.torch_device)
        
        # Encode
        with torch.no_grad():
            latent = self.ae.encode(image_tensor).to(torch.bfloat16)
        
        # Move encoder back to CPU if offloading
        if self.offload:
            self.ae.encoder.cpu()
            torch.cuda.empty_cache()
        
        return latent
    
    def _decode_latent(self, latent: torch.Tensor, width: int, height: int) -> Image.Image:
        """
        Decode a latent tensor to an image.
        
        Args:
            latent: Latent tensor
            width: Target image width
            height: Target image height
            
        Returns:
            PIL Image
        """
        # Move decoder to device if offloading
        if self.offload:
            self.ae.decoder.to(latent.device)
        
        # Unpack and decode
        with torch.no_grad():
            with torch.autocast(device_type=self.torch_device.type, dtype=torch.bfloat16):
                batch_x = unpack(latent.float(), width, height)
                x = self.ae.decode(batch_x)
        
        # Convert to PIL image
        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        
        # Move decoder back to CPU if offloading
        if self.offload:
            self.ae.decoder.cpu()
            torch.cuda.empty_cache()
        
        return img
    
    def _prepare_empty_latent(self, height: int, width: int, device='cuda', dtype=torch.bfloat16) -> torch.Tensor:
        """
        Prepare an empty latent tensor for generation.
        """
        latent_h = height // 8
        latent_w = width // 8
        latent_channels = 16  # Flux uses 16 latent channels
        return torch.randn(1, latent_channels, latent_h, latent_w, device=device, dtype=dtype)


    def _prepare_inputs(self, prompt: str, latent: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Prepare inputs for the diffusion model.
        
        Args:
            prompt: Text prompt
            image: Optional encoded image tensor
            
        Returns:
            Dictionary of prepared inputs
        """
        if latent is None and latent is None:
            raise ValueError("Either image or latent must be provided")
        
        # Move text encoders to device if offloading
        if self.offload:
            self.t5.to(self.torch_device)
            self.clip.to(self.torch_device)
        
        # Prepare inputs
        inp = prepare(self.t5, 
                      self.clip, 
                      img=latent, 
                      prompt=prompt)
        
        # Move text encoders back to CPU if offloading
        if self.offload:
            self.t5.cpu()
            self.clip.cpu()
            torch.cuda.empty_cache()
        
        return inp
    
    @torch.inference_mode()
    def invert(
        self,
        image_path: Optional[str] = None,
        source_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_steps: int = 50,
        guidance: float = 3.5,
        seed: Optional[int] = None,
        order: int = 2,
        save_intermediates: bool = True,
        feature_path: str = "feature",
        inject_step: int = 0,
        start_latent: Optional[torch.Tensor] = None,
        start_timestep: int = None
    ) -> InversionResult:
        """
        Perform DDIM inversion on an image.
        
        Args:
            image_path: Path to the input image
            source_prompt: Source prompt describing the image
            num_steps: Number of inversion steps
            guidance: Guidance scale
            seed: Random seed (optional)
            order: Solver order (1 or 2)
            save_intermediates: Whether to save intermediate latents
            feature_path: Path for feature storage
            inject_step: Step for feature injection
            
        Returns:
            Tuple of (inverted_latent, intermediate_data_dict)
        """
        # Load and preprocess image

        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.encoder.to('cuda')

        if start_latent is None:
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)
            
            # Ensure dimensions are multiples of 16
            h, w = image_np.shape[:2]
            new_h = h - (h % 16) if h % 16 != 0 else h
            new_w = w - (w % 16) if w % 16 != 0 else w
            image_np = image_np[:new_h, :new_w, :]
            
            # Encode image
            encoded_image = self._encode_image(image_np)
        
        else:
            assert start_latent is not None

            new_h = height
            new_w = width

            start_latent = start_latent.to(self.torch_device)
            start_latent = unpack(start_latent, new_w, new_h)
            encoded_image = start_latent
        # Prepare inputs
        inp = self._prepare_inputs(source_prompt, encoded_image)
        
        # Get timestep schedule
        timesteps = get_schedule(num_steps, encoded_image.shape[1], shift=(self.name != "flux-schnell"))
        
        # Setup info dictionary
        info = {
            'feature_path': feature_path,
            'feature': {},
            'inject_step': inject_step
        }
        
        # Create feature directory if needed
        os.makedirs(feature_path, exist_ok=True)
        
        # Move model to device if offloading
        if self.offload:
            self.ae.encoder.cpu()
            torch.cuda.empty_cache()
            self.model.to(self.torch_device)
        
        # Setup callback for intermediate results
        intermediate_latents = {}
        intermediate_scores = {}
        
        def callback(log):
            if save_intermediates:
                t = log["t"]
                intermediate_latents[t] = log["latent"].cpu()
                intermediate_scores[t] = log["score"].cpu()
        
        # Perform inversion
        print("Starting inversion...")
        inverted_latent, info = denoise(
            self.model,
            **inp,
            timesteps=timesteps,
            start_timestep=start_timestep,
            guidance=guidance,
            inverse=True,
            info=info,
            order=order,
            callback=callback if save_intermediates else None
        )
        
        # Move model back to CPU if offloading
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to('cuda')
                
        # Decode final latent to image
        final_image = self._decode_latent(inverted_latent, new_w, new_h)

        # Prepare intermediate data
        intermediate_data = {
            'intermediate_latents': intermediate_latents,
            'intermediate_scores': intermediate_scores,
            'timesteps': timesteps.cpu() if hasattr(timesteps, 'cpu') else timesteps,
            'final_latent': inverted_latent.cpu(),
            'final_image': final_image,
            'encoded_image': encoded_image.cpu(),
            'metadata': {
                'image_path': image_path,
                'source_prompt': source_prompt,
                'width': new_w,
                'height': new_h,
                'num_steps': num_steps,
                'guidance': guidance,
                'seed': seed,
                'order': order,
                'model_name': self.name
            }
        }
        
        return InversionResult(
            final_latent=inverted_latent.cpu(),
            final_image=final_image,
            initial_image=encoded_image.cpu(),
            intermediate_latents=intermediate_latents,
            intermediate_scores=intermediate_scores,
            timesteps=timesteps.cpu() if hasattr(timesteps, 'cpu') else timesteps,
            metadata=intermediate_data['metadata']
        )

    @torch.inference_mode()
    def invert_with_timestep_skipping(
            self,
            image_path: Optional[str] = None,
            source_prompt: str = "",
            width: int = 1024,
            height: int = 1024,
            num_steps: int = 50,
            guidance: float = 3.5,
            seed: Optional[int] = None,
            order: int = 2,
            save_intermediates: bool = True,
            feature_path: str = "feature",
            inject_step: int = 0,
            start_latent: Optional[torch.Tensor] = None,
            timestep_to_skip_to: int = None
        ) -> InversionResult:
            """
            Perform Heun's method inversion on an image.
            LoRA model is used to skip to the timestep to skip to.
            
            Args:
                image_path: Path to the input image
                source_prompt: Source prompt describing the image
                num_steps: Number of inversion steps
                guidance: Guidance scale
                seed: Random seed (optional)
                order: Solver order (1 or 2)
                save_intermediates: Whether to save intermediate latents
                feature_path: Path for feature storage
                inject_step: Step for feature injection
                
            Returns:
                Tuple of (inverted_latent, intermediate_data_dict)
            """
            # Load and preprocess image

            if self.model_with_lora is None:
                raise ValueError("Model_with_lora must be provided at initialization of FluxModelManager Class. This is the model to use for the first step of inversion.")
            
            if self.model is None:
                raise ValueError("Model must be provided. This is the model to use for the second step of inversion.")

            if timestep_to_skip_to is None:
                raise ValueError("Argument timestep_to_skip_to must be provided. This is timestep to skip to.")

            if start_latent is None:
                image = Image.open(image_path).convert('RGB')
                image_np = np.array(image)
                
                # Ensure dimensions are multiples of 16
                h, w = image_np.shape[:2]
                new_h = h - (h % 16) if h % 16 != 0 else h
                new_w = w - (w % 16) if w % 16 != 0 else w
                image_np = image_np[:new_h, :new_w, :]
                
                # Encode image
                encoded_image = self._encode_image(image_np)
            
            else:
                assert start_latent is not None

                new_h = height
                new_w = width

                start_latent = start_latent.to(self.torch_device)
                start_latent = unpack(start_latent, new_w, new_h)
                encoded_image = start_latent
            # Prepare inputs

            inp = self._prepare_inputs(source_prompt, encoded_image)
            
            # Get timestep schedule
            # timesteps = get_schedule(num_steps, inp["img"].shape[1], shift=(self.name != "flux-schnell"))
            timesteps = get_schedule(num_steps, encoded_image.shape[1], shift=(self.name != "flux-schnell"))
            
            # Setup info dictionary
            info = {
                'feature_path': feature_path,
                'feature': {},
                'inject_step': inject_step
            }
            
            # Create feature directory if needed
            os.makedirs(feature_path, exist_ok=True)
            
            # Move model to device if offloading
            
            # Setup callback for intermediate results
            intermediate_latents = {}
            intermediate_scores = {}
            
            def callback(log):
                if save_intermediates:
                    t = log["t"]
                    intermediate_latents[t] = log["latent"].cpu()
                    intermediate_scores[t] = log["score"].cpu()
            
            # Perform inversion
            print("Starting inversion...")

            # First step Invert with Lora
            inverted_latent, info = denoise_with_timestep_skipping_lora(
                self.model,
                self.model_with_lora,
                timestep_idx_to_inference=1,
                **inp,
                timesteps=timesteps,
                start_timestep=timestep_to_skip_to,
                guidance=guidance,
                inverse=True,
                info=info,
                order=order,
                callback=callback if save_intermediates else None
            )

            # Move model back to CPU if offloading
            if self.offload:
                self.model.cpu()
                self.model_with_lora.cpu()                
                torch.cuda.empty_cache()
                self.ae.encoder.to('cuda')
                    
            # Decode final latent to image
            final_image = self._decode_latent(inverted_latent, new_w, new_h)

            # Prepare intermediate data
            intermediate_data = {
                'intermediate_latents': intermediate_latents,
                'intermediate_scores': intermediate_scores,
                'timesteps': timesteps.cpu() if hasattr(timesteps, 'cpu') else timesteps,
                'final_latent': inverted_latent.cpu(),
                'final_image': final_image,
                'encoded_image': encoded_image.cpu(),
                'metadata': {
                    'image_path': image_path,
                    'source_prompt': source_prompt,
                    'width': new_w,
                    'height': new_h,
                    'num_steps': num_steps,
                    'guidance': guidance,
                    'seed': seed,
                    'order': order,
                    'model_name': self.name
                }
            }
            
            return InversionResult(
                final_latent=inverted_latent.cpu(),
                final_image=final_image,
                initial_image=encoded_image.cpu(),
                intermediate_latents=intermediate_latents,
                intermediate_scores=intermediate_scores,
                timesteps=timesteps.cpu() if hasattr(timesteps, 'cpu') else timesteps,
                metadata=intermediate_data['metadata']
            )

    @torch.inference_mode()
    def invert_with_straight_timestep_skipping(
            self,
            image_path: Optional[str] = None,
            source_prompt: str = "",
            width: int = 1024,
            height: int = 1024,
            num_steps: int = 50,
            guidance: float = 3.5,
            seed: Optional[int] = None,
            order: int = 2,
            save_intermediates: bool = True,
            feature_path: str = "feature",
            inject_step: int = 0,
            start_latent: Optional[torch.Tensor] = None,
            timestep_to_skip_to: int = None,
            gt_velocity: torch.Tensor = None
        ) -> InversionResult:
            """
            Perform Heun's method inversion on an image.
            LoRA model is used to skip to the timestep to skip to.
            
            Args:
                image_path: Path to the input image
                source_prompt: Source prompt describing the image
                num_steps: Number of inversion steps
                guidance: Guidance scale
                seed: Random seed (optional)
                order: Solver order (1 or 2)
                save_intermediates: Whether to save intermediate latents
                feature_path: Path for feature storage
                inject_step: Step for feature injection
                
            Returns:
                Tuple of (inverted_latent, intermediate_data_dict)
            """
            # Load and preprocess image
            
            if self.model is None:
                raise ValueError("Model must be provided. This is the model to use for the second step of inversion.")

            if timestep_to_skip_to is None:
                raise ValueError("Argument timestep_to_skip_to must be provided. This is timestep to skip to.")

            if gt_velocity is None:
                raise ValueError("gt_velocity is required for straight timestep skipping")

            if start_latent is None:
                image = Image.open(image_path).convert('RGB')
                image_np = np.array(image)
                
                # Ensure dimensions are multiples of 16
                h, w = image_np.shape[:2]
                new_h = h - (h % 16) if h % 16 != 0 else h
                new_w = w - (w % 16) if w % 16 != 0 else w
                image_np = image_np[:new_h, :new_w, :]
                
                # Encode image
                encoded_image = self._encode_image(image_np)
            
            else:
                assert start_latent is not None

                new_h = height
                new_w = width

                start_latent = start_latent.to(self.torch_device)
                start_latent = unpack(start_latent, new_w, new_h)
                encoded_image = start_latent
            # Prepare inputs

            inp = self._prepare_inputs(source_prompt, encoded_image)
            
            # Get timestep schedule
            # timesteps = get_schedule(num_steps, inp["img"].shape[1], shift=(self.name != "flux-schnell"))
            timesteps = get_schedule(num_steps, encoded_image.shape[1], shift=(self.name != "flux-schnell"))
            
            # Setup info dictionary
            info = {
                'feature_path': feature_path,
                'feature': {},
                'inject_step': inject_step
            }
            
            # Create feature directory if needed
            os.makedirs(feature_path, exist_ok=True)
            
            # Move model to device if offloading
            
            # Setup callback for intermediate results
            intermediate_latents = {}
            intermediate_scores = {}
            
            def callback(log):
                if save_intermediates:
                    t = log["t"]
                    intermediate_latents[t] = log["latent"].cpu()
                    intermediate_scores[t] = log["score"].cpu()
            
            # Perform inversion
            print("Starting inversion...")

            # First step Invert with Lora
            inverted_latent, info = denoise_with_straight_timestep_skipping(
                self.model,
                timestep_idx_to_inference=1,
                **inp,
                timesteps=timesteps,
                start_timestep=timestep_to_skip_to,
                gt_velocity=gt_velocity,
                guidance=guidance,
                inverse=True,
                info=info,
                order=order,
                callback=callback if save_intermediates else None
            )

            # Move model back to CPU if offloading
            if self.offload:
                self.model.cpu()
                torch.cuda.empty_cache()
                self.ae.encoder.to('cuda')
                    
            # Decode final latent to image
            final_image = self._decode_latent(inverted_latent, new_w, new_h)

            # Prepare intermediate data
            intermediate_data = {
                'intermediate_latents': intermediate_latents,
                'intermediate_scores': intermediate_scores,
                'timesteps': timesteps.cpu() if hasattr(timesteps, 'cpu') else timesteps,
                'final_latent': inverted_latent.cpu(),
                'final_image': final_image,
                'encoded_image': encoded_image.cpu(),
                'metadata': {
                    'image_path': image_path,
                    'source_prompt': source_prompt,
                    'width': new_w,
                    'height': new_h,
                    'num_steps': num_steps,
                    'guidance': guidance,
                    'seed': seed,
                    'order': order,
                    'model_name': self.name
                }
            }
            
            return InversionResult(
                final_latent=inverted_latent.cpu(),
                final_image=final_image,
                initial_image=encoded_image.cpu(),
                intermediate_latents=intermediate_latents,
                intermediate_scores=intermediate_scores,
                timesteps=timesteps.cpu() if hasattr(timesteps, 'cpu') else timesteps,
                metadata=intermediate_data['metadata']
            )


    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        bs: int = 1,
        width: int = 1024,
        height: int = 1024,
        num_steps: int = 50,
        guidance: float = 3.5,
        seed: Optional[int] = None,
        order: int = 2,
        start_timestep: int = None,
        save_intermediates: bool = True,
        start_latent: Optional[torch.Tensor] = None,
        feature_path: str = "feature",
        inject_step: int = 0
    ) -> GenerationResult:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text prompt for generation
            width: Image width (must be multiple of 16)
            height: Image height (must be multiple of 16)
            num_steps: Number of denoising steps
            guidance: Guidance scale
            seed: Random seed (optional)
            order: Solver order (1 or 2)
            save_intermediates: Whether to save intermediate latents
            starting_latent: Optional starting latent (if None, random noise is used)
            feature_path: Path for feature storage
            inject_step: Step for feature injection
            
        Returns:
            GenerationResult object containing the generated image and intermediate data
        """
        # Validate dimensions
        if width % 16 != 0 or height % 16 != 0:
            raise ValueError("Width and height must be multiples of 16")

        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # Create or use starting latent
        if start_latent is None:
            start_latent = self._prepare_empty_latent(height, width, device=self.torch_device)
        else:
            start_latent = start_latent.to(self.torch_device)
        
        if len(start_latent.shape) == 3:
            start_latent = unpack(start_latent, width, height)
        
        # Prepare inputs (no image needed for generation)
        inp = self._prepare_inputs(prompt, latent=start_latent)
        
        # Get timestep schedule
        timesteps = get_schedule(num_steps, start_latent.shape[1], shift=(self.name != "flux-schnell"))
        
        # Setup info dictionary
        info = {
            'feature_path': feature_path,
            'feature': {},
            'inject_step': inject_step
        }
        
        # Create feature directory if needed
        os.makedirs(feature_path, exist_ok=True)
        
        # Move model to device if offloading
        if self.offload:
            self.model.to(self.torch_device)
        
        # Setup callback for intermediate results
        intermediate_latents = {}
        intermediate_scores = {}
        
        def callback(log):
            if save_intermediates:
                t = log["t"]
                intermediate_latents[t] = log["latent"].cpu()
                intermediate_scores[t] = log["score"].cpu()
        
        # Perform generation
        print("Starting generation...")
        final_latent, _ = denoise(
            self.model,
            **inp,
            timesteps=timesteps,
            start_timestep=start_timestep,
            guidance=guidance,
            inverse=False,
            info=info,
            order=order,
            callback=callback if save_intermediates else None
        )
        
        # Move model back to CPU if offloading
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
        
        # Decode final latent to image
        final_image = self._decode_latent(final_latent, width, height)
        
        # Create result object
        result = GenerationResult(
            final_image=final_image,
            final_latent=final_latent.cpu(),
            initial_latent=start_latent.cpu(),
            intermediate_latents=intermediate_latents,
            intermediate_scores=intermediate_scores,
            timesteps=timesteps.cpu() if hasattr(timesteps, 'cpu') else timesteps,
            metadata={
                'prompt': prompt,
                'width': width,
                'height': height,
                'num_steps': num_steps,
                'guidance': guidance,
                'seed': seed,
                'order': order,
                'model_name': self.name
            }
        )
        
        return result
    
    def save_intermediate_data(self, data: Dict[str, Any], output_path: str):
        """
        Save intermediate diffusion data to a .pt file.
        
        Args:
            data: Dictionary containing intermediate data
            output_path: Path to save the .pt file
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save data
        torch.save(data, output_path)
        print(f"Intermediate data saved to: {output_path}")
    
    def save_generation_result(self, result: GenerationResult, output_dir: str, filename_prefix: str = "generated"):
        """
        Save generation results including image and intermediate data.
        
        Args:
            result: GenerationResult object
            output_dir: Directory to save outputs
            filename_prefix: Prefix for output files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save final image
        image_path = os.path.join(output_dir, f"{filename_prefix}_final.png")
        result.final_image.save(image_path, quality=95, subsampling=0)
        print(f"Final image saved to: {image_path}")
        
        # Save intermediate data
        intermediate_data = {
            'final_image': result.final_image,
            'final_latent': result.final_latent,
            'initial_latent': result.initial_latent,
            'intermediate_latents': result.intermediate_latents,
            'intermediate_scores': result.intermediate_scores,
            'timesteps': result.timesteps,
            'metadata': result.metadata
        }
        
        data_path = os.path.join(output_dir, f"{filename_prefix}_data.pt")
        self.save_intermediate_data(intermediate_data, data_path)
        
        # Save metadata as JSON for easy inspection
        metadata_path = os.path.join(output_dir, f"{filename_prefix}_metadata.json")
        with open(metadata_path, 'w') as f:
            # Convert non-serializable items for JSON
            json_metadata = result.metadata.copy()
            if 'timesteps' in json_metadata and hasattr(json_metadata['timesteps'], 'tolist'):
                json_metadata['timesteps'] = json_metadata['timesteps'].tolist()
            json.dump(json_metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")
    
    def save_inversion_results(self, result: InversionResult, output_dir: str, filename_prefix: str = "inverted"):
        """
        Save inversion results including image and intermediate data.
        
        Args:
            result: InversionResult object
            output_dir: Directory to save outputs
            filename_prefix: Prefix for output files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save final image
        image_path = os.path.join(output_dir, f"{filename_prefix}_final.png")
        result.final_image.save(image_path, quality=95, subsampling=0)
        print(f"Final image saved to: {image_path}")
        
        # Save intermediate data
        intermediate_data = {
            'final_latent': result.final_latent,
            'initial_image': result.initial_image,
            'intermediate_latents': result.intermediate_latents,
            'intermediate_scores': result.intermediate_scores,
            'timesteps': result.timesteps,
            'metadata': result.metadata
        }
        
        data_path = os.path.join(output_dir, f"{filename_prefix}_data.pt")
        self.save_intermediate_data(intermediate_data, data_path)
        
        # Save metadata as JSON for easy inspection
        metadata_path = os.path.join(output_dir, f"{filename_prefix}_metadata.json")
        with open(metadata_path, 'w') as f:
            # Convert non-serializable items for JSON
            json_metadata = result.metadata.copy()
            if 'timesteps' in json_metadata and hasattr(json_metadata['timesteps'], 'tolist'):
                json_metadata['timesteps'] = json_metadata['timesteps'].tolist()
            json.dump(json_metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")

    def invert_image_starting_from_timestep(self, timestep_idx, inversion_data, prompt:str) -> InversionResult:
        """
            This function inverts an image starting from a given timestep.
            It uses the intermediate latents and timesteps from diffusion generation intermediate results.

            Args:
                timestep_idx: The index of the timestep to start from.
                timestep_idx = 0 means t = 0.0, and timestep_idx = 31 means t = 1.0, if num_steps = 30.
                inversion_data: The intermediate results from diffusion generation.
                prompt: The prompt to use for inversion.
            Returns:
                result: Inversion result object.
        """

        inversion_data_intermediate_latents = inversion_data.intermediate_latents
        inversion_data_intermediate_latents_timesteps = sorted(inversion_data.timesteps)

        previous_timestep = inversion_data_intermediate_latents_timesteps[timestep_idx - 1]
        intermediate_latent = inversion_data_intermediate_latents[previous_timestep]

        result = self.invert(
            source_prompt=prompt,
            num_steps=50,
            guidance=3.5,
            seed=42,
            save_intermediates=True,
            start_latent=intermediate_latent,
            start_timestep=previous_timestep,
            order=2
        )

        return result
    
    def generate_image_starting_from_timestep(self, timestep_idx, generation_data, prompt:str) -> GenerationResult:
        """
            This function generates an image starting from a given timestep.
            It uses the intermediate latents and timesteps from diffusion inversion intermediate results.

            Args:
                timestep_idx: The index of the timestep to start from.
                timestep_idx = 0 means t = 0.0, and timestep_idx = 31 means t = 1.0, if num_steps = 30.
                generation_data: The intermediate results from diffusion inversion.
                prompt: The prompt to use for generation.
            Returns:
                result: Generation result object.
        """
        generation_data_intermediate_latents = generation_data.intermediate_latents
        generation_data_intermediate_latents_timesteps = sorted(generation_data.timesteps)

        current_timestep = generation_data_intermediate_latents_timesteps[timestep_idx]
        next_timestep = generation_data_intermediate_latents_timesteps[timestep_idx + 1]
        intermediate_latent = generation_data_intermediate_latents[next_timestep]

        result = self.generate(
            prompt=prompt,
            width=1024,
            height=1024,
            num_steps=50,
            guidance=3.5,
            seed=42,
            save_intermediates=True,
            start_latent=intermediate_latent,
            start_timestep=current_timestep,
            order=2
        )

        return result
    
    def sanity_check(self, prompt:str="A cat holding a sign that says hello world."):
        generation_result = self.generate(prompt=prompt, width=1024, height=1024, num_steps=50, guidance=3.5, seed=42, save_intermediates=True)

        inversion_result = self.invert(
            source_prompt=prompt,
            width=1024,
            height=1024,
            num_steps=50,
            guidance=3.5,
            start_latent=generation_result.final_latent,
        )

        result_generation_from_inversion = self.generate_image_starting_from_timestep(
            timestep_idx=30,
            generation_data=generation_result,
            prompt=prompt,
        )

        result_inversion_from_generation = self.invert_image_starting_from_timestep(
            timestep_idx=30,
            inversion_data=inversion_result,
            prompt=prompt,
        )

        print(f"Inversion test passed: {torch.allclose(inversion_result.final_latent, result_inversion_from_generation.final_latent)}")
        print(f"Generation test passed: {torch.allclose(generation_result.final_latent, result_generation_from_inversion.final_latent)}")