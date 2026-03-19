import os
import logging
import folder_paths
from nodes import LoraLoader, LoraLoaderModelOnly

logger = logging.getLogger(__name__)

def get_model_type(model):
    """Detects if a model is a Nunchaku-wrapped model and returns its type."""
    if not hasattr(model, 'model') or not hasattr(model.model, 'diffusion_model'):
        return 'standard'
    
    # We check the wrapper class name to identify the specific Nunchaku model type
    class_name = model.model.diffusion_model.__class__.__name__
    if class_name == 'ComfyFluxWrapper':
        return 'flux'
    elif class_name == 'ComfyQwenImageWrapper':
        return 'qwen'
    elif class_name == 'ComfyZImageWrapper':
        return 'zimage'
    
    return 'standard'

class NunchakuUniversalLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model (standard or Nunchaku)."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model to be patched."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "Select the LoRA file."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01, "tooltip": "Strength for the diffusion model."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01, "tooltip": "Strength for the CLIP model."}),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "If disabled, the LoRA will not be applied and the inputs will be passed through."}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    CATEGORY = "Nunchaku/Tools"
    TITLE = "Nunchaku Universal LoRA Loader"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip, enabled=True):
        if not enabled or (strength_model == 0 and strength_clip == 0):
            return (model, clip)

        model_type = get_model_type(model)
        
        if model_type == 'flux':
            from ..lora.flux import NunchakuFluxLoraLoader
            (new_model,) = NunchakuFluxLoraLoader().load_lora(model, lora_name, strength_model)
            # Standard CLIP patching if strength_clip != 0
            if strength_clip != 0:
                _, new_clip = LoraLoader().load_lora(new_model, clip, lora_name, 0.0, strength_clip)
                return (new_model, new_clip)
            return (new_model, clip)
            
        elif model_type == 'qwen':
            from ..lora.qwenimage import NunchakuQwenImageLoraLoader
            (new_model,) = NunchakuQwenImageLoraLoader().load_lora(model, lora_name, strength_model)
            if strength_clip != 0:
                _, new_clip = LoraLoader().load_lora(new_model, clip, lora_name, 0.0, strength_clip)
                return (new_model, new_clip)
            return (new_model, clip)
            
        elif model_type == 'zimage':
            from ..lora.zimage import NunchakuZImageLoraLoader
            (new_model,) = NunchakuZImageLoraLoader().load_lora(model, lora_name, strength_model)
            if strength_clip != 0:
                _, new_clip = LoraLoader().load_lora(new_model, clip, lora_name, 0.0, strength_clip)
                return (new_model, new_clip)
            return (new_model, clip)
            
        else:
            # Fallback to standard ComfyUI LoRA loader
            return LoraLoader().load_lora(model, clip, lora_name, strength_model, strength_clip)

class NunchakuUniversalLoraLoaderModelOnly:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model (standard or Nunchaku)."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "Select the LoRA file."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01, "tooltip": "Strength for the diffusion model."}),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "If disabled, the LoRA will not be applied and the input model will be passed through."}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "Nunchaku/Tools"
    TITLE = "Nunchaku Universal LoRA Loader (Model Only)"

    def load_lora(self, model, lora_name, strength_model, enabled=True):
        if not enabled or strength_model == 0:
            return (model,)

        model_type = get_model_type(model)
        
        if model_type == 'flux':
            from ..lora.flux import NunchakuFluxLoraLoader
            return NunchakuFluxLoraLoader().load_lora(model, lora_name, strength_model)
        elif model_type == 'qwen':
            from ..lora.qwenimage import NunchakuQwenImageLoraLoader
            return NunchakuQwenImageLoraLoader().load_lora(model, lora_name, strength_model)
        elif model_type == 'zimage':
            from ..lora.zimage import NunchakuZImageLoraLoader
            return NunchakuZImageLoraLoader().load_lora(model, lora_name, strength_model)
        else:
            # Fallback to standard ComfyUI LoRA loader (model only)
            return LoraLoaderModelOnly().load_lora_model_only(model, lora_name, strength_model)
