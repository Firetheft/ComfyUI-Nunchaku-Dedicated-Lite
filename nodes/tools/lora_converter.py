import logging
import os
from pathlib import Path

import torch
import safetensors.torch
import comfy.utils
from ..utils import folder_paths, get_full_path_or_raise

logger = logging.getLogger(__name__)

class NunchakuLoRAConverter:
    """
    A tool node to convert LoKr (Low-Rank Kronecker) adapters to standard LoRA format
    using SVD approximation, making them compatible with Nunchaku.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "Select a LoKr type LoRA to convert."}),
                "target_rank": ("INT", {"default": 64, "min": 8, "max": 256, "step": 8, "tooltip": "The rank for the output LoRA approximation. Higher rank means better quality but larger file size."}),
                "manual_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Additional scaling factor. If your output is black or too artifacts-heavy, try reducing this."}),
                "swap_kron": ("BOOLEAN", {"default": False, "tooltip": "Swap the order of Kronecker product (w2 x w1 instead of w1 x w2). Try this if the default results in black images or no effect."}),
                "device": (["cuda", "cpu"], {"default": "cuda", "tooltip": "Device to perform the SVD conversion on. CUDA is much faster."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_name",)
    FUNCTION = "convert"
    CATEGORY = "Nunchaku/Tools"
    TITLE = "Nunchaku LoRA Converter"

    def convert(self, lora_name, target_rank, manual_scale, swap_kron, device):
        lora_path = Path(get_full_path_or_raise("loras", lora_name))
        
        # Include rank and other settings in filename to avoid naming conflicts
        suffix = f"_r{target_rank}_nb"
        if swap_kron:
            suffix += "_swapped"
        if manual_scale != 1.0:
            suffix += f"_s{manual_scale}".replace(".", "_")
        
        save_name = lora_path.stem + suffix + ".safetensors"
        
        # Determine save directory - same as source
        save_directory = lora_path.parent
        save_path = save_directory / save_name
        
        logger.info(f"Converting LoKr adapter {lora_path} (rank={target_rank}, scale={manual_scale}, swap={swap_kron})...")
        
        sd = safetensors.torch.load_file(lora_path, device="cpu")
        new_sd = {}
        processed_pairs = set()
        
        keys = list(sd.keys())
        
        # Check if it's already a standard LoRA (contains .lora_A.weight keys)
        if any(k.endswith(".lora_A.weight") for k in keys):
            logger.info(f"LoRA {lora_path} is already in standard format. Skipping conversion.")
            return (lora_name,)

        # We only really care about the number of weight pairs
        lokr_pairs = sorted([k for k in keys if k.endswith(".lokr_w1")])
        if not lokr_pairs:
            logger.warning(f"No LoKr layers found in {lora_path}. This model might be in an unsupported format.")
            return (lora_name,)
            
        pbar = comfy.utils.ProgressBar(len(lokr_pairs))
        
        # Set computational device
        comp_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        
        for idx, w1_key in enumerate(lokr_pairs):
            base_key = w1_key[:-len(".lokr_w1")]
            w2_key = base_key + ".lokr_w2"
            alpha_key = base_key + ".alpha"
            
            if w2_key not in sd:
                continue
            
            # Load weights to comp device
            w1 = sd[w1_key].to(device=comp_device, dtype=torch.float32)
            w2 = sd[w2_key].to(device=comp_device, dtype=torch.float32)
            alpha = sd.get(alpha_key, torch.tensor(1.0)).to(device=comp_device, dtype=torch.float32)
            
            # Safety check for extremely large alpha values (some LoKr training scripts leave junk here)
            if alpha > 1000.0:
                logger.warning(f"Layer {base_key} has suspiciously large alpha ({alpha.item()}). Defaulting to 1.0.")
                alpha = torch.tensor(1.0, device=comp_device, dtype=torch.float32)
            
            # Apply user's manual scale
            alpha = alpha * manual_scale
            
            try:
                # W = (w1 (x) w2) * alpha
                if swap_kron:
                    W = torch.kron(w2, w1) * alpha
                else:
                    W = torch.kron(w1, w2) * alpha
                
                # Perform SVD for low-rank approximation
                # W = U @ S @ V_h
                U, S, V_h = torch.linalg.svd(W, full_matrices=False)
                
                # Truncate to target rank
                rank = min(target_rank, len(S))
                U_r = U[:, :rank]
                S_r = S[:rank]
                V_rh = V_h[:rank, :] # [rank, in_features]
                
                # Standard LoRA: A = U * sqrt(S), B = sqrt(S) * V
                # In safetensors lora_A (down) is [rank, in], lora_B (up) is [out, rank]
                sqrt_S = torch.sqrt(S_r)
                
                # A (down) = sqrt(S) * V_rh  [rank, in]
                # B (up) = U_r * sqrt(S)     [out, rank]
                A = sqrt_S.view(-1, 1) * V_rh
                B = U_r * sqrt_S
                
                new_sd[f"{base_key}.lora_A.weight"] = A.to(device="cpu", dtype=torch.float16).contiguous()
                new_sd[f"{base_key}.lora_B.weight"] = B.to(device="cpu", dtype=torch.float16).contiguous()
                # Set alpha to rank so that loaders using alpha/rank apply a scale of 1.0 (baked in)
                new_sd[f"{base_key}.alpha"] = torch.tensor(float(rank), device="cpu", dtype=torch.float32)
                
            except Exception as e:
                logger.error(f"Error converting layer {base_key}: {e}")
            
            pbar.update(1)
            processed_pairs.add(base_key)

        # Copy over any non-LoKr keys if they exist
        for k, v in sd.items():
            is_processed = False
            for base in processed_pairs:
                if k.startswith(base):
                    is_processed = True
                    break
            if not is_processed:
                new_sd[k] = v
        
        # Save output
        safetensors.torch.save_file(new_sd, str(save_path))
        logger.info(f"Successfully converted {len(processed_pairs)} layers. Saved to: {save_path}")
        
        return (save_name,)
