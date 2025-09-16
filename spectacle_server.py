import os

# Image server directory: Where the prompt request files appear, and where the 
# generated images and error response files need to be written.
os.environ['IMAGE_SERVER_DIR_PATH'] = '../image-server'
os.environ['IMAGE_SERVER_GEN_TIMEOUT_SECONDS'] = f'{45 * 60}'

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['HF_HUB_OFFLINE'] = '0'

# For better CPU threading .. Example: For on Ryzen 7950X (16-core)
os.environ['OMP_NUM_THREADS'] = '15'
os.environ['MKL_NUM_THREADS'] = '15'
os.environ['NUMEXPR_NUM_THREADS'] = '15'
os.environ['VECLIB_MAXIMUM_THREADS'] = '15'
os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'

import torch
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
import math
import gc
import signal
import random
import time

directory_path = os.environ['IMAGE_SERVER_DIR_PATH']
# TODO: attempt to mkdir if it doesn't exist, or exit with error.
timeout_seconds = int(os.environ['IMAGE_SERVER_GEN_TIMEOUT_SECONDS'])

# Define a timeout exception
class TimeoutException(Exception):
    pass

# Timeout handler
def timeout_handler(signum, frame):
    raise TimeoutException("Image generation timed out")

# Scheduler configuration (required for Qwen-Image-Lightning)
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),  # shift=3 in distillation
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),  # shift=3 in distillation
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

# Load with low precision (halves memory needs)
pipe = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image",  # Or your specific checkpoint
    scheduler=scheduler,
    torch_dtype=torch.bfloat16,
)
pipe = pipe.to("cpu")

# Load LoRA weights for using Qwen Image Lightning variant
pipe.load_lora_weights(
    "lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-8steps-V1.0.safetensors"
)

# Enable memory optimizations
#pipe.enable_attention_slicing(slice_size=1)  # Slice attention layers (big win for large models)
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()  # Process VAE in slices

#pipe.enable_model_cpu_offload()  # Offload unused submodels to CPU (reduces VRAM, but uses some RAM—start here)
pipe.enable_sequential_cpu_offload()  # More aggressive offloading if above isn't enough (slower but saves more VRAM/RAM)

aspect_ratios = {
    "1:1": (640, 640),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

# defaults
width, height = aspect_ratios["4:3"]

positive_magic = {
  "en": "cinematic composition.",
  "zh": "电影级构图."
}

# Main loop
print("Watching for prompt files in the " + directory_path + " directory.")
while True:
    # Get list of .txt files in the directory
    files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    
    # Sort files to process in a consistent order (e.g., alphabetical)
    for file in sorted(files):
        full_path = os.path.join(directory_path, file)
        base = file[:-4]  # Remove .txt extension
        
        # Parse filename: <uuid>-<message_id>-<width>x<height>
        parts = base.split('-')
        if len(parts) != 3:
            continue  # Ignore if not exactly 3 parts
        
        uuid, msg_id, dims = parts
        if 'x' not in dims.lower():
            continue  # Ignore if no 'x' in dimensions
        
        dim_parts = dims.lower().split('x')
        if len(dim_parts) != 2:
            continue  # Ignore if not exactly two dimensions
        
        try:
            width = int(dim_parts[0])
            height = int(dim_parts[1])
        except ValueError:
            continue  # Ignore if dimensions are not integers
        
        # Read the prompt from the file
        try:
            with open(full_path, 'r') as f:
                prompt = f.read().strip()  # Basic sanitization: strip whitespace
            # Additional sanitization if needed (e.g., remove special characters), but minimal for image gen
        except Exception:
            continue  # Ignore if can't read file
        
        # Generate image with timeout
        try:
            # Set up timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)

            print("Processing image: " +  base + ".png ...")
            print("          prompt: " + prompt)
            full_prompt = prompt + positive_magic["en"]
            negative_prompt = " "
            with torch.no_grad():
                image = pipe(
                    full_prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=6,  # Fewer steps = faster/less memory
                    true_cfg_scale=4.0,
                    generator=torch.Generator(device="cpu").manual_seed(random.randint(0, 255))
                ).images[0]
            
            # Disable alarm after successful generation
            signal.alarm(0)
            
            # Delete the prompt file
            os.remove(full_path)

            # Save the image
            image_path = os.path.join(directory_path, base + '.png')
            image.save(image_path)

            print("Saved " + base + '.png')

        except (Exception, TimeoutException) as e:
            # Disable alarm in case of any error
            signal.alarm(0)
            
            # Create error file
            error_path = os.path.join(directory_path, base + '-error.txt')
            with open(error_path, 'w') as f:
                f.write(str(e))
            
            # Delete the prompt file
            os.remove(full_path)

            print("Error on " + base + '.png')
        
        # Clean up memory after each generation
        torch.cuda.empty_cache()
        gc.collect()
    
    # Sleep for a short time before checking again
    time.sleep(1)

