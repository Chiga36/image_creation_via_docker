import asyncio
import torch
import time
import random
import os
import sys
import gc
import psutil
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Sequence, Mapping, Any, Union, Optional, Dict, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
import uvicorn
from pydantic import BaseModel
from fastapi.responses import FileResponse
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import functools
# ============================================================================
# CONFIGURATION - Set your exact absolute paths here
# ============================================================================
CUDA_DEVICE = 0

# Main directories - use absolute paths
COMFYUI_PATH = os.environ.get("COMFYUI_PATH", "/home/comfyuser/ComfyUI")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/home/comfyuser/ComfyUI/output")

# Model file paths - use absolute paths directly
UNET_MODEL = os.environ.get("UNET_MODEL", "flux1-dev-Q5_K_S.gguf")
CLIP_L_MODEL = os.environ.get("CLIP_L_MODEL", "clip_l.safetensors")
T5_MODEL = os.environ.get("T5_MODEL", "t5xxl_fp8_e4m3fn_scaled.safetensors")
VAE_MODEL = os.environ.get("VAE_MODEL", "ae.safetensors")
LORA_MODEL = os.environ.get("LORA_MODEL", "Ghibli_v6.safetensors")
# ============================================================================
# Optional: Full paths for verification (not used by loaders)
UNET_MODEL_FULL_PATH = os.environ.get("UNET_MODEL_FULL_PATH", "/home/comfyuser/ComfyUI/models/unet/flux1-dev-Q5_K_S.gguf")
CLIP_L_MODEL_FULL_PATH = os.environ.get("CLIP_L_MODEL_FULL_PATH", "/home/comfyuser/ComfyUI/models/clip/clip_l.safetensors")
T5_MODEL_FULL_PATH = os.environ.get("T5_MODEL_FULL_PATH", "/home/comfyuser/ComfyUI/models/clip/t5xxl_fp8_e4m3fn_scaled.safetensors")
VAE_MODEL_FULL_PATH = os.environ.get("VAE_MODEL_FULL_PATH", "/home/comfyuser/ComfyUI/models/vae/ae.safetensors")
LORA_MODEL_FULL_PATH = os.environ.get("LORA_MODEL_FULL_PATH", "/home/comfyuser/ComfyUI/models/loras/Ghibli_v6.safetensors")
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)


# Configure logging
log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'flux_worker_gpu0.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Verify paths on startup
def verify_paths():
    """Verify all configured paths exist"""
    logger.info("Verifying configured paths...")
    logger.info(f"ComfyUI Path: {COMFYUI_PATH}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")
    logger.info(f"UNET Model: {UNET_MODEL}")
    logger.info(f"CLIP-L Model: {CLIP_L_MODEL}")
    logger.info(f"T5 Model: {T5_MODEL}")
    logger.info(f"VAE Model: {VAE_MODEL}")
    logger.info(f"LoRA Model: {LORA_MODEL}")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(COMFYUI_PATH):
        logger.error(f"ComfyUI path does not exist: {COMFYUI_PATH}")
        raise FileNotFoundError(f"ComfyUI path not found: {COMFYUI_PATH}")
    
    # Check if model files exist
    model_files = {
        "UNET": UNET_MODEL_FULL_PATH,
        "CLIP-L": CLIP_L_MODEL_FULL_PATH,
        "T5": T5_MODEL_FULL_PATH,
        "VAE": VAE_MODEL_FULL_PATH,
        "LoRA": LORA_MODEL_FULL_PATH
    }
    
    missing_models = []
    for model_name, model_path in model_files.items():
        if not os.path.exists(model_path):
            logger.warning(f"{model_name} model not found: {model_path}")
            missing_models.append(f"{model_name}: {model_path}")
    
    if missing_models:
        logger.warning(f"Missing models: {missing_models}")
        logger.warning("These models must exist for image generation to work")
    
    if not os.access(OUTPUT_DIR, os.W_OK):
        logger.error(f"Output directory is not writable: {OUTPUT_DIR}")
        raise PermissionError(f"Cannot write to output directory: {OUTPUT_DIR}")
    
    logger.info("Path verification completed successfully")
    if missing_models:
        logger.warning("WARNING: Some model files are missing. Please verify paths.")
    else:
        logger.info("All model files found!")

# Job Management Configuration
MAX_CONCURRENT_JOBS = 2
MAX_QUEUE_SIZE = 10
JOB_TIMEOUT = 300
CLEANUP_INTERVAL = 600
generation_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS)
# Global job management
job_queue = deque()
active_jobs = {}
job_history = {}
job_lock = threading.Lock()

def find_path(name: str, path: str = None) -> str:
    """Recursively looks at parent folders starting from the given path until it finds the given name."""
    if path is None:
        path = os.getcwd()

    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        logger.info(f"{name} found: {path_name}")
        return path_name

    parent_directory = os.path.dirname(path)
    if parent_directory == path:
        return None

    return find_path(name, parent_directory)

# ComfyUI helper functions
def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping."""
    try:
        return obj[index]
    except (KeyError, IndexError, TypeError):
        pass
    
    # Try accessing as a dict with 'result' key
    try:
        return obj["result"][index]
    except (KeyError, IndexError, TypeError):
        pass
    
    # If obj is a tuple/list, try direct access
    try:
        if isinstance(obj, (tuple, list)):
            return obj[index]
    except (IndexError, TypeError):
        pass
    
    # Last resort: return the object itself if it's not indexable
    logger.warning(f"Could not index into object of type {type(obj)}, returning as-is")
    return obj

def is_gpu_busy() -> bool:
    """Check if GPU is currently processing"""
    with job_lock:
        return any(job["status"] == JobStatus.PROCESSING for job in active_jobs.values())
    
def add_comfyui_to_path():
    """Add ComfyUI to sys.path"""
    if COMFYUI_PATH not in sys.path:
        sys.path.append(COMFYUI_PATH)
        logger.info(f"Added {COMFYUI_PATH} to sys.path")

def add_extra_model_paths() -> None:
    """Load extra model paths configuration if available"""
    try:
        from main import load_extra_path_config
    except ImportError:
        try:
            from utils.extra_config import load_extra_path_config
        except ImportError:
            logger.info("load_extra_path_config not available, skipping")
            return

    extra_config_path = os.path.join(COMFYUI_PATH, "extra_model_paths.yaml")
    if os.path.exists(extra_config_path):
        load_extra_path_config(extra_config_path)
        logger.info(f"Loaded extra model paths from {extra_config_path}")
    else:
        logger.info("No extra_model_paths.yaml found")

async def import_custom_nodes() -> None:
    """Import ComfyUI custom nodes"""
    import execution
    from nodes import init_extra_nodes
    import server

    loop = asyncio.get_running_loop()
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)
    await init_extra_nodes()

# Initialize paths
verify_paths()
add_comfyui_to_path()
add_extra_model_paths()

# Global variables for models
models_loaded = False
NODE_CLASS_MAPPINGS = None

class JobStatus:
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class JobResponse(BaseModel):
    job_id: str
    status: str
    output_filename: Optional[str] = None
    message: Optional[str] = None
    estimated_wait_time: Optional[int] = None
    queue_position: Optional[int] = None
    result_url: Optional[str] = None
    download_url: Optional[str] = None
    generation_details: Optional[dict] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    processing_time: Optional[float] = None

class ImageGenerationRequest(BaseModel):
    prompt: str
    height: int = 512
    width: int = 512
    steps: int = 20
    guidance_scale: float = 7.5
    model: str = "flux-dev"
    priority: str = "normal"
    seed: Optional[int] = None
    batch_size: int = 1
    api_key: Optional[str] = None

# API Key Management
_api_keys_lock = threading.Lock()

def load_api_keys() -> set:
    """Load API keys with robust error handling and file locking"""
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    os.makedirs(config_dir, exist_ok=True)
    api_keys_file = os.path.join(config_dir, "api_keys.json")
    
    # Default keys to use if file is missing or corrupted
    default_keys = {"master_key_123","flux_key_456"}
    
    # Thread lock to prevent race conditions
    with _api_keys_lock:
        try:
            # Check if file exists and is readable
            if not os.path.exists(api_keys_file):
                logger.info("API keys file not found, creating default")
                with open(api_keys_file, "w") as f:
                    json.dump({"keys": list(default_keys)}, f, indent=2)
                return default_keys
            
            # Try to read the file
            with open(api_keys_file, "r") as f:
                content = f.read().strip()
                
                # Check if file is empty
                if not content:
                    logger.warning("API keys file is empty, recreating")
                    with open(api_keys_file, "w") as f:
                        json.dump({"keys": list(default_keys)}, f, indent=2)
                    return default_keys
                
                # Parse JSON
                data = json.loads(content)
                keys = set(data.get("keys", []))
                
                if not keys:
                    logger.warning("No keys found in file, using defaults")
                    return default_keys
                
                logger.info(f"✓ Loaded {len(keys)} API keys")
                return keys
                
        except json.JSONDecodeError as e:
            logger.error(f"⚠️  Corrupted API keys file: {e}")
            logger.error(f"    Error at line 1, column {e.pos}")
            
            # Backup corrupted file
            backup_file = f"{api_keys_file}.corrupted.{int(time.time())}"
            try:
                import shutil
                shutil.copy2(api_keys_file, backup_file)
                logger.info(f"    Backed up to: {backup_file}")
            except:
                pass
            
            # Remove corrupted file and create fresh one
            try:
                os.remove(api_keys_file)
            except:
                pass
            
            with open(api_keys_file, "w") as f:
                json.dump({"keys": list(default_keys)}, f, indent=2)
            
            logger.info("    Created fresh API keys file")
            return default_keys
            
        except Exception as e:
            logger.error(f"Unexpected error loading API keys: {e}")
            logger.warning("Using default keys without writing to file")
            return default_keys

def save_api_keys(api_keys: set):
    """Save API keys with atomic write"""
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    api_keys_file = os.path.join(config_dir, "api_keys.json")
    
    with _api_keys_lock:
        try:
            # Write to temp file first (atomic operation)
            temp_file = f"{api_keys_file}.tmp.{os.getpid()}"
            with open(temp_file, "w") as f:
                json.dump({"keys": list(api_keys)}, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic rename
            os.replace(temp_file, api_keys_file)
            logger.info(f"✓ Saved {len(api_keys)} API keys")
            
        except Exception as e:
            logger.error(f"Error saving API keys: {e}")
            # Clean up temp file
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

valid_api_keys = load_api_keys()

def cleanup_gpu_memory():
    """Comprehensive GPU memory cleanup - thread-safe"""
    try:
        if torch.cuda.is_available():
            # Ensure we're on the correct device
            with torch.cuda.device(CUDA_DEVICE):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU memory cleaned - Allocated: {allocated:.2f}GB / {total:.2f}GB")
    except Exception as e:
        logger.error(f"Error during GPU cleanup: {e}")

def get_current_load() -> dict:
    """Get current system load information"""
    with job_lock:
        active_count = len([job for job in active_jobs.values() if job["status"] == JobStatus.PROCESSING])
        queued_count = len(job_queue)
        
    return {
        "active_jobs": active_count,
        "queued_jobs": queued_count,
        "max_concurrent": MAX_CONCURRENT_JOBS,
        "available_slots": MAX_CONCURRENT_JOBS - active_count,
        "queue_capacity": MAX_QUEUE_SIZE - queued_count
    }

async def periodic_cleanup_task():
    """Background task for periodic cleanup"""
    while True:
        try:
            cleanup_old_jobs()
            await asyncio.sleep(CLEANUP_INTERVAL)
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(60)

def estimate_wait_time(queue_position: int) -> int:
    """Estimate wait time based on queue position"""
    base_time_per_job = 45
    current_load = get_current_load()
    active_jobs_time = current_load["active_jobs"] * base_time_per_job
    queue_wait_time = queue_position * base_time_per_job
    return active_jobs_time + queue_wait_time

async def monitor_stuck_jobs():
    """Monitor and handle stuck jobs"""
    while True:
        try:
            with job_lock:
                now = datetime.now()
                for job_id, job_data in list(active_jobs.items()):
                    if job_data["status"] == JobStatus.PROCESSING:
                        started_at = job_data.get("started_at")
                        if started_at:
                            elapsed = (now - started_at).total_seconds()
                            if elapsed > JOB_TIMEOUT:
                                logger.warning(f"Job {job_id} exceeded timeout, marking as timed out")
                                job_data.update({
                                    "status": JobStatus.TIMEOUT,
                                    "error": f"Job exceeded timeout of {JOB_TIMEOUT}s",
                                    "completed_at": now,
                                    "processing_time": elapsed
                                })
            
            await asyncio.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            logger.error(f"Error in stuck job monitor: {e}")
            await asyncio.sleep(60)

def cleanup_old_jobs():
    """Clean up old completed/failed jobs from memory"""
    with job_lock:
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        jobs_to_move = []
        for job_id, job_data in active_jobs.items():
            if job_data.get("completed_at") and job_data["completed_at"] < cutoff_time:
                jobs_to_move.append(job_id)
        
        for job_id in jobs_to_move:
            job_history[job_id] = active_jobs.pop(job_id)
        
        history_to_remove = []
        for job_id, job_data in job_history.items():
            if job_data.get("created_at", datetime.now()) < cutoff_time:
                history_to_remove.append(job_id)
        
        for job_id in history_to_remove:
            job_history.pop(job_id)
        
        if jobs_to_move or history_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_move)} active jobs and {len(history_to_remove)} history jobs")

async def load_models():
    """Load all ComfyUI models and nodes for FLUX"""
    global models_loaded, NODE_CLASS_MAPPINGS
    
    if models_loaded:
        return
    
    try:
        logger.info("Loading FLUX image models on GPU 0...")
        cleanup_gpu_memory()
        
        await import_custom_nodes()
        
        from nodes import (
            NODE_CLASS_MAPPINGS as NCM,
            LoraLoader,
            DualCLIPLoader,
            KSampler,
            EmptyLatentImage,
            VAEDecode,
            CLIPTextEncode,
            SaveImage,
            VAELoader,
        )
        NODE_CLASS_MAPPINGS = NCM
        
        logger.info("FLUX image models loaded successfully on GPU 0")
        models_loaded = True
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        raise

async def process_job_queue():
    """Background task to process job queue"""
    while True:
        try:
            current_load = get_current_load()
            
            if current_load["available_slots"] > 0 and len(job_queue) > 0:
                with job_lock:
                    if job_queue:
                        job_data = job_queue.popleft()
                        job_id = job_data["job_id"]
                        
                        if job_id in active_jobs:
                            active_jobs[job_id]["status"] = JobStatus.PROCESSING
                            active_jobs[job_id]["started_at"] = datetime.now()
                            
                            logger.info(f"Starting job {job_id} - Queue size: {len(job_queue)}")
                
                # Start job processing outside the lock
                asyncio.create_task(
                    process_image_generation_job_async(job_id, job_data["request"])
                )
            
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Error in job queue processor: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(5)


async def process_image_generation_job_async(job_id: str, request: ImageGenerationRequest):
    """Process a single image generation job in thread pool"""
    loop = asyncio.get_event_loop()
    
    try:
        # Use functools.partial to bind the arguments
        await asyncio.wait_for(
            loop.run_in_executor(
                generation_executor,
                functools.partial(process_image_generation_job_sync, job_id, request)
            ),
            timeout=JOB_TIMEOUT
        )
    except asyncio.TimeoutError:
        logger.error(f"Job {job_id} timed out after {JOB_TIMEOUT}s")
        with job_lock:
            if job_id in active_jobs:
                active_jobs[job_id].update({
                    "status": JobStatus.TIMEOUT,
                    "error": f"Job exceeded timeout of {JOB_TIMEOUT}s",
                    "completed_at": datetime.now(),
                    "processing_time": JOB_TIMEOUT
                })
    except Exception as e:
        logger.error(f"Unexpected error in job {job_id}: {e}")
        import traceback
        traceback.print_exc()
        with job_lock:
            if job_id in active_jobs:
                active_jobs[job_id].update({
                    "status": JobStatus.FAILED,
                    "error": f"Unexpected error: {str(e)}",
                    "completed_at": datetime.now()
                })
                
def process_image_generation_job_sync(job_id: str, request: ImageGenerationRequest):
    """
    Synchronous image generation job - runs in thread pool to avoid blocking.
    This is your existing generation code but made synchronous.
    """
    start_time = time.time()
    max_attempts = 2
    
    for attempt in range(max_attempts):
        try:
            logger.info(f"Processing job {job_id} (attempt {attempt + 1}/{max_attempts}): {request.prompt[:100]}...")
            cleanup_gpu_memory()
            
            if request.seed is None:
                request.seed = random.randint(1, 2**64)
            
            with torch.inference_mode():
                from nodes import (
                    LoraLoader,
                    DualCLIPLoader,
                    KSampler,
                    EmptyLatentImage,
                    VAEDecode,
                    CLIPTextEncode,
                    SaveImage,
                    VAELoader,
                )
                
                # Create empty latent image
                emptylatentimage = EmptyLatentImage()
                emptylatentimage_5 = emptylatentimage.generate(
                    width=request.width, 
                    height=request.height, 
                    batch_size=request.batch_size
                )

                # Load UNET model
                unetloadergguf = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
                unetloadergguf_27 = unetloadergguf.load_unet(unet_name=UNET_MODEL)
                easy_cleangpuused = NODE_CLASS_MAPPINGS["easy cleanGpuUsed"]()

                # Load CLIP models
                dualcliploader = DualCLIPLoader()
                dualcliploader_11 = dualcliploader.load_clip(
                    clip_name1=CLIP_L_MODEL,
                    clip_name2=T5_MODEL,
                    type="flux",
                    device="default",
                )

                # Load LoRA
                loraloader = LoraLoader()
                loraloader_29 = loraloader.load_lora(
                    lora_name=LORA_MODEL,
                    strength_model=1,
                    strength_clip=1,
                    model=get_value_at_index(unetloadergguf_27, 0),
                    clip=get_value_at_index(dualcliploader_11, 0),
                )

                # Encode positive prompt
                cliptextencode = CLIPTextEncode()
                cliptextencode_6 = cliptextencode.encode(
                    text=request.prompt,
                    clip=get_value_at_index(loraloader_29, 1),
                )

                # Load VAE
                vaeloader = VAELoader()
                vaeloader_10 = vaeloader.load_vae(vae_name=VAE_MODEL)

                # Encode negative prompt
                cliptextencode_31 = cliptextencode.encode(
                    text="Blurry, low-resolution, multiple heads, deformed, out of frame, extra limbs, bad anatomy, ugly, tiling, cropped, signature, watermark, photo realistic",
                    clip=get_value_at_index(loraloader_29, 1),
                )

                cleanup_gpu_memory()

                # Sample
                ksampler = KSampler()
                ksampler_30 = ksampler.sample(
                    seed=request.seed,
                    steps=request.steps,
                    cfg=request.guidance_scale,
                    sampler_name="euler",
                    scheduler="normal",
                    denoise=1,
                    model=get_value_at_index(unetloadergguf_27, 0),
                    positive=get_value_at_index(cliptextencode_6, 0),
                    negative=get_value_at_index(cliptextencode_31, 0),
                    latent_image=get_value_at_index(emptylatentimage_5, 0),
                )

                # Decode VAE
                vaedecode = VAEDecode()
                vaedecode_8 = vaedecode.decode(
                    samples=get_value_at_index(ksampler_30, 0),
                    vae=get_value_at_index(vaeloader_10, 0),
                )

                # Save image
                saveimage = SaveImage()
                timestamp = int(time.time())
                filename_prefix = f"gpu0_{timestamp}_FLUX_{job_id[:8]}"
                
                saveimage_9 = saveimage.save_images(
                    filename_prefix=filename_prefix,
                    images=get_value_at_index(vaedecode_8, 0)
                )
                
                unique_id = random.randint(1, 2**63)
                
                easy_cleangpuused_93 = easy_cleangpuused.empty_cache(
                    anything=get_value_at_index(saveimage_9, 0),
                    unique_id=unique_id,
                )
                
                cleanup_gpu_memory()

            # Find the generated image
            image_filename = f'{filename_prefix}_00001_.png'
            full_image_path = os.path.join(OUTPUT_DIR, image_filename)
            
            if not os.path.exists(full_image_path):
                alternative_names = [
                    f'{filename_prefix}.png',
                    f'{filename_prefix}_00001.png',
                    f'{filename_prefix}_1.png',
                    f'{filename_prefix}_00000_.png',
                ]
                
                for alt_name in alternative_names:
                    alt_path = os.path.join(OUTPUT_DIR, alt_name)
                    if os.path.exists(alt_path):
                        image_filename = alt_name
                        full_image_path = alt_path
                        logger.info(f"Found image with alternative name: {image_filename}")
                        break
                else:
                    try:
                        all_files = os.listdir(OUTPUT_DIR)
                        matching_files = [f for f in all_files if filename_prefix in f]
                        logger.error(f"Expected image not found: {image_filename}")
                        logger.error(f"Files in output directory: {matching_files if matching_files else 'No matching files'}")
                        
                        if matching_files:
                            image_filename = matching_files[0]
                            full_image_path = os.path.join(OUTPUT_DIR, image_filename)
                            logger.info(f"Using first matching file: {image_filename}")
                        else:
                            raise FileNotFoundError(f"No generated image found with prefix: {filename_prefix}")
                    except Exception as list_error:
                        logger.error(f"Error listing output directory: {list_error}")
                        raise FileNotFoundError(f"Generated image not found: {image_filename}")
            
            logger.info(f"Image saved successfully: {full_image_path}")
            processing_time = time.time() - start_time
            
            with job_lock:
                if job_id in active_jobs:
                    active_jobs[job_id].update({
                        "status": JobStatus.COMPLETED,
                        "result_url": image_filename,
                        "download_url": f"/download/{image_filename}",
                        "output_filename": image_filename,
                        "completed_at": datetime.now(),
                        "processing_time": processing_time,
                        "generation_details": {
                            "seed": request.seed,
                            "model": request.model,
                            "width": request.width,
                            "height": request.height,
                            "steps": request.steps,
                            "guidance_scale": request.guidance_scale,
                            "processing_time": f"{processing_time:.2f}s",
                            "gpu": CUDA_DEVICE
                        }
                    })
            
            logger.info(f"Job {job_id} completed successfully in {processing_time:.2f}s: {image_filename}")
            return
            
        except torch.cuda.OutOfMemoryError as oom_error:
            logger.warning(f"CUDA out of memory for job {job_id} on attempt {attempt + 1}/{max_attempts}: {oom_error}")
            cleanup_gpu_memory()
            if attempt == max_attempts - 1:
                with job_lock:
                    if job_id in active_jobs:
                        active_jobs[job_id].update({
                            "status": JobStatus.FAILED,
                            "error": f"GPU out of memory after {max_attempts} attempts",
                            "completed_at": datetime.now(),
                            "processing_time": time.time() - start_time
                        })
                logger.error(f"Job {job_id} failed due to GPU OOM after {max_attempts} attempts")
                
        except Exception as e:
            logger.error(f"Error processing job {job_id} on attempt {attempt + 1}/{max_attempts}: {e}")
            import traceback
            traceback.print_exc()
            cleanup_gpu_memory()
            if attempt == max_attempts - 1:
                with job_lock:
                    if job_id in active_jobs:
                        active_jobs[job_id].update({
                            "status": JobStatus.FAILED,
                            "error": str(e),
                            "completed_at": datetime.now(),
                            "processing_time": time.time() - start_time
                        })
                logger.error(f"Job {job_id} failed after {max_attempts} attempts: {str(e)}")

async def cleanup_task():
    """Background task for periodic cleanup"""
    while True:
        try:
            cleanup_old_jobs()
            await asyncio.sleep(CLEANUP_INTERVAL)
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Self-Managing FLUX Worker (GPU 0)...")
    
    await load_models()
    
    # Start all background tasks
    queue_processor_task = asyncio.create_task(process_job_queue())
    cleanup_task_instance = asyncio.create_task(periodic_cleanup_task())
    stuck_job_monitor_task = asyncio.create_task(monitor_stuck_jobs())
    
    yield
    
    logger.info("Shutting down FLUX worker...")
    
    # Shutdown executor gracefully
    generation_executor.shutdown(wait=True, cancel_futures=False)
    
    # Cancel all background tasks
    for task in [queue_processor_task, cleanup_task_instance, stuck_job_monitor_task]:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        
app = FastAPI(
    title="Self-Managing FLUX Image Generation Worker",
    description="FLUX image generation with built-in job queue and load management",
    version="2.0.0",
    lifespan=lifespan
)

def validate_api_key(api_key: Optional[str] = None) -> bool:
    """Validate API key"""
    if not valid_api_keys:
        return True
    return api_key in valid_api_keys

@app.post("/generate")
async def generate_image(request: ImageGenerationRequest):
    """Submit image generation job to queue"""
    
    if not validate_api_key(request.api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not models_loaded or NODE_CLASS_MAPPINGS is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Please try again in a moment.")
    
    current_load = get_current_load()
    
    if current_load["queued_jobs"] >= MAX_QUEUE_SIZE:
        raise HTTPException(
            status_code=429, 
            detail=f"Queue is full. Maximum {MAX_QUEUE_SIZE} jobs can be queued. Please try again later."
        )
    
    job_id = str(uuid.uuid4())
    queue_position = current_load["queued_jobs"] + 1
    estimated_wait = estimate_wait_time(queue_position)
    
    job_entry = {
        "job_id": job_id,
        "status": JobStatus.QUEUED,
        "created_at": datetime.now(),
        "request": request,
        "estimated_wait_time": estimated_wait,
        "queue_position": queue_position
    }
    
    with job_lock:
        active_jobs[job_id] = job_entry.copy()
        job_queue.append({
            "job_id": job_id,
            "request": request,
            "priority": request.priority
        })
    
    logger.info(f"Job {job_id} added to queue. Queue size: {len(job_queue)}, Position: {queue_position}")
    
    return {
        "job_id": job_id,
        "status": JobStatus.QUEUED,
        "message": "Job added to queue successfully",
        "status_url": f"/job/{job_id}",
        "estimated_wait_time": estimated_wait,
        "queue_position": queue_position,
        "current_load": current_load
    }


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a specific job - ALWAYS responds"""
    
    # CRITICAL: Yield control immediately
    await asyncio.sleep(0)
    
    try:
        # Use a timeout on the lock to prevent hanging
        lock_acquired = job_lock.acquire(timeout=0.1)
        
        if not lock_acquired:
            # If we can't get the lock quickly, return a "busy" response
            raise HTTPException(
                status_code=503,
                detail="System busy - unable to retrieve job status. Try again in a moment."
            )
        
        try:
            # Check active jobs first
            if job_id in active_jobs:
                job_data = active_jobs[job_id].copy()

                # Update queue position and estimated wait time if queued
                if job_data["status"] == JobStatus.QUEUED:
                    for i, queued_job in enumerate(job_queue):
                        if queued_job["job_id"] == job_id:
                            job_data["queue_position"] = i + 1
                            job_data["estimated_wait_time"] = estimate_wait_time(i + 1)
                            break
                
                # If job is processing, estimate remaining time
                elif job_data["status"] == JobStatus.PROCESSING:
                    job_data["queue_position"] = 0
                    job_data["estimated_wait_time"] = estimate_wait_time(1)
                            
                # Remove internal fields from response
                job_data.pop("request", None)
                job_data.pop("started_at", None)

                return JobResponse(**job_data)

            # Check job history
            elif job_id in job_history:
                history_data = job_history[job_id].copy()
                history_data.pop("request", None)
                history_data.pop("started_at", None)
                return JobResponse(**history_data)
            
            else:
                raise HTTPException(status_code=404, detail="Job not found")
                
        finally:
            # Always release the lock
            job_lock.release()
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error getting job status for {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving job status: {str(e)}"
        )

def update_job_status(job_id: str, updates: dict):
    """Safely update job status with thread lock"""
    with job_lock:
        if job_id in active_jobs:
            active_jobs[job_id].update(updates)
            return True
    return False

@app.delete("/job/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a queued job"""
    
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = active_jobs[job_id]
    
    if job_data["status"] == JobStatus.QUEUED:
        with job_lock:
            # Remove from queue
            job_queue_list = list(job_queue)
            job_queue.clear()
            
            for queued_job in job_queue_list:
                if queued_job["job_id"] != job_id:
                    job_queue.append(queued_job)
            
            # Update job status
            active_jobs[job_id].update({
                "status": JobStatus.CANCELLED,
                "completed_at": datetime.now(),
                "error": "Job cancelled by user"
            })
        
        logger.info(f"Job {job_id} cancelled successfully")
        
        return {
            "job_id": job_id,
            "status": JobStatus.CANCELLED,
            "message": "Job cancelled successfully"
        }
    
    elif job_data["status"] == JobStatus.PROCESSING:
        raise HTTPException(status_code=400, detail="Cannot cancel job that is currently processing")
    
    else:
        raise HTTPException(status_code=400, detail=f"Cannot cancel job with status: {job_data['status']}")

@app.get("/queue")
async def get_queue_status():
    """Get current queue status and statistics"""
    await asyncio.sleep(0)
    current_load = get_current_load()
    
    # Get queue details
    with job_lock:
        queue_jobs = []
        for i, queued_job in enumerate(job_queue):
            job_id = queued_job["job_id"]
            if job_id in active_jobs:
                job_data = active_jobs[job_id]
                queue_jobs.append({
                    "job_id": job_id,
                    "position": i + 1,
                    "created_at": job_data["created_at"].isoformat(),
                    "estimated_wait_time": estimate_wait_time(i + 1),
                    "priority": queued_job.get("priority", "normal")
                })
    
    return {
        "queue_length": len(job_queue),
        "max_queue_size": MAX_QUEUE_SIZE,
        "available_queue_slots": MAX_QUEUE_SIZE - len(job_queue),
        "active_jobs": current_load["active_jobs"],
        "gpu_busy": is_gpu_busy(),
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "available_processing_slots": current_load["available_slots"],
        "queue_jobs": queue_jobs,
        "system_status": "healthy" if models_loaded else "loading"
    }

@app.get("/stats")
async def get_statistics():
    """Get comprehensive system statistics"""
    if is_gpu_busy():
        current_load = get_current_load()
        return {
            "status": "gpu_busy",
            "message": "GPU is currently processing. Try again in a moment.",
            "current_load": current_load,
            "active_jobs": current_load["active_jobs"],
            "queued_jobs": current_load["queued_jobs"]
        }
    
    current_load = get_current_load()
    
    # Calculate job statistics
    with job_lock:
        total_jobs = len(active_jobs) + len(job_history)
        completed_jobs = sum(1 for job in list(active_jobs.values()) + list(job_history.values()) 
                           if job["status"] == JobStatus.COMPLETED)
        failed_jobs = sum(1 for job in list(active_jobs.values()) + list(job_history.values()) 
                        if job["status"] == JobStatus.FAILED)
        
        # Calculate average processing time for completed jobs
        completed_job_times = [
            job.get("processing_time", 0) 
            for job in list(active_jobs.values()) + list(job_history.values()) 
            if job["status"] == JobStatus.COMPLETED and job.get("processing_time")
        ]
        avg_processing_time = sum(completed_job_times) / len(completed_job_times) if completed_job_times else 0
    
    # GPU memory info
    gpu_memory = {}
    if torch.cuda.is_available():
        gpu_memory = {
            "total_memory_gb": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}",
            "allocated_gb": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f}",
            "reserved_gb": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f}",
            "utilization": f"{(torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory) * 100:.1f}%"
        }
    
    # System info
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    return {
        "service_info": {
            "name": "Self-Managing FLUX Worker",
            "version": "2.0.0",
            "gpu_id": CUDA_DEVICE,
            "models_loaded": models_loaded
        },
        "current_load": current_load,
        "job_statistics": {
            "total_jobs_processed": total_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "success_rate": f"{(completed_jobs/total_jobs)*100:.1f}%" if total_jobs > 0 else "0%",
            "average_processing_time": f"{avg_processing_time:.2f}s"
        },
        "system_resources": {
            "gpu_memory": gpu_memory,
            "cpu_usage": f"{cpu_percent}%",
            "ram_usage": f"{memory.percent}%",
            "ram_available_gb": f"{memory.available / 1024**3:.2f}"
        },
        "configuration": {
            "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
            "max_queue_size": MAX_QUEUE_SIZE,
            "job_timeout": JOB_TIMEOUT,
            "cleanup_interval": CLEANUP_INTERVAL
        }
    }

@app.post("/cleanup_gpu")
async def manual_gpu_cleanup(api_key: Optional[str] = None):
    """
    Manually trigger GPU memory cleanup.
    Useful for clearing memory between jobs or when GPU memory is stuck.
    """
    
    # Validate API key if system is using API keys
    if valid_api_keys and api_key not in valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if is_gpu_busy():
        raise HTTPException(
            status_code=503,
            detail="Cannot cleanup GPU while jobs are processing. Please wait."
        )
    try:
        # Check if any jobs are currently processing
        current_load = get_current_load()
        
        if current_load["active_jobs"] > 0:
            logger.warning("GPU cleanup requested while jobs are processing")
            return {
                "status": "warning",
                "message": "Cleanup performed, but jobs are currently processing",
                "active_jobs": current_load["active_jobs"],
                "note": "For best results, wait until all jobs complete"
            }
        
        # Get memory state before cleanup
        if torch.cuda.is_available():
            memory_before = {
                "allocated_gb": torch.cuda.memory_allocated(0) / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved(0) / 1024**3,
                "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
        else:
            raise HTTPException(status_code=503, detail="CUDA not available")
        
        # Perform aggressive cleanup
        logger.info("Manual GPU cleanup triggered")
        
        # Multiple rounds of cleanup
        for i in range(5):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            if i < 4:
                time.sleep(0.1)
        
        # Try IPC collection
        try:
            torch.cuda.ipc_collect()
        except:
            pass
        
        # Additional garbage collection
        for _ in range(3):
            gc.collect()
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        
        # Get memory state after cleanup
        memory_after = {
            "allocated_gb": torch.cuda.memory_allocated(0) / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved(0) / 1024**3,
            "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
        
        memory_freed = memory_before["allocated_gb"] - memory_after["allocated_gb"]
        
        logger.info(f"Manual cleanup completed - freed {memory_freed:.2f}GB")
        
        return {
            "status": "success",
            "message": "GPU memory cleanup completed",
            "memory_before": {
                "allocated": f"{memory_before['allocated_gb']:.2f} GB",
                "reserved": f"{memory_before['reserved_gb']:.2f} GB",
                "total": f"{memory_before['total_gb']:.2f} GB"
            },
            "memory_after": {
                "allocated": f"{memory_after['allocated_gb']:.2f} GB",
                "reserved": f"{memory_after['reserved_gb']:.2f} GB",
                "total": f"{memory_after['total_gb']:.2f} GB"
            },
            "memory_freed": f"{memory_freed:.2f} GB",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error during manual GPU cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@app.get("/gpu_memory")
async def get_gpu_memory_status():
    """
    Get current GPU memory status without performing cleanup.
    Useful for monitoring memory usage.
    """
    if is_gpu_busy():
        raise HTTPException(
            status_code=503, 
            detail="GPU busy - memory stats unavailable during image generation"
        )
    
    try:
        if not torch.cuda.is_available():
            raise HTTPException(status_code=503, detail="CUDA not available")
        
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated
        
        # Get peak memory stats
        peak_allocated = torch.cuda.max_memory_allocated(0) / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved(0) / 1024**3
        
        return {
            "gpu_id": CUDA_DEVICE,
            "current_memory": {
                "allocated_gb": f"{allocated:.2f}",
                "reserved_gb": f"{reserved:.2f}",
                "free_gb": f"{free:.2f}",
                "total_gb": f"{total:.2f}",
                "utilization_percent": f"{(allocated/total)*100:.1f}%"
            },
            "peak_memory": {
                "allocated_gb": f"{peak_allocated:.2f}",
                "reserved_gb": f"{peak_reserved:.2f}"
            },
            "current_load": get_current_load(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting GPU memory status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory status: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    await asyncio.sleep(0)
    current_load = get_current_load()
    
    # Determine health status
    health_status = "healthy"
    if not models_loaded:
        health_status = "loading"
    elif current_load["available_slots"] == 0 and current_load["queue_capacity"] == 0:
        health_status = "overloaded"
    
    gpu_memory = {}
    if torch.cuda.is_available():
        gpu_memory = {
            "total_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB",
            "allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
            "reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB"
        }
    
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    return {
        "status": health_status,
        "gpu_id": CUDA_DEVICE,
        "service": "flux_image_generation_self_managing",
        "models_loaded": models_loaded,
        "current_load": current_load,
        "gpu_memory": gpu_memory,
        "gpu_busy": is_gpu_busy(),
        "cpu_usage": f"{cpu_percent}%",
        "ram_usage": f"{memory.percent}%",
        "cuda_available": torch.cuda.is_available(),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/download/{filename}")
async def download_image(filename: str):
    """Download a generated image."""
    try:
        # Check multiple possible output directories
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "output", filename),
            os.path.join(os.path.dirname(__file__), "..", "output", filename),
        ]
        
        # Try ComfyUI default output directory
        comfyui_path = find_path("ComfyUI")
        if comfyui_path:
            possible_paths.append(os.path.join(comfyui_path, "output", filename))
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
        
        if not file_path:
            logger.error(f"Image not found: {filename}")
            raise HTTPException(status_code=404, detail="Image not found")
        
        logger.info(f"Downloading image: {filename}")
        return FileResponse(file_path, media_type="image/png", filename=filename)
        
    except Exception as e:
        logger.error(f"Error downloading image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/generate_api_key")
async def generate_api_key(request: dict):
    """Generate new API key"""
    email = request.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email required")
    
    new_key = f"flux_key_{uuid.uuid4().hex[:16]}"
    
    global valid_api_keys
    valid_api_keys.add(new_key)
    save_api_keys(valid_api_keys)
    
    logger.info(f"Generated new API key for {email}")
    
    return {
        "api_key": new_key,
        "email": email,
        "service": "flux_image_generation",
        "created_at": datetime.now().isoformat()
    }

@app.get("/list_jobs")
async def list_jobs(limit: int = 50, status_filter: Optional[str] = None):
    """List recent jobs with optional status filter"""
    
    with job_lock:
        all_jobs = list(active_jobs.values()) + list(job_history.values())
    
    # Filter by status if provided
    if status_filter:
        all_jobs = [job for job in all_jobs if job.get("status") == status_filter]
    
    # Sort by creation time (newest first)
    all_jobs.sort(key=lambda x: x.get("created_at", datetime.min), reverse=True)
    
    # Limit results
    limited_jobs = all_jobs[:limit]
    
    # Format for response
    formatted_jobs = []
    for job in limited_jobs:
        formatted_job = {
            "job_id": job["job_id"],
            "status": job["status"],
            "created_at": job["created_at"].isoformat(),
            "processing_time": job.get("processing_time"),
        }
        
        if job.get("completed_at"):
            formatted_job["completed_at"] = job["completed_at"].isoformat()
        
        if job.get("error"):
            formatted_job["error"] = job["error"]
        
        if job.get("result_url"):
            formatted_job["result_url"] = job["result_url"]
        
        formatted_jobs.append(formatted_job)
    
    return {
        "total_jobs": len(all_jobs),
        "returned_jobs": len(formatted_jobs),
        "jobs": formatted_jobs
    }
@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "paths": {
            "comfyui": COMFYUI_PATH,
            "output": OUTPUT_DIR
        },
        "models": {
            "filenames": {
                "unet": UNET_MODEL,
                "clip_l": CLIP_L_MODEL,
                "t5": T5_MODEL,
                "vae": VAE_MODEL,
                "lora": LORA_MODEL
            },
            "full_paths_for_verification": {
                "unet": UNET_MODEL_FULL_PATH,
                "clip_l": CLIP_L_MODEL_FULL_PATH,
                "t5": T5_MODEL_FULL_PATH,
                "vae": VAE_MODEL_FULL_PATH,
                "lora": LORA_MODEL_FULL_PATH
            }
        },
        "limits": {
            "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
            "max_queue_size": MAX_QUEUE_SIZE,
            "job_timeout": JOB_TIMEOUT
        },
        "gpu": {
            "device": CUDA_DEVICE,
            "cuda_available": torch.cuda.is_available()
        },
        "note": "ComfyUI loaders use filenames only. Full paths are for verification."
    }

@app.get("/ping")
async def ping():
    """Ultra-fast ping - NEVER blocks"""
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat(),
        "gpu_busy": is_gpu_busy()
    }

@app.post("/batch_generate")
async def batch_generate(requests: List[ImageGenerationRequest]):
    """Submit multiple image generation jobs"""
    
    if len(requests) > MAX_QUEUE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"Batch size cannot exceed {MAX_QUEUE_SIZE} jobs"
        )
    
    # Check if we have enough queue capacity
    current_load = get_current_load()
    if current_load["queue_capacity"] < len(requests):
        raise HTTPException(
            status_code=429, 
            detail=f"Not enough queue capacity. Available: {current_load['queue_capacity']}, Requested: {len(requests)}"
        )
    
    # Validate all API keys first
    for request in requests:
        if not validate_api_key(request.api_key):
            raise HTTPException(status_code=401, detail="Invalid API key in one or more requests")
    
    # Submit all jobs
    job_ids = []
    for request in requests:
        job_id = str(uuid.uuid4())
        
        queue_position = current_load["queued_jobs"] + len(job_ids) + 1
        estimated_wait = estimate_wait_time(queue_position)
        
        job_entry = {
            "job_id": job_id,
            "status": JobStatus.QUEUED,
            "created_at": datetime.now(),
            "request": request,
            "estimated_wait_time": estimated_wait,
            "queue_position": queue_position
        }
        
        with job_lock:
            active_jobs[job_id] = job_entry.copy()
            job_queue.append({
                "job_id": job_id,
                "request": request,
                "priority": request.priority
            })
        
        job_ids.append(job_id)
    
    logger.info(f"Batch of {len(job_ids)} jobs submitted. Total queue size: {len(job_queue)}")
    
    return {
        "message": f"Successfully submitted {len(job_ids)} jobs",
        "job_ids": job_ids,
        "total_jobs_queued": len(job_queue),
        "estimated_completion_time": estimate_wait_time(len(job_queue))
    }

@app.get("/")
async def root():
    """Root endpoint with service information"""
    current_load = get_current_load()
    
    return {
        "service": "Self-Managing FLUX Image Generation Worker",
        "version": "2.0.0",
        "gpu_id": CUDA_DEVICE,
        "status": "healthy" if models_loaded else "loading",
        "capabilities": [
            "flux-dev",
            "flux-schnell", 
            "text-to-image",
            "batch-processing",
            "job-queue-management",
            "auto-load-balancing"
        ],
        "current_load": current_load,
        "endpoints": {
            "generate_single": "/generate",
            "generate_batch": "/batch_generate",
            "job_status": "/job/{job_id}",
            "cancel_job": "/job/{job_id} [DELETE]",
            "queue_status": "/queue",
            "list_jobs": "/list_jobs",
            "health": "/health",
            "stats": "/stats",
            "download": "/download/{filename}",
            "generate_api_key": "/generate_api_key"
        },
        "features": {
            "built_in_queue": "Automatic job queuing when at capacity",
            "load_balancing": "Smart job processing based on system load",
            "gpu_memory_management": "Automatic memory cleanup between jobs",
            "job_persistence": "Job status tracking and history",
            "batch_processing": "Submit multiple jobs at once",
            "api_key_management": "Optional API key authentication"
        },
        "limits": {
            "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
            "max_queue_size": MAX_QUEUE_SIZE,
            "job_timeout_seconds": JOB_TIMEOUT
        }
    }


if __name__ == "__main__":
    # ==================== HARDCODED TEST GENERATION ====================
    print("="*70)
    print("RUNNING FLUX DEV TEST GENERATION")
    print("="*70)

    TEST_PROMPT = "A majestic black panther standing on a rocky cliff edge in a heroic brave position, muscles tensed, piercing golden eyes staring intensely forward, sleek fur glistening under dramatic sunset lighting, powerful stance with one paw raised, misty mountains in the background, cinematic composition, highly detailed, 8k quality"

    try:
        import asyncio
        from datetime import datetime

        # Create test output directory
        test_output_dir = os.path.join(COMFYUI_PATH, "test_outputs")
        os.makedirs(test_output_dir, exist_ok=True)

        print(f"Test Prompt: {TEST_PROMPT[:80]}...")
        print(f"Output Directory: {test_output_dir}")

        # Create a minimal test request
        test_request = {
            "prompt": TEST_PROMPT,
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 20,
            "guidance_scale": 3.5,
            "seed": 42
        }

        # Run the generation (using the existing workflow queue_prompt function)
        print("Generating test output...")
        result = asyncio.run(queue_prompt(test_request))

        print(f"✓ FLUX Dev test generation completed successfully!")
        print(f"✓ Output saved to: {test_output_dir}")

    except Exception as e:
        print(f"⚠ Test generation failed: {str(e)}")
        print("Continuing to start server...")

    print("="*70)
    # ==================== END TEST GENERATION ====================


    uvicorn.run(app, host="0.0.0.0", port=9090, log_level="info")