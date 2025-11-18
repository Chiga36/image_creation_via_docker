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
import base64
import io
from PIL import Image
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
# CONFIGURATION - Set your exact paths here
# ============================================================================
CUDA_DEVICE = 0

# Main directories - use absolute paths
COMFYUI_PATH = os.environ.get("COMFYUI_PATH", "/home/comfyuser/ComfyUI")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/home/comfyuser/ComfyUI/output")
INPUT_DIR = os.environ.get("INPUT_DIR", "/home/comfyuser/ComfyUI/input")

# Model filenames - ComfyUI loaders expect just the filename, not full paths
UNET_MODEL = os.environ.get("UNET_MODEL", "flux1-dev-kontext_fp8_scaled.safetensors")
CLIP_L_MODEL = os.environ.get("CLIP_L_MODEL", "clip_l.safetensors")
T5_MODEL = os.environ.get("T5_MODEL", "t5xxl_fp8_e4m3fn_scaled.safetensors")
VAE_MODEL = os.environ.get("VAE_MODEL", "ae.safetensors")

# Optional: Full paths for verification
UNET_MODEL_FULL_PATH = os.environ.get("UNET_MODEL_FULL_PATH", "/home/comfyuser/ComfyUI/models/unet/flux1-dev-kontext_fp8_scaled.safetensors")
CLIP_L_MODEL_FULL_PATH = os.environ.get("CLIP_L_MODEL_FULL_PATH", "/home/comfyuser/ComfyUI/models/clip/clip_l.safetensors")
T5_MODEL_FULL_PATH = os.environ.get("T5_MODEL_FULL_PATH", "/home/comfyuser/ComfyUI/models/clip/t5xxl_fp8_e4m3fn_scaled.safetensors")
VAE_MODEL_FULL_PATH = os.environ.get("VAE_MODEL_FULL_PATH", "/home/comfyuser/ComfyUI/models/vae/ae.safetensors")
# ============================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)

# Configure logging
log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'flux_kontext_worker_gpu0.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Verify paths on startup
def verify_paths():
    """Verify all configured paths exist"""
    logger.info("=" * 70)
    logger.info("VERIFYING FLUX KONTEXT CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"ComfyUI Path: {COMFYUI_PATH}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")
    logger.info(f"Input Directory: {INPUT_DIR}")
    logger.info("")
    logger.info("Model Filenames (ComfyUI will search for these):")
    logger.info(f"  UNET Model: {UNET_MODEL}")
    logger.info(f"  CLIP-L Model: {CLIP_L_MODEL}")
    logger.info(f"  T5 Model: {T5_MODEL}")
    logger.info(f"  VAE Model: {VAE_MODEL}")
    logger.info("=" * 70)
    
    # Create directories if they don't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)
    
    if not os.path.exists(COMFYUI_PATH):
        logger.error(f"ComfyUI path does not exist: {COMFYUI_PATH}")
        raise FileNotFoundError(f"ComfyUI path not found: {COMFYUI_PATH}")
    
    # Verify model files exist in expected locations
    model_checks = {
        "UNET (Kontext)": UNET_MODEL_FULL_PATH,
        "CLIP-L": CLIP_L_MODEL_FULL_PATH,
        "T5": T5_MODEL_FULL_PATH,
        "VAE": VAE_MODEL_FULL_PATH
    }
    
    logger.info("")
    logger.info("Checking model files exist:")
    missing_models = []
    for model_name, full_path in model_checks.items():
        exists = os.path.exists(full_path)
        status = "✓ FOUND" if exists else "✗ MISSING"
        logger.info(f"  {status} - {model_name}: {full_path}")
        
        if exists:
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            logger.info(f"           Size: {size_mb:.2f} MB")
        else:
            missing_models.append(f"{model_name}: {full_path}")
    
    if not os.access(OUTPUT_DIR, os.W_OK):
        logger.error(f"Output directory is not writable: {OUTPUT_DIR}")
        raise PermissionError(f"Cannot write to output directory: {OUTPUT_DIR}")
    
    if not os.access(INPUT_DIR, os.W_OK):
        logger.error(f"Input directory is not writable: {INPUT_DIR}")
        raise PermissionError(f"Cannot write to input directory: {INPUT_DIR}")
    
    logger.info("=" * 70)
    if missing_models:
        logger.error("CRITICAL: Some model files are MISSING!")
        for missing in missing_models:
            logger.error(f"  - {missing}")
        logger.error("Image generation will FAIL without these models!")
    else:
        logger.info("✓ All model files verified successfully!")
    logger.info("=" * 70)

# Job Management Configuration
MAX_CONCURRENT_JOBS = 2
MAX_QUEUE_SIZE = 10
JOB_TIMEOUT = 500
CLEANUP_INTERVAL = 600
generation_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS)
# Global job management
job_queue = deque()
active_jobs = {}
job_history = {}
job_lock = threading.Lock()

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

class KontextImageGenerationRequest(BaseModel):
    prompt: str = ""
    height: int = 1024
    width: int = 1024
    steps: int = 25
    guidance_scale: float = 2.0
    model: str = "flux-kontext"
    priority: str = "normal"
    seed: Optional[int] = None
    batch_size: int = 1
    megapixel: str = "1.0"
    aspect_ratio: str = "1:1 (Perfect Square)"
    reference_image_base64: str
    secondary_image_base64: Optional[str] = None
    api_key: Optional[str] = None

# API Key Management
_api_keys_lock = threading.Lock()

def load_api_keys() -> set:
    """Load API keys with robust error handling and file locking"""
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    os.makedirs(config_dir, exist_ok=True)
    api_keys_file = os.path.join(config_dir, "api_keys.json")
    
    # Default keys to use if file is missing or corrupted
    default_keys = {"master_key_123","kontext_key_456"}
    
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

def save_uploaded_image(image_base64: str, filename: str) -> str:
    """Save base64 encoded image to ComfyUI input directory - thread-safe"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Ensure filename is unique
        base_name, ext = os.path.splitext(filename)
        image_path = os.path.join(INPUT_DIR, filename)
        
        # Handle collision with atomic operation
        counter = 0
        while os.path.exists(image_path):
            counter += 1
            new_filename = f"{base_name}_{counter}{ext}"
            image_path = os.path.join(INPUT_DIR, new_filename)
            filename = new_filename
        
        # Save image
        image.save(image_path)
        
        logger.info(f"Image saved: {image_path}")
        return filename
        
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

gpu_cleanup_lock = threading.Lock()

def cleanup_gpu_memory():
    """Comprehensive GPU memory cleanup - thread-safe"""
    with gpu_cleanup_lock:
        try:
            if torch.cuda.is_available():
                with torch.cuda.device(CUDA_DEVICE):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
                
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"GPU memory cleaned - Allocated: {allocated:.2f}GB / {total:.2f}GB")
        except Exception as e:
            logger.error(f"Error during GPU cleanup: {e}")

def is_gpu_busy() -> bool:
    """Check if GPU is currently processing"""
    with job_lock:
        return any(job["status"] == JobStatus.PROCESSING for job in active_jobs.values())
    
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

def estimate_wait_time(queue_position: int) -> int:
    """Estimate wait time based on queue position"""
    base_time_per_job = 60  # Kontext takes longer
    current_load = get_current_load()
    active_jobs_time = current_load["active_jobs"] * base_time_per_job
    queue_wait_time = queue_position * base_time_per_job
    return active_jobs_time + queue_wait_time

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
    """Load all ComfyUI models and nodes for FLUX Kontext"""
    global models_loaded, NODE_CLASS_MAPPINGS
    
    if models_loaded:
        return
    
    try:
        logger.info("Loading FLUX Kontext models on GPU 0...")
        cleanup_gpu_memory()
        
        await import_custom_nodes()
        
        from nodes import (
            NODE_CLASS_MAPPINGS as NCM,
            VAELoader,
            VAEEncode,
            UNETLoader,
            CLIPTextEncode,
            KSampler,
            DualCLIPLoader,
            VAEDecode,
            ConditioningZeroOut,
            SaveImage,
            LoadImage,
        )
        NODE_CLASS_MAPPINGS = NCM
        
        logger.info("FLUX Kontext models loaded successfully on GPU 0")
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
                            
                            logger.info(f"Starting Kontext job {job_id} - Queue size: {len(job_queue)}")
                            asyncio.create_task(process_kontext_job(job_id, job_data["request"]))
            
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Error in job queue processor: {e}")
            await asyncio.sleep(5)

async def process_kontext_job(job_id: str, request: KontextImageGenerationRequest):
    """Async wrapper for Kontext job processing"""
    loop = asyncio.get_event_loop()
    
    try:
        await asyncio.wait_for(
            loop.run_in_executor(
                generation_executor,
                process_kontext_job_sync,
                job_id,
                request
            ),
            timeout=JOB_TIMEOUT
        )
    except asyncio.TimeoutError:
        logger.error(f"Kontext job {job_id} timed out")
        with job_lock:
            if job_id in active_jobs:
                active_jobs[job_id].update({
                    "status": JobStatus.TIMEOUT,
                    "error": f"Job exceeded timeout of {JOB_TIMEOUT}s",
                    "completed_at": datetime.now()
                })
    except Exception as e:
        logger.error(f"Error in Kontext job wrapper {job_id}: {e}")

def process_kontext_job_sync(job_id: str, request: KontextImageGenerationRequest):
    """Process a single Kontext image generation job"""
    start_time = time.time()
    max_attempts = 2
    
    for attempt in range(max_attempts):
        try:
            logger.info(f"Processing Kontext job {job_id} (attempt {attempt + 1}/{max_attempts}): {request.prompt[:100] if request.prompt else 'Reference-based generation'}...")
            cleanup_gpu_memory()
            
            if request.seed is None:
                request.seed = random.randint(1, 2**64)
            
            # Save uploaded reference images
            timestamp = int(time.time())
            reference_image_name = f"ref_image_{timestamp}_{request.seed}.png"
            saved_ref_image = save_uploaded_image(request.reference_image_base64, reference_image_name)
            
            secondary_image_name = None
            if request.secondary_image_base64:
                secondary_image_name = f"sec_image_{timestamp}_{request.seed}.png"
                saved_sec_image = save_uploaded_image(request.secondary_image_base64, secondary_image_name)
            
            with torch.inference_mode():
                from nodes import (
                    VAELoader,
                    VAEEncode,
                    UNETLoader,
                    CLIPTextEncode,
                    KSampler,
                    DualCLIPLoader,
                    VAEDecode,
                    ConditioningZeroOut,
                    SaveImage,
                    LoadImage,
                )

                # Load reference images
                loadimage = LoadImage()
                loadimage_2 = loadimage.load_image(image=saved_ref_image)
                
                loadimage_3 = None
                if secondary_image_name:
                    loadimage_3 = loadimage.load_image(image=saved_sec_image)

                # Load CLIP models
                dualcliploader = DualCLIPLoader()
                dualcliploader_10 = dualcliploader.load_clip(
                    clip_name1=CLIP_L_MODEL,
                    clip_name2=T5_MODEL,
                    type="flux",
                    device="default",
                )

                # Load VAE
                vaeloader = VAELoader()
                vaeloader_11 = vaeloader.load_vae(vae_name=VAE_MODEL)
######################################################################################
                # Calculate resolution
                fluxresolutionnode = NODE_CLASS_MAPPINGS["FluxResolutionNode"]()
                fluxresolutionnode_23 = fluxresolutionnode.calculate_dimensions(
                    megapixel=request.megapixel,
                    aspect_ratio=request.aspect_ratio,
                    divisible_by="64",
                    custom_ratio=False,
                    custom_aspect_ratio="1:1",
                )

                # Scale reference image
                fluxkontextimagescale = NODE_CLASS_MAPPINGS["FluxKontextImageScale"]()
                fluxkontextimagescale_26 = fluxkontextimagescale.scale(
                    image=get_value_at_index(loadimage_2, 0)
                )

                # Encode reference image
                vaeencode = VAEEncode()
                vaeencode_27 = vaeencode.encode(
                    pixels=get_value_at_index(fluxkontextimagescale_26, 0), 
                    vae=get_value_at_index(vaeloader_11, 0)
                )

                # Encode secondary image if provided
                vaeencode_28 = None
                if loadimage_3:
                    fluxkontextimagescale_28 = fluxkontextimagescale.scale(
                        image=get_value_at_index(loadimage_3, 0)
                    )
                    vaeencode_28 = vaeencode.encode(
                        pixels=get_value_at_index(fluxkontextimagescale_28, 0), 
                        vae=get_value_at_index(vaeloader_11, 0)
                    )

                # Encode text prompt
                cliptextencode = CLIPTextEncode()
                cliptextencode_33 = cliptextencode.encode(
                    text=request.prompt, 
                    clip=get_value_at_index(dualcliploader_10, 0)
                )

                # Load UNET
                unetloader = UNETLoader()
                unetloader_36 = unetloader.load_unet(
                    unet_name=UNET_MODEL, 
                    weight_dtype="default"
                )

                # Initialize nodes
                referencelatent = NODE_CLASS_MAPPINGS["ReferenceLatent"]()
                fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
                conditioningzeroout = ConditioningZeroOut()
                emptysd3latentimage = NODE_CLASS_MAPPINGS["EmptySD3LatentImage"]()
                easy_cleangpuused = NODE_CLASS_MAPPINGS["easy cleanGpuUsed"]()

                cleanup_gpu_memory()

                # Build reference latent conditioning
                if vaeencode_28 and loadimage_3:
                    referencelatent_31 = referencelatent.execute(
                        conditioning=get_value_at_index(cliptextencode_33, 0),
                        latent=get_value_at_index(vaeencode_28, 0),
                    )
                    referencelatent_30 = referencelatent.execute(
                        conditioning=get_value_at_index(referencelatent_31, 0),
                        latent=get_value_at_index(vaeencode_27, 0),
                    )
                else:
                    referencelatent_30 = referencelatent.execute(
                        conditioning=get_value_at_index(cliptextencode_33, 0),
                        latent=get_value_at_index(vaeencode_27, 0),
                    )

                # Apply guidance
                fluxguidance_32 = fluxguidance.append(
                    guidance=request.guidance_scale, 
                    conditioning=get_value_at_index(referencelatent_30, 0)
                )

                # Zero out negative conditioning
                conditioningzeroout_34 = conditioningzeroout.zero_out(
                    conditioning=get_value_at_index(cliptextencode_33, 0)
                )

                # Create empty latent
                emptysd3latentimage_19 = emptysd3latentimage.generate(
                    width=get_value_at_index(fluxresolutionnode_23, 0),
                    height=get_value_at_index(fluxresolutionnode_23, 1),
                    batch_size=request.batch_size,
                )

                cleanup_gpu_memory()

                # Sample
                ksampler = KSampler()
                ksampler_16 = ksampler.sample(
                    seed=request.seed,
                    steps=request.steps,
                    cfg=1,
                    sampler_name="euler",
                    scheduler="beta",
                    denoise=1,
                    model=get_value_at_index(unetloader_36, 0),
                    positive=get_value_at_index(fluxguidance_32, 0),
                    negative=get_value_at_index(conditioningzeroout_34, 0),
                    latent_image=get_value_at_index(emptysd3latentimage_19, 0),
                )

                # Decode VAE
                vaedecode = VAEDecode()
                vaedecode_20 = vaedecode.decode(
                    samples=get_value_at_index(ksampler_16, 0),
                    vae=get_value_at_index(vaeloader_11, 0),
                )

                # Save image
                saveimage = SaveImage()
                filename_prefix = f"gpu0_{timestamp}_FLUX_Kontext_{job_id[:8]}"
                
                saveimage_21 = saveimage.save_images(
                    filename_prefix=filename_prefix,
                    images=get_value_at_index(vaedecode_20, 0)
                )
                unique_id = random.randint(1, 2**63)
                
                easy_cleangpuused_93 = easy_cleangpuused.empty_cache(
                    anything=get_value_at_index(vaedecode_20, 0),
                    unique_id=unique_id,
                )
                cleanup_gpu_memory()

            # Find the generated image file
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
                        "output_filename": image_filename,
                        "download_url": f"/download/{image_filename}",
                        "completed_at": datetime.now(),
                        "processing_time": processing_time,
                        "generation_details": {
                            "seed": request.seed,
                            "model": request.model,
                            "width": get_value_at_index(fluxresolutionnode_23, 0),
                            "height": get_value_at_index(fluxresolutionnode_23, 1),
                            "steps": request.steps,
                            "guidance_scale": request.guidance_scale,
                            "processing_time": f"{processing_time:.2f}s",
                            "gpu": CUDA_DEVICE,
                            "reference_images_used": 2 if secondary_image_name else 1
                        }
                    })
            
            logger.info(f"Kontext job {job_id} completed successfully in {processing_time:.2f}s: {image_filename}")
            return
            
        except torch.cuda.OutOfMemoryError as oom_error:
            logger.warning(f"CUDA out of memory for Kontext job {job_id} on attempt {attempt + 1}/{max_attempts}: {oom_error}")
            cleanup_gpu_memory()
            time.sleep(2)
            if attempt == max_attempts - 1:
                with job_lock:
                    if job_id in active_jobs:
                        active_jobs[job_id].update({
                            "status": JobStatus.FAILED,
                            "error": f"GPU out of memory after {max_attempts} attempts",
                            "completed_at": datetime.now(),
                            "processing_time": time.time() - start_time
                        })
                logger.error(f"Kontext job {job_id} failed due to GPU OOM after {max_attempts} attempts")
                
        except Exception as e:
            logger.error(f"Error processing Kontext job {job_id} on attempt {attempt + 1}/{max_attempts}: {e}")
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
                logger.error(f"Kontext job {job_id} failed after {max_attempts} attempts: {str(e)}")

async def cleanup_task():
    """Background task for periodic cleanup"""
    while True:
        try:
            cleanup_old_jobs()
            await asyncio.sleep(CLEANUP_INTERVAL)
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(60)

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


async def run_startup_test():
    """Run a test generation after server startup - non-blocking"""
    try:
        await asyncio.sleep(5)  # Wait for server to be fully ready

        test_prompt = "A majestic black panther standing on a rocky cliff edge in a heroic brave position, muscles tensed, piercing golden eyes staring intensely forward, wind blowing through its sleek black fur, dramatic cloudy sky background, cinematic lighting, photorealistic, 8k quality"

        print("=" * 70)
        print("RUNNING FLUX KONTEXT TEST GENERATION")
        print("=" * 70)
        print(f"Test Prompt: {test_prompt[:80]}...")
        print(f"Output Directory: {OUTPUT_DIR}")

        # Create test request
        test_request = ImageGenerationRequest(
            prompt=test_prompt,
            width=1024,
            height=1024,
            steps=20,
            seed=42,
            apikey="masterkey123"
        )

        # Generate job ID
        job_id = f"test_{int(time.time())}"

        # Create job entry and add to queue
        with job_lock:
            active_jobs[job_id] = {
                "job_id": job_id,
                "status": JobStatus.QUEUED,
                "request": test_request,
                "created_at": datetime.now(),
                "message": "Test generation"
            }
            job_queue.append({"job_id": job_id, "request": test_request})

        print("Generating test output...")

        # Wait for completion (max 5 minutes)
        for _ in range(150):  # 150 * 2 = 300 seconds
            await asyncio.sleep(2)
            with job_lock:
                if job_id in active_jobs:
                    job = active_jobs[job_id]
                    if job["status"] == JobStatus.COMPLETED:
                        print(f"✓ Test generation completed: {job.get('output_filename')}")
                        print("="  * 70)
                        return
                    elif job["status"] in [JobStatus.FAILED, JobStatus.TIMEOUT]:
                        print(f"⚠ Test generation failed: {job.get('error')}")
                        print("=" * 70)
                        return

        print("⚠ Test generation timed out")
        print("=" * 70)

    except Exception as e:
        print(f"⚠ Test generation failed: {e}")
        print("=" * 70)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FLUX Kontext Worker (GPU 0)...")
    
    await load_models()
    
    # Start all background tasks
    queue_processor_task = asyncio.create_task(process_job_queue())
    cleanup_task_instance = asyncio.create_task(cleanup_task())
    stuck_job_monitor_task = asyncio.create_task(monitor_stuck_jobs())
    
    
    # Run startup test in background
    asyncio.create_task(run_startup_test())
    
    yield
    
    logger.info("Shutting down FLUX Kontext worker...")
    
    generation_executor.shutdown(wait=True, cancel_futures=False)
    
    # Cancel all tasks
    for task in [queue_processor_task, cleanup_task_instance, stuck_job_monitor_task]:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="Self-Managing FLUX Kontext Worker",
    description="FLUX Kontext image generation with built-in job queue and load management",
    version="2.0.0",
    lifespan=lifespan
)

def validate_api_key(api_key: Optional[str] = None) -> bool:
    """Validate API key"""
    if not valid_api_keys:
        return True
    return api_key in valid_api_keys

@app.post("/generate")
async def generate_image(request: KontextImageGenerationRequest):
    """Submit Kontext image generation job to queue"""
    
    if not validate_api_key(request.api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not models_loaded or NODE_CLASS_MAPPINGS is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Please try again in a moment.")
    
    if not request.reference_image_base64:
        raise HTTPException(status_code=400, detail="Reference image is required for Kontext generation")
    
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
    
    logger.info(f"Kontext job {job_id} added to queue. Queue size: {len(job_queue)}, Position: {queue_position}")
    
    return {
        "job_id": job_id,
        "status": JobStatus.QUEUED,
        "message": "Kontext job added to queue successfully",
        "estimated_wait_time": estimated_wait,
        "queue_position": queue_position,
        "current_load": current_load
    }

@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a specific job"""
    
    with job_lock:
        if job_id in active_jobs:
            job_data = active_jobs[job_id].copy()

            if job_data["status"] == JobStatus.QUEUED:
                for i, queued_job in enumerate(job_queue):
                    if queued_job["job_id"] == job_id:
                        job_data["queue_position"] = i + 1
                        job_data["estimated_wait_time"] = estimate_wait_time(i + 1)
                        break
            
            elif job_data["status"] == JobStatus.PROCESSING:
                queue_pos = 0
                for i, queued_job in enumerate(job_queue):
                    if queued_job["job_id"] == job_id:
                        queue_pos = i
                        break
                job_data["queue_position"] = 0
                job_data["estimated_wait_time"] = estimate_wait_time(1)
                        
            job_data.pop("request", None)
            job_data.pop("started_at", None)

            return JobResponse(**job_data)

        elif job_id in job_history:
            history_data = job_history[job_id].copy()
            history_data.pop("request", None)
            history_data.pop("started_at", None)
            return JobResponse(**history_data)

    raise HTTPException(status_code=404, detail="Job not found")

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
            job_queue_list = list(job_queue)
            job_queue.clear()
            
            for queued_job in job_queue_list:
                if queued_job["job_id"] != job_id:
                    job_queue.append(queued_job)
            
            active_jobs[job_id].update({
                "status": JobStatus.CANCELLED,
                "completed_at": datetime.now(),
                "error": "Job cancelled by user"
            })
        
        logger.info(f"Kontext job {job_id} cancelled successfully")
        
        return {
            "job_id": job_id,
            "status": JobStatus.CANCELLED,
            "message": "Kontext job cancelled successfully"
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
    
    with job_lock:
        total_jobs = len(active_jobs) + len(job_history)
        completed_jobs = sum(1 for job in list(active_jobs.values()) + list(job_history.values()) 
                           if job["status"] == JobStatus.COMPLETED)
        failed_jobs = sum(1 for job in list(active_jobs.values()) + list(job_history.values()) 
                        if job["status"] == JobStatus.FAILED)
        
        completed_job_times = [
            job.get("processing_time", 0) 
            for job in list(active_jobs.values()) + list(job_history.values()) 
            if job["status"] == JobStatus.COMPLETED and job.get("processing_time")
        ]
        avg_processing_time = sum(completed_job_times) / len(completed_job_times) if completed_job_times else 0
    
    gpu_memory = {}
    if torch.cuda.is_available():
        gpu_memory = {
            "total_memory_gb": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}",
            "allocated_gb": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f}",
            "reserved_gb": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f}",
            "utilization": f"{(torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory) * 100:.1f}%"
        }
    
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    return {
        "service_info": {
            "name": "Self-Managing FLUX Kontext Worker",
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
        "service": "flux_kontext_generation_self_managing",
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
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "output", filename),
            os.path.join(os.path.dirname(__file__), "..", "output", filename),
        ]
        
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
        
        logger.info(f"Downloading Kontext image: {filename}")
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
    
    new_key = f"flux_kontext_key_{uuid.uuid4().hex[:16]}"
    
    global valid_api_keys
    valid_api_keys.add(new_key)
    save_api_keys(valid_api_keys)
    
    logger.info(f"Generated new API key for {email}")
    
    return {
        "api_key": new_key,
        "email": email,
        "service": "flux_kontext_generation",
        "created_at": datetime.now().isoformat()
    }

@app.get("/list_jobs")
async def list_jobs(limit: int = 50, status_filter: Optional[str] = None):
    """List recent jobs with optional status filter"""
    
    with job_lock:
        all_jobs = list(active_jobs.values()) + list(job_history.values())
    
    if status_filter:
        all_jobs = [job for job in all_jobs if job.get("status") == status_filter]
    
    all_jobs.sort(key=lambda x: x.get("created_at", datetime.min), reverse=True)
    
    limited_jobs = all_jobs[:limit]
    
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
            "comfyui": COMFYUI_PATH,  # Use configured path
            "input": INPUT_DIR,       # Use configured path
            "output": OUTPUT_DIR      # Use configured path
        },
        "models": {
            "filenames": {
                "unet": UNET_MODEL,
                "clip_l": CLIP_L_MODEL,
                "t5": T5_MODEL,
                "vae": VAE_MODEL
            },
            "full_paths_for_verification": {
                "unet": UNET_MODEL_FULL_PATH,
                "clip_l": CLIP_L_MODEL_FULL_PATH,
                "t5": T5_MODEL_FULL_PATH,
                "vae": VAE_MODEL_FULL_PATH
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
                                logger.warning(f"Kontext job {job_id} exceeded timeout")
                                job_data.update({
                                    "status": JobStatus.TIMEOUT,
                                    "error": f"Job exceeded timeout of {JOB_TIMEOUT}s",
                                    "completed_at": now,
                                    "processing_time": elapsed
                                })
            
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"Error in stuck job monitor: {e}")
            await asyncio.sleep(60)

@app.post("/batch_generate")
async def batch_generate(requests: List[KontextImageGenerationRequest]):
    """Submit multiple Kontext image generation jobs"""
    
    if len(requests) > MAX_QUEUE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"Batch size cannot exceed {MAX_QUEUE_SIZE} jobs"
        )
    
    current_load = get_current_load()
    if current_load["queue_capacity"] < len(requests):
        raise HTTPException(
            status_code=429, 
            detail=f"Not enough queue capacity. Available: {current_load['queue_capacity']}, Requested: {len(requests)}"
        )
    
    for request in requests:
        if not validate_api_key(request.api_key):
            raise HTTPException(status_code=401, detail="Invalid API key in one or more requests")
        if not request.reference_image_base64:
            raise HTTPException(status_code=400, detail="Reference image is required for all Kontext generation requests")
    
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
    
    logger.info(f"Batch of {len(job_ids)} Kontext jobs submitted. Total queue size: {len(job_queue)}")
    
    return {
        "message": f"Successfully submitted {len(job_ids)} Kontext jobs",
        "job_ids": job_ids,
        "total_jobs_queued": len(job_queue),
        "estimated_completion_time": estimate_wait_time(len(job_queue))
    }

@app.get("/")
async def root():
    """Root endpoint with service information"""
    current_load = get_current_load()
    
    return {
        "service": "Self-Managing FLUX Kontext Worker",
        "version": "2.0.0",
        "gpu_id": CUDA_DEVICE,
        "status": "healthy" if models_loaded else "loading",
        "capabilities": [
            "flux-kontext",
            "image-to-image",
            "reference-based-generation", 
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
            "generate_api_key": "/generate_api_key",
            "config": "/config"
        },
        "features": {
            "built_in_queue": "Automatic job queuing when at capacity",
            "load_balancing": "Smart job processing based on system load",
            "gpu_memory_management": "Automatic memory cleanup between jobs",
            "job_persistence": "Job status tracking and history",
            "batch_processing": "Submit multiple jobs at once",
            "api_key_management": "Optional API key authentication",
            "reference_image_support": "Support for one or two reference images",
            "kontext_specific": "Optimized for FLUX Kontext model workflows"
        },
        "limits": {
            "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
            "max_queue_size": MAX_QUEUE_SIZE,
            "job_timeout_seconds": JOB_TIMEOUT
        },
        "kontext_features": {
            "reference_images": "Required primary reference image, optional secondary image",
            "resolution_calculation": "Automatic resolution calculation based on megapixel and aspect ratio",
            "guidance_system": "FLUX guidance system for reference conditioning",
            "scheduler": "Beta scheduler optimized for Kontext"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9091, log_level="info")
