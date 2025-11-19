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
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, UploadFile, File
import uvicorn
from pydantic import BaseModel
from fastapi.responses import FileResponse
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import functools
import shutil
from pathlib import Path

# ============================================================================
# CONFIGURATION - Set your exact absolute paths here
# ============================================================================
CUDA_DEVICE = 0

# Main directories - use absolute paths
COMFYUI_PATH = os.environ.get("COMFYUI_PATH", "/home/comfyuser/ComfyUI")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/home/comfyuser/ComfyUI/output")
TEMP_UPLOAD_DIR = os.environ.get("TEMP_UPLOAD_DIR", os.path.join(COMFYUI_PATH, "input"))
# Model file paths - use filenames only (ComfyUI loaders handle paths)
CLIP_MODEL = os.environ.get("CLIP_MODEL", "umt5-xxl-encoder-Q5_K_M.gguf")
VAE_MODEL = os.environ.get("VAE_MODEL", "wan_2.1_vae.safetensors")
UNET_HIGH_NOISE = os.environ.get("UNET_HIGH_NOISE", "Wan2.2-I2V-A14B-HighNoise-Q3_K_S.gguf")
UNET_LOW_NOISE = os.environ.get("UNET_LOW_NOISE", "Wan2.2-I2V-A14B-LowNoise-Q3_K_S.gguf")
LORA_HIGH_NOISE = os.environ.get("LORA_HIGH", "Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors")
LORA_LOW_NOISE = os.environ.get("LORA_LOW", "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors")
RIFE_MODEL = os.environ.get("RIFE_CKPT", "rife47.pth")

# Optional: Full paths for verification (not used by loaders)
CLIP_MODEL_FULL_PATH = os.environ.get("CLIP_MODEL_FULL_PATH", "/home/comfyuser/ComfyUI/models/clip/umt5-xxl-encoder-Q5_K_M.gguf")
VAE_MODEL_FULL_PATH = os.environ.get("VAE_MODEL_FULL_PATH", "/home/comfyuser/ComfyUI/models/vae/wan_2.1_vae.safetensors")
UNET_HIGH_NOISE_FULL_PATH = os.environ.get("UNET_HIGH_NOISE_FULL_PATH", "/home/comfyuser/ComfyUI/models/unet/Wan2.2-I2V-A14B-HighNoise-Q3_K_S.gguf")
UNET_LOW_NOISE_FULL_PATH = os.environ.get("UNET_LOW_NOISE_FULL_PATH", "/home/comfyuser/ComfyUI/models/unet/Wan2.2-I2V-A14B-LowNoise-Q3_K_S.gguf")
LORA_HIGH_FULL_PATH = os.environ.get("LORA_HIGH_FULL_PATH", "/home/comfyuser/ComfyUI/models/loras/Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors")
LORA_LOW_FULL_PATH = os.environ.get("LORA_LOW_FULL_PATH", "/home/comfyuser/ComfyUI/models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors")

os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)

# Configure logging
log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'wan22_worker_gpu0.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Job Management Configuration
MAX_CONCURRENT_JOBS = 1  # Video generation is more resource intensive
MAX_QUEUE_SIZE = 5
JOB_TIMEOUT = 720  # 12 minutes for video generation
CLEANUP_INTERVAL = 600
generation_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS)

# Global job management
job_queue = deque()
active_jobs = {}
job_history = {}
job_lock = threading.Lock()

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

class VideoGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    width: int = 640
    height: int = 640
    video_length: int = 49  # Number of frames
    frame_rate: int = 24
    seed: Optional[int] = None
    shift: float = 8.0
    steps: int = 4
    cfg: float = 1.0
    interpolation_multiplier: int = 2  # RIFE interpolation
    priority: str = "normal"
    api_key: Optional[str] = None

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

def verify_paths():
    """Verify all configured paths exist"""
    logger.info("Verifying configured paths...")
    logger.info(f"ComfyUI Path: {COMFYUI_PATH}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")
    logger.info(f"Temp Upload Directory: {TEMP_UPLOAD_DIR}")
    
    # Create directories if they don't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    
    if not os.path.exists(COMFYUI_PATH):
        logger.error(f"ComfyUI path does not exist: {COMFYUI_PATH}")
        raise FileNotFoundError(f"ComfyUI path not found: {COMFYUI_PATH}")
    
    # Check model files
    model_files = {
        "CLIP": CLIP_MODEL_FULL_PATH,
        "VAE": VAE_MODEL_FULL_PATH,
        "UNET High Noise": UNET_HIGH_NOISE_FULL_PATH,
        "UNET Low Noise": UNET_LOW_NOISE_FULL_PATH
    }
    
    missing_models = []
    for model_name, model_path in model_files.items():
        if not os.path.exists(model_path):
            logger.warning(f"{model_name} model not found: {model_path}")
            missing_models.append(f"{model_name}: {model_path}")
    
    if missing_models:
        logger.warning(f"Missing models: {missing_models}")
    else:
        logger.info("All model files found!")
    
    logger.info("Path verification completed")

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping."""
    try:
        return obj[index]
    except (KeyError, IndexError, TypeError):
        pass
    
    try:
        return obj["result"][index]
    except (KeyError, IndexError, TypeError):
        pass
    
    try:
        if isinstance(obj, (tuple, list)):
            return obj[index]
    except (IndexError, TypeError):
        pass
    
    logger.warning(f"Could not index into object of type {type(obj)}, returning as-is")
    return obj

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

# API Key Management
_api_keys_lock = threading.Lock()

def load_api_keys() -> set:
    """Load API keys with robust error handling"""
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    os.makedirs(config_dir, exist_ok=True)
    api_keys_file = os.path.join(config_dir, "api_keys.json")
    
    default_keys = {"master_key_123","wan_key_456"}
    
    with _api_keys_lock:
        try:
            if not os.path.exists(api_keys_file):
                logger.info("API keys file not found, creating default")
                with open(api_keys_file, "w") as f:
                    json.dump({"keys": list(default_keys)}, f, indent=2)
                return default_keys
            
            with open(api_keys_file, "r") as f:
                content = f.read().strip()
                if not content:
                    logger.warning("API keys file is empty, recreating")
                    with open(api_keys_file, "w") as f:
                        json.dump({"keys": list(default_keys)}, f, indent=2)
                    return default_keys
                
                data = json.loads(content)
                keys = set(data.get("keys", []))
                
                if not keys:
                    logger.warning("No keys found in file, using defaults")
                    return default_keys
                
                logger.info(f"✓ Loaded {len(keys)} API keys")
                return keys
                
        except json.JSONDecodeError as e:
            logger.error(f"⚠️  Corrupted API keys file: {e}")
            backup_file = f"{api_keys_file}.corrupted.{int(time.time())}"
            try:
                shutil.copy2(api_keys_file, backup_file)
                logger.info(f"Backed up to: {backup_file}")
            except:
                pass
            
            try:
                os.remove(api_keys_file)
            except:
                pass
            
            with open(api_keys_file, "w") as f:
                json.dump({"keys": list(default_keys)}, f, indent=2)
            
            logger.info("Created fresh API keys file")
            return default_keys
            
        except Exception as e:
            logger.error(f"Unexpected error loading API keys: {e}")
            return default_keys

def save_api_keys(api_keys: set):
    """Save API keys with atomic write"""
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    api_keys_file = os.path.join(config_dir, "api_keys.json")
    
    with _api_keys_lock:
        try:
            temp_file = f"{api_keys_file}.tmp.{os.getpid()}"
            with open(temp_file, "w") as f:
                json.dump({"keys": list(api_keys)}, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            
            os.replace(temp_file, api_keys_file)
            logger.info(f"✓ Saved {len(api_keys)} API keys")
            
        except Exception as e:
            logger.error(f"Error saving API keys: {e}")
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

valid_api_keys = load_api_keys()

def cleanup_gpu_memory():
    """Comprehensive GPU memory cleanup"""
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
    """Estimate wait time based on queue position (video generation is slower)"""
    base_time_per_job = 180  # 3 minutes per video
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
    """Load all ComfyUI models and nodes for Wan 2.2"""
    global models_loaded, NODE_CLASS_MAPPINGS
    
    if models_loaded:
        return
    
    try:
        logger.info("Loading Wan 2.2 video models on GPU 0...")
        cleanup_gpu_memory()
        
        await import_custom_nodes()
        
        from nodes import (
            NODE_CLASS_MAPPINGS as NCM,
            VAELoader,
            KSamplerAdvanced,
            LoraLoaderModelOnly,
            VAEDecode,
            CLIPTextEncode,
            LoadImage,
        )
        NODE_CLASS_MAPPINGS = NCM
        
        logger.info("Wan 2.2 video models loaded successfully on GPU 0")
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
                
                asyncio.create_task(
                    process_video_generation_job_async(
                        job_id, 
                        job_data["request"], 
                        job_data["input_image_path"]
                    )
                )
            
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Error in job queue processor: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(5)

async def process_video_generation_job_async(job_id: str, request: VideoGenerationRequest, input_image_path: str):
    """Process a single video generation job in thread pool"""
    loop = asyncio.get_event_loop()
    
    try:
        await asyncio.wait_for(
            loop.run_in_executor(
                generation_executor,
                functools.partial(process_video_generation_job_sync, job_id, request, input_image_path)
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

def process_video_generation_job_sync(job_id: str, request: VideoGenerationRequest, input_image_path: str):
    """Synchronous video generation job - runs in thread pool"""
    start_time = time.time()
    max_attempts = 2
    
    for attempt in range(max_attempts):
        try:
            logger.info(f"Processing video job {job_id} (attempt {attempt + 1}/{max_attempts})")
            cleanup_gpu_memory()
            
            if request.seed is None:
                request.seed = random.randint(1, 2**64)
            
            with torch.inference_mode():
                from nodes import (
                    VAELoader,
                    KSamplerAdvanced,
                    LoraLoaderModelOnly,
                    VAEDecode,
                    CLIPTextEncode,
                    LoadImage,
                )
                
                # Load CLIP model
                cliploadergguf = NODE_CLASS_MAPPINGS["CLIPLoaderGGUF"]()
                cliploadergguf_84 = cliploadergguf.load_clip(
                    clip_name=CLIP_MODEL,
                    type="wan"
                )
                
                # Encode prompts
                cliptextencode = CLIPTextEncode()
                cliptextencode_6 = cliptextencode.encode(
                    text=request.prompt,
                    clip=get_value_at_index(cliploadergguf_84, 0),
                )
                
                cliptextencode_7 = cliptextencode.encode(
                    text=request.negative_prompt,
                    clip=get_value_at_index(cliploadergguf_84, 0),
                )
                
                # Load VAE
                vaeloader = VAELoader()
                vaeloader_39 = vaeloader.load_vae(vae_name=VAE_MODEL)
                
                # Load input image
                loadimage = LoadImage()
                # ComfyUI's LoadImage expects just the filename - it searches in the input directory
                # The file should already be in COMFYUI_PATH/input/ folder
                image_filename = os.path.basename(input_image_path)
                
                # Verify the file exists in ComfyUI's input directory
                expected_path = os.path.join(COMFYUI_PATH, "input", image_filename)
                if not os.path.exists(expected_path):
                    # If not there, copy it
                    if os.path.exists(input_image_path):
                        shutil.copy2(input_image_path, expected_path)
                        logger.info(f"Copied image to ComfyUI input dir: {expected_path}")
                    else:
                        raise FileNotFoundError(f"Input image not found: {input_image_path}")
                
                loadimage_52 = loadimage.load_image(image=image_filename)
                
                # Load UNET models
                unetloadergguf = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
                unetloadergguf_61 = unetloadergguf.load_unet(unet_name=UNET_HIGH_NOISE)
                unetloadergguf_62 = unetloadergguf.load_unet(unet_name=UNET_LOW_NOISE)
                
                # Load LoRA models
                loraloadermodelonly = LoraLoaderModelOnly()
                loraloadermodelonly_64 = loraloadermodelonly.load_lora_model_only(
                    lora_name=LORA_HIGH_NOISE,
                    strength_model=1,
                    model=get_value_at_index(unetloadergguf_61, 0),
                )
                
                loraloadermodelonly_66 = loraloadermodelonly.load_lora_model_only(
                    lora_name=LORA_LOW_NOISE,
                    strength_model=1,
                    model=get_value_at_index(unetloadergguf_62, 0),
                )
                
                # Initialize nodes
                modelsamplingsd3 = NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
                unloadmodel = NODE_CLASS_MAPPINGS["UnloadModel"]()
                wanimagetovideo = NODE_CLASS_MAPPINGS["WanImageToVideo"]()
                ksampleradvanced = KSamplerAdvanced()
                vaedecode = VAEDecode()
                rife_vfi = NODE_CLASS_MAPPINGS["RIFE VFI"]()
                vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
                easy_cleangpuused = NODE_CLASS_MAPPINGS["easy cleanGpuUsed"]()
                
                # Patch models
                modelsamplingsd3_68 = modelsamplingsd3.patch(
                    shift=request.shift,
                    model=get_value_at_index(loraloadermodelonly_66, 0),
                )
                
                unloadmodel_85 = unloadmodel.route(
                    value=get_value_at_index(cliptextencode_6, 0)
                )
                
                # Image to video conversion
                wanimagetovideo_50 = wanimagetovideo.EXECUTE_NORMALIZED(
                    width=request.width,
                    height=request.height,
                    length=request.video_length,
                    batch_size=1,
                    positive=get_value_at_index(unloadmodel_85, 0),
                    negative=get_value_at_index(cliptextencode_7, 0),
                    vae=get_value_at_index(vaeloader_39, 0),
                    start_image=get_value_at_index(loadimage_52, 0),
                )
                
                modelsamplingsd3_67 = modelsamplingsd3.patch(
                    shift=request.shift,
                    model=get_value_at_index(loraloadermodelonly_64, 0),
                )
                
                cleanup_gpu_memory()
                
                # First sampling pass (high noise)
                ksampleradvanced_57 = ksampleradvanced.sample(
                    add_noise="enable",
                    noise_seed=request.seed,
                    steps=request.steps,
                    cfg=request.cfg,
                    sampler_name="uni_pc",
                    scheduler="simple",
                    start_at_step=0,
                    end_at_step=2,
                    return_with_leftover_noise="enable",
                    model=get_value_at_index(modelsamplingsd3_67, 0),
                    positive=get_value_at_index(wanimagetovideo_50, 0),
                    negative=get_value_at_index(wanimagetovideo_50, 1),
                    latent_image=get_value_at_index(wanimagetovideo_50, 2),
                )
                
                unloadmodel_87 = unloadmodel.route(
                    value=get_value_at_index(ksampleradvanced_57, 0),
                    model=get_value_at_index(unetloadergguf_61, 0),
                )
                
                # Second sampling pass (low noise)
                ksampleradvanced_58 = ksampleradvanced.sample(
                    add_noise="disable",
                    noise_seed=random.randint(1, 2**64),
                    steps=request.steps,
                    cfg=request.cfg,
                    sampler_name="uni_pc",
                    scheduler="sgm_uniform",
                    start_at_step=2,
                    end_at_step=4,
                    return_with_leftover_noise="disable",
                    model=get_value_at_index(modelsamplingsd3_68, 0),
                    positive=get_value_at_index(wanimagetovideo_50, 0),
                    negative=get_value_at_index(wanimagetovideo_50, 1),
                    latent_image=get_value_at_index(unloadmodel_87, 0),
                )
                
                unloadmodel_88 = unloadmodel.route(
                    value=get_value_at_index(ksampleradvanced_58, 0),
                    model=get_value_at_index(unetloadergguf_62, 0),
                )
                
                # Decode VAE
                vaedecode_8 = vaedecode.decode(
                    samples=get_value_at_index(unloadmodel_88, 0),
                    vae=get_value_at_index(vaeloader_39, 0),
                )
                
                cleanup_gpu_memory()
                
                # RIFE frame interpolation
                rife_vfi_83 = rife_vfi.vfi(
                    ckpt_name=RIFE_MODEL,
                    clear_cache_after_n_frames=10,
                    multiplier=request.interpolation_multiplier,
                    fast_mode=True,
                    ensemble=True,
                    scale_factor=1,
                    frames=get_value_at_index(vaedecode_8, 0),
                )
                
                # Combine video
                timestamp = int(time.time())
                filename_prefix = f"gpu0_{timestamp}_Wan22_{job_id[:8]}"
                
                vhs_videocombine_82 = vhs_videocombine.combine_video(
                    frame_rate=request.frame_rate,
                    loop_count=0,
                    filename_prefix=filename_prefix,
                    format="video/h264-mp4",
                    pix_fmt="yuv420p",
                    crf=15,
                    save_metadata=True,
                    trim_to_audio=False,
                    pingpong=False,
                    save_output=True,
                    images=get_value_at_index(rife_vfi_83, 0),
                    unique_id=random.randint(1, 2**63),
                )
                
                unique_id = random.randint(1, 2**63)
                easy_cleangpuused_93 = easy_cleangpuused.empty_cache(
                    anything=get_value_at_index(vhs_videocombine_82, 0),
                    unique_id=unique_id,
                )
                
                cleanup_gpu_memory()
            
            # Find the generated video
            video_filename = None
            output_files = vhs_videocombine_82.get("ui", {}).get("gifs", [])
            
            if output_files and len(output_files) > 0:
                video_info = output_files[0]
                video_filename = video_info.get("filename")
            
            if not video_filename:
                # Try to find the video file in output directory
                possible_names = [
                    f'{filename_prefix}.mp4',
                    f'{filename_prefix}_00001.mp4',
                ]
                
                for name in possible_names:
                    full_path = os.path.join(OUTPUT_DIR, name)
                    if os.path.exists(full_path):
                        video_filename = name
                        break
                
                if not video_filename:
                    # Search for any matching files
                    try:
                        all_files = os.listdir(OUTPUT_DIR)
                        matching_files = [f for f in all_files if filename_prefix in f and f.endswith('.mp4')]
                        if matching_files:
                            video_filename = matching_files[0]
                            logger.info(f"Found video with name: {video_filename}")
                        else:
                            raise FileNotFoundError(f"No generated video found with prefix: {filename_prefix}")
                    except Exception as list_error:
                        logger.error(f"Error listing output directory: {list_error}")
                        raise FileNotFoundError(f"Generated video not found: {filename_prefix}")
            
            full_video_path = os.path.join(OUTPUT_DIR, video_filename)
            logger.info(f"Video saved successfully: {full_video_path}")
            
            processing_time = time.time() - start_time
            
            with job_lock:
                if job_id in active_jobs:
                    active_jobs[job_id].update({
                        "status": JobStatus.COMPLETED,
                        "result_url": video_filename,
                        "download_url": f"/download/{video_filename}",
                        "output_filename": video_filename,
                        "completed_at": datetime.now(),
                        "processing_time": processing_time,
                        "generation_details": {
                            "seed": request.seed,
                            "width": request.width,
                            "height": request.height,
                            "video_length": request.video_length,
                            "frame_rate": request.frame_rate,
                            "interpolation_multiplier": request.interpolation_multiplier,
                            "processing_time": f"{processing_time:.2f}s",
                            "gpu": CUDA_DEVICE
                        }
                    })
            
            logger.info(f"Job {job_id} completed successfully in {processing_time:.2f}s: {video_filename}")
            
            # Clean up input image
            try:
                if os.path.exists(input_image_path):
                    os.remove(input_image_path)
                    logger.info(f"Cleaned up input image: {input_image_path}")
            except Exception as cleanup_error:
                logger.warning(f"Could not clean up input image: {cleanup_error}")
            
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
            
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"Error in stuck job monitor: {e}")
            await asyncio.sleep(60)

async def periodic_cleanup_task():
    """Background task for periodic cleanup"""
    while True:
        try:
            cleanup_old_jobs()
            await asyncio.sleep(CLEANUP_INTERVAL)
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(60)


async def run_startup_test():
    """Run a test generation after server startup - non-blocking"""
    try:
        await asyncio.sleep(5)  # Wait for server to be fully ready

        test_prompt = "A majestic black panther standing on a rocky cliff edge in a heroic brave position, muscles tensed, piercing golden eyes staring intensely forward, wind blowing through its sleek black fur, dramatic cloudy sky background, cinematic lighting, photorealistic, 8k quality"

        print("=" * 70)
        print("RUNNING WAN 2.2 VIDEO TEST GENERATION")
        print("=" * 70)
        print(f"Test Prompt: {test_prompt[:80]}...")
        print(f"Output Directory: {OUTPUT_DIR}")

        # Create a simple black reference image for testing
        import tempfile
        from PIL import Image
        img = Image.new('RGB', (640, 640), color='black')
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=TEMP_UPLOAD_DIR) as tmp:
            img.save(tmp.name)
            test_image_path = tmp.name

        # Create test request - NOTE: Using VideoGenerationRequest
        test_request = VideoGenerationRequest(
            prompt=test_prompt,
            width=640,
            height=640,
            videolength=49,
            framerate=24,
            steps=4,
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
                "input_image_path": test_image_path,
                "created_at": datetime.now(),
                "message": "Test generation"
            }
            job_queue.append({"job_id": job_id, "request": test_request, "input_image_path": test_image_path})

        print("Generating test output...")

        # Wait for completion (max 10 minutes for video)
        for _ in range(300):  # 300 * 2 = 600 seconds = 10 minutes
            await asyncio.sleep(2)
            with job_lock:
                if job_id in active_jobs:
                    job = active_jobs[job_id]
                    if job["status"] == JobStatus.COMPLETED:
                        print(f"✓ Test video generation completed: {job.get('output_filename')}")
                        print("=" * 70)
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
    logger.info("Starting Self-Managing Wan 2.2 I2V Worker (GPU 0)...")
    
    await load_models()
    
    # Start all background tasks
    queue_processor_task = asyncio.create_task(process_job_queue())
    cleanup_task_instance = asyncio.create_task(periodic_cleanup_task())
    stuck_job_monitor_task = asyncio.create_task(monitor_stuck_jobs())
    
    
    # Run startup test in background
    asyncio.create_task(run_startup_test())
    
    yield
    
    logger.info("Shutting down Wan 2.2 worker...")
    
    generation_executor.shutdown(wait=True, cancel_futures=False)
    
    for task in [queue_processor_task, cleanup_task_instance, stuck_job_monitor_task]:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

app = FastAPI(
    title="Self-Managing Wan 2.2 I2V Generation Worker",
    description="Wan 2.2 Image-to-Video generation with built-in job queue and load management",
    version="1.0.0",
    lifespan=lifespan
)

def validate_api_key(api_key: Optional[str] = None) -> bool:
    """Validate API key"""
    if not valid_api_keys:
        return True
    return api_key in valid_api_keys

@app.post("/generate")
async def generate_video(
    prompt: str,
    image: UploadFile = File(...),
    negative_prompt: Optional[str] = None,
    width: int = 640,
    height: int = 640,
    video_length: int = 49,
    frame_rate: int = 24,
    seed: Optional[int] = None,
    shift: float = 8.0,
    steps: int = 4,
    cfg: float = 1.0,
    interpolation_multiplier: int = 2,
    priority: str = "normal",
    api_key: Optional[str] = None
):
    """Submit video generation job to queue"""
    
    if not validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not models_loaded or NODE_CLASS_MAPPINGS is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Please try again in a moment.")
    
    current_load = get_current_load()
    
    if current_load["queued_jobs"] >= MAX_QUEUE_SIZE:
        raise HTTPException(
            status_code=429, 
            detail=f"Queue is full. Maximum {MAX_QUEUE_SIZE} jobs can be queued. Please try again later."
        )
    
    # Validate image file
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    
    # Save uploaded image
    job_id = str(uuid.uuid4())
    image_extension = image.filename.split('.')[-1] if '.' in image.filename else 'jpg'
    temp_image_filename = f"input_{job_id[:8]}.{image_extension}"
    temp_image_path = os.path.join(TEMP_UPLOAD_DIR, temp_image_filename)
    
    try:
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        logger.info(f"Saved input image: {temp_image_path}")
    except Exception as e:
        logger.error(f"Error saving uploaded image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded image: {str(e)}")
    
    # Create request object
    if negative_prompt is None:
        negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    
    request = VideoGenerationRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        video_length=video_length,
        frame_rate=frame_rate,
        seed=seed,
        shift=shift,
        steps=steps,
        cfg=cfg,
        interpolation_multiplier=interpolation_multiplier,
        priority=priority,
        api_key=api_key
    )
    
    queue_position = current_load["queued_jobs"] + 1
    estimated_wait = estimate_wait_time(queue_position)
    
    job_entry = {
        "job_id": job_id,
        "status": JobStatus.QUEUED,
        "created_at": datetime.now(),
        "request": request,
        "input_image_path": temp_image_path,
        "estimated_wait_time": estimated_wait,
        "queue_position": queue_position
    }
    
    with job_lock:
        active_jobs[job_id] = job_entry.copy()
        job_queue.append({
            "job_id": job_id,
            "request": request,
            "input_image_path": temp_image_path,
            "priority": priority
        })
    
    logger.info(f"Job {job_id} added to queue. Queue size: {len(job_queue)}, Position: {queue_position}")
    
    return {
        "job_id": job_id,
        "status": JobStatus.QUEUED,
        "message": "Video generation job added to queue successfully",
        "status_url": f"/job/{job_id}",
        "estimated_wait_time": estimated_wait,
        "queue_position": queue_position,
        "current_load": current_load
    }

@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a specific job"""
    
    await asyncio.sleep(0)
    
    try:
        lock_acquired = job_lock.acquire(timeout=0.1)
        
        if not lock_acquired:
            raise HTTPException(
                status_code=503,
                detail="System busy - unable to retrieve job status. Try again in a moment."
            )
        
        try:
            if job_id in active_jobs:
                job_data = active_jobs[job_id].copy()

                if job_data["status"] == JobStatus.QUEUED:
                    for i, queued_job in enumerate(job_queue):
                        if queued_job["job_id"] == job_id:
                            job_data["queue_position"] = i + 1
                            job_data["estimated_wait_time"] = estimate_wait_time(i + 1)
                            break
                
                elif job_data["status"] == JobStatus.PROCESSING:
                    job_data["queue_position"] = 0
                    job_data["estimated_wait_time"] = estimate_wait_time(1)
                            
                job_data.pop("request", None)
                job_data.pop("started_at", None)
                job_data.pop("input_image_path", None)

                return JobResponse(**job_data)

            elif job_id in job_history:
                history_data = job_history[job_id].copy()
                history_data.pop("request", None)
                history_data.pop("started_at", None)
                history_data.pop("input_image_path", None)
                return JobResponse(**history_data)
            
            else:
                raise HTTPException(status_code=404, detail="Job not found")
                
        finally:
            job_lock.release()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status for {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving job status: {str(e)}"
        )

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
            
            # Clean up input image
            input_image_path = job_data.get("input_image_path")
            if input_image_path and os.path.exists(input_image_path):
                try:
                    os.remove(input_image_path)
                    logger.info(f"Cleaned up input image: {input_image_path}")
                except Exception as e:
                    logger.warning(f"Could not clean up input image: {e}")
        
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
            "name": "Self-Managing Wan 2.2 I2V Worker",
            "version": "1.0.0",
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
    """Manually trigger GPU memory cleanup"""
    
    if valid_api_keys and api_key not in valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if is_gpu_busy():
        raise HTTPException(
            status_code=503,
            detail="Cannot cleanup GPU while jobs are processing. Please wait."
        )
    
    try:
        current_load = get_current_load()
        
        if current_load["active_jobs"] > 0:
            logger.warning("GPU cleanup requested while jobs are processing")
            return {
                "status": "warning",
                "message": "Cleanup performed, but jobs are currently processing",
                "active_jobs": current_load["active_jobs"]
            }
        
        if torch.cuda.is_available():
            memory_before = {
                "allocated_gb": torch.cuda.memory_allocated(0) / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved(0) / 1024**3,
                "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
        else:
            raise HTTPException(status_code=503, detail="CUDA not available")
        
        logger.info("Manual GPU cleanup triggered")
        
        for i in range(5):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            if i < 4:
                time.sleep(0.1)
        
        try:
            torch.cuda.ipc_collect()
        except:
            pass
        
        for _ in range(3):
            gc.collect()
        
        torch.cuda.reset_peak_memory_stats()
        
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
    """Get current GPU memory status"""
    if is_gpu_busy():
        raise HTTPException(
            status_code=503, 
            detail="GPU busy - memory stats unavailable during video generation"
        )
    
    try:
        if not torch.cuda.is_available():
            raise HTTPException(status_code=503, detail="CUDA not available")
        
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated
        
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
        "service": "wan22_i2v_generation_self_managing",
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
async def download_video(filename: str):
    """Download a generated video"""
    try:
        possible_paths = [
            os.path.join(OUTPUT_DIR, filename),
            os.path.join(COMFYUI_PATH, "output", filename),
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
        
        if not file_path:
            logger.error(f"Video not found: {filename}")
            raise HTTPException(status_code=404, detail="Video not found")
        
        logger.info(f"Downloading video: {filename}")
        return FileResponse(file_path, media_type="video/mp4", filename=filename)
        
    except Exception as e:
        logger.error(f"Error downloading video: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/generate_api_key")
async def generate_api_key(request: dict):
    """Generate new API key"""
    email = request.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email required")
    
    new_key = f"wan22_key_{uuid.uuid4().hex[:16]}"
    
    global valid_api_keys
    valid_api_keys.add(new_key)
    save_api_keys(valid_api_keys)
    
    logger.info(f"Generated new API key for {email}")
    
    return {
        "api_key": new_key,
        "email": email,
        "service": "wan22_i2v_generation",
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
            "comfyui": COMFYUI_PATH,
            "output": OUTPUT_DIR,
            "temp_upload": TEMP_UPLOAD_DIR
        },
        "models": {
            "filenames": {
                "clip": CLIP_MODEL,
                "vae": VAE_MODEL,
                "unet_high_noise": UNET_HIGH_NOISE,
                "unet_low_noise": UNET_LOW_NOISE,
                "lora_high_noise": LORA_HIGH_NOISE,
                "lora_low_noise": LORA_LOW_NOISE,
                "rife": RIFE_MODEL
            },
            "full_paths_for_verification": {
                "clip": CLIP_MODEL_FULL_PATH,
                "vae": VAE_MODEL_FULL_PATH,
                "unet_high_noise": UNET_HIGH_NOISE_FULL_PATH,
                "unet_low_noise": UNET_LOW_NOISE_FULL_PATH
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

@app.get("/")
async def root():
    """Root endpoint with service information"""
    current_load = get_current_load()
    
    return {
        "service": "Self-Managing Wan 2.2 Image-to-Video Generation Worker",
        "version": "1.0.0",
        "gpu_id": CUDA_DEVICE,
        "status": "healthy" if models_loaded else "loading",
        "capabilities": [
            "wan-2.2",
            "image-to-video",
            "high-quality-i2v",
            "frame-interpolation",
            "job-queue-management",
            "auto-load-balancing"
        ],
        "current_load": current_load,
        "endpoints": {
            "generate": "/generate (POST with image upload)",
            "job_status": "/job/{job_id}",
            "cancel_job": "/job/{job_id} [DELETE]",
            "queue_status": "/queue",
            "list_jobs": "/list_jobs",
            "health": "/health",
            "stats": "/stats",
            "download": "/download/{filename}",
            "generate_api_key": "/generate_api_key",
            "cleanup_gpu": "/cleanup_gpu",
            "gpu_memory": "/gpu_memory"
        },
        "features": {
            "built_in_queue": "Automatic job queuing when at capacity",
            "load_balancing": "Smart job processing based on system load",
            "gpu_memory_management": "Automatic memory cleanup between jobs",
            "job_persistence": "Job status tracking and history",
            "image_upload": "Direct image upload for I2V conversion",
            "frame_interpolation": "RIFE-based frame interpolation",
            "api_key_management": "Optional API key authentication"
        },
        "limits": {
            "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
            "max_queue_size": MAX_QUEUE_SIZE,
            "job_timeout_seconds": JOB_TIMEOUT
        },
        "video_parameters": {
            "default_resolution": "640x640",
            "default_video_length": "49 frames",
            "default_frame_rate": "24 fps",
            "interpolation_multiplier": "2x (RIFE)",
            "output_format": "H.264 MP4"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9094, log_level="info")
