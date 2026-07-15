import cv2
import onnxruntime as rt
import sys
sys.path.insert(1, './recognition')
from scrfd import SCRFD
from arcface_onnx import ArcFaceONNX
import face_align
import os.path as osp
import os
import requests
from tqdm import tqdm
import ffmpeg
import random
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from insightface.model_zoo.inswapper import INSwapper
import psutil
from enum import Enum
from insightface.app.common import Face
from insightface.utils.storage import ensure_available
import re
import subprocess
from PIL import Image
import numpy as np
import time
from codeformer_wrapper import enhance_image, enhance_image_memory
import tempfile
import hashlib
import json
import shutil
from pathlib import Path
from collections import defaultdict

gc = __import__('gc')

# Preload NVIDIA DLLs if Windows
if sys.platform in ("win32", "win64"):
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
            os.add_dll_directory(r"C:\Program Files\NVIDIA\CUDNN\v9.4\bin\12.6")
        except Exception as e:
            print(f"[INFO] Failed to add CUDA or CUDNN DLL directory: {e}")
            print("[INFO] This error can be ignored if running in CPU mode. Otherwise, make sure the paths are correct.")

    if hasattr(rt, "preload_dlls"):
        rt.preload_dlls()

class RefacerMode(Enum):
    CPU, CUDA, COREML, TENSORRT = range(1, 5)

class Refacer:
    def __init__(self, force_cpu=False, colab_performance=False):
        self.disable_similarity = False
        self.multiple_faces_mode = False
        self.first_face = False
        self.force_cpu = force_cpu
        self.colab_performance = colab_performance
        self.use_num_cpus = mp.cpu_count()
        self.__check_encoders()
        self.__check_providers()
        self.total_mem = psutil.virtual_memory().total
        self.__init_apps()

        # VRAM detection for dynamic batch sizing (currently disabled, using fixed batch size)
        self.vram_gb = self._detect_vram()

        # Cache infrastructure
        self.cache_dir = Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.pipeline_version = "1.0"
        
        # Performance profiling
        self.profiling_enabled = True
        self.profile_times = defaultdict(float)
        self.profile_counts = defaultdict(int)
        
        # Ultra-lightweight in-memory cache for short videos
        self.light_cache = {}  # {video_hash: {frame_idx: {"bboxes": ..., "kpss": ...}}}
        self.light_cache_max_videos = 1  # Keep cache for current video only
        # Optimized for Tesla T4 (16GB VRAM) - increased limits for better cache hit rate
        self.light_cache_max_memory_mb = 2000  # Max 2GB for light cache (was 500MB)
        self.light_cache_max_frames_per_video = 5000  # Max frames per video in cache (was 1000)
        self.current_video_hash = None  # Track current video to clear cache on change
    
    def _profile_start(self, name):
        """Start profiling a section."""
        if self.profiling_enabled:
            return time.time()
        return None
    
    def _profile_end(self, name, start_time):
        """End profiling a section."""
        if self.profiling_enabled and start_time is not None:
            elapsed = time.time() - start_time
            self.profile_times[name] += elapsed
            self.profile_counts[name] += 1
    
    def _print_profile_summary(self):
        """Print profiling summary."""
        if not self.profiling_enabled:
            return
        
        print("\n=== PERFORMANCE PROFILE ===")
        total_time = sum(self.profile_times.values())
        
        for name in sorted(self.profile_times.keys(), key=lambda x: self.profile_times[x], reverse=True):
            elapsed = self.profile_times[name]
            count = self.profile_counts[name]
            avg = elapsed / count if count > 0 else 0
            pct = (elapsed / total_time * 100) if total_time > 0 else 0
            print(f"{name:30s}: {elapsed:6.2f}s ({pct:5.1f}%) | count={count:4d} | avg={avg:6.4f}s")
        print(f"{'TOTAL':30s}: {total_time:6.2f}s (100.0%)")
        print("============================\n")
    
    def _get_light_cache(self, video_hash, frame_idx):
        """Get cached detection data from lightweight in-memory cache."""
        if video_hash in self.light_cache and frame_idx in self.light_cache[video_hash]:
            return self.light_cache[video_hash][frame_idx]
        return None
    
    def _set_light_cache(self, video_hash, frame_idx, bboxes, kpss):
        """Set detection data in lightweight in-memory cache with memory protection."""
        # Clear cache if video changed
        if self.current_video_hash != video_hash:
            self._clear_light_cache()
            self.current_video_hash = video_hash
        
        if video_hash not in self.light_cache:
            self.light_cache[video_hash] = {}
        
        # Check if video has too many frames cached
        if len(self.light_cache[video_hash]) >= self.light_cache_max_frames_per_video:
            return  # Don't cache if video exceeds frame limit
        
        # Only cache bboxes and kpss (lightweight, no embeddings)
        self.light_cache[video_hash][frame_idx] = {
            "bboxes": bboxes.copy() if bboxes is not None else None,
            "kpss": kpss.copy() if kpss is not None else None
        }
        
        # Estimate cache size and enforce memory limit
        self._enforce_light_cache_memory_limit()
    
    def _estimate_light_cache_size_mb(self):
        """Estimate current light cache size in MB."""
        total_size = 0
        for video_hash, frames in self.light_cache.items():
            for frame_idx, data in frames.items():
                # Estimate: bboxes (n_faces * 5 floats) + kpss (n_faces * 5 * 2 floats)
                # Each float = 8 bytes
                if data["bboxes"] is not None:
                    total_size += data["bboxes"].nbytes
                if data["kpss"] is not None:
                    total_size += data["kpss"].nbytes
        return total_size / (1024 * 1024)  # Convert to MB
    
    def _enforce_light_cache_memory_limit(self):
        """Enforce memory limit by evicting oldest entries if needed."""
        cache_size_mb = self._estimate_light_cache_size_mb()
        
        if cache_size_mb > self.light_cache_max_memory_mb:
            print(f"[CACHE] Warning: Light cache at {cache_size_mb:.1f}MB, evicting oldest entries...")
            
            # Evict oldest videos until under limit
            while cache_size_mb > self.light_cache_max_memory_mb * 0.8 and len(self.light_cache) > 0:
                oldest_key = next(iter(self.light_cache))
                del self.light_cache[oldest_key]
                cache_size_mb = self._estimate_light_cache_size_mb()
    
    def _clear_light_cache(self, video_hash=None):
        """Clear lightweight cache for specific video or all videos."""
        if video_hash:
            if video_hash in self.light_cache:
                del self.light_cache[video_hash]
        else:
            self.light_cache.clear()
    
    def __get_faces_with_light_cache(self, frame, video_hash, frame_idx, max_num=8):
        """Get faces with lightweight in-memory cache for detection only."""
        # Check light cache first
        cached = self._get_light_cache(video_hash, frame_idx)
        if cached is not None:
            # Cache hit: use cached detection, compute embedding only
            bboxes = cached["bboxes"]
            kpss = cached["kpss"]
            
            if bboxes.shape[0] == 0:
                return []
            
            ret = []
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i, 0:4]
                det_score = bboxes[i, 4]
                kps = kpss[i] if kpss is not None else None
                face = Face(bbox=bbox, kps=kps, det_score=det_score)
                
                # Still need to compute embedding (depends on source face)
                start_embed = self._profile_start("face_embedding")
                face.embedding = self.rec_app.get(frame, kps)
                self._profile_end("face_embedding", start_embed)
                
                ret.append(face)
            
            return ret
        
        # Cache miss: do full detection and cache result
        start_detect = self._profile_start("face_detection")
        bboxes, kpss = self.face_detector.detect(frame, max_num=max_num, metric='default')
        self._profile_end("face_detection", start_detect)
        
        # Cache detection result (bboxes + kpss only, no embeddings)
        self._set_light_cache(video_hash, frame_idx, bboxes, kpss)
        
        if bboxes.shape[0] == 0:
            return []
        
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = kpss[i] if kpss is not None else None
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            
            start_embed = self._profile_start("face_embedding")
            face.embedding = self.rec_app.get(frame, kps)
            self._profile_end("face_embedding", start_embed)
            
            ret.append(face)
        
        return ret
    
    def reface_with_light_cache(self, video_path, faces, output_path,
                                preview=False, disable_similarity=False,
                                multiple_faces_mode=False, partial_reface_ratio=0.0):
        """Reface video with ultra-lightweight in-memory cache (detection only).
        
        This caches ONLY face detection results (bboxes + kpss) in RAM.
        Embeddings are still computed because they depend on the source face.
        No disk I/O, no preprocessing, lazy evaluation.
        
        Ideal for: Short videos (5-15s) where the same target is reused multiple times.
        """
        
        # Prepare source faces
        self.prepare_faces(faces, disable_similarity, multiple_faces_mode)
        self.first_face = False if multiple_faces_mode else (faces[0].get("origin") is None or disable_similarity)
        self.partial_reface_ratio = partial_reface_ratio
        
        # Compute video hash for cache key
        video_hash = self._compute_video_hash(video_path)
        
        # Setup video writer
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Use mp4v for OpenCV (H264 support is unreliable in OpenCV)
        # FFmpeg will handle H264 encoding in __convert_video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Dynamic batch size based on VRAM
        batch_size = self._calculate_optimal_batch_size(frame_width, frame_height, fps)
        frames = []
        frame_index = 0
        skip_rate = 10 if preview else 1
        cache_hits = 0
        cache_misses = 0
        
        start_total = self._profile_start("total_processing")
        
        with tqdm(total=total_frames, desc="Processing frames (light cache)") as pbar:
            while cap.isOpened():
                flag, frame = cap.read()
                if not flag:
                    break
                
                if frame_index % skip_rate != 0:
                    frame_index += 1
                    pbar.update()
                    continue
                
                # Use light cache for detection
                faces_in_frame = self.__get_faces_with_light_cache(frame, video_hash, frame_index, max_num=8)
                
                # Track cache hits/misses
                if self._get_light_cache(video_hash, frame_index) is not None:
                    cache_hits += 1
                else:
                    cache_misses += 1
                
                # Process faces
                if not faces_in_frame:
                    processed_frame = frame
                else:
                    processed_frame = self._process_faces_with_cached_data(frame, faces_in_frame)
                
                frames.append(processed_frame)
                
                if len(frames) > batch_size:
                    for f in frames:
                        output.write(f)
                    frames = []
                    gc.collect()
                
                frame_index += 1
                pbar.update()
        
        cap.release()
        if frames:
            for f in frames:
                output.write(f)
        output.release()
        
        self._profile_end("total_processing", start_total)
        
        print(f"[LIGHT CACHE] Cache hits: {cache_hits}, misses: {cache_misses}")
        print(f"[LIGHT CACHE] Hit rate: {cache_hits/(cache_hits+cache_misses)*100:.1f}%")
        
        converted_path = self.__convert_video(video_path, output_path, preview=preview)
        return converted_path

    def _detect_vram(self):
        """Detect available VRAM for dynamic batch sizing."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / 1e9
        except:
            pass
        # Conservative estimate if detection fails
        return 8.0  # Assume 8GB if unable to detect

    def _calculate_optimal_batch_size(self, frame_width, frame_height, fps):
        """Calculate optimal batch size based on VRAM availability and frame properties.
        
        Optimized for Tesla T4 (16GB VRAM) but works for any GPU.
        """
        try:
            import torch
            if torch.cuda.is_available():
                # Get VRAM info
                vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                vram_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
                vram_available = vram_total - vram_allocated
                
                # Calculate frame size in MB (RGB)
                frame_size_mb = (frame_width * frame_height * 3) / (1024**2)
                
                # Use 70% of available VRAM for frames
                vram_for_frames = vram_available * 1024 * 0.7  # MB
                max_frames_by_vram = int(vram_for_frames / frame_size_mb)
                
                # Also consider FPS (batch should represent 1-2 seconds of video)
                min_frames_by_fps = max(fps, 30)  # At least 1 second
                max_frames_by_fps = fps * 2  # At most 2 seconds
                
                # Apply hard limits
                min_batch = 100
                max_batch = 1000
                
                # Calculate final batch size
                batch_size = min(max_frames_by_vram, max_frames_by_fps)
                batch_size = max(min(batch_size, max_batch), min_batch)
                
                print(f"[BATCH] Dynamic batch size: {batch_size} (VRAM: {vram_available:.1f}GB, Frame: {frame_size_mb:.1f}MB)")
                return batch_size
        except Exception as e:
            print(f"[BATCH] Failed to calculate dynamic batch size: {e}")
        
        # Fallback to conservative default
        return 300

    def _partial_face_blend(self, original_frame, swapped_frame, face):
        h_frame, w_frame = original_frame.shape[:2]
    
        x1, y1, x2, y2 = map(int, face.bbox)
        x1 = max(0, min(x1, w_frame-1))
        y1 = max(0, min(y1, h_frame-1))
        x2 = max(0, min(x2, w_frame))
        y2 = max(0, min(y2, h_frame))
    
        if x2 <= x1 or y2 <= y1:
            print(f"Invalid bbox: {x1},{y1},{x2},{y2}")
            return swapped_frame
    
        w = x2 - x1
        h = y2 - y1
        cutoff = int(h * (1.0 - self.blend_height_ratio))
    
        swap_crop = swapped_frame[y1:y2, x1:x2].copy()
        orig_crop = original_frame[y1:y2, x1:x2].copy()
    
        mask = np.ones((h, w, 3), dtype=np.float32)
        transition = 40
    
        if cutoff < h:
            blend_start = max(cutoff - transition // 2, 0)
            blend_end = min(cutoff + transition // 2, h)
    
            if blend_end > blend_start:
                alpha = np.linspace(1.0, 0.0, blend_end - blend_start)[:, np.newaxis, np.newaxis]
                mask[blend_start:blend_end, :, :] = alpha
            mask[blend_end:, :, :] = 0.0
    
        blended_crop = (swap_crop.astype(np.float32) * mask + orig_crop.astype(np.float32) * (1.0 - mask)).astype(np.uint8)
    
        blended_frame = swapped_frame.copy()
        blended_frame[y1:y2, x1:x2] = blended_crop
    
        return blended_frame
    

    def __download_with_progress(self, url, output_path):
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {os.path.basename(output_path)}")

        with open(output_path, 'wb') as f:
            for data in response.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()

        if total_size != 0 and t.n != total_size:
            raise Exception("ERROR, something went wrong downloading the model!")

    def __check_providers(self):
        available_providers = rt.get_available_providers()

        if self.force_cpu:
            self.providers = ['CPUExecutionProvider']
        else:
            # Prioritize CUDA for Tesla T4 optimization
            self.providers = []
            for p in ['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider']:
                if p in available_providers:
                    self.providers.append(p)

        rt.set_default_logger_severity(4)
        self.sess_options = rt.SessionOptions()
        # Optimized for GPU: SEQUENTIAL mode is better for CUDA, PARALLEL for CPU
        self.sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        self.sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Ensure buffalo_l model directory and det_10g.onnx are downloaded/available
        # before attempting to create the test session.
        try:
            ensure_available('models', 'buffalo_l', root='~/.insightface')
        except Exception as e:
            print(f"[WARNING] Failed to ensure buffalo_l models are downloaded during provider check: {e}")

        test_model = os.path.expanduser("~/.insightface/models/buffalo_l/det_10g.onnx")
        try:
            test_session = rt.InferenceSession(test_model, self.sess_options, providers=self.providers)
            active_provider = test_session.get_providers()[0]
        except Exception as e:
            print(f"[ERROR] Failed to create test session: {e}")
            active_provider = 'CPUExecutionProvider'

        if active_provider == 'CUDAExecutionProvider':
            self.mode = RefacerMode.CUDA
            self.use_num_cpus = 2
            self.sess_options.intra_op_num_threads = 1
            self.sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        elif active_provider == 'CoreMLExecutionProvider':
            self.mode = RefacerMode.COREML
            self.use_num_cpus = max(mp.cpu_count() - 1, 1)
            self.sess_options.intra_op_num_threads = int(self.use_num_cpus / 2)
        elif self.colab_performance:
            self.mode = RefacerMode.TENSORRT
            self.use_num_cpus = max(mp.cpu_count() - 1, 1)
            self.sess_options.intra_op_num_threads = int(self.use_num_cpus / 2)
        else:
            self.mode = RefacerMode.CPU
            self.use_num_cpus = max(mp.cpu_count() - 1, 1)
            self.sess_options.intra_op_num_threads = int(self.use_num_cpus / 2)

        print(f"Available providers: {available_providers}")
        print(f"Using providers: {self.providers}")
        print(f"Active provider: {active_provider}")
        print(f"Mode: {self.mode}")

    def __init_apps(self):
        assets_dir = ensure_available('models', 'buffalo_l', root='~/.insightface')

        model_path = os.path.join(assets_dir, 'det_10g.onnx')
        sess_face = rt.InferenceSession(model_path, self.sess_options, providers=self.providers)
        print(f"Face Detector providers: {sess_face.get_providers()}")
        self.face_detector = SCRFD(model_path, sess_face)
        self.face_detector.prepare(0, input_size=(640, 640))

        model_path = os.path.join(assets_dir, 'w600k_r50.onnx')
        sess_rec = rt.InferenceSession(model_path, self.sess_options, providers=self.providers)
        print(f"Face Recognizer providers: {sess_rec.get_providers()}")
        self.rec_app = ArcFaceONNX(model_path, sess_rec)
        self.rec_app.prepare(0)

        model_dir = os.path.join('weights', 'inswapper')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'inswapper_128.onnx')

        if not os.path.exists(model_path):
            print(f"Model {model_path} not found. Downloading from HuggingFace...")
            url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
            try:
                self.__download_with_progress(url, model_path)
                print(f"Downloaded {model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download {model_path}. Error: {e}")

        sess_swap = rt.InferenceSession(model_path, self.sess_options, providers=self.providers)
        print(f"Face Swapper providers: {sess_swap.get_providers()}")
        self.face_swapper = INSwapper(model_path, sess_swap)

    def _compute_video_hash(self, video_path):
        """Fast video fingerprinting using first and last 1MB plus file size."""
        size = os.path.getsize(video_path)
        
        # Read first and last 1MB (handle small files)
        with open(video_path, 'rb') as f:
            first_mb = f.read(1024*1024)
            
            # For files smaller than 2MB, read from middle instead of end
            if size < 2 * 1024 * 1024:
                f.seek(max(0, size // 2 - 512 * 1024))
            else:
                f.seek(-1024*1024, 2)
            last_mb = f.read()
        
        hash_input = f"{size}-{first_mb}-{last_mb}".encode()
        return hashlib.sha256(hash_input).hexdigest()
    
    def _get_cache_dir(self, video_hash):
        """Get cache directory for a specific video hash."""
        return self.cache_dir / "video_analysis" / video_hash
    
    def _cache_valid(self, cache_dir):
        """Check if cache exists and is valid."""
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.exists():
            return False
        
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            
            # Check version compatibility
            if manifest.get("pipeline_version") != self.pipeline_version:
                return False
            
            return True
        except (json.JSONDecodeError, IOError):
            return False
    
    def _load_cached_analysis(self, cache_dir):
        """Load cached analysis metadata."""
        metadata_path = cache_dir / "metadata.json"
        with open(metadata_path) as f:
            return json.load(f)
    
    def clear_cache(self, video_path=None):
        """Clear cache for specific video or all videos."""
        if video_path:
            video_hash = self._compute_video_hash(video_path)
            cache_dir = self._get_cache_dir(video_hash)
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                print(f"[CACHE] Cleared cache for video {video_hash[:8]}...")
        else:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(exist_ok=True)
                print("[CACHE] Cleared all cache...")

    def _cleanup_old_video_caches(self, current_hash):
        """Remove any cache directories and in-memory caches that do not match the current video hash to save space."""
        # Disk cleanup
        video_analysis_dir = self.cache_dir / "video_analysis"
        if video_analysis_dir.exists():
            for path in video_analysis_dir.iterdir():
                if path.is_dir() and path.name != current_hash:
                    try:
                        shutil.rmtree(path)
                        print(f"[CACHE] Automatically cleaned up old cache folder: {path.name[:8]}...")
                    except Exception as e:
                        print(f"[CACHE] Warning: Failed to clean up old cache {path.name}: {e}")
        
        # In-memory cleanup
        keys_to_remove = [k for k in self.light_cache if k != current_hash]
        for k in keys_to_remove:
            del self.light_cache[k]
            print(f"[CACHE] Automatically cleaned up in-memory cache for video: {k[:8]}...")

    def analyze_target_video(self, video_path, max_num_faces=8, force_reanalyze=False, cache_embeddings=True):
        """Analyze target video and cache face detection and embedding results.
        
        Args:
            video_path: Path to video file
            max_num_faces: Maximum number of faces to detect
            force_reanalyze: Force re-analysis even if cache exists
            cache_embeddings: If False, only cache detection (bboxes/kpss), not embeddings
        """
        video_hash = self._compute_video_hash(video_path)
        cache_dir = self._get_cache_dir(video_hash)
        
        if not force_reanalyze and self._cache_valid(cache_dir):
            print(f"[CACHE] Loading cached analysis for {video_hash[:8]}...")
            return self._load_cached_analysis(cache_dir)
        
        print(f"[ANALYSIS] Analyzing target video (cache_embeddings={cache_embeddings})...")
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        metadata = {
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "pipeline_version": self.pipeline_version,
            "max_num_faces": max_num_faces,
            "cache_embeddings": cache_embeddings
        }
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        frame_cache_dir = cache_dir / "frame_analysis"
        frame_cache_dir.mkdir(exist_ok=True)
        
        # Batch frames for more efficient I/O
        batch_size = 100
        batch_data = []
        
        # Analyze frames
        with tqdm(total=total_frames, desc="Analyzing frames") as pbar:
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect faces
                bboxes, kpss = self.face_detector.detect(frame, max_num=max_num_faces)
                
                frame_data = {
                    "bboxes": bboxes,
                    "kpss": kpss,
                    "embeddings": [],
                    "transforms": []
                }
                
                # Only compute embeddings if requested (expensive operation)
                if cache_embeddings:
                    for i in range(len(bboxes)):
                        kps = kpss[i] if kpss is not None else None
                        embedding = self.rec_app.get(frame, kps)
                        transform = face_align.estimate_norm(kps, image_size=112)[0]
                        
                        frame_data["embeddings"].append(embedding)
                        frame_data["transforms"].append(transform)
                
                batch_data.append((frame_idx, frame_data))
                
                # Save in batches to reduce I/O overhead
                if len(batch_data) >= batch_size:
                    for batch_idx, batch_frame_data in batch_data:
                        np.savez_compressed(
                            frame_cache_dir / f"frame_{batch_idx:06d}.npz",
                            **batch_frame_data
                        )
                    batch_data = []
                
                pbar.update(1)
        
        # Save remaining frames in batch
        if batch_data:
            for batch_idx, batch_frame_data in batch_data:
                np.savez_compressed(
                    frame_cache_dir / f"frame_{batch_idx:06d}.npz",
                    **batch_frame_data
                )
        
        # Save metadata
        with open(cache_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        # Save manifest
        manifest = {
            "pipeline_version": self.pipeline_version,
            "timestamp": time.time(),
            "video_hash": video_hash
        }
        with open(cache_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f)
        
        cap.release()
        
        print(f"[ANALYSIS] Completed. Cached {total_frames} frames.")
        return metadata


    def _process_faces_with_cached_data(self, frame, faces):
        """Process frame using pre-detected and pre-embedded faces (cached data)."""
        if not faces:
            return frame
        
        faces = sorted(faces, key=lambda face: face.bbox[0])
        
        if self.multiple_faces_mode:
            for idx, face in enumerate(faces):
                if idx >= len(self.replacement_faces):
                    break
                swapped = self.face_swapper.get(frame, face, self.replacement_faces[idx][1], paste_back=True)
                if hasattr(self, 'partial_reface_ratio') and self.partial_reface_ratio > 0.0:
                    self.blend_height_ratio = self.partial_reface_ratio
                    frame = self._partial_face_blend(frame, swapped, face)
                else:
                    frame = swapped
        elif self.disable_similarity:
            for face in faces:
                swapped = self.face_swapper.get(frame, face, self.replacement_faces[0][1], paste_back=True)
                if hasattr(self, 'partial_reface_ratio') and self.partial_reface_ratio > 0.0:
                    self.blend_height_ratio = self.partial_reface_ratio
                    frame = self._partial_face_blend(frame, swapped, face)
                else:
                    frame = swapped
        else:
            for rep_face in self.replacement_faces:
                for i in range(len(faces) - 1, -1, -1):
                    sim = self.rec_app.compute_sim(rep_face[0], faces[i].embedding)
                    if sim >= rep_face[2]:
                        swapped = self.face_swapper.get(frame, faces[i], rep_face[1], paste_back=True)
                        if hasattr(self, 'partial_reface_ratio') and self.partial_reface_ratio > 0.0:
                            self.blend_height_ratio = self.partial_reface_ratio
                            frame = self._partial_face_blend(frame, swapped, faces[i])
                        else:
                            frame = swapped
                        del faces[i]
                        break
        return frame

    def reface_with_cache(self, video_path, faces, output_path, 
                          preview=False, disable_similarity=False,
                          multiple_faces_mode=False, partial_reface_ratio=0.0,
                          cache_embeddings=True):
        """Reface video using cached target analysis for faster processing.
        
        Args:
            cache_embeddings: If True, cache embeddings (expensive). If False, only cache detection.
        """
        
        # Prepare source faces (not cached - depends on source)
        self.prepare_faces(faces, disable_similarity, multiple_faces_mode)
        self.first_face = False if multiple_faces_mode else (faces[0].get("origin") is None or disable_similarity)
        self.partial_reface_ratio = partial_reface_ratio
        
        # Load or create cache
        video_hash = self._compute_video_hash(video_path)
        cache_dir = self._get_cache_dir(video_hash)
        
        if not self._cache_valid(cache_dir):
            print("[CACHE] No valid cache found, analyzing target video...")
            self.analyze_target_video(video_path, max_num_faces=8, cache_embeddings=cache_embeddings)
        
        # Load cached analysis
        metadata = json.load(open(cache_dir / "metadata.json"))
        frame_cache_dir = cache_dir / "frame_analysis"
        
        # Check if max_num_faces matches and cache_embeddings matches
        if metadata.get("max_num_faces", 8) != 8 or metadata.get("cache_embeddings", True) != cache_embeddings:
            print("[CACHE] Cache was created with different settings, reanalyzing...")
            self.analyze_target_video(video_path, max_num_faces=8, force_reanalyze=True, cache_embeddings=cache_embeddings)
            metadata = json.load(open(cache_dir / "metadata.json"))
        
        # Setup video writer
        cap = cv2.VideoCapture(video_path)
        # Use mp4v for OpenCV (H264 support is unreliable in OpenCV)
        # FFmpeg will handle H264 encoding in __convert_video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(
            output_path, fourcc, 
            metadata["fps"], 
            (metadata["width"], metadata["height"])
        )
        
        # Process frames using cached data
        total_frames = metadata["total_frames"]
        # Dynamic batch size based on VRAM
        batch_size = self._calculate_optimal_batch_size(metadata["width"], metadata["height"], metadata["fps"])
        frames = []
        skip_rate = 10 if preview else 1
        
        with tqdm(total=total_frames, desc="Processing frames (cached)") as pbar:
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % skip_rate != 0:
                    pbar.update(1)
                    continue
                
                # Load cached face data
                cache_file = frame_cache_dir / f"frame_{frame_idx:06d}.npz"
                
                try:
                    cached_data = np.load(cache_file)
                    bboxes = cached_data["bboxes"]
                    kpss = cached_data["kpss"]
                    cached_embeddings = cached_data["embeddings"] if "embeddings" in cached_data and len(cached_data["embeddings"]) > 0 else None
                except (FileNotFoundError, KeyError):
                    # Cache miss for this frame, fall back to regular processing
                    print(f"[CACHE] Miss for frame {frame_idx}, falling back to regular processing...")
                    processed_frame = self.process_faces(frame) if not self.first_face else self.process_first_face(frame)
                    frames.append(processed_frame)
                    if len(frames) >= batch_size:
                        for f in frames:
                            output.write(f)
                        frames = []
                        gc.collect()
                    pbar.update(1)
                    continue
                
                # Reconstruct Face objects from cached data
                faces_in_frame = []
                for i in range(len(bboxes)):
                    face = Face(
                        bbox=bboxes[i],
                        kps=kpss[i] if len(kpss) > i else None,
                        det_score=bboxes[i][4]
                    )
                    
                    # Use cached embeddings if available, otherwise compute on-the-fly
                    if cached_embeddings is not None and i < len(cached_embeddings):
                        face.embedding = cached_embeddings[i]
                    else:
                        # Compute embedding on-the-fly (detection was cached, but not embedding)
                        kps = kpss[i] if kpss is not None and len(kpss) > i else None
                        face.embedding = self.rec_app.get(frame, kps)
                    
                    faces_in_frame.append(face)
                
                # Process faces using cached data (swap only, no detection/embedding)
                processed_frame = self._process_faces_with_cached_data(frame, faces_in_frame)
                
                frames.append(processed_frame)
                
                if len(frames) >= batch_size:
                    for f in frames:
                        output.write(f)
                    frames = []
                    gc.collect()
                
                pbar.update(1)
        
        # Write remaining frames
        for f in frames:
            output.write(f)
        
        cap.release()
        output.release()
        
        # Convert video (add audio if needed)
        converted_path = self.__convert_video(video_path, output_path, preview)
        return converted_path

    def prepare_faces(self, faces, disable_similarity=False, multiple_faces_mode=False):
        self.replacement_faces = []
        self.disable_similarity = disable_similarity
        self.multiple_faces_mode = multiple_faces_mode

        for face in faces:
            if "destination" not in face or face["destination"] is None:
                print("Skipping face config: No destination face provided.")
                continue

            _faces = self.__get_faces(face['destination'], max_num=1)
            if len(_faces) < 1:
                raise Exception('No face detected on "Destination face" image')

            if multiple_faces_mode:
                self.replacement_faces.append((None, _faces[0], 0.0))
            else:
                if "origin" in face and face["origin"] is not None and not disable_similarity:
                    face_threshold = face['threshold']
                    # Use autodetect for robust face detection in prepare phase (called once per video)
                    # Single-scale optimization is only applied in __get_faces for video frame processing
                    bboxes1, kpss1 = self.face_detector.autodetect(face['origin'], max_num=1)
                    if len(kpss1) < 1:
                        raise Exception('No face detected on "Face to replace" image')
                    feat_original = self.rec_app.get(face['origin'], kpss1[0])
                else:
                    face_threshold = 0
                    self.first_face = True
                    feat_original = None

                self.replacement_faces.append((feat_original, _faces[0], face_threshold))

    def __get_faces(self, frame, max_num=8):
        # Limit max_num to avoid detecting unnecessary faces (default 8 from app.py)
        start_detect = self._profile_start("face_detection")
        bboxes, kpss = self.face_detector.detect(frame, max_num=max_num, metric='default')
        self._profile_end("face_detection", start_detect)
        
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = kpss[i] if kpss is not None else None
            face = Face(bbox=bbox, kps=kps, det_score=det_score)

            start_embed = self._profile_start("face_embedding")
            face.embedding = self.rec_app.get(frame, kps)
            self._profile_end("face_embedding", start_embed)

            ret.append(face)

        return ret

    def process_first_face(self, frame):
        return self.process_faces(frame)

    def process_faces(self, frame):
        faces = self.__get_faces(frame, max_num=8)
        if not faces:
            return frame
 
        faces = sorted(faces, key=lambda face: face.bbox[0])
 
        start_swap = self._profile_start("face_swap")
        if self.multiple_faces_mode:
            for idx, face in enumerate(faces):
                if idx >= len(self.replacement_faces):
                    break
                swapped = self.face_swapper.get(frame, face, self.replacement_faces[idx][1], paste_back=True)
                if hasattr(self, 'partial_reface_ratio') and self.partial_reface_ratio > 0.0:
                    self.blend_height_ratio = self.partial_reface_ratio
                    frame = self._partial_face_blend(frame, swapped, face)
                else:
                    frame = swapped
        elif self.disable_similarity:
            if len(self.replacement_faces) > 1:
                for idx, face in enumerate(faces):
                    if idx >= len(self.replacement_faces):
                        break
                    swapped = self.face_swapper.get(frame, face, self.replacement_faces[idx][1], paste_back=True)
                    if hasattr(self, 'partial_reface_ratio') and self.partial_reface_ratio > 0.0:
                        self.blend_height_ratio = self.partial_reface_ratio
                        frame = self._partial_face_blend(frame, swapped, face)
                    else:
                        frame = swapped
            else:
                for face in faces:
                    swapped = self.face_swapper.get(frame, face, self.replacement_faces[0][1], paste_back=True)
                    if hasattr(self, 'partial_reface_ratio') and self.partial_reface_ratio > 0.0:
                        self.blend_height_ratio = self.partial_reface_ratio
                        frame = self._partial_face_blend(frame, swapped, face)
                    else:
                        frame = swapped
        else:
            for rep_face in self.replacement_faces:
                for i in range(len(faces) - 1, -1, -1):
                    sim = self.rec_app.compute_sim(rep_face[0], faces[i].embedding)
                    if sim >= rep_face[2]:
                        swapped = self.face_swapper.get(frame, faces[i], rep_face[1], paste_back=True)
                        if hasattr(self, 'partial_reface_ratio') and self.partial_reface_ratio > 0.0:
                            self.blend_height_ratio = self.partial_reface_ratio
                            frame = self._partial_face_blend(frame, swapped, faces[i])
                        else:
                            frame = swapped
                        del faces[i]
                        break
        self._profile_end("face_swap", start_swap)
        return frame

    def reface_group(self, faces, frames, output):
        # Use all available CPUs - GPU-bound tasks still benefit from parallel preprocessing
        max_workers = self.use_num_cpus

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            if self.first_face:
                results = list(tqdm(executor.map(self.process_first_face, frames), total=len(frames), desc="Processing frames"))
            else:
                results = list(tqdm(executor.map(self.process_faces, frames), total=len(frames), desc="Processing frames"))
            for result in results:
                output.write(result)
                            if len(self.replacement_faces) > 1:
                                for idx, face in enumerate(faces):
                                    if idx >= len(self.replacement_faces):
                                        break
                                    swapped = self.face_swapper.get(frame, face, self.replacement_faces[idx][1], paste_back=True)
                                    if hasattr(self, 'partial_reface_ratio') and self.partial_reface_ratio > 0.0:
                                        self.blend_height_ratio = self.partial_reface_ratio
                                        frame = self._partial_face_blend(frame, swapped, face)
                                    else:
                                        frame = swapped
                            else:
                                for face in faces:
                                    swapped = self.face_swapper.get(frame, face, self.replacement_faces[0][1], paste_back=True)
                                    if hasattr(self, 'partial_reface_ratio') and self.partial_reface_ratio > 0.0:
                                        self.blend_height_ratio = self.partial_reface_ratio
                                        frame = self._partial_face_blend(frame, swapped, face)
                                    else:
                                        frame = swapped

    def reface(self, video_path, faces, preview=False, disable_similarity=False, multiple_faces_mode=False, partial_reface_ratio=0.0, use_cache=False):
        """Reface video with optional caching for faster subsequent runs.
        
        Args:
            video_path: Path to input video
            faces: List of face configurations
            preview: If True, skip 90% of frames for faster preview
            disable_similarity: If True, disable face similarity matching
            multiple_faces_mode: If True, use multiple faces mode
            partial_reface_ratio: Ratio for partial face blending (0-0.5)
            use_cache: If True, use cached target analysis for faster processing (disabled by default)
        """
        original_name = osp.splitext(osp.basename(video_path))[0]
        timestamp = str(int(time.time()))
        filename = f"{original_name}_preview.mp4" if preview else f"{original_name}_{timestamp}.mp4"
    
        self.__check_video_has_audio(video_path)
    
        if preview:
            os.makedirs("output/preview", exist_ok=True)
            output_video_path = os.path.join('output', 'preview', filename)
        else:
            os.makedirs("output", exist_ok=True)
            output_video_path = os.path.join('output', filename)
        
        # Check if cache should be used based on video duration
        if use_cache:
            # Clean up old caches to keep disk and RAM usage low
            video_hash = self._compute_video_hash(video_path)
            self._cleanup_old_video_caches(video_hash)

            # Get video duration
            cap_check = cv2.VideoCapture(video_path)
            total_frames_check = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_check = cap_check.get(cv2.CAP_PROP_FPS)
            duration_seconds = total_frames_check / fps_check if fps_check > 0 else 0
            cap_check.release()
            
            # For short videos (5-15s): use ultra-lightweight in-memory cache (detection only)
            if duration_seconds < 15:
                print(f"[LIGHT CACHE] Using ultra-lightweight in-memory cache for short video ({duration_seconds:.1f}s)")
                converted_path = self.reface_with_light_cache(
                    video_path, faces, output_video_path,
                    preview=preview,
                    disable_similarity=disable_similarity,
                    multiple_faces_mode=multiple_faces_mode,
                    partial_reface_ratio=partial_reface_ratio
                )
            elif duration_seconds < 30:
                # Medium videos: use ultra-lightweight in-memory cache (detection only)
                print(f"[LIGHT CACHE] Using ultra-lightweight in-memory cache for medium video ({duration_seconds:.1f}s)")
                converted_path = self.reface_with_light_cache(
                    video_path, faces, output_video_path,
                    preview=preview,
                    disable_similarity=disable_similarity,
                    multiple_faces_mode=multiple_faces_mode,
                    partial_reface_ratio=partial_reface_ratio
                )
            elif duration_seconds < 60:
                # Long videos: use selective disk cache (detection only, not embeddings)
                print(f"[CACHE] Using selective disk cache (detection only) for long video ({duration_seconds:.1f}s)")
                converted_path = self.reface_with_cache(
                    video_path, faces, output_video_path,
                    preview=preview,
                    disable_similarity=disable_similarity,
                    multiple_faces_mode=multiple_faces_mode,
                    partial_reface_ratio=partial_reface_ratio,
                    cache_embeddings=False
                )
            else:
                # Very long videos: use full disk cache (detection + embeddings)
                print(f"[CACHE] Using full disk cache for very long video ({duration_seconds:.1f}s)")
                converted_path = self.reface_with_cache(
                    video_path, faces, output_video_path,
                    preview=preview,
                    disable_similarity=disable_similarity,
                    multiple_faces_mode=multiple_faces_mode,
                    partial_reface_ratio=partial_reface_ratio,
                    cache_embeddings=True
                )
        else:
            # Use original non-cached implementation
            self.prepare_faces(faces, disable_similarity=disable_similarity, multiple_faces_mode=multiple_faces_mode)
            self.first_face = False if multiple_faces_mode else (faces[0].get("origin") is None or disable_similarity)
            self.partial_reface_ratio = partial_reface_ratio

            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)  # Increase buffer size for smoother I/O
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Use mp4v for OpenCV (H264 support is unreliable in OpenCV)
            # FFmpeg will handle H264 encoding in __convert_video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

            # Dynamic batch size based on VRAM
            batch_size = self._calculate_optimal_batch_size(frame_width, frame_height, fps)

            frames = []
            frame_index = 0
            skip_rate = 10 if preview else 1

            with tqdm(total=total_frames, desc="Extracting frames") as pbar:
                while cap.isOpened():
                    flag, frame = cap.read()
                    if not flag:
                        break
                    if frame_index % skip_rate == 0:
                        frames.append(frame)
                        if len(frames) > batch_size:
                            self.reface_group(faces, frames, output)
                            frames = []
                            gc.collect()
                    frame_index += 1
                    pbar.update()
        
            cap.release()
            if frames:
                self.reface_group(faces, frames, output)
            output.release()
        
            converted_path = self.__convert_video(video_path, output_video_path, preview=preview)
    
        if video_path.lower().endswith(".gif"):
            if preview:
                gif_output_path = os.path.join("output", "preview", os.path.basename(converted_path).replace(".mp4", ".gif"))
            else:
                gif_output_path = os.path.join("output", "gifs", os.path.basename(converted_path).replace(".mp4", ".gif"))
    
            self.__generate_gif(converted_path, gif_output_path)
            return converted_path, gif_output_path
    
        return converted_path, None
    
   
  


    def __generate_gif(self, video_path, gif_output_path):
        os.makedirs(os.path.dirname(gif_output_path), exist_ok=True)
        print(f"Generating GIF at {gif_output_path}")
        (
            ffmpeg
            .input(video_path)
            .output(gif_output_path, vf='fps=10,scale=512:-1:flags=lanczos', loop=0)
            .overwrite_output()
            .run(quiet=True)
        )

    def __convert_video(self, video_path, output_video_path, preview=False):
        if self.video_has_audio and not preview:
            new_path = output_video_path + str(random.randint(0, 999)) + "_c.mp4"
            in1 = ffmpeg.input(output_video_path)
            in2 = ffmpeg.input(video_path)
            out = ffmpeg.output(in1.video, in2.audio, new_path, video_bitrate=self.ffmpeg_video_bitrate, vcodec=self.ffmpeg_video_encoder)
            out.run(overwrite_output=True, quiet=True)
        else:
            new_path = output_video_path
        print(f"Refaced video saved at: {os.path.abspath(new_path)}")
        return new_path

    def reface_image(self, image_path, faces, disable_similarity=False, multiple_faces_mode=False, partial_reface_ratio=0.0):
         self.prepare_faces(faces, disable_similarity=disable_similarity, multiple_faces_mode=multiple_faces_mode)
         self.first_face = False if multiple_faces_mode else (faces[0].get("origin") is None or disable_similarity)
         self.partial_reface_ratio = partial_reface_ratio
 
         ext = osp.splitext(image_path)[1].lower()
         os.makedirs("output", exist_ok=True)
         original_name = osp.splitext(osp.basename(image_path))[0]
         timestamp = str(int(time.time()))
 
         if ext in ['.tif', '.tiff']:
             pil_img = Image.open(image_path)
             frames = []
 
             page_count = 0
             try:
                 while True:
                     pil_img.seek(page_count)
                     page_count += 1
             except EOFError:
                 pass
 
             pil_img = Image.open(image_path)
 
             with tqdm(total=page_count, desc="Processing TIFF pages") as pbar:
                 for page in range(page_count):
                     pil_img.seek(page)
                     bgr_image = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
                     refaced_bgr = self.process_first_face(bgr_image.copy()) if self.first_face else self.process_faces(bgr_image.copy())
                     enhanced_bgr = enhance_image_memory(refaced_bgr)
                     enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
                     enhanced_pil = Image.fromarray(enhanced_rgb)
                     frames.append(enhanced_pil)
                     pbar.update(1)
 
             output_path = os.path.join("output", f"{original_name}_{timestamp}.tif")
             frames[0].save(output_path, save_all=True, append_images=frames[1:], compression="tiff_deflate")
             print(f"Saved multipage refaced TIFF to {output_path}")
             return output_path
 
         else:
             bgr_image = cv2.imread(image_path)
             if bgr_image is None:
                 raise ValueError("Failed to read input image")
 
             refaced_bgr = self.process_first_face(bgr_image.copy()) if self.first_face else self.process_faces(bgr_image.copy())
             refaced_rgb = cv2.cvtColor(refaced_bgr, cv2.COLOR_BGR2RGB)
             pil_img = Image.fromarray(refaced_rgb)
             filename = f"{original_name}_{timestamp}.jpg"
             output_path = os.path.join("output", filename)
             pil_img.save(output_path, format='JPEG', quality=100, subsampling=0)
             output_path = enhance_image(output_path)
             print(f"Saved refaced image to {output_path}")
             return output_path


    def extract_faces_from_image(self, image_path, max_faces=5):
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError("Failed to read input image for face extraction.")

        faces = self.__get_faces(frame, max_num=max_faces)
        cropped_faces = []

        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, frame.shape[1])
            y2 = min(y2, frame.shape[0])

            cropped = frame[y1:y2, x1:x2]
            pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

            temp_file = tempfile.NamedTemporaryFile(delete=False, dir="./tmp", suffix=".png")
            pil_img.save(temp_file.name)
            cropped_faces.append(temp_file.name)

            if len(cropped_faces) >= max_faces:
                break

        return cropped_faces

    def __try_ffmpeg_encoder(self, vcodec):
        command = ['ffmpeg', '-y', '-f', 'lavfi', '-i', 'testsrc=duration=1:size=1280x720:rate=30', '-vcodec', vcodec, 'testsrc.mp4']
        try:
            subprocess.run(command, check=True, capture_output=True).stderr
        except subprocess.CalledProcessError:
            return False
        return True

    def __check_encoders(self):
        self.ffmpeg_video_encoder = 'libx264'
        self.ffmpeg_video_bitrate = '0'
        pattern = r"encoders: ([a-zA-Z0-9_]+(?: [a-zA-Z0-9_]+)*)"
        command = ['ffmpeg', '-codecs', '--list-encoders']
        commandout = subprocess.run(command, check=True, capture_output=True).stdout
        result = commandout.decode('utf-8').split('\n')
        for r in result:
            if "264" in r or "265" in r:
                encoders = re.search(pattern, r)
                if encoders:
                    # Prioritize hardware encoders for Tesla T4
                    for v_c in Refacer.VIDEO_CODECS:
                        for v_k in encoders.group(1).split(' '):
                            if v_c == v_k and self.__try_ffmpeg_encoder(v_k):
                                self.ffmpeg_video_encoder = v_k
                                self.ffmpeg_video_bitrate = Refacer.VIDEO_CODECS[v_k]
                                print(f"[FFMPEG] Using hardware encoder: {v_k}")
                                return

    VIDEO_CODECS = {
        'h264_nvenc': '0',      # NVIDIA GPU encoder (Tesla T4) - highest priority
        'hevc_nvenc': '0',      # NVIDIA H.265 encoder
        'h264_videotoolbox': '0',  # Apple hardware encoder
        'libx264': '0'          # Software encoder (fallback)
    }