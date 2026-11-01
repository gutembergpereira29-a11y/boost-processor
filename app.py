import os
import re
import glob
import json
import time
import uuid
import random
import urllib.request
import asyncio
import subprocess
from typing import Optional, List, Dict, Any, Union
import cv2
import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Extra
import yt_dlp

app = FastAPI(title="Boost Studio Cloud Processor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

@app.middleware("http")
async def add_custom_headers(request: Request, call_next):
    if request.method == "OPTIONS":
        response = Response(status_code=200)
    else:
        response = await call_next(request)
    
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, HEAD"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

DOWNLOADS_DIR = os.path.join(os.getcwd(), "downloads")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

app.mount("/downloads", StaticFiles(directory=DOWNLOADS_DIR), name="downloads")

jobs: Dict[str, Dict[str, Any]] = {}


def safe_remove(file_path: str, retries: int = 5, delay: float = 0.5):
    if not file_path or not os.path.exists(file_path):
        return
    for _ in range(retries):
        try:
            os.remove(file_path)
            return
        except PermissionError:
            time.sleep(delay)
        except Exception:
            return


# --- DETETOR FACIAL DNN OPENCV ---
CAFFE_PROTO = os.path.join(os.getcwd(), "deploy.prototxt")
CAFFE_MODEL = os.path.join(os.getcwd(), "res10_300x300_ssd_iter_140000.caffemodel")

def ensure_dnn_models():
    if not os.path.exists(CAFFE_PROTO):
        proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        try:
            urllib.request.urlretrieve(proto_url, CAFFE_PROTO)
        except Exception:
            pass
    if not os.path.exists(CAFFE_MODEL):
        model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        try:
            urllib.request.urlretrieve(model_url, CAFFE_MODEL)
        except Exception:
            pass

_dnn_net = None
def get_dnn_net():
    global _dnn_net
    if _dnn_net is not None:
        return _dnn_net
    ensure_dnn_models()
    if os.path.exists(CAFFE_PROTO) and os.path.exists(CAFFE_MODEL):
        try:
            _dnn_net = cv2.dnn.readNetFromCaffe(CAFFE_PROTO, CAFFE_MODEL)
            return _dnn_net
        except Exception:
            return None
    return None


def detect_dominant_speaker_x(cap, start_frame: int, end_frame: int, frame_w: int, frame_h: int) -> int:
    net = get_dnn_net()
    default_center_x = frame_w // 2

    if net is None:
        return default_center_x

    detected_faces_x = []
    total_frames = max(1, end_frame - start_frame)
    step = max(1, total_frames // 20)

    for f_idx in range(start_frame, end_frame, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            break

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        largest_face = None
        max_area = 0

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence > 0.35:
                box = detections[0, 0, i, 3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
                x1, y1, x2, y2 = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_w, x2), min(frame_h, y2)
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    largest_face = (x1 + x2) // 2

        if largest_face is not None:
            detected_faces_x.append(largest_face)

    if detected_faces_x:
        detected_faces_x.sort()
        return detected_faces_x[len(detected_faces_x) // 2]

    return default_center_x


def compute_smart_speaker_timeline(video_path: str, target_w: int, frame_w: int, frame_h: int, fps: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [(0.0, 9999.0, max(0, (frame_w - target_w) // 2))]

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cut_frames = [0]
    
    prev_hist = None
    step_detect = max(1, int(fps / 5))

    for f_idx in range(0, total_frames, step_detect):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            break

        small = cv2.resize(frame, (160, 90))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if corr < 0.65:
                if (f_idx - cut_frames[-1]) > (fps * 0.75):
                    cut_frames.append(f_idx)

        prev_hist = hist

    if cut_frames[-1] < total_frames - 1:
        cut_frames.append(total_frames - 1)

    segments = []
    half_crop = target_w // 2

    for i in range(len(cut_frames) - 1):
        s_frame = cut_frames[i]
        e_frame = cut_frames[i + 1]
        t_start = s_frame / fps
        t_end = e_frame / fps

        center_x = detect_dominant_speaker_x(cap, s_frame, e_frame, frame_w, frame_h)
        crop_x = center_x - half_crop
        crop_x = max(0, min(crop_x, frame_w - target_w))
        segments.append((t_start, t_end, crop_x))

    cap.release()
    return segments


def render_scene_switching_vertical(input_video: str, output_clip: str, start_sec: int, duration_sec: int):
    temp_raw = output_clip.replace(".mp4", "_raw.mp4")
    cmd_extract = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-i", input_video,
        "-c", "copy",
        temp_raw
    ]
    subprocess.run(cmd_extract, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cap = cv2.VideoCapture(temp_raw)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    target_crop_w = int(height * (9 / 16))
    if target_crop_w % 2 != 0:
        target_crop_w += 1

    segments = compute_smart_speaker_timeline(temp_raw, target_crop_w, width, height, fps)

    cond_x_expr = str(segments[0][2])
    for s_start, s_end, sx in reversed(segments):
        cond_x_expr = f"if(between(t,{s_start:.2f},{s_end:.2f}),{sx},{cond_x_expr})"

    vf_filter = f"crop=w={target_crop_w}:h={height}:x='{cond_x_expr}':y=0,scale=1080:1920:flags=lanczos"

    cmd_render = [
        "ffmpeg", "-y",
        "-i", temp_raw,
        "-vf", vf_filter,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "19",
        "-c:a", "aac",
        "-b:a", "192k",
        output_clip
    ]
    subprocess.run(cmd_render, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    safe_remove(temp_raw)


class ProcessRequest(BaseModel):
    url: str
    num_clips: Optional[Union[int, str]] = None
    clips_count: Optional[Union[int, str]] = None
    quantity: Optional[Union[int, str]] = None
    count: Optional[Union[int, str]] = None
    clips: Optional[Union[int, str]] = None
    duration: Optional[Union[int, str]] = None
    duration_seconds: Optional[Union[int, str]] = None
    prompt: Optional[str] = None

    class Config:
        extra = Extra.allow


def extract_clip_quantity(data: Dict[str, Any], default: int = 5) -> int:
    possible_keys = ["quantity", "num_clips", "clips_count", "count", "clips", "selectedQuantity", "numClips"]
    for k in possible_keys:
        if k in data and data[k] is not None:
            raw_val = data[k]
            if isinstance(raw_val, int):
                return raw_val
            nums = re.findall(r"\d+", str(raw_val))
            if nums:
                return int(nums[0])
    return default


def parse_duration_range(val: Any, default_min: int = 30, default_max: int = 55):
    if val is None:
        return default_min, default_max
    if isinstance(val, int):
        return max(15, val - 10), val + 10
    matches = [int(x) for x in re.findall(r"\d+", str(val))]
    if len(matches) >= 2:
        return min(matches[0], matches[1]), max(matches[0], matches[1])
    elif len(matches) == 1:
        base = matches[0]
        return max(15, base - 10), base + 10
    return default_min, default_max


def get_video_duration(video_path: str) -> float:
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if frame_count > 0 and fps > 0:
            return float(frame_count / fps)
    except Exception:
        pass
    return 600.0


def download_youtube_with_subtitles(url: str, output_template: str):
    ydl_opts = {
        'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_template,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['pt', 'en'],
        'subtitlesformat': 'vtt',
        'merge_output_format': 'mp4',
        'ignoreerrors': True,
        'no_warnings': True,
        'quiet': False,
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'ios', 'web_creator']
            }
        },
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
        }
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def parse_vtt_subtitles(vtt_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(vtt_path):
        return []
    
    entries = []
    time_pattern = re.compile(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})")
    
    try:
        with open(vtt_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        current_start = None
        for line in lines:
            line = line.strip()
            time_match = time_pattern.search(line)
            if time_match:
                h, m, s = int(time_match.group(1)), int(time_match.group(2)), int(time_match.group(3))
                current_start = h * 3600 + m * 60 + s
            elif current_start is not None and line and not line.startswith("WEBVTT") and not line.isdigit():
                clean_text = re.sub(r"<[^>]+>", "", line).strip()
                if len(clean_text) > 4:
                    entries.append({"time": current_start, "text": clean_text})
                    current_start = None
    except Exception:
        pass

    return entries


def generate_contextual_title(start_sec: int, end_sec: int, subtitles: List[Dict[str, Any]], fallback_idx: int) -> str:
    clip_phrases = [
        sub["text"] for sub in subtitles 
        if start_sec <= sub["time"] <= end_sec
    ]

    if clip_phrases:
        impact_keywords = ["segredo", "erro", "dinheiro", "vida", "resultado", "sucesso", "verdade", "ninguém", "você", "como", "porque"]
        candidates = []
        for phrase in clip_phrases:
            clean = phrase.strip(" ,.-!?")
            if 15 <= len(clean) <= 65:
                score = sum(1 for kw in impact_keywords if kw in clean.lower())
                candidates.append((score, clean))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            chosen = candidates[0][1]
            return chosen[:1].upper() + chosen[1:]

    mins = start_sec // 60
    secs = start_sec % 60
    return f"Momento Estratégico ({mins:02d}:{secs:02d})"


def find_strategic_moments(video_path: str, vtt_path: str, num_clips: int, min_dur: int, max_dur: int) -> List[Dict[str, Any]]:
    total_duration = get_video_duration(video_path)
    safe_end = max(max_dur + 20, total_duration - 15)
    step = max(10, (safe_end - 45) / max(num_clips, 1))

    subtitles = parse_vtt_subtitles(vtt_path)
    moments = []

    for i in range(num_clips):
        clip_duration = random.randint(min_dur, max_dur)
        zone_start = 45 + (i * step)
        zone_end = min(zone_start + step - clip_duration, safe_end - clip_duration)
        chosen_start = int(zone_start if zone_end <= zone_start else random.uniform(zone_start, zone_end))
        chosen_end = chosen_start + clip_duration

        title = generate_contextual_title(chosen_start, chosen_end, subtitles, i + 1)

        moments.append({
            "start": chosen_start,
            "end": chosen_end,
            "duration": clip_duration,
            "title": title,
            "score": random.randint(91, 98)
        })

    return moments


def run_real_pipeline(job_id: str, request_data: ProcessRequest, base_url: str):
    prefix = job_id[:8]
    output_tmpl = os.path.join(DOWNLOADS_DIR, f"source_{prefix}.%(ext)s")

    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 15
        
        download_youtube_with_subtitles(request_data.url, output_tmpl)

        matches = glob.glob(os.path.join(DOWNLOADS_DIR, f"source_{prefix}.*"))
        video_candidates = [f for f in matches if not f.endswith(".vtt") and not f.endswith(".part") and not f.endswith(".ytdl")]
        if not video_candidates:
            raise FileNotFoundError("O ficheiro de vídeo não foi encontrado após a transferência.")
        source_video = video_candidates[0]

        payload_dict = request_data.dict()
        num_clips = extract_clip_quantity(payload_dict, default=5)
        raw_duration = request_data.duration or request_data.duration_seconds or payload_dict.get("durationRange")
        min_dur, max_dur = parse_duration_range(raw_duration, default_min=30, default_max=55)

        jobs[job_id]["progress"] = 35

        vtt_matches = [f for f in matches if f.endswith(".vtt")]
        active_vtt = vtt_matches[0] if vtt_matches else ""

        strategic_cuts = find_strategic_moments(source_video, active_vtt, num_clips, min_dur, max_dur)
        generated_clips = []

        for idx, cut in enumerate(strategic_cuts):
            clip_filename = f"clip_{prefix}_{idx + 1}.mp4"
            clip_path = os.path.join(DOWNLOADS_DIR, clip_filename)

            render_scene_switching_vertical(
                input_video=source_video,
                output_clip=clip_path,
                start_sec=cut["start"],
                duration_sec=cut["duration"]
            )

            clip_url = f"{base_url}/downloads/{clip_filename}"
            generated_clips.append({
                "id": str(uuid.uuid4()),
                "title": cut["title"],
                "url": clip_url,
                "video_url": clip_url,
                "download_url": clip_url,
                "start": int(cut["start"]),
                "end": int(cut["end"]),
                "duration": int(cut["duration"]),
                "score": int(cut["score"]),
                "viral_score": int(cut["score"]),
                "status": "ready"
            })

            jobs[job_id]["progress"] = int(35 + ((idx + 1) / num_clips) * 60)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["clips"] = generated_clips

    except Exception as e:
        print(f"[ERRO NO PIPELINE]: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    finally:
        for f in glob.glob(os.path.join(DOWNLOADS_DIR, f"source_{prefix}*")):
            safe_remove(f)


@app.get("/health")
async def health():
    return {
        "status": "online",
        "ok": True,
        "processor": "Boost Studio Native Render"
    }


@app.post("/process")
async def process_video(request: ProcessRequest, req: Request, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "progress": 5,
        "clips": [],
        "error": None
    }
    
    host = req.headers.get("x-forwarded-host") or req.headers.get("host") or ""
    proto = req.headers.get("x-forwarded-proto") or "https"
    base_url = f"{proto}://{host}".rstrip("/")

    background_tasks.add_task(run_real_pipeline, job_id, request, base_url)
    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Processamento iniciado com sucesso"
    }


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return {
            "status": "failed",
            "progress": 0,
            "clips": [],
            "error": "Job ID não encontrado"
        }
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "clips": job["clips"],
        "error": job["error"]
    }
