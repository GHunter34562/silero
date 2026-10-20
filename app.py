import os
import logging
import subprocess
from contextlib import asynccontextmanager
from typing import Literal
import traceback
import gc

os.environ["TORCH_HOME"] = "/tmp/torch"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536" 

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import imageio_ffmpeg
import torch
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("silero_api")

SAMPLE_RATE = 24_000
MAX_TEXT_LENGTH = 2_000
MAX_AUDIO_BYTES = 15 * 1024 * 1024

model = None

# Загружаем модель прямо при старте приложения, чтобы видеть реальную ошибку в логах хоста
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        logger.info("Starting up and loading Silero model...")
        os.makedirs("/tmp/torch", exist_ok=True)
        
        loaded_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker="v4_ru",
            trust_repo=True,
        )
        loaded_model.to(torch.device("cpu"))
        loaded_model.eval()
        model = loaded_model
        logger.info("Silero model loaded successfully during startup!")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load model at startup: {e}", exc_info=True)
        # Не падаем наглухо при старте, чтобы хост не ушел в бесконечный рестартный цикличный шторм,
        # но модель останется None
    yield

app = FastAPI(title="Silero TTS Startup Debug API", version="2.1.2", lifespan=lifespan)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    speaker: Literal["kseniya", "baya", "xenia", "eugene", "aidar"] = "kseniya"


def encode_mp3(pcm_float32: bytes) -> bytes:
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel", "error",
            "-f", "f32le",
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-i", "pipe:0",
            "-vn",
            "-codec:a", "libmp3lame",
            "-b:a", "64k",
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-f", "mp3",
            "pipe:1",
        ],
        input=pcm_float32,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"MP3 encoding failed: {detail}")

    return result.stdout


@app.get("/")
def ping():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/generate")
def generate(data: TTSRequest):
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model is not loaded. Check hosting startup logs for torch.hub.load errors."
        )

    text = data.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text must not be empty")

    try:
        gc.collect()
        with torch.inference_mode():
            audio = model.apply_tts(
                text=text,
                speaker=data.speaker,
                sample_rate=SAMPLE_RATE,
            )

        pcm_float32 = (
            audio.detach()
            .cpu()
            .numpy()
            .astype("<f4", copy=False)
            .tobytes(order="C")
        )
        del audio
        gc.collect()

        mp3_data = encode_mp3(pcm_float32)
        del pcm_float32
        gc.collect()

        return Response(
            content=mp3_data,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": 'inline; filename="voice.mp3"',
                "Cache-Control": "no-store",
            },
        )

    except Exception as exc:
        err_msg = "".join(traceback.format_exception(type(exc), exc, exc.traceback))
        logger.exception("Detailed TTS generation failed:\n%s", err_msg)
        raise HTTPException(status_code=500, detail=f"{type(exc).name}: {str(exc)}")
