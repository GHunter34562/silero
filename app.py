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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        logger.info("Setting up Silero model (Direct Download mode)...")
        os.makedirs("/tmp/torch", exist_ok=True)
        model_path = "/tmp/torch/v4_ru.pt"
        
        # Скачиваем модель напрямую по ссылке, минуя git и torch.hub.load
        if not os.path.isfile(model_path):
            logger.info("Downloading v4_ru.pt from official Silero repository...")
            torch.hub.download_url_to_file(
                "https://models.silero.ai/models/tts/ru/v4_ru.pt",
                model_path
            )
            logger.info("Model downloaded successfully.")

        logger.info("Loading model via PackageImporter...")
        loaded_model = torch.package.PackageImporter(model_path).load_pickle("tts_models", "model")
        loaded_model.to(torch.device("cpu"))
        loaded_model.eval()
        
        model = loaded_model
        logger.info("Silero model loaded and ready!")
        
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load model: {e}", exc_info=True)
    yield

app = FastAPI(title="Silero TTS Direct API", version="2.1.3", lifespan=lifespan)

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
            detail="Model is not loaded. Check hosting startup logs."
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
