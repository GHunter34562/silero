import os
import logging
import subprocess
from threading import Lock
from typing import Literal

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import imageio_ffmpeg
import torch
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("silero_api")

SAMPLE_RATE = 24_000
MAX_TEXT_LENGTH = 8_000
MAX_AUDIO_BYTES = 30 * 1024 * 1024

app = FastAPI(title="Silero TTS API", version="2.0.0")
torch.set_num_threads(1)

model = None
_model_lock = Lock()
_inference_lock = Lock()


def get_model():
    """Загрузить модель Silero v4 только один раз."""
    global model

    if model is None:
        with _model_lock:
            if model is None:
                logger.info("Loading Silero v4 Russian TTS model")

                loaded_model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-models",
                    model="silero_tts",
                    language="ru",
                    speaker="v4_ru",
                    trust_repo=True,
                )

                loaded_model.to(torch.device("cpu"))
                model = loaded_model

                logger.info("Silero model loaded")

    return model


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    speaker: Literal[
        "kseniya",
        "baya",
        "xenia",
        "eugene",
        "aidar",
    ] = "kseniya"


def encode_mp3(pcm_float32: bytes) -> bytes:
    """Кодировать mono 24 kHz float32 PCM прямо в MP3, без промежуточного WAV."""
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
        timeout=40,
        check=False,
    )

    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"MP3 encoding failed: {detail}")

    if len(result.stdout) > MAX_AUDIO_BYTES:
        raise RuntimeError("Generated MP3 is too large")

    is_mp3 = result.stdout.startswith(b"ID3") or (
        len(result.stdout) > 2
        and result.stdout[0] == 0xFF
        and (result.stdout[1] & 0xE0) == 0xE0
    )

    if not is_mp3:
        raise RuntimeError("Encoder returned invalid MP3 data")

    return result.stdout


@app.get("/")
def ping():
    return {"status": "ok"}


@app.post("/generate")
def generate(data: TTSRequest):
    text = data.text.strip()

    if not text:
        raise HTTPException(
            status_code=422,
            detail="Text must not be empty",
        )

    try:
        tts_model = get_model()

        # Не запускаем несколько генераций одновременно,
        # чтобы избежать резких пиков нагрузки.
        with _inference_lock:
            audio = tts_model.apply_tts(
                text=text,
                speaker=data.speaker,
                sample_rate=SAMPLE_RATE,
            )

        # Silero возвращает Tensor. Передаём float32 PCM прямо в FFmpeg.
        pcm_float32 = (
            audio.detach()
            .cpu()
            .numpy()
            .astype("<f4", copy=False)
            .tobytes(order="C")
        )

        mp3_data = encode_mp3(pcm_float32)

        return Response(
            content=mp3_data,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": 'inline; filename="voice.mp3"',
                "Cache-Control": "no-store",
            },
        )

    except HTTPException:
        raise

    except subprocess.TimeoutExpired as exc:
        logger.exception("MP3 encoder timed out")
        raise HTTPException(
            status_code=504,
            detail="MP3 encoding timed out",
        ) from exc

    except Exception as exc:
        logger.exception("TTS generation failed")
        raise HTTPException(
            status_code=500,
            detail="Audio generation failed",
        ) from exc
