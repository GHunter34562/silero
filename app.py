import os
import logging
import subprocess
from threading import Lock
from typing import Literal

# Жестко ограничиваем потоки для CPU, чтобы не плодить оверхед по памяти
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import imageio_ffmpeg
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("silero_onnx_api")

SAMPLE_RATE = 24_000
MAX_TEXT_LENGTH = 8_000
MAX_AUDIO_BYTES = 30 * 1024 * 1024

app = FastAPI(title="Silero ONNX TTS API", version="2.1.0")

_session = None
_model_lock = Lock()
_inference_lock = Lock()

ONNX_MODEL_PATH = "silero_v4.onnx"  # Путь к твоему скачанному ONNX файлу модели


def get_onnx_session():
    """Ленивая загрузка ONNX-сессии (без тяжелого PyTorch)."""
    global _session
    if _session is None:
        with _model_lock:
            if _session is None:
                logger.info("Loading Silero ONNX model...")
                
                if not os.path.exists(ONNX_MODEL_PATH):
                    raise RuntimeError(
                        f"ONNX model file not found at {ONNX_MODEL_PATH}. "
                        "Download it from the official Silero repository and place it in the app directory."
                    )

                # Настройка сессии с упором на минимальное потребление RAM
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                _session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
                logger.info("Silero ONNX model loaded successfully!")
    return _session


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    speaker: Literal[
        "kseniya",
        "baya",
        "xenia",
        "eugene",
        "aidar",
    ] = "kseniya"


def text_to_tokens(text: str) -> np.ndarray:
    """
    Примитивная/стандартная токенизация под Silero. 
    (Символьный маппинг: каждый символ переводится в его ASCII/Unicode ID или согласно словарю модели).
    """
    # Пример базового маппинга символов для ONNX входа Silero
    # (в зависимости от конкретной экспортированной модели входной тензор ожидает массив int64)
    cleaned_text = text.lower().strip()
    tokens = [ord(char) for char in cleaned_text]
    # Добавляем батч-дименшн [1, len]
    return np.array([tokens], dtype=np.int64)


def encode_mp3(pcm_float32: bytes) -> bytes:
    """Кодирование PCM float32 в MP3 через FFmpeg."""
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

    return result.stdout


@app.get("/")
def ping():
    return {"status": "ok", "engine": "onnxruntime"}


@app.post("/generate")
def generate(data: TTSRequest):
    text = data.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text must not be empty")

    try:
        session = get_onnx_session()
        input_tensor = text_to_tokens(text)

        # Получаем имена входов модели динамически
        input_name = session.get_inputs()[0].name
        
        # Некоторые версии ONNX моделей Silero требуют также передать speaker_id или sample_rate
        inputs = {input_name: input_tensor}

        with _inference_lock:
            # Инференс через ONNX Runtime без участия PyTorch
            outputs = session.run(None, inputs)
            audio_array = outputs[0].squeeze() # Убираем лишние размерности

        # Переводим numpy array в байты float32 PCM
        pcm_float32 = audio_array.astype("<f4", copy=False).tobytes(order="C")
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
    except Exception as exc:
        logger.exception("TTS ONNX generation failed")
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(exc)}")
