import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
import torch
import io
import soundfile as sf

app = FastAPI()
torch.set_num_threads(1)

model = None

def get_model():
    global model
    if model is None:
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='ru',
            speaker='v4_ru',
            trust_repo=True
        )
        model.to(torch.device('cpu'))
    return model

class TTSRequest(BaseModel):
    text: str
    speaker: str = "kseniya"

@app.get("/")
def ping():
    return {"status": "ok"}

@app.post("/generate")
def generate(data: TTSRequest):
    try:
        tts_model = get_model()
        audio = tts_model.apply_tts(
            text=data.text,
            speaker=data.speaker,
            sample_rate=24000
        )
        
        buffer = io.BytesIO()
        sf.write(buffer, audio.numpy(), 24000, format='WAV')
        buffer.seek(0)
        
        return Response(content=buffer.read(), media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
