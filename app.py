from fastapi import FastAPI, Response
from pydantic import BaseModel
import torch
import io
import soundfile as sf

app = FastAPI()

# Ограничиваем CPU, чтобы не выходить за рамки бесплатного тарифа
torch.set_num_threads(2)

# Загружаем модель Silero v5 с trust_repo=True, чтобы деплой на Render не падал
model, _ = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language='ru',
    speaker='v5_4_ru',
    trust_repo=True
)
model.to(torch.device('cpu'))

class TTSRequest(BaseModel):
    text: str
    speaker: str = "kseniya" # Доступные голоса: kseniya, baya, xenia, eugene, aidar

# Пинг-эндпоинт, чтобы сервер не уходил в спячку каждые 15 минут
@app.get("/")
def ping():
    return {"status": "ok"}

@app.post("/generate")
def generate(data: TTSRequest):
    audio = model.apply_tts(
        text=data.text,
        speaker=data.speaker,
        sample_rate=48000
    )
    
    buffer = io.BytesIO()
    sf.write(buffer, audio.numpy(), 48000, format='WAV')
    buffer.seek(0)
    
    return Response(content=buffer.read(), media_type="audio/wav")
