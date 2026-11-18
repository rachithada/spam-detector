from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from fastapi.responses import FileResponse

app = FastAPI(
    title="AI Spam Detector",
    description="NLP-based SMS Spam Detection API",
    version="1.0"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "spam_model.joblib"

model = joblib.load(MODEL_PATH)


class MessageInput(BaseModel):
    message: str = Field(min_length=1)


@app.get("/")
def home():
    return FileResponse(
        BASE_DIR / "app" / "static" / "index.html"
    )


@app.post("/predict")
def predict_spam(data: MessageInput):
    prediction = int(model.predict([data.message])[0])

    probabilities = model.predict_proba([data.message])[0]
    class_index = list(model.classes_).index(prediction)
    confidence = float(probabilities[class_index])

    return {
        "message": data.message,
        "prediction": "Spam" if prediction == 1 else "Not Spam",
        "confidence": round(confidence * 100, 2)
    }