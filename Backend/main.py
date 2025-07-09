from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services.chatbot_logic import get_remedy_reply
import joblib
from collections import OrderedDict
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
from services import report_interpreter
import json
import tempfile
import os
# Load trained model and encoders
model = joblib.load("model/knn_model.pkl")
mlb = joblib.load("model/symptom_binarizer.pkl")
le = joblib.load("model/label_encoder.pkl")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SymptomRequest(BaseModel):
    symptoms: list[str]


@app.post("/predict")
def predict(request: SymptomRequest):
    try:
        input_vector = mlb.transform([request.symptoms])
        if input_vector.sum() == 0:
            return {
                "error": (
                    "None of the provided symptoms match our database."
                )
            }

        distances, indices = model.kneighbors(input_vector, n_neighbors=15)
        predicted_diseases = le.inverse_transform(model._y[indices[0]])

        results = []
        for disease, distance in zip(predicted_diseases, distances[0]):
            score = round(100 / (1 + distance), 2)
            results.append({
                "disease": disease,
                "confidence": f"{score}%",
                "description": f"Match based on symptoms: {disease}.",
                "recommendations": [
                    "Consult a doctor for confirmation.",
                    "Stay hydrated and rest.",
                    "Do not rely solely on online tools for diagnosis."
                ],
                "severity": "Varies"
            })

        # Remove duplicates and keep highest-ranked match
        results = list(
            OrderedDict((r["disease"], r) for r in results).values()
        )
        # Sort by confidence (highest first)
        results = sorted(
            results,
            key=lambda r: float(r["confidence"].strip('%')),
            reverse=True
        )

        # Optionally limit to top 5
        results = results[:5]

        return {"predictions": results}

    except Exception as e:
        return {"error": str(e)}


