"use client"

import { useState } from "react"
import "./Predictor.css"
import { useNavigate } from "react-router-dom";


const Predictor = () => {
  const [selectedSymptoms, setSelectedSymptoms] = useState([])
  const [prediction, setPrediction] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const navigate = useNavigate();


  const symptoms = [
    "Abdominal Pain", "Acidity", "Anxiety", "Back Pain", "Blood in Sputum",
    "Blurred Vision", "Breathlessness", "Chills", "Chest Pain", "Confusion",
    "Constipation", "Cough", "Cramps", "Cold hands and feets", "Dark Urine", "Dehydration", "Depression", "Diarrhea", "Dischronic Patches",
    "Dizziness", "Extra Marital Contacts", "Fast Heart Rate", "Fatigue", "Fever", "Frequent Urination", "Ghost Feeling",
    "Headache", "High Blood Pressure", "Indigestion", "Insomnia", "Itching",
    "Itchy Eyes", "Irregular sugar level", "Joint Pain", "Loss of Appetite",
    "Lethargy", "Memory Problems", "Mood Swings", "Muscle Weakness", "Muscle Wasting", "Nausea", "Neck Pain",
    "Nosebleeds", "Night Sweats", "nodal skin eruptions", "Painful Urination",
    "Patches in Throat", "Pus Filled Pimples" , "Restlessness", "Runny Nose", "Scurring", "Shortness of Breath",
    "Skin Rash", "Sneezing", "Sore Throat", "Stomach Pain", "Sweating", "Swollen Legs", "Ulcers on Tongue",
    "Vomiting", "Weight Gain", "Weight Loss", "Watering From Eyes", "Yellowing of Eyes",
    "Yellowish Skin"
  ]

  const handleSymptomChange = (symptom) => {
    setSelectedSymptoms((prev) => prev.includes(symptom)
      ? prev.filter((s) => s !== symptom)
      : [...prev, symptom])
  }

  const handlePredict = async () => {
    if (selectedSymptoms.length === 0) {
      alert("Please select at least one symptom.")
      return
    }

    setIsLoading(true)
    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symptoms: selectedSymptoms.map((s) =>
            s.toLowerCase().trim().replace(/\s+/g, "_")
          ),
        }),
      })
      const data = await response.json()
      setPrediction(data)
    } catch (error) {
      console.error("Prediction failed:", error)
      alert("Something went wrong.")
    } finally {
      setIsLoading(false)
    }
  }

  const resetPrediction = () => {
    setSelectedSymptoms([])
    setPrediction(null)
  }

  return (
    <div className="predictor-container">
      <div className="predictor-header">
        <h1>AI Disease Predictor</h1>
        <p>Select your symptoms to get AI-powered health insights</p>
      </div>
      <div className="predictor-content">
        <div className="symptoms-section">
          <h2>Select Your Symptoms</h2>
          <div className="symptoms-grid">
            {symptoms.map((symptom) => (
              <label key={symptom} className="symptom-checkbox">
                <input
                  type="checkbox"
                  checked={selectedSymptoms.includes(symptom)}
                  onChange={() => handleSymptomChange(symptom)}
                />
                <span className="checkmark"></span>
                {symptom}
              </label>
            ))}
          </div>
          <div className="selected-symptoms">
            <h3>Selected Symptoms ({selectedSymptoms.length})</h3>
            <div className="selected-tags">
              {selectedSymptoms.map((symptom) => (
                <span key={symptom} className="symptom-tag">
                  {symptom}
                  <button onClick={() => handleSymptomChange(symptom)}>×</button>
                </span>
              ))}
            </div>
          </div>
          <div className="action-buttons">
            <button className="predict-btn" onClick={handlePredict} disabled={isLoading}>
              {isLoading ? "Analyzing..." : "Predict Disease"}
            </button>
            <button className="reset-btn" onClick={resetPrediction}>
              <i className="fas fa-sync-alt" style={{ marginRight: "8px" }}></i>
              Reset
            </button>

          </div>
        </div>
        {prediction && prediction.predictions && (
          <div className="prediction-section">
            <h2>Prediction Results</h2>
            {prediction.predictions.map((item, index) => (
              <div className="prediction-card" key={index}>
                <div className="prediction-header">
                  <h3>{item.disease}</h3>
                  {/* <span className="confidence">Confidence: {item.confidence}</span> */}
                </div>
                <div className="prediction-details">
                  <div className="severity">
                    <strong>Severity:</strong>
                    <span className="severity-badge">{item.severity}</span>
                  </div>
                  <div className="description">
                    <strong>Description:</strong>
                    <p>{item.description}</p>
                  </div>
                  <div className="recommendations">
                    <strong>Recommendations:</strong>
                    <ul>
                      {item.recommendations.map((rec, idx) => (
                        <li key={idx}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            ))}
            <div className="disclaimer">
              <p><strong>Disclaimer:</strong> These results are for informational purposes only. Please consult a doctor.</p>
            </div>
          </div>
        )}

      </div>
      <button
        className="chatbot-toggle"
        onClick={() => navigate("/chatbot")}
        title="Go to AI Chatbot"
      >
        💬 Chat with AI
      </button>

    </div>
  )
}

export default Predictor