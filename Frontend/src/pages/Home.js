"use client"

import { Link } from "react-router-dom"
import useScrollReveal from "../hooks/useScrollReveal"
import { useTheme } from "../contexts/ThemeContext"
import "./Home.css"
import Three3DBackground from "../components/Three3DBackground"

const Home = () => {
  useScrollReveal()
  const { isDarkMode } = useTheme()

  return (
    <div className={`home ${isDarkMode ? "dark" : ""}`}>
      <section id="home" className="hero-section">
        <Three3DBackground />

        <div className="hero-content">
          <div className="hero-text">
            <h1 className="hero-title">
              Your Personal <span className="gradient-text">AI Health</span> Assistant
            </h1>
            <p className="hero-description">
              Get instant medical insights, disease predictions, and personalized health advice powered by advanced AI
              technology. Your health, simplified.
            </p>
          </div>

          <div className="feature-cards">
            <Link to="/predictor" className="feature-card">
              <div className="card-icon">🔍</div>
              <h3>Disease Predictor</h3>
              <p>Analyze symptoms and get AI-powered health predictions</p>
              <div className="card-arrow">→</div>
            </Link>
            <Link to="/report-scanner" className="feature-card">
              <div className="card-icon">📋</div>
              <h3>Report Scanner</h3>
              <p>Upload and analyze medical reports with AI insights</p>
              <div className="card-arrow">→</div>
            </Link>
            {/* <Link to="/advisor" className="feature-card">
              <div className="card-icon">💊</div>
              <h3>Medicine Advisor</h3>
              <p>Get detailed information about medications and treatments</p>
              <div className="card-arrow">→</div>
            </Link> */}

            <Link to="/chatbot" className="feature-card">
              <div className="card-icon">🤖</div>
              <h3>AI Health Chat</h3>
              <p>Chat with AI for remedies and medical report analysis</p>
              <div className="card-arrow">→</div>
            </Link>
          </div>
        </div>
      </section>

      <section id="about" className="about-section">
        <div className="container">
          <h2 className="section-title scroll-reveal fade-up">About HomeDoc AI</h2>
          <div className="about-content">
            <div className="about-text scroll-reveal fade-up">
              <p>
                HomeDoc AI is your trusted digital health companion, leveraging cutting-edge artificial intelligence to
                provide personalized medical insights and guidance. Our platform combines advanced machine learning
                algorithms with comprehensive medical databases to offer accurate health predictions and
                recommendations.
              </p>
              <div className="features-grid">
                <div className="feature-item scroll-reveal fade-left">
                  <h4>🎯 Accurate Predictions</h4>
                  <p>AI-powered symptom analysis for reliable health insights</p>
                </div>
                <div className="feature-item scroll-reveal fade-up">
                  <h4>📱 Easy to Use</h4>
                  <p>Intuitive interface designed for everyone</p>
                </div>
                <div className="feature-item scroll-reveal fade-up">
                  <h4>🔒 Secure & Private</h4>
                  <p>Your health data is protected with enterprise-grade security</p>
                </div>
                <div className="feature-item scroll-reveal fade-right">
                  <h4>⚡ Instant Results</h4>
                  <p>Get immediate health insights and recommendations</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

export default Home