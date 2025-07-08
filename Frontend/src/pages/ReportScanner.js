"use client";

import { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { useTheme } from "../contexts/ThemeContext";
import axios from "axios";
import jsPDF from "jspdf";
import "./ReportScanner.css";

const ReportScanner = () => {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scanResult, setScanResult] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const { isDarkMode } = useTheme();
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  const handleFileUpload = (file) => {
    if (file && (file.type.includes("pdf") || file.type.includes("image"))) {
      setUploadedFile(file);
      setScanResult(null);
    } else {
      alert("Please upload a valid PDF or image file.");
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  };

  const handleScanReport = async () => {
    if (!uploadedFile) {
      alert("Please upload a file first.");
      return;
    }

    setIsScanning(true);
    setScanResult(null);

    const formData = new FormData();
    formData.append("file", uploadedFile);

    try {
      const response = await axios.post(`${process.env.REACT_APP_API_URL}/interpret-report`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      setScanResult(response.data);
    } catch (error) {
      alert(error.response?.data?.error || "Something went wrong. Please try again.");
    } finally {
      setIsScanning(false);
    }
  };

  const downloadPDF = () => {
    if (!scanResult) return;

    const doc = new jsPDF();
    doc.setFontSize(16);
    doc.text("Medical Report Summary", 10, 10);

    doc.setFontSize(12);
    doc.text(`Report Type: ${scanResult.reportType}`, 10, 20);
    doc.text(`File Name: ${uploadedFile?.name || "N/A"}`, 10, 30);
    doc.text(`\nKey Findings:`, 10, 40);
    scanResult.keyFindings.forEach((item, i) => {
      doc.text(`- ${item}`, 12, 50 + i * 7);
    });

    let offset = 50 + scanResult.keyFindings.length * 7;
    doc.text("\nSummary:", 10, offset += 10);
    doc.text(doc.splitTextToSize(scanResult.detailedAnalysis.summary, 180), 12, offset += 5);

    offset += 25;
    doc.text("Recommendations:", 10, offset);
    scanResult.detailedAnalysis.recommendations.forEach((r, i) => {
      doc.text(`- ${r}`, 12, offset + 10 + i * 7);
    });

    offset = offset + 10 + scanResult.detailedAnalysis.recommendations.length * 7;
    doc.text("Risk Assessment:", 10, offset);
    scanResult.detailedAnalysis.riskFactors.forEach((r, i) => {
      doc.text(`- ${r.factor}: ${r.explanation}`, 12, offset + 10 + i * 7);
    });

    doc.save("Health_Report_Summary.pdf");
  };


  const removeFile = () => {
    setUploadedFile(null);
    setScanResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const navigateToChat = () => {
    navigate("/chatbot");
  };

  const getFileIcon = (file) => {
    if (file.type.includes("pdf")) return "📄";
    if (file.type.includes("image")) return "🖼️";
    return "📋";
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Number.parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };
  // const getRiskLevelTag = (confidence) => {
  //   const score = parseInt(confidence);
  //   if (score >= 90) return <span style={{ color: "green", fontWeight: 600 }}>✅ Low Risk</span>;
  //   if (score >= 70) return <span style={{ color: "orange", fontWeight: 600 }}>🟡 Moderate Risk</span>;
  //   return <span style={{ color: "red", fontWeight: 600 }}>⚠️ High Risk</span>;
  // };

  const renderTooltip = (text) => {
    return <span title={text} style={{ textDecoration: "underline dotted", cursor: "help" }}>{text}</span>;
  };

  return (
    <div className={`report-scanner-container ${isDarkMode ? "dark" : ""}`}>
      <h1>Medical Report Scanner</h1>
      <p>Upload your medical reports for AI-powered, personalized health insights</p>

      <div className="scanner-content">
        <div className="upload-section">
          <div
            className={`upload-area ${dragActive ? "drag-active" : ""} ${uploadedFile ? "has-file" : ""}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.jpg,.jpeg,.png,.bmp,.tiff"
              onChange={handleFileInputChange}
              style={{ display: "none" }}
            />

            {!uploadedFile ? (
              <div className="upload-placeholder">
                <div className="upload-icon">
                  <i className="fas fa-cloud-upload-alt"></i>
                </div>
                <h3>Drop your medical report here</h3>
                <p>or click to browse files</p>
                <div className="supported-formats">
                  <span>Supported: PDF, JPG, PNG, TIFF</span>
                </div>
              </div>
            ) : (
              <div className="uploaded-file">
                <div className="file-info">
                  <div className="file-icon">{getFileIcon(uploadedFile)}</div>
                  <div className="file-details">
                    <h4>{uploadedFile.name}</h4>
                    <p>{formatFileSize(uploadedFile.size)}</p>
                  </div>
                  <button className="remove-file" onClick={(e) => { e.stopPropagation(); removeFile(); }}>
                    <i className="fas fa-times"></i>
                  </button>
                </div>
                <div className="file-preview">
                  <div className="preview-placeholder">
                    <i className="fas fa-file-medical"></i>
                    <span>Ready for analysis</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          <button
            className={`scan-button ${!uploadedFile ? "disabled" : ""} ${isScanning ? "scanning" : ""}`}
            onClick={handleScanReport}
            disabled={!uploadedFile || isScanning}
          >
            {isScanning ? (
              <>
                <div className="scanning-spinner"></div>
                <span>Analyzing Report...</span>
              </>
            ) : (
              <>
                <i className="fas fa-search"></i>
                <span>Scan Report</span>
              </>
            )}
          </button>
        </div>

        {scanResult && (
          <div className="analysis-section">
            <div className="analysis-header">
              <h2>Analysis Results</h2>
              {/* <div className="analysis-meta">
                <span className="confidence">Confidence: {scanResult.confidence}</span>
                <span className="processing-time">Processed in {scanResult.processingTime}</span>
                <span>{getRiskLevelTag(scanResult.confidence)}</span>
              </div> */}
            </div>

            <div className="analysis-content">
              <div className="overview-card">
                <h3>Report Overview</h3>
                <div className="overview-details">
                  <div className="detail-item">
                    <span className="label">Report Type:</span>
                    <span className="value">{renderTooltip(scanResult.reportType)}</span>
                  </div>
                  <div className="detail-item">
                    <span className="label">File Name:</span>
                    <span className="value">{uploadedFile.name}</span>
                  </div>
                  
                </div>
              </div>

              <div className="findings-card">
                <h3>Key Findings</h3>
                <ul className="findings-list">
                  {scanResult.keyFindings.map((finding, index) => (
                    <li key={index} className="finding-item">
                      <i className="fas fa-check-circle"></i>
                      <span>{finding}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="analysis-card">
                <h3>Detailed Analysis</h3>
                <div className="analysis-summary">
                  <h4>Summary</h4>
                  <p>{scanResult.detailedAnalysis.summary}</p>
                </div>
                <div className="recommendations">
                  <h4>Recommendations</h4>
                  <ul>
                    {scanResult.detailedAnalysis.recommendations.map((rec, index) => (
                      <li key={index}>{rec}</li>
                    ))}
                  </ul>
                </div>
                <div className="risk-factors">
                  <h4>Risk Assessment</h4>
                  <ul>
                    {scanResult.detailedAnalysis.riskFactors.map((risk, index) => (
                      <li key={index}><strong>{risk.factor}</strong>: {risk.explanation}</li>
                    ))}
                  </ul>
                </div>
              </div>

              <div className="action-buttons">
                <button className="chat-button" onClick={navigateToChat}>
                  <i className="fas fa-comments"></i>
                  <span>Chat with our AI</span>
                </button>
                <button className="download-button" onClick={downloadPDF}>
                  <i className="fas fa-download"></i>
                  <span>Download Summary</span>
                </button>
              </div>

              <div className="disclaimer">
                <div className="disclaimer-content">
                  <i className="fas fa-exclamation-triangle"></i>
                  <div>
                    <h4>Important Disclaimer</h4>
                    <p>
                      This AI analysis is for informational purposes only and should not replace
                      professional medical advice. Always consult with qualified healthcare providers
                      for proper diagnosis and treatment.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ReportScanner;