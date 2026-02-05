/**
 * Model Evaluator Component
 * Provides two modes:
 * 1. Evaluate: Upload CSV with actual/predicted columns
 * 2. Predict: Upload test data to run inference with loaded model
 */

import React, { useState, useCallback } from 'react'
import { useSession } from '../contexts/SessionContext'
import {
  BarChart, Bar, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, PieChart, Pie, Legend
} from 'recharts'
import './ModelEvaluator.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

const COLORS = ['#22c55e', '#ef4444', '#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899']

export default function ModelEvaluator({ modelId, modelInfo, onEvaluationComplete }) {
  const { sessionId } = useSession()
  const [mode, setMode] = useState('evaluate') // 'evaluate' or 'predict'
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [results, setResults] = useState(null)
  const [dragActive, setDragActive] = useState(false)

  // Handle file selection
  const handleFileSelect = useCallback((selectedFile) => {
    if (selectedFile && (selectedFile.name.endsWith('.csv') || 
        selectedFile.name.endsWith('.xlsx') || 
        selectedFile.name.endsWith('.xls'))) {
      setFile(selectedFile)
      setError(null)
    } else {
      setError('Please upload a CSV or Excel file')
    }
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragActive(false)
    const droppedFile = e.dataTransfer.files[0]
    handleFileSelect(droppedFile)
  }, [handleFileSelect])

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setDragActive(true)
  }, [])

  const handleDragLeave = useCallback((e) => {
    e.preventDefault()
    setDragActive(false)
  }, [])

  // Submit for evaluation
  const handleSubmit = async () => {
    if (!file) {
      setError('Please select a file')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      let endpoint
      if (mode === 'evaluate') {
        // Evaluate from pre-computed predictions
        endpoint = `${API_BASE}/models/evaluate-predictions`
      } else {
        // Run model inference
        endpoint = `${API_BASE}/models/${modelId}/evaluate-with-file`
      }

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'X-Session-Id': sessionId
        },
        body: formData
      })

      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(err.detail || 'Evaluation failed')
      }

      const data = await response.json()
      setResults(data)
      
      if (onEvaluationComplete) {
        onEvaluationComplete(data)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // Reset to upload new file
  const handleReset = () => {
    setFile(null)
    setResults(null)
    setError(null)
  }

  return (
    <div className="model-evaluator">
      {/* Mode Toggle */}
      <div className="evaluator-header">
        <h3>🧪 Evaluate Model</h3>
        <div className="mode-toggle">
          <button 
            className={`mode-btn ${mode === 'evaluate' ? 'active' : ''}`}
            onClick={() => setMode('evaluate')}
          >
            📊 Upload Predictions
          </button>
          <button 
            className={`mode-btn ${mode === 'predict' ? 'active' : ''}`}
            onClick={() => setMode('predict')}
            disabled={!modelId}
            title={!modelId ? 'Upload a model first' : ''}
          >
            🤖 Run Inference
          </button>
        </div>
      </div>

      {/* Mode Description */}
      <div className="mode-description">
        {mode === 'evaluate' ? (
          <p>
            <strong>Upload Predictions Mode:</strong> Upload a CSV with <code>actual</code> and <code>predicted</code> columns 
            to calculate accuracy, F1-score, and confusion matrix.
          </p>
        ) : (
          <p>
            <strong>Run Inference Mode:</strong> Upload test data with feature columns. 
            The model will make predictions and show metrics if a <code>target</code> column exists.
          </p>
        )}
      </div>

      {!results ? (
        <>
          {/* File Upload Zone */}
          <div 
            className={`upload-zone ${dragActive ? 'drag-active' : ''} ${file ? 'has-file' : ''}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => document.getElementById('eval-file-input').click()}
          >
            <input
              id="eval-file-input"
              type="file"
              accept=".csv,.xlsx,.xls"
              onChange={(e) => handleFileSelect(e.target.files[0])}
              style={{ display: 'none' }}
            />
            
            {file ? (
              <div className="file-preview">
                <span className="file-icon">📄</span>
                <div className="file-info">
                  <span className="file-name">{file.name}</span>
                  <span className="file-size">{(file.size / 1024).toFixed(1)} KB</span>
                </div>
                <button className="remove-file" onClick={(e) => {
                  e.stopPropagation()
                  setFile(null)
                }}>✕</button>
              </div>
            ) : (
              <div className="upload-prompt">
                <span className="upload-icon">📂</span>
                <p>Drop your {mode === 'evaluate' ? 'predictions' : 'test data'} CSV here</p>
                <p className="hint">or click to browse</p>
              </div>
            )}
          </div>

          {/* Expected Format */}
          <div className="format-hint">
            <h4>Expected Format:</h4>
            {mode === 'evaluate' ? (
              <code>
                actual,predicted<br/>
                1,1<br/>
                0,1<br/>
                1,0<br/>
                ...
              </code>
            ) : (
              <code>
                feature1,feature2,target<br/>
                0.5,1.2,1<br/>
                0.8,0.9,0<br/>
                ...
              </code>
            )}
          </div>

          {error && <div className="error-message">⚠️ {error}</div>}

          {/* Submit Button */}
          <button 
            className="evaluate-btn"
            onClick={handleSubmit}
            disabled={!file || loading}
          >
            {loading ? (
              <>
                <span className="spinner-small"></span>
                {mode === 'evaluate' ? 'Calculating Metrics...' : 'Running Inference...'}
              </>
            ) : (
              mode === 'evaluate' ? '📈 Calculate Metrics' : '🚀 Run Model'
            )}
          </button>
        </>
      ) : (
        /* Results Display */
        <EvaluationResults 
          results={results} 
          onReset={handleReset}
          mode={mode}
        />
      )}
    </div>
  )
}

// Results Display Component
function EvaluationResults({ results, onReset, mode }) {
  const isClassification = results.task === 'classification'
  
  return (
    <div className="evaluation-results animate-fade-in">
      {/* Header */}
      <div className="results-header">
        <div className="results-title">
          <span className="success-icon">✅</span>
          <h4>Evaluation Complete</h4>
        </div>
        <button className="btn-new-eval" onClick={onReset}>
          📂 New Evaluation
        </button>
      </div>

      {/* Summary */}
      <div className="results-summary">
        <div className="summary-item">
          <span className="label">File</span>
          <span className="value">{results.filename}</span>
        </div>
        <div className="summary-item">
          <span className="label">Samples</span>
          <span className="value">{results.n_samples?.toLocaleString() || results.n_test_samples?.toLocaleString()}</span>
        </div>
        <div className="summary-item">
          <span className="label">Task</span>
          <span className="value badge">{results.task || 'classification'}</span>
        </div>
      </div>

      {isClassification ? (
        <ClassificationMetrics results={results} />
      ) : (
        <RegressionMetrics results={results} />
      )}
    </div>
  )
}

// Classification Metrics
function ClassificationMetrics({ results }) {
  return (
    <>
      {/* Main Metrics */}
      <div className="metrics-grid">
        <MetricCard 
          label="Accuracy" 
          value={results.accuracy} 
          format="percent"
          color={results.accuracy >= 80 ? '#22c55e' : results.accuracy >= 60 ? '#f59e0b' : '#ef4444'}
        />
        <MetricCard 
          label="Precision" 
          value={results.precision} 
          format="percent"
          color="#3b82f6"
        />
        <MetricCard 
          label="Recall" 
          value={results.recall} 
          format="percent"
          color="#8b5cf6"
        />
        <MetricCard 
          label="F1 Score" 
          value={results.f1_score} 
          format="percent"
          color="#ec4899"
        />
      </div>

      {/* Confusion Matrix */}
      {results.confusion_matrix && (
        <div className="chart-section">
          <h5>Confusion Matrix</h5>
          <ConfusionMatrix 
            matrix={results.confusion_matrix}
            labels={results.class_labels || []}
          />
        </div>
      )}

      {/* Prediction Distribution */}
      {results.prediction_distribution && (
        <div className="chart-section">
          <h5>Prediction Distribution</h5>
          <PredictionDistributionChart data={results.prediction_distribution} />
        </div>
      )}
    </>
  )
}

// Regression Metrics
function RegressionMetrics({ results }) {
  return (
    <>
      {/* Main Metrics */}
      <div className="metrics-grid">
        <MetricCard 
          label="R² Score" 
          value={results.r2} 
          format="number"
          color={results.r2 >= 0.8 ? '#22c55e' : results.r2 >= 0.5 ? '#f59e0b' : '#ef4444'}
        />
        <MetricCard 
          label="RMSE" 
          value={results.rmse} 
          format="number"
          color="#3b82f6"
        />
        <MetricCard 
          label="MAE" 
          value={results.mae} 
          format="number"
          color="#8b5cf6"
        />
        <MetricCard 
          label="MSE" 
          value={results.mse} 
          format="number"
          color="#ec4899"
        />
      </div>

      {/* Scatter Plot */}
      {results.scatter_data && results.scatter_data.length > 0 && (
        <div className="chart-section">
          <h5>Actual vs Predicted</h5>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 40, left: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis 
                type="number" 
                dataKey="actual" 
                name="Actual"
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                label={{ value: 'Actual', position: 'bottom', fill: '#94a3b8' }}
              />
              <YAxis 
                type="number" 
                dataKey="predicted" 
                name="Predicted"
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                label={{ value: 'Predicted', angle: -90, position: 'left', fill: '#94a3b8' }}
              />
              <Tooltip 
                contentStyle={{ 
                  background: 'rgba(30, 41, 59, 0.95)', 
                  border: '1px solid #475569', 
                  borderRadius: '8px' 
                }}
                formatter={(value) => value.toFixed(4)}
              />
              <Scatter data={results.scatter_data} fill="#3b82f6" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Residual Stats */}
      {results.residual_stats && (
        <div className="residual-stats">
          <h5>Residual Statistics</h5>
          <div className="stats-row">
            <span>Mean: <strong>{results.residual_stats.mean}</strong></span>
            <span>Std: <strong>{results.residual_stats.std}</strong></span>
            <span>Min: <strong>{results.residual_stats.min}</strong></span>
            <span>Max: <strong>{results.residual_stats.max}</strong></span>
          </div>
        </div>
      )}
    </>
  )
}

// Metric Card
function MetricCard({ label, value, format, color }) {
  const displayValue = format === 'percent' 
    ? `${value?.toFixed(1) || '0'}%`
    : value?.toFixed(4) || '0'
  
  return (
    <div className="metric-card" style={{ borderColor: color }}>
      <div className="metric-value" style={{ color }}>{displayValue}</div>
      <div className="metric-label">{label}</div>
    </div>
  )
}

// Confusion Matrix Visualization
function ConfusionMatrix({ matrix, labels }) {
  const maxVal = Math.max(...matrix.flat())
  
  return (
    <div className="confusion-matrix">
      <div className="matrix-labels-x">
        <span className="axis-label">Predicted</span>
      </div>
      <div className="matrix-container">
        <div className="matrix-labels-y">
          <span className="axis-label">Actual</span>
        </div>
        <div className="matrix-grid" style={{ 
          gridTemplateColumns: `repeat(${matrix[0]?.length || 2}, 1fr)` 
        }}>
          {matrix.map((row, i) => 
            row.map((val, j) => {
              const intensity = maxVal > 0 ? val / maxVal : 0
              const bgColor = i === j 
                ? `rgba(34, 197, 94, ${0.2 + intensity * 0.6})` // Green for diagonal
                : `rgba(239, 68, 68, ${0.1 + intensity * 0.5})` // Red for off-diagonal
              
              return (
                <div 
                  key={`${i}-${j}`} 
                  className="matrix-cell"
                  style={{ backgroundColor: bgColor }}
                  title={`Actual: ${labels[i] || i}, Predicted: ${labels[j] || j}`}
                >
                  {val}
                </div>
              )
            })
          )}
        </div>
      </div>
      {/* Labels below */}
      <div className="matrix-footer">
        {labels.map((label, i) => (
          <span key={i} className="class-label">{label}</span>
        ))}
      </div>
    </div>
  )
}

// Prediction Distribution Chart
function PredictionDistributionChart({ data }) {
  const chartData = Object.entries(data).map(([label, count]) => ({
    name: label,
    count: count
  }))
  
  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={chartData} margin={{ top: 10, right: 20, left: 20, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} />
        <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
        <Tooltip 
          contentStyle={{ 
            background: 'rgba(30, 41, 59, 0.95)', 
            border: '1px solid #475569', 
            borderRadius: '8px' 
          }}
        />
        <Bar dataKey="count" radius={[4, 4, 0, 0]}>
          {chartData.map((entry, index) => (
            <Cell key={index} fill={COLORS[index % COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
