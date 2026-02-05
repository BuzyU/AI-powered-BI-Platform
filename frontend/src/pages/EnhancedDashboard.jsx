/**
 * Enhanced Dashboard v2.0
 * Comprehensive persona-aware dashboard with dynamic rendering
 * 
 * Business → Power BI style (KPIs, revenue trends, profit analysis)
 * Analytics → EDA style (distributions, correlations, outliers)
 * ML → Streamlit style (accuracy, confusion matrix, ROC curves)
 * CV → Computer Vision (mAP, IoU, detection metrics)
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react'
import {
  BarChart, Bar, LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell, ScatterChart, Scatter, RadarChart, Radar,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis, Treemap, ComposedChart
} from 'recharts'
import { useSession } from '../contexts/SessionContext'
import ModelEvaluator from '../components/ModelEvaluator'
import './EnhancedDashboard.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

// Color schemes per persona
const PERSONA_COLORS = {
  business: {
    primary: '#3b82f6',
    secondary: '#10b981',
    accent: '#f59e0b',
    danger: '#ef4444',
    chart: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']
  },
  analytics: {
    primary: '#6366f1',
    secondary: '#8b5cf6',
    accent: '#a855f7',
    danger: '#f43f5e',
    chart: ['#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899', '#f43f5e']
  },
  ml_engineer: {
    primary: '#14b8a6',
    secondary: '#22c55e',
    accent: '#eab308',
    danger: '#ef4444',
    chart: ['#14b8a6', '#22c55e', '#84cc16', '#eab308', '#f97316', '#ef4444']
  },
  computer_vision: {
    primary: '#0891b2',
    secondary: '#06b6d4',
    accent: '#22d3ee',
    danger: '#f43f5e',
    chart: ['#0891b2', '#06b6d4', '#22d3ee', '#67e8f9', '#a5f3fc', '#cffafe']
  },
  data_scientist: {
    primary: '#7c3aed',
    secondary: '#8b5cf6',
    accent: '#a78bfa',
    danger: '#f43f5e',
    chart: ['#7c3aed', '#8b5cf6', '#a78bfa', '#c4b5fd', '#ddd6fe', '#ede9fe']
  },
  default: {
    primary: '#64748b',
    secondary: '#94a3b8',
    accent: '#cbd5e1',
    danger: '#ef4444',
    chart: ['#64748b', '#94a3b8', '#cbd5e1', '#e2e8f0', '#f1f5f9', '#f8fafc']
  }
}

// Persona display info
const PERSONA_INFO = {
  business: { icon: '💼', name: 'Business Intelligence', color: '#3b82f6' },
  analytics: { icon: '📊', name: 'Data Analytics', color: '#6366f1' },
  ml_engineer: { icon: '🤖', name: 'ML Performance', color: '#14b8a6' },
  computer_vision: { icon: '🖼️', name: 'Computer Vision', color: '#0891b2' },
  data_scientist: { icon: '🔬', name: 'Data Science', color: '#7c3aed' },
  developer: { icon: '💻', name: 'Developer', color: '#64748b' },
  unknown: { icon: '📈', name: 'General Analytics', color: '#94a3b8' }
}

export default function EnhancedDashboard() {
  const { sessionId } = useSession()
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [dashboard, setDashboard] = useState(null)
  const [showModelEvaluator, setShowModelEvaluator] = useState(false)

  // Fetch enhanced dashboard
  const fetchDashboard = useCallback(async (forceRefresh = false) => {
    if (!sessionId) return
    
    setLoading(true)
    setError(null)
    
    try {
      const url = `${API_BASE}/dashboard/adaptive${forceRefresh ? '?force_refresh=true' : ''}`
      const response = await fetch(url, {
        headers: { 'X-Session-Id': sessionId }
      })
      
      if (!response.ok) {
        if (response.status === 400) {
          setDashboard(null)
          setLoading(false)
          return
        }
        const errData = await response.json().catch(() => ({}))
        throw new Error(errData.detail || 'Failed to load dashboard')
      }
      
      const data = await response.json()
      setDashboard(data)
    } catch (err) {
      console.error('Dashboard fetch error:', err)
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [sessionId])

  useEffect(() => {
    fetchDashboard()
  }, [fetchDashboard])

  // Refresh dashboard after evaluation
  const handleEvaluationComplete = useCallback(() => {
    fetchDashboard(true)
    setShowModelEvaluator(false)
  }, [fetchDashboard])

  if (loading) {
    return (
      <div className="enhanced-dashboard-loading">
        <div className="loading-spinner"></div>
        <h3>Analyzing Your Data</h3>
        <p>Detecting persona and generating personalized dashboard...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="enhanced-dashboard-error">
        <span className="error-icon">⚠️</span>
        <h2>Dashboard Error</h2>
        <p>{error}</p>
        <button className="btn-retry" onClick={() => fetchDashboard()}>
          Try Again
        </button>
      </div>
    )
  }

  if (!dashboard) {
    return (
      <div className="enhanced-dashboard-empty">
        <span className="empty-icon">📊</span>
        <h2>No Dashboard Available</h2>
        <p>Upload data to generate your personalized dashboard.</p>
      </div>
    )
  }

  const persona = dashboard.persona_detection?.persona || 'unknown'
  const personaInfo = PERSONA_INFO[persona] || PERSONA_INFO.unknown
  const colors = PERSONA_COLORS[persona] || PERSONA_COLORS.default

  return (
    <div className={`enhanced-dashboard dashboard-${persona}`}>
      {/* Header */}
      <header className="dashboard-header">
        <div className="header-left">
          <span className="persona-icon">{personaInfo.icon}</span>
          <div>
            <h1>{dashboard.title || personaInfo.name}</h1>
            <p className="dashboard-subtitle">
              {dashboard.subtitle || dashboard.persona_detection?.summary}
            </p>
          </div>
        </div>
        
        <div className="header-right">
          <div className="persona-badge" style={{ backgroundColor: personaInfo.color }}>
            {personaInfo.name}
          </div>
          
          {dashboard.has_model && (
            <button 
              className="btn-evaluate"
              onClick={() => setShowModelEvaluator(!showModelEvaluator)}
            >
              {showModelEvaluator ? 'Hide Evaluator' : '📈 Evaluate Model'}
            </button>
          )}
          
          <button className="btn-refresh" onClick={() => fetchDashboard(true)}>
            🔄 Refresh
          </button>
        </div>
      </header>

      {/* Model Evaluator */}
      {showModelEvaluator && dashboard.has_model && (
        <div className="model-evaluator-container">
          <ModelEvaluator 
            modelId={dashboard.model_info?.id}
            modelInfo={dashboard.model_info}
            onEvaluationComplete={handleEvaluationComplete}
          />
        </div>
      )}

      {/* Main Content - Render sections based on dashboard type */}
      <main className="dashboard-main">
        {dashboard.sections?.map((section, idx) => (
          <DashboardSection 
            key={section.id || idx}
            section={section}
            colors={colors}
            persona={persona}
          />
        ))}
      </main>

      {/* Recommendations */}
      {dashboard.persona_detection?.recommended_analysis && (
        <aside className="recommendations-panel">
          <h3>💡 Recommended Analysis</h3>
          <ul>
            {dashboard.persona_detection.recommended_analysis.map((rec, idx) => (
              <li key={idx}>{rec}</li>
            ))}
          </ul>
        </aside>
      )}
    </div>
  )
}

// Dashboard Section Component
function DashboardSection({ section, colors, persona }) {
  const renderSectionContent = () => {
    switch (section.type) {
      case 'kpi_grid':
      case 'stats_grid':
        return <KPIGrid data={section.stats || section.charts} colors={colors} />
      
      case 'metric_cards':
        return <MetricCards metrics={section.metrics} colors={colors} />
      
      case 'chart_grid':
        return <ChartGrid charts={section.charts} colors={colors} />
      
      case 'confusion_matrix':
        return <ConfusionMatrixChart data={section.data} colors={colors} />
      
      case 'heatmap':
        return <HeatmapChart data={section.data} colors={colors} />
      
      case 'stats_table':
      case 'metrics_table':
      case 'feature_table':
        return <DataTable data={section.data} columns={section.columns} />
      
      case 'data_table':
        return <DataTable data={section.data?.rows} columns={section.data?.columns} />
      
      case 'outlier_table':
      case 'missing_table':
        return <DataTable data={section.data} />
      
      case 'scatter':
        return <ScatterPlot data={section.data} colors={colors} />
      
      case 'residual_analysis':
        return <ResidualAnalysis stats={section.stats} histogram={section.histogram} colors={colors} />
      
      case 'prompt':
        return <ActionPrompt content={section.content} />
      
      case 'message':
        return <MessageBox content={section.content} />
      
      case 'model_info':
        return <ModelInfoCard data={section.data} />
      
      case 'readiness_score':
        return <ReadinessScore score={section.score} issues={section.issues} suggestions={section.suggestions} />
      
      case 'data_profile':
        return <DataProfile data={section.data} />
      
      case 'error_analysis':
        return <ErrorAnalysis data={section.data} />
      
      default:
        return <GenericSection section={section} colors={colors} />
    }
  }

  return (
    <section className={`dashboard-section section-${section.type}`}>
      {section.title && (
        <div className="section-header">
          <h2>{section.title}</h2>
          {section.subtitle && <p>{section.subtitle}</p>}
        </div>
      )}
      <div className="section-content">
        {renderSectionContent()}
      </div>
    </section>
  )
}

// KPI Grid Component
function KPIGrid({ data, colors }) {
  if (!data || data.length === 0) return null

  return (
    <div className="kpi-grid">
      {data.map((kpi, idx) => (
        <div key={kpi.id || idx} className="kpi-card">
          <div className="kpi-icon">{kpi.icon || '📊'}</div>
          <div className="kpi-info">
            <span className="kpi-label">{kpi.label}</span>
            <span className="kpi-value">{kpi.value}</span>
            {kpi.subtitle && <span className="kpi-subtitle">{kpi.subtitle}</span>}
          </div>
          {kpi.trend && (
            <div className={`kpi-trend trend-${kpi.trend}`}>
              {kpi.trend === 'up' ? '↑' : kpi.trend === 'down' ? '↓' : '→'}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

// Metric Cards Component (for ML dashboards)
function MetricCards({ metrics, colors }) {
  if (!metrics || metrics.length === 0) return null

  return (
    <div className="metric-cards">
      {metrics.map((metric, idx) => (
        <div key={idx} className="metric-card">
          <div className="metric-icon">{metric.icon || '📈'}</div>
          <div className="metric-content">
            <span className="metric-label">{metric.label}</span>
            <span className="metric-value" style={{ color: colors.primary }}>
              {metric.value}
            </span>
          </div>
        </div>
      ))}
    </div>
  )
}

// Chart Grid Component
function ChartGrid({ charts, colors }) {
  if (!charts || charts.length === 0) return null

  return (
    <div className="chart-grid">
      {charts.map((chart, idx) => (
        <div key={chart.id || idx} className="chart-wrapper">
          <h4>{chart.title}</h4>
          <ResponsiveChart chart={chart} colors={colors} />
        </div>
      ))}
    </div>
  )
}

// Responsive Chart Component
function ResponsiveChart({ chart, colors }) {
  const data = chart.data?.values ? 
    chart.data.labels.map((label, i) => ({ name: label, value: chart.data.values[i] })) :
    chart.data

  switch (chart.type) {
    case 'bar':
      return (
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill={colors.primary} />
          </BarChart>
        </ResponsiveContainer>
      )
    
    case 'line':
      return (
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="value" stroke={colors.primary} />
          </LineChart>
        </ResponsiveContainer>
      )
    
    case 'pie':
      return (
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              labelLine={false}
              outerRadius={100}
              fill={colors.primary}
              dataKey="value"
              label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={colors.chart[index % colors.chart.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      )
    
    case 'histogram':
      const histData = chart.data?.counts ? 
        chart.data.counts.map((count, i) => ({
          bin: `${chart.data.bins[i].toFixed(1)}-${chart.data.bins[i+1]?.toFixed(1) || ''}`,
          count
        })) :
        chart.data?.values?.slice(0, 30).map((v, i) => ({ bin: i, count: v })) || []
      
      return (
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={histData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="bin" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill={colors.secondary} />
          </BarChart>
        </ResponsiveContainer>
      )
    
    default:
      return (
        <div className="chart-placeholder">
          <p>Chart type: {chart.type}</p>
        </div>
      )
  }
}

// Confusion Matrix Component
function ConfusionMatrixChart({ data, colors }) {
  if (!data?.matrix) return <p>No confusion matrix data</p>

  const { matrix, labels } = data

  return (
    <div className="confusion-matrix">
      <table>
        <thead>
          <tr>
            <th></th>
            {labels.map((label, i) => <th key={i}>Pred: {label}</th>)}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th>Actual: {labels[i]}</th>
              {row.map((cell, j) => {
                const maxVal = Math.max(...matrix.flat())
                const intensity = cell / maxVal
                const bgColor = i === j ? 
                  `rgba(34, 197, 94, ${0.2 + intensity * 0.6})` : 
                  `rgba(239, 68, 68, ${intensity * 0.6})`
                return (
                  <td key={j} style={{ backgroundColor: bgColor }}>
                    {cell}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// Heatmap Chart Component
function HeatmapChart({ data, colors }) {
  if (!data?.matrix) return <p>No heatmap data</p>

  const { matrix, columns } = data

  return (
    <div className="heatmap-container">
      <table className="heatmap-table">
        <thead>
          <tr>
            <th></th>
            {columns.map((col, i) => <th key={i}>{col}</th>)}
          </tr>
        </thead>
        <tbody>
          {columns.map((rowLabel, i) => (
            <tr key={i}>
              <th>{rowLabel}</th>
              {columns.map((_, j) => {
                const value = matrix[rowLabel]?.[columns[j]] || 0
                const intensity = Math.abs(value)
                const bgColor = value >= 0 ? 
                  `rgba(59, 130, 246, ${intensity})` : 
                  `rgba(239, 68, 68, ${intensity})`
                return (
                  <td key={j} style={{ backgroundColor: bgColor }}>
                    {value.toFixed(2)}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// Data Table Component
function DataTable({ data, columns }) {
  if (!data || data.length === 0) return <p>No data available</p>

  const cols = columns || Object.keys(data[0] || {})

  return (
    <div className="data-table-container">
      <table className="data-table">
        <thead>
          <tr>
            {cols.map((col, i) => <th key={i}>{col}</th>)}
          </tr>
        </thead>
        <tbody>
          {data.slice(0, 50).map((row, i) => (
            <tr key={i}>
              {cols.map((col, j) => (
                <td key={j}>
                  {typeof row[col] === 'number' ? 
                    (Number.isInteger(row[col]) ? row[col] : row[col].toFixed(4)) : 
                    row[col]?.toString() || '-'}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {data.length > 50 && <p className="table-footer">Showing 50 of {data.length} rows</p>}
    </div>
  )
}

// Scatter Plot Component
function ScatterPlot({ data, colors }) {
  if (!data || data.length === 0) return <p>No scatter data</p>

  return (
    <ResponsiveContainer width="100%" height={400}>
      <ScatterChart>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" dataKey="actual" name="Actual" />
        <YAxis type="number" dataKey="predicted" name="Predicted" />
        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
        <Scatter name="Predictions" data={data} fill={colors.primary} />
        {/* Perfect prediction line */}
        <Line 
          type="linear" 
          dataKey="actual" 
          stroke="#888" 
          strokeDasharray="5 5"
        />
      </ScatterChart>
    </ResponsiveContainer>
  )
}

// Residual Analysis Component
function ResidualAnalysis({ stats, histogram, colors }) {
  const histData = histogram?.counts?.map((count, i) => ({
    bin: `${histogram.bin_edges[i].toFixed(2)}`,
    count
  })) || []

  return (
    <div className="residual-analysis">
      <div className="residual-stats">
        <div className="stat-item">
          <span className="stat-label">Mean</span>
          <span className="stat-value">{stats?.mean?.toFixed(4)}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Std Dev</span>
          <span className="stat-value">{stats?.std?.toFixed(4)}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Skewness</span>
          <span className="stat-value">{stats?.skewness?.toFixed(4)}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Kurtosis</span>
          <span className="stat-value">{stats?.kurtosis?.toFixed(4)}</span>
        </div>
      </div>
      
      {histData.length > 0 && (
        <div className="residual-histogram">
          <h4>Residual Distribution</h4>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={histData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="bin" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill={colors.secondary} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}

// Action Prompt Component
function ActionPrompt({ content }) {
  return (
    <div className="action-prompt">
      <p className="prompt-message">{content?.message}</p>
      <div className="prompt-actions">
        {content?.actions?.map((action, idx) => (
          <button key={idx} className="btn-action">
            {action.label}
          </button>
        )) || (
          <button className="btn-action">{content?.action || 'Take Action'}</button>
        )}
      </div>
    </div>
  )
}

// Message Box Component
function MessageBox({ content }) {
  return (
    <div className="message-box">
      <p>{typeof content === 'string' ? content : content?.message || 'No content'}</p>
    </div>
  )
}

// Model Info Card Component
function ModelInfoCard({ data }) {
  return (
    <div className="model-info-card">
      <div className="info-item">
        <span className="info-label">File</span>
        <span className="info-value">{data?.filename || 'Unknown'}</span>
      </div>
      <div className="info-item">
        <span className="info-label">Type</span>
        <span className="info-value">{data?.model_type || 'Unknown'}</span>
      </div>
      <div className="info-item">
        <span className="info-label">Framework</span>
        <span className="info-value">{data?.framework || 'Unknown'}</span>
      </div>
      {data?.task && (
        <div className="info-item">
          <span className="info-label">Task</span>
          <span className="info-value">{data.task}</span>
        </div>
      )}
      {data?.parameters && (
        <div className="info-item">
          <span className="info-label">Parameters</span>
          <span className="info-value">{data.parameters?.toLocaleString()}</span>
        </div>
      )}
    </div>
  )
}

// Readiness Score Component
function ReadinessScore({ score, issues, suggestions }) {
  const getScoreColor = (s) => {
    if (s >= 80) return '#22c55e'
    if (s >= 60) return '#f59e0b'
    return '#ef4444'
  }

  return (
    <div className="readiness-score">
      <div className="score-display">
        <div 
          className="score-circle"
          style={{ 
            background: `conic-gradient(${getScoreColor(score)} ${score * 3.6}deg, #e5e7eb 0deg)` 
          }}
        >
          <span className="score-value">{score}</span>
        </div>
        <span className="score-label">ML Readiness</span>
      </div>
      
      {issues?.length > 0 && (
        <div className="score-issues">
          <h4>⚠️ Issues</h4>
          <ul>
            {issues.map((issue, idx) => <li key={idx}>{issue}</li>)}
          </ul>
        </div>
      )}
      
      {suggestions?.length > 0 && (
        <div className="score-suggestions">
          <h4>💡 Suggestions</h4>
          <ul>
            {suggestions.map((sug, idx) => <li key={idx}>{sug}</li>)}
          </ul>
        </div>
      )}
    </div>
  )
}

// Data Profile Component
function DataProfile({ data }) {
  return (
    <div className="data-profile">
      <div className="profile-grid">
        <div className="profile-item">
          <span className="profile-label">Rows</span>
          <span className="profile-value">{data?.shape?.rows?.toLocaleString()}</span>
        </div>
        <div className="profile-item">
          <span className="profile-label">Columns</span>
          <span className="profile-value">{data?.shape?.columns}</span>
        </div>
        <div className="profile-item">
          <span className="profile-label">Missing</span>
          <span className="profile-value">
            {Object.values(data?.missing || {}).reduce((a, b) => a + b, 0)}
          </span>
        </div>
      </div>
    </div>
  )
}

// Error Analysis Component
function ErrorAnalysis({ data }) {
  return (
    <div className="error-analysis">
      <div className="error-summary">
        <div className="error-stat">
          <span className="stat-label">Total Errors</span>
          <span className="stat-value error">{data?.total_errors}</span>
        </div>
        <div className="error-stat">
          <span className="stat-label">Error Rate</span>
          <span className="stat-value">{data?.error_rate?.toFixed(2)}%</span>
        </div>
      </div>
      
      {data?.confused_pairs?.length > 0 && (
        <div className="confused-pairs">
          <h4>Most Confused Pairs</h4>
          <table>
            <thead>
              <tr>
                <th>Class A</th>
                <th>Class B</th>
                <th>A→B</th>
                <th>B→A</th>
                <th>Total</th>
              </tr>
            </thead>
            <tbody>
              {data.confused_pairs.map((pair, idx) => (
                <tr key={idx}>
                  <td>{pair.class_a}</td>
                  <td>{pair.class_b}</td>
                  <td>{pair.a_as_b}</td>
                  <td>{pair.b_as_a}</td>
                  <td>{pair.total}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

// Generic Section Fallback
function GenericSection({ section, colors }) {
  return (
    <div className="generic-section">
      <p>Section type: {section.type}</p>
      <pre>{JSON.stringify(section, null, 2)}</pre>
    </div>
  )
}
