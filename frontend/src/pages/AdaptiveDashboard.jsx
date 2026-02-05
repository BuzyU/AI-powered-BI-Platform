/**
 * Adaptive Dashboard - Completely transforms based on user persona
 * 
 * Business User → Revenue, Customers, Products KPIs (Power BI style)
 * Analytics User → EDA, Distributions, Correlations (Jupyter style)
 * ML Engineer → Model Metrics, Confusion Matrix (MLflow style)
 */

import React, { useState, useEffect, useCallback } from 'react'
import {
  BarChart, Bar, LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell, ScatterChart, Scatter, RadarChart, Radar,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis, Treemap
} from 'recharts'
import { useSession } from '../contexts/SessionContext'
import ModelEvaluator from '../components/ModelEvaluator'
import './AdaptiveDashboard.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

// Color palettes for different personas
const COLORS = {
  business: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'],
  analytics: ['#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899', '#f43f5e'],
  ml: ['#14b8a6', '#22c55e', '#84cc16', '#eab308', '#f97316', '#ef4444'],
  default: ['#64748b', '#94a3b8', '#cbd5e1', '#e2e8f0', '#f1f5f9', '#f8fafc']
}

// Icons for different personas
const PersonaIcons = {
  BUSINESS: '💼',
  ANALYTICS: '📊',
  ML_ENGINEER: '🤖',
  DATA_SCIENTIST: '🔬',
  DEVELOPER: '💻',
  GENERAL: '📈'
}

const PersonaDescriptions = {
  BUSINESS: 'Business Intelligence Dashboard - Revenue, Customers & Performance Metrics',
  ANALYTICS: 'Exploratory Data Analysis Dashboard - Statistical Distributions & Correlations',
  ML_ENGINEER: 'Model Performance Dashboard - Accuracy, Confusion Matrix & Feature Importance',
  DATA_SCIENTIST: 'Data Science Dashboard - Combined Analytics & Model Evaluation',
  DEVELOPER: 'Developer Dashboard - Data Structure & API Insights',
  GENERAL: 'General Analytics Dashboard'
}

export default function AdaptiveDashboard() {
  const { sessionId } = useSession()
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [dashboard, setDashboard] = useState(null)
  const [activeTab, setActiveTab] = useState(null)
  const [tabData, setTabData] = useState({})
  const [loadingTab, setLoadingTab] = useState(false)
  const [filters, setFilters] = useState({})

  // Fetch dashboard configuration
  const fetchDashboard = useCallback(async () => {
    if (!sessionId) return
    
    setLoading(true)
    setError(null)
    
    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 30000) // 30 second timeout
      
      const response = await fetch(`${API_BASE}/session-dashboard`, {
        headers: { 'X-Session-Id': sessionId },
        signal: controller.signal
      })
      
      clearTimeout(timeoutId)
      
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        // Handle no data case gracefully
        if (response.status === 400) {
          setLoading(false)
          setDashboard(null)
          return
        }
        throw new Error(errData.detail || 'Failed to load dashboard')
      }
      
      const data = await response.json()
      setDashboard(data)
      
      // Set first tab as active
      if (data.tabs && data.tabs.length > 0) {
        setActiveTab(data.tabs[0].id)
      }
    } catch (err) {
      console.error('Dashboard fetch error:', err)
      if (err.name === 'AbortError') {
        setError('Dashboard loading timed out. The server may be slow. Please try again.')
      } else {
        setError(err.message)
      }
    } finally {
      setLoading(false)
    }
  }, [sessionId])

  // Fetch data for active tab
  const fetchTabData = useCallback(async (tabId) => {
    if (!sessionId || !tabId || tabData[tabId]) return
    
    setLoadingTab(true)
    
    try {
      const response = await fetch(`${API_BASE}/session-dashboard/tabs/${tabId}`, {
        headers: { 'X-Session-Id': sessionId }
      })
      
      if (!response.ok) throw new Error('Failed to load tab data')
      
      const data = await response.json()
      setTabData(prev => ({ ...prev, [tabId]: data }))
    } catch (err) {
      console.error('Tab data fetch error:', err)
    } finally {
      setLoadingTab(false)
    }
  }, [sessionId, tabData])

  useEffect(() => {
    fetchDashboard()
  }, [fetchDashboard])

  useEffect(() => {
    if (activeTab) {
      fetchTabData(activeTab)
    }
  }, [activeTab, fetchTabData])

  // Regenerate dashboard
  const handleRegenerate = async () => {
    setLoading(true)
    setTabData({})
    
    try {
      const response = await fetch(`${API_BASE}/session-dashboard/regenerate`, {
        method: 'POST',
        headers: { 'X-Session-Id': sessionId }
      })
      
      if (!response.ok) throw new Error('Failed to regenerate')
      
      const data = await response.json()
      setDashboard(data)
      
      if (data.tabs && data.tabs.length > 0) {
        setActiveTab(data.tabs[0].id)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="dashboard-loading">
        <div className="spinner"></div>
        <p>Analyzing your data and building personalized dashboard...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="dashboard-error">
        <h2>⚠️ Dashboard Error</h2>
        <p>{error}</p>
        <button className="btn-primary" onClick={fetchDashboard}>Retry</button>
      </div>
    )
  }

  if (!dashboard) {
    return (
      <div className="dashboard-empty">
        <h2>📊 No Dashboard Available</h2>
        <p>Upload data first to generate your personalized dashboard.</p>
      </div>
    )
  }

  const persona = dashboard.persona_detection || {}
  const personaType = persona.persona || 'GENERAL'
  const colorPalette = COLORS[personaType.toLowerCase()] || COLORS.default

  return (
    <div className="adaptive-dashboard animate-fade-in">
      {/* Persona Header */}
      <header className="persona-header">
        <div className="persona-info">
          <span className="persona-icon">{PersonaIcons[personaType]}</span>
          <div>
            <h1>{dashboard.title || 'Dashboard'}</h1>
            <p className="persona-description">
              {PersonaDescriptions[personaType]}
            </p>
          </div>
        </div>
        <div className="persona-badge">
          <span className={`badge badge-${personaType.toLowerCase()}`}>
            {personaType.replace('_', ' ')}
          </span>
          <span className="confidence">
            {Math.round((persona.confidence || 0) * 100)}% confidence
          </span>
        </div>
        <button className="btn-refresh" onClick={handleRegenerate} title="Regenerate Dashboard">
          🔄
        </button>
      </header>

      {/* Recommendations */}
      {persona.recommended_analysis && persona.recommended_analysis.length > 0 && (
        <div className="recommendations-bar">
          <span className="rec-label">💡 Recommended:</span>
          {persona.recommended_analysis.slice(0, 4).map((rec, i) => (
            <span key={i} className="rec-item">{rec}</span>
          ))}
        </div>
      )}

      {/* Model Evaluator - Always shown for ML personas or when models are present */}
      {/* Also add a toggle button to show it on demand */}
      <ModelEvaluatorSection 
        personaType={personaType}
        dashboard={dashboard}
        sessionId={sessionId}
        onEvaluationComplete={() => {
          // Clear tab data and refresh dashboard after evaluation
          setTabData({})
          fetchDashboard()
        }}
      />

      {/* Tab Navigation */}
      <nav className="dashboard-tabs">
        {dashboard.tabs?.map(tab => (
          <button
            key={tab.id}
            className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            <span className="tab-icon">{tab.icon}</span>
            <span className="tab-title">{tab.title}</span>
          </button>
        ))}
      </nav>

      {/* Tab Content */}
      <div className="tab-content">
        {loadingTab ? (
          <div className="tab-loading">
            <div className="spinner small"></div>
            <span>Loading charts...</span>
          </div>
        ) : tabData[activeTab] ? (
          <TabContent 
            tab={tabData[activeTab]} 
            colors={colorPalette}
            personaType={personaType}
            filters={filters}
            onFilterChange={setFilters}
          />
        ) : (
          <div className="tab-placeholder">
            <p>Select a tab to view data</p>
          </div>
        )}
      </div>
    </div>
  )
}

// Model Evaluator Section with toggle
function ModelEvaluatorSection({ personaType, dashboard, sessionId, onEvaluationComplete }) {
  const [showEvaluator, setShowEvaluator] = useState(false)
  
  // Auto-show for ML personas or when model is present
  const autoShow = personaType === 'ML_ENGINEER' || dashboard.has_model
  
  // If we have evaluation results, show a summary badge
  const hasEvaluation = dashboard.has_evaluation
  
  return (
    <div className="model-evaluator-section">
      {!autoShow && !showEvaluator && (
        <button 
          className="btn-evaluate-toggle"
          onClick={() => setShowEvaluator(true)}
        >
          <span>🧪</span>
          Evaluate Model Predictions
          {hasEvaluation && <span className="eval-badge">✓</span>}
        </button>
      )}
      
      {(autoShow || showEvaluator) && (
        <div className="evaluator-wrapper">
          {!autoShow && (
            <button 
              className="btn-close-evaluator"
              onClick={() => setShowEvaluator(false)}
              title="Close evaluator"
            >
              ✕
            </button>
          )}
          <ModelEvaluator 
            modelId={dashboard.model_id}
            modelInfo={dashboard.model_info}
            onEvaluationComplete={onEvaluationComplete}
          />
        </div>
      )}
    </div>
  )
}

// Tab Content Component
function TabContent({ tab, colors, personaType, filters, onFilterChange }) {
  if (!tab || !tab.charts) return null

  // Group charts by layout hints
  const kpiCharts = tab.charts.filter(c => c.type === 'kpi')
  const regularCharts = tab.charts.filter(c => c.type !== 'kpi')

  return (
    <div className="tab-content-inner animate-slide-up">
      {tab.description && (
        <p className="tab-description">{tab.description}</p>
      )}

      {/* KPI Row */}
      {kpiCharts.length > 0 && (
        <div className="kpi-row">
          {kpiCharts.map(chart => (
            <KPICard 
              key={chart.id} 
              chart={chart} 
              colors={colors}
              personaType={personaType}
            />
          ))}
        </div>
      )}

      {/* Charts Grid */}
      <div className={`charts-grid persona-${personaType.toLowerCase()}`}>
        {regularCharts.map(chart => (
          <ChartCard 
            key={chart.id} 
            chart={chart} 
            colors={colors}
            personaType={personaType}
          />
        ))}
      </div>
    </div>
  )
}

// KPI Card Component
function KPICard({ chart, colors, personaType }) {
  const data = chart.chart_data || {}
  const config = chart.config || {}
  
  const iconMap = {
    'currency': '💰',
    'users': '👥',
    'chart': '📈',
    'box': '📦',
    'percent': '📊',
    'hash': '#',
    'accuracy': '🎯',
    'error': '⚠️',
    'cpu': '🤖',
    'database': '🗄️',
    'help-circle': '❓'
  }
  
  const icon = iconMap[config.icon] || iconMap[data.icon] || '📊'
  
  return (
    <div className={`kpi-card persona-${personaType.toLowerCase()}`}>
      <div className="kpi-icon">{icon}</div>
      <div className="kpi-content">
        <div className="kpi-value">{data.formatted_value || data.value || 'N/A'}</div>
        <div className="kpi-label">{chart.title}</div>
        {data.subtitle && <div className="kpi-subtitle">{data.subtitle}</div>}
      </div>
    </div>
  )
}

// Chart Card Component
function ChartCard({ chart, colors, personaType }) {
  const data = chart.chart_data || {}
  const chartType = data.type || chart.type
  
  return (
    <div className={`chart-card persona-${personaType.toLowerCase()}`}>
      <div className="chart-header">
        <h3>{chart.title}</h3>
        {chart.description && <p>{chart.description}</p>}
      </div>
      <div className="chart-body">
        <RenderChart 
          type={chartType} 
          data={data} 
          colors={colors} 
          personaType={personaType}
        />
      </div>
    </div>
  )
}

// Dynamic Chart Renderer
function RenderChart({ type, data, colors, personaType }) {
  if (!data || data.error) {
    return (
      <div className="chart-error">
        <span>⚠️</span>
        <p>{data?.error || 'No data available'}</p>
      </div>
    )
  }

  // Handle message-only responses (model-only sessions)
  if (data.message && (!data.data || data.data.length === 0)) {
    return (
      <div className="chart-placeholder">
        <span className="placeholder-icon">📊</span>
        <p>{data.message}</p>
      </div>
    )
  }

  const chartData = data.data || []
  
  // Handle empty data
  if (chartData.length === 0 && type !== 'gauge') {
    return (
      <div className="chart-placeholder">
        <span className="placeholder-icon">📊</span>
        <p>{data.message || 'No data to display'}</p>
      </div>
    )
  }
  
  switch (type) {
    case 'bar':
      return <BarChartComponent data={chartData} colors={colors} xLabel={data.x_label} yLabel={data.y_label} />
    
    case 'line':
      return <LineChartComponent data={chartData} colors={colors} xLabel={data.x_label} yLabel={data.y_label} />
    
    case 'area':
      return <AreaChartComponent data={chartData} colors={colors} xLabel={data.x_label} yLabel={data.y_label} />
    
    case 'pie':
    case 'donut':
      return <PieChartComponent data={chartData} colors={colors} isDonut={type === 'donut'} />
    
    case 'scatter':
      return <ScatterChartComponent data={chartData} colors={colors} xLabel={data.x_label} yLabel={data.y_label} />
    
    case 'histogram':
      return <HistogramComponent data={chartData} colors={colors} xLabel={data.x_label} />
    
    case 'boxplot':
      return <BoxPlotComponent data={data.data} colors={colors} column={data.column} />
    
    case 'heatmap':
      return <HeatmapComponent data={chartData} colors={colors} columns={data.columns} />
    
    case 'gauge':
      return <GaugeComponent value={data.value} max={data.max} thresholds={data.thresholds} colors={colors} message={data.message} />
    
    case 'table':
      return <TableComponent data={chartData} columns={data.columns} totalRows={data.total_rows} />
    
    default:
      return <BarChartComponent data={chartData} colors={colors} xLabel={data.x_label} yLabel={data.y_label} />
  }
}

// Individual Chart Components
// Custom tick component for truncated labels
const CustomXAxisTick = ({ x, y, payload }) => {
  const label = String(payload.value)
  const truncated = label.length > 12 ? label.substring(0, 10) + '...' : label
  return (
    <g transform={`translate(${x},${y})`}>
      <text
        x={0}
        y={0}
        dy={12}
        textAnchor="end"
        fill="#94a3b8"
        fontSize={11}
        transform="rotate(-35)"
      >
        {truncated}
      </text>
    </g>
  )
}

function BarChartComponent({ data, colors, xLabel, yLabel }) {
  if (!data || data.length === 0) return <NoData />
  
  // Limit to top 15 items for readability
  const displayData = data.slice(0, 15)
  
  return (
    <ResponsiveContainer width="100%" height={320}>
      <BarChart data={displayData} margin={{ top: 20, right: 20, left: 20, bottom: 70 }}>
        <defs>
          <linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={colors[0]} stopOpacity={1}/>
            <stop offset="100%" stopColor={colors[0]} stopOpacity={0.7}/>
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
        <XAxis 
          dataKey="label" 
          tick={<CustomXAxisTick />}
          interval={0}
          axisLine={{ stroke: '#475569' }}
          tickLine={{ stroke: '#475569' }}
        />
        <YAxis 
          tick={{ fill: '#94a3b8', fontSize: 11 }}
          axisLine={{ stroke: '#475569' }}
          tickLine={{ stroke: '#475569' }}
          tickFormatter={(value) => value >= 1000 ? `${(value/1000).toFixed(1)}k` : value}
        />
        <Tooltip 
          contentStyle={{ 
            background: 'rgba(30, 41, 59, 0.95)', 
            border: '1px solid #475569', 
            borderRadius: '8px',
            boxShadow: '0 10px 25px rgba(0,0,0,0.3)'
          }}
          labelStyle={{ color: '#f8fafc', fontWeight: 600, marginBottom: '4px' }}
          itemStyle={{ color: '#94a3b8' }}
          formatter={(value) => [typeof value === 'number' ? value.toLocaleString() : value, 'Value']}
        />
        <Bar 
          dataKey="value" 
          fill="url(#barGradient)" 
          radius={[6, 6, 0, 0]}
          animationDuration={800}
        />
      </BarChart>
    </ResponsiveContainer>
  )
}

function LineChartComponent({ data, colors, xLabel, yLabel }) {
  if (!data || data.length === 0) return <NoData />
  
  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={data} margin={{ top: 20, right: 20, left: 20, bottom: 70 }}>
        <defs>
          <linearGradient id="lineGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={colors[0]} stopOpacity={0.3}/>
            <stop offset="100%" stopColor={colors[0]} stopOpacity={0}/>
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
        <XAxis 
          dataKey="label" 
          tick={<CustomXAxisTick />}
          interval={Math.max(0, Math.floor(data.length / 10))}
          axisLine={{ stroke: '#475569' }}
        />
        <YAxis 
          tick={{ fill: '#94a3b8', fontSize: 11 }}
          axisLine={{ stroke: '#475569' }}
          tickFormatter={(value) => value >= 1000 ? `${(value/1000).toFixed(1)}k` : value}
        />
        <Tooltip 
          contentStyle={{ 
            background: 'rgba(30, 41, 59, 0.95)', 
            border: '1px solid #475569', 
            borderRadius: '8px',
            boxShadow: '0 10px 25px rgba(0,0,0,0.3)'
          }}
          labelStyle={{ color: '#f8fafc', fontWeight: 600 }}
        />
        <Area 
          type="monotone" 
          dataKey="value" 
          stroke="none"
          fill="url(#lineGradient)" 
        />
        <Line 
          type="monotone" 
          dataKey="value" 
          stroke={colors[0]} 
          strokeWidth={2.5} 
          dot={{ fill: colors[0], strokeWidth: 0, r: 3 }}
          activeDot={{ r: 6, fill: colors[0], stroke: '#fff', strokeWidth: 2 }}
          animationDuration={800}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}

function AreaChartComponent({ data, colors, xLabel, yLabel }) {
  if (!data || data.length === 0) return <NoData />
  
  return (
    <ResponsiveContainer width="100%" height={300}>
      <AreaChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
        <defs>
          <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={colors[0]} stopOpacity={0.8}/>
            <stop offset="95%" stopColor={colors[0]} stopOpacity={0.1}/>
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis 
          dataKey="label" 
          tick={{ fill: '#94a3b8', fontSize: 12 }}
          angle={-45}
          textAnchor="end"
        />
        <YAxis tick={{ fill: '#94a3b8' }} />
        <Tooltip 
          contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
        />
        <Area type="monotone" dataKey="value" stroke={colors[0]} fill="url(#areaGradient)" />
      </AreaChart>
    </ResponsiveContainer>
  )
}

function PieChartComponent({ data, colors, isDonut }) {
  if (!data || data.length === 0) return <NoData />
  
  // Custom label renderer for better display
  const renderCustomLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent, label }) => {
    if (percent < 0.05) return null // Don't show labels for tiny slices
    const RADIAN = Math.PI / 180
    const radius = innerRadius + (outerRadius - innerRadius) * 1.3
    const x = cx + radius * Math.cos(-midAngle * RADIAN)
    const y = cy + radius * Math.sin(-midAngle * RADIAN)
    const truncatedLabel = label?.length > 10 ? label.substring(0, 8) + '...' : label

    return (
      <text 
        x={x} 
        y={y} 
        fill="#94a3b8" 
        textAnchor={x > cx ? 'start' : 'end'} 
        dominantBaseline="central"
        fontSize={11}
      >
        {truncatedLabel}: {(percent * 100).toFixed(0)}%
      </text>
    )
  }
  
  return (
    <ResponsiveContainer width="100%" height={320}>
      <PieChart>
        <Pie
          data={data}
          dataKey="value"
          nameKey="label"
          cx="50%"
          cy="50%"
          outerRadius={110}
          innerRadius={isDonut ? 60 : 0}
          label={renderCustomLabel}
          labelLine={{ stroke: '#475569', strokeWidth: 1 }}
          animationDuration={800}
        >
          {data.map((_, index) => (
            <Cell 
              key={index} 
              fill={colors[index % colors.length]}
              stroke="#1e293b"
              strokeWidth={2}
            />
          ))}
        </Pie>
        <Tooltip 
          contentStyle={{ 
            background: 'rgba(30, 41, 59, 0.95)', 
            border: '1px solid #475569', 
            borderRadius: '8px',
            boxShadow: '0 10px 25px rgba(0,0,0,0.3)'
          }}
          formatter={(value, name) => [value.toLocaleString(), name]}
        />
        <Legend 
          wrapperStyle={{ fontSize: '11px', color: '#94a3b8' }}
          iconType="circle"
          formatter={(value) => <span style={{ color: '#94a3b8' }}>{value?.length > 15 ? value.substring(0, 12) + '...' : value}</span>}
        />
      </PieChart>
    </ResponsiveContainer>
  )
}

function ScatterChartComponent({ data, colors, xLabel, yLabel }) {
  if (!data || data.length === 0) return <NoData />
  
  return (
    <ResponsiveContainer width="100%" height={300}>
      <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis 
          dataKey="x" 
          name={xLabel} 
          tick={{ fill: '#94a3b8' }}
          label={{ value: xLabel, position: 'bottom', fill: '#94a3b8' }}
        />
        <YAxis 
          dataKey="y" 
          name={yLabel} 
          tick={{ fill: '#94a3b8' }}
          label={{ value: yLabel, angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
        />
        <Tooltip 
          cursor={{ strokeDasharray: '3 3' }}
          contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
        />
        <Scatter data={data} fill={colors[0]} />
      </ScatterChart>
    </ResponsiveContainer>
  )
}

function HistogramComponent({ data, colors, xLabel }) {
  if (!data || data.length === 0) return <NoData />
  
  // Format bin labels to be shorter
  const formattedData = data.map(item => ({
    ...item,
    shortBin: String(item.bin).length > 10 ? String(item.bin).substring(0, 8) + '...' : item.bin
  }))
  
  return (
    <ResponsiveContainer width="100%" height={320}>
      <BarChart data={formattedData} margin={{ top: 20, right: 20, left: 20, bottom: 70 }}>
        <defs>
          <linearGradient id="histGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={colors[0]} stopOpacity={0.9}/>
            <stop offset="100%" stopColor={colors[0]} stopOpacity={0.6}/>
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
        <XAxis 
          dataKey="shortBin" 
          tick={{ fill: '#94a3b8', fontSize: 10 }}
          angle={-35}
          textAnchor="end"
          interval={0}
          axisLine={{ stroke: '#475569' }}
        />
        <YAxis 
          tick={{ fill: '#94a3b8', fontSize: 11 }}
          axisLine={{ stroke: '#475569' }}
          tickFormatter={(value) => value >= 1000 ? `${(value/1000).toFixed(1)}k` : value}
          label={{ value: 'Frequency', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }}
        />
        <Tooltip 
          contentStyle={{ 
            background: 'rgba(30, 41, 59, 0.95)', 
            border: '1px solid #475569', 
            borderRadius: '8px',
            boxShadow: '0 10px 25px rgba(0,0,0,0.3)'
          }}
          labelStyle={{ color: '#f8fafc', fontWeight: 600 }}
          formatter={(value, name) => [value.toLocaleString(), 'Count']}
          labelFormatter={(label) => `Bin: ${label}`}
        />
        <Bar 
          dataKey="count" 
          fill="url(#histGradient)"
          radius={[4, 4, 0, 0]}
          animationDuration={600}
        />
      </BarChart>
    </ResponsiveContainer>
  )
}

function BoxPlotComponent({ data, colors, column }) {
  if (!data || !data.median) {
    return <NoData />
  }
  
  // Simplified box plot visualization
  return (
    <div className="boxplot-container">
      <div className="boxplot-visual">
        <div className="boxplot-whisker" style={{ left: '10%', width: '80%' }}>
          <div className="whisker-line"></div>
          <div className="box" style={{
            left: `${((data.q1 - data.min) / (data.max - data.min)) * 100}%`,
            width: `${((data.q3 - data.q1) / (data.max - data.min)) * 100}%`
          }}>
            <div className="median" style={{
              left: `${((data.median - data.q1) / (data.q3 - data.q1)) * 100}%`
            }}></div>
          </div>
        </div>
      </div>
      <div className="boxplot-stats">
        <div><span>Min:</span> {data.min?.toFixed(2)}</div>
        <div><span>Q1:</span> {data.q1?.toFixed(2)}</div>
        <div><span>Median:</span> {data.median?.toFixed(2)}</div>
        <div><span>Q3:</span> {data.q3?.toFixed(2)}</div>
        <div><span>Max:</span> {data.max?.toFixed(2)}</div>
      </div>
    </div>
  )
}

function HeatmapComponent({ data, colors, columns }) {
  if (!data || data.length === 0) return <NoData />
  
  // Get unique x and y values
  const xValues = [...new Set(data.map(d => d.x))]
  const yValues = [...new Set(data.map(d => d.y))]
  
  // Create value lookup
  const valueLookup = {}
  data.forEach(d => {
    valueLookup[`${d.y}-${d.x}`] = d.value
  })
  
  // Color scale
  const getColor = (value) => {
    if (value === undefined) return '#1e293b'
    // For correlation: -1 to 1
    if (value >= -1 && value <= 1) {
      if (value < 0) return `rgba(239, 68, 68, ${Math.abs(value)})`
      return `rgba(34, 197, 94, ${value})`
    }
    // For confusion matrix or counts
    const max = Math.max(...data.map(d => d.value))
    const intensity = max > 0 ? value / max : 0
    return `rgba(59, 130, 246, ${0.2 + intensity * 0.8})`
  }
  
  return (
    <div className="heatmap-container">
      <div className="heatmap-grid" style={{
        gridTemplateColumns: `80px repeat(${xValues.length}, 1fr)`,
        gridTemplateRows: `repeat(${yValues.length + 1}, minmax(30px, 1fr))`
      }}>
        {/* Header row */}
        <div className="heatmap-cell header corner"></div>
        {xValues.map(x => (
          <div key={x} className="heatmap-cell header">{x}</div>
        ))}
        
        {/* Data rows */}
        {yValues.map(y => (
          <React.Fragment key={y}>
            <div className="heatmap-cell row-label">{y}</div>
            {xValues.map(x => {
              const val = valueLookup[`${y}-${x}`]
              return (
                <div 
                  key={`${y}-${x}`} 
                  className="heatmap-cell data"
                  style={{ backgroundColor: getColor(val) }}
                  title={`${y} × ${x}: ${val?.toFixed(2) ?? 'N/A'}`}
                >
                  {val !== undefined ? val.toFixed(2) : ''}
                </div>
              )
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  )
}

function GaugeComponent({ value, max, thresholds, colors, message }) {
  // Handle no value case
  if (value === null || value === undefined || value === 0) {
    return (
      <div className="gauge-container">
        <div className="gauge-placeholder">
          <span className="gauge-icon">🎯</span>
          <p>{message || 'Upload evaluation data to see accuracy'}</p>
        </div>
      </div>
    )
  }
  
  const percentage = Math.min((value / (max || 100)) * 100, 100)
  
  // Determine color based on thresholds
  let gaugeColor = colors[2] // Default yellow
  if (thresholds) {
    if (value >= thresholds[2]) gaugeColor = '#22c55e' // Green
    else if (value >= thresholds[1]) gaugeColor = '#eab308' // Yellow
    else if (value >= thresholds[0]) gaugeColor = '#f97316' // Orange
    else gaugeColor = '#ef4444' // Red
  }
  
  return (
    <div className="gauge-container">
      <div className="gauge">
        <div 
          className="gauge-fill" 
          style={{ 
            background: `conic-gradient(${gaugeColor} ${percentage * 3.6}deg, #334155 0deg)` 
          }}
        />
        <div className="gauge-center">
          <span className="gauge-value">{value?.toFixed(1)}%</span>
        </div>
      </div>
    </div>
  )
}

function TableComponent({ data, columns, totalRows }) {
  if (!data || data.length === 0) return <NoData />
  
  const displayCols = columns || Object.keys(data[0] || {})
  
  return (
    <div className="table-container">
      <table className="data-table">
        <thead>
          <tr>
            {displayCols.map(col => (
              <th key={col}>{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.slice(0, 50).map((row, i) => (
            <tr key={i}>
              {displayCols.map(col => (
                <td key={col}>{formatCell(row[col])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {totalRows > 50 && (
        <div className="table-footer">
          Showing 50 of {totalRows} rows
        </div>
      )}
    </div>
  )
}

function NoData() {
  return (
    <div className="no-data">
      <span>📭</span>
      <p>No data available for this chart</p>
    </div>
  )
}

// Helper function
function formatCell(value) {
  if (value === null || value === undefined) return '-'
  if (typeof value === 'number') {
    return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(2)
  }
  return String(value)
}
