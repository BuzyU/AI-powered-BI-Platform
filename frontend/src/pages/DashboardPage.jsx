import { useState, useEffect } from 'react'
import {
    BarChart, Bar, LineChart, Line, AreaChart, Area,
    XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
    PieChart, Pie, Cell, ScatterChart, Scatter, ZAxis
} from 'recharts'
import './DashboardPage.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#6366f1']

function DashboardPage({ analysisData }) {
    const [loading, setLoading] = useState(!analysisData)
    const [data, setData] = useState(analysisData)
    const [activeTab, setActiveTab] = useState('overview')

    const [error, setError] = useState(null)
    const [filters, setFilters] = useState({})
    const [isFiltering, setIsFiltering] = useState(false)
    const [customCharts, setCustomCharts] = useState([])
    const [builderConfig, setBuilderConfig] = useState({
        type: 'bar', xCol: '', yCol: '', aggregation: 'sum'
    })
    const [pythonCode, setPythonCode] = useState("sns.pairplot(df)")
    const [plotResult, setPlotResult] = useState(null)
    const [availableVars, setAvailableVars] = useState([])

    useEffect(() => {
        if (!analysisData) {
            fetchAnalysis()
        } else {
            // Extract columns for python cheat sheet
            if (analysisData.datasets && analysisData.datasets.length > 0) {
                setAvailableVars(analysisData.datasets[0].columns || [])
            }
        }
    }, [analysisData])

    const fetchAnalysis = async (currentFilters = null) => {
        setError(null)
        try {
            if (currentFilters) setIsFiltering(true)

            const url = currentFilters ? `${API_BASE}/analyze` : `${API_BASE}/analysis`
            const method = currentFilters ? 'POST' : 'GET'
            const body = currentFilters ? JSON.stringify({ filters: currentFilters }) : null

            const response = await fetch(url, {
                method,
                headers: { 'Content-Type': 'application/json' },
                body
            })

            if (response.ok) {
                const result = await response.json()
                setData(result)
                if (result.datasets && result.datasets.length > 0) {
                    setAvailableVars(result.datasets[0].columns || [])
                }
            } else {
                // Removed debugger statement for production
                const errText = await response.text()
                console.error('Analysis fetch failed:', errText)
                throw new Error(errText || "Failed to load analysis")
            }
        } catch (error) {
            console.error('Failed to fetch analysis:', error)
            setError(error.message)
        } finally {
            setLoading(false)
            setIsFiltering(false)
        }
    }

    const handleFilterChange = (column, value) => {
        const newFilters = { ...filters }
        if (value === '' || value === null || value === 'All') {
            delete newFilters[column]
        } else {
            newFilters[column] = value
        }
        setFilters(newFilters)
        fetchAnalysis(newFilters)
    }

    const tabs = [
        { id: 'overview', label: 'Overview' },
        { id: 'charts', label: 'Visualizations' },
        { id: 'custom', label: 'Custom Builder' },
        { id: 'python', label: 'Python Plotter' },
        { id: 'insights', label: 'Insights' },
    ]

    const handleGenerateCustomChart = async () => {
        if (!builderConfig.xCol) return
        setLoading(true)
        try {
            const validDatasetId = data.datasets?.[0]?.id
            const res = await fetch(`${API_BASE}/analyze/chart/custom`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    type: builderConfig.type,
                    x_col: builderConfig.xCol,
                    y_col: builderConfig.yCol,
                    aggregation: builderConfig.aggregation,
                    dataset_id: validDatasetId
                })
            })
            if (res.ok) {
                const newChart = await res.json()
                setCustomCharts([...customCharts, newChart])
            }
        } catch (err) {
            console.error(err)
            setError("Failed to generate custom chart")
        } finally {
            setLoading(false)
        }
    }

    const handleRunPython = async () => {
        setLoading(true)
        setPlotResult(null)
        try {
            const validDatasetId = data.datasets?.[0]?.id
            const res = await fetch(`${API_BASE}/analyze/python`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    code: pythonCode,
                    dataset_id: validDatasetId
                })
            })
            if (res.ok) {
                const result = await res.json()
                setPlotResult(result.image)
            } else {
                const err = await res.json()
                setError(err.detail)
            }
        } catch (err) {
            console.error(err)
            setError("Failed to run python code")
        } finally {
            setLoading(false)
        }
    }

    if (loading && !data) {
        return (
            <div className="dashboard-loading">
                <div className="spinner"></div>
                <p>Loading analysis...</p>
            </div>
        )
    }

    if (error && !data) {
        return (
            <div className="dashboard-error">
                <h2>Analysis Error</h2>
                <p>{error}</p>
                <button className="btn-primary" onClick={() => fetchAnalysis()}>Retry</button>
            </div>
        )
    }

    if (!data) {
        return (
            <div className="dashboard-empty">
                <h2>No Analysis Available</h2>
                <p>Upload data and run analysis first.</p>
            </div>
        )
    }

    return (
        <div className={`dashboard-page animate-fade-in ${isFiltering ? 'filtering' : ''}`}>
            {isFiltering && (
                <div className="dashboard-overlay">
                    <div className="spinner"></div>
                </div>
            )}

            <header className="page-header">
                <div className="header-content">
                    <h1>Analysis Dashboard</h1>
                    <p>Interactive insights & visualizations</p>
                </div>
                <div className="header-stats">
                    <div className="stat-item">
                        <span className="stat-value">{data.summary?.total_rows?.toLocaleString() || 0}</span>
                        <span className="stat-label">Rows</span>
                    </div>
                </div>
            </header>

            {/* Slicers Bar */}
            {data.slicers && data.slicers.length > 0 && (
                <div className="slicers-bar">
                    {data.slicers.map(slicer => (
                        <div key={slicer.column} className="slicer-item">
                            <label>{slicer.column}</label>
                            <select
                                value={filters[slicer.column] || ''}
                                onChange={(e) => handleFilterChange(slicer.column, e.target.value)}
                            >
                                <option value="All">All</option>
                                {slicer.options.map(opt => (
                                    <option key={opt} value={opt}>{opt}</option>
                                ))}
                            </select>
                        </div>
                    ))}
                    {Object.keys(filters).length > 0 && (
                        <button className="btn-clear-filters" onClick={() => {
                            setFilters({})
                            fetchAnalysis({})
                        }}>
                            Clear Filters
                        </button>
                    )}
                </div>
            )}

            {/* Tabs */}
            <div className="tabs">
                {tabs.map(tab => (
                    <button
                        key={tab.id}
                        className={`tab ${activeTab === tab.id ? 'active' : ''}`}
                        onClick={() => setActiveTab(tab.id)}
                    >
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Overview Tab */}
            {activeTab === 'overview' && (
                <div className="tab-content animate-slide-up">
                    <div className="overview-header">
                        <h3>Key Metrics (What)</h3>
                        <p>High-level performance indicators</p>
                    </div>

                    <div className="kpi-grid">
                        {data.kpis && data.kpis.map((kpi, i) => (
                            <div key={i} className="kpi-card card">
                                <div className="kpi-label">{kpi.label}</div>
                                <div className="kpi-value">{formatKpiValue(kpi.value, kpi.format)}</div>
                                {kpi.column && <div className="kpi-source">Source: {kpi.column}</div>}
                            </div>
                        ))}
                    </div>

                    <div className="overview-header" style={{ marginTop: '2rem' }}>
                        <h3>Visual Trends (How)</h3>
                        <p>Distributions and patterns</p>
                    </div>

                    <div className="quick-charts">
                        {data.charts && data.charts.slice(0, 2).map((chart, i) => (
                            <div key={i} className="chart-wrapper-half">
                                <DynamicChart chart={chart} onFilter={handleFilterChange} />
                            </div>
                        ))}
                    </div>

                    <div className="overview-header" style={{ marginTop: '2rem' }}>
                        <h3>AI Summary (Why)</h3>
                        <p>Automated reasoning</p>
                    </div>
                    {data.ai_summary ? (
                        <div className="ai-summary-card card">
                            <p>{data.ai_summary}</p>
                        </div>
                    ) : (
                        <div className="card"><p>AI Summary unavailable.</p></div>
                    )}
                </div>
            )}

            {/* Charts Tab */}
            {activeTab === 'charts' && (
                <div className="tab-content animate-slide-up">
                    <div className="charts-grid-full">
                        {data.charts && data.charts.map((chart, i) => (
                            <DynamicChart key={i} chart={chart} onFilter={handleFilterChange} />
                        ))}
                    </div>
                </div>
            )}

            {/* Custom Builder Tab */}
            {activeTab === 'custom' && (
                <div className="tab-content animate-slide-up">
                    <div className="builder-controls card">
                        <h3>Create Custom Chart</h3>
                        <div className="controls-row">
                            <div className="control-group">
                                <label>Chart Type</label>
                                <select value={builderConfig.type} onChange={e => setBuilderConfig({ ...builderConfig, type: e.target.value })}>
                                    <option value="bar">Bar Chart</option>
                                    <option value="line">Line Chart</option>
                                    <option value="area">Area Chart</option>
                                    <option value="donut">Donut Chart</option>
                                    <option value="scatter">Scatter Plot</option>
                                </select>
                            </div>
                            <div className="control-group">
                                <label>X Axis (Category/Time)</label>
                                <select value={builderConfig.xCol} onChange={e => setBuilderConfig({ ...builderConfig, xCol: e.target.value })}>
                                    <option value="">Select Column</option>
                                    {availableVars.map(c => (
                                        <option key={c.name} value={c.name}>{c.name} ({c.semantic_type})</option>
                                    ))}
                                </select>
                            </div>
                            <div className="control-group">
                                <label>Y Axis (Value)</label>
                                <select value={builderConfig.yCol} onChange={e => setBuilderConfig({ ...builderConfig, yCol: e.target.value })}>
                                    <option value="">None (Count)</option>
                                    {availableVars
                                        .filter(c => ['numeric', 'currency', 'percentage'].includes(c.semantic_type))
                                        .map(c => (
                                            <option key={c.name} value={c.name}>{c.name}</option>
                                        ))}
                                </select>
                            </div>
                            <div className="control-group">
                                <label>Aggregation</label>
                                <select value={builderConfig.aggregation} onChange={e => setBuilderConfig({ ...builderConfig, aggregation: e.target.value })}>
                                    <option value="sum">Sum</option>
                                    <option value="mean">Average</option>
                                    <option value="count">Count</option>
                                    <option value="min">Min</option>
                                    <option value="max">Max</option>
                                </select>
                            </div>
                            <button className="btn-primary" onClick={handleGenerateCustomChart}>Generate Interactive Plot</button>
                        </div>
                    </div>

                    <div className="custom-charts-grid" style={{ marginTop: '2rem' }}>
                        {customCharts.map((chart, i) => (
                            <div key={i} className="custom-chart-wrapper" style={{ position: 'relative', marginBottom: '2rem' }}>
                                <div style={{ position: 'absolute', top: 0, right: 0, zIndex: 10 }}>
                                    <button onClick={() => setCustomCharts(customCharts.filter((_, idx) => idx !== i))}>✕</button>
                                </div>
                                <DynamicChart chart={chart} onFilter={() => { }} />
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Python Plotter Tab */}
            {activeTab === 'python' && (
                <div className="tab-content animate-slide-up" style={{ display: 'flex', gap: '1rem' }}>
                    <div className="python-sidebar card" style={{ flex: '0 0 250px' }}>
                        <h4>Available Variables</h4>
                        <ul className="var-list">
                            {availableVars.map(v => (
                                <li key={v.name} className="var-item">
                                    <span className="var-name">{v.name}</span>
                                    <span className={`var-type ${v.semantic_type}`}>{v.semantic_type}</span>
                                </li>
                            ))}
                        </ul>
                        <p style={{ marginTop: '1rem', fontSize: '0.8rem' }}>DataFrame variable: <code>df</code></p>
                    </div>

                    <div style={{ flex: 1 }}>
                        <div className="python-editor card">
                            <h3>Python Code Editor</h3>
                            <textarea
                                value={pythonCode}
                                onChange={(e) => setPythonCode(e.target.value)}
                                style={{
                                    width: '100%',
                                    height: '200px',
                                    fontFamily: 'monospace',
                                    padding: '1rem',
                                    background: '#1e293b',
                                    color: '#f8fafc',
                                    border: '1px solid #334155',
                                    borderRadius: '4px'
                                }}
                            />
                            <button className="btn-primary" onClick={handleRunPython} style={{ marginTop: '1rem' }}>Run Code</button>
                        </div>

                        {plotResult && (
                            <div className="plot-result card" style={{ marginTop: '2rem', textAlign: 'center' }}>
                                <img src={`data:image/png;base64,${plotResult}`} alt="Generated Plot" style={{ maxWidth: '100%', borderRadius: '0.5rem' }} />
                            </div>
                        )}
                    </div>
                </div>
            )}
            {/* Insights Tab */}
            {activeTab === 'insights' && (
                <div className="tab-content animate-slide-up">
                    {data.ai_summary && (
                        <div className="card ai-summary-highlight">
                            <h3>AI Executive Summary</h3>
                            <p>{data.ai_summary}</p>
                        </div>
                    )}
                    <div className="insights-list" style={{ marginTop: '1rem' }}>
                        {data.insights.map((insight, i) => (
                            <div key={i} className={`insight-card card ${insight.type}`}>
                                <h4>{insight.title}</h4>
                                <p>{insight.description}</p>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
}

function DynamicChart({ chart, onFilter }) {
    if (!chart || !chart.data) return null
    if (chart.data.length === 0) return <div className="card">No data for chart {chart.title}</div>

    const CustomTooltip = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            return (
                <div className="custom-tooltip" style={{ background: '#fff', padding: '10px', boxShadow: '0 2px 5px rgba(0,0,0,0.1)', borderRadius: '4px', border: '1px solid #ccc' }}>
                    <p className="label">{`${payload[0].name} : ${payload[0].value.toLocaleString()}`}</p>
                    <p className="desc">{label}</p>
                </div>
            );
        }
        return null;
    };

    return (
        <div className="chart-card card" style={{ height: '400px', width: '100%', padding: '1rem' }}>
            <h3>{chart.title}</h3>
            <ResponsiveContainer width="100%" height="90%">
                {renderRechart(chart, onFilter, CustomTooltip)}
            </ResponsiveContainer>
        </div>
    )
}

function renderRechart(chart, onFilter, CustomTooltip) {
    const { type, data, xLabel, yLabel } = chart

    switch (type) {
        case 'bar':
            return (
                <BarChart data={data} onClick={(e) => e && e.activeLabel && onFilter(xLabel, e.activeLabel)}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="label" label={{ value: xLabel, position: 'insideBottom', offset: -5 }} />
                    <YAxis label={{ value: yLabel, angle: -90, position: 'insideLeft' }} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} cursor="pointer" />
                </BarChart>
            )
        case 'line':
            return (
                <LineChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="x" label={{ value: xLabel, position: 'insideBottom', offset: -5 }} />
                    <YAxis label={{ value: yLabel, angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <Line type="monotone" dataKey="y" stroke="#10b981" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 8 }} />
                </LineChart>
            )
        case 'area':
            return (
                <AreaChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="x" />
                    <YAxis />
                    <Tooltip />
                    <Area type="monotone" dataKey="y" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.3} />
                </AreaChart>
            )
        case 'donut':
            return (
                <PieChart>
                    <Pie
                        data={data}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        fill="#8884d8"
                        paddingAngle={5}
                        dataKey="value"
                        onClick={(e, index) => onFilter(xLabel, data[index].label)}
                        cursor="pointer"
                    >
                        {data.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                </PieChart>
            )
        case 'scatter':
            return (
                <ScatterChart>
                    <CartesianGrid />
                    <XAxis type="number" dataKey="x" name={xLabel} label={{ value: xLabel, position: 'insideBottom', offset: -5 }} />
                    <YAxis type="number" dataKey="y" name={yLabel} label={{ value: yLabel, angle: -90, position: 'insideLeft' }} />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    <Scatter name="Data" data={data} fill="#8884d8" />
                </ScatterChart>
            )
        case 'heatmap':
            // Recharts doesn't natively support heatmap well, fallback to scatter with sizes/colors or simplified
            return (
                <ScatterChart>
                    <CartesianGrid />
                    <XAxis dataKey="x" type="category" allowDuplicatedCategory={false} />
                    <YAxis dataKey="y" type="category" allowDuplicatedCategory={false} />
                    <ZAxis dataKey="value" range={[50, 400]} />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    <Scatter data={data} fill="#8884d8" />
                </ScatterChart>
            )
        default:
            return <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>Chart type {type} not supported yet</div>
    }
}


function formatKpiValue(value, format) {
    if (value === null || value === undefined) return '-'
    switch (format) {
        case 'currency':
            if (value >= 1000000) return `$${(value / 1000000).toFixed(2)}M`
            if (value >= 1000) return `$${(value / 1000).toFixed(1)}K`
            return `$${value.toFixed(2)}`
        case 'percent': return `${value.toFixed(1)}%`
        default: return value.toLocaleString()
    }
}

export default DashboardPage
