import { useState, useEffect } from 'react'
import './DashboardPage.css'

const API_BASE = 'http://localhost:8000/api'

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

    useEffect(() => {
        if (!analysisData) {
            fetchAnalysis()
        }
    }, [analysisData])

    const fetchAnalysis = async (currentFilters = null) => {
        setError(null)
        try {
            if (currentFilters) setIsFiltering(true)

            // Check if we are doing initial fetch or filtering
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
            } else {
                throw new Error("Failed to load analysis")
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
        { id: 'data', label: 'Data Quality' },
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

    if (loading) {
        return (
            <div className="dashboard-loading">
                <div className="spinner"></div>
                <p>Loading analysis...</p>
            </div>
        )
    }

    if (error) {
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
            <div className="dashboard-overlay" style={{ display: isFiltering ? 'flex' : 'none' }}>
                <div className="spinner"></div>
            </div>

            <header className="page-header">
                <div className="header-content">
                    <h1>Analysis Dashboard</h1>
                    <p>Insights generated from your data</p>
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
                    {/* ... (KPIs, existing content) ... */}


                    {/* KPI Grid */}
                    {data.kpis && data.kpis.length > 0 && (
                        <div className="kpi-grid">
                            {data.kpis.map((kpi, i) => (
                                <div key={i} className="kpi-card card">
                                    <div className="kpi-label">{kpi.label}</div>
                                    <div className="kpi-value">{formatKpiValue(kpi.value, kpi.format)}</div>
                                    {kpi.column && (
                                        <div className="kpi-source">from: {kpi.column}</div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Quick Charts */}
                    {data.charts && data.charts.length > 0 && (
                        <div className="quick-charts">
                            {data.charts.slice(0, 2).map((chart, i) => (
                                <DynamicChart key={i} chart={chart} onFilter={handleFilterChange} />
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* Charts Tab */}
            {activeTab === 'charts' && (
                <div className="tab-content animate-slide-up">
                    {data.charts && data.charts.length > 0 ? (
                        <div className="charts-grid">
                            {data.charts.map((chart, i) => (
                                <DynamicChart key={i} chart={chart} onFilter={handleFilterChange} />
                            ))}
                        </div>
                    ) : (
                        <div className="empty-state card">
                            <p>No visualizations available for current filters.</p>
                        </div>
                    )}
                </div>
            )}

            {/* Custom Builder Tab */}
            {
                activeTab === 'custom' && (
                    <div className="tab-content animate-slide-up">
                        <div className="builder-controls card">
                            <h3>Create Custom Chart</h3>
                            <div className="controls-row" style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'end' }}>
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
                                        {data.datasets?.[0]?.columns?.map(c => (
                                            <option key={c.name} value={c.name}>{c.name} ({c.semantic_type})</option>
                                        ))}
                                    </select>
                                </div>
                                <div className="control-group">
                                    <label>Y Axis (Value)</label>
                                    <select value={builderConfig.yCol} onChange={e => setBuilderConfig({ ...builderConfig, yCol: e.target.value })}>
                                        <option value="">None (Count)</option>
                                        {data.datasets?.[0]?.columns?.filter(c => ['numeric', 'currency', 'percentage'].includes(c.semantic_type)).map(c => (
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
                                <button className="btn-primary" onClick={handleGenerateCustomChart}>Generate</button>
                            </div>
                        </div>

                        <div className="custom-charts-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '1.5rem', marginTop: '2rem' }}>
                            {customCharts.map((chart, i) => (
                                <div key={i} className="custom-chart-wrapper" style={{ position: 'relative' }}>
                                    <button
                                        onClick={() => setCustomCharts(customCharts.filter((_, idx) => idx !== i))}
                                        style={{ position: 'absolute', top: '10px', right: '10px', zIndex: 10, background: '#fee2e2', border: 'none', borderRadius: '4px', cursor: 'pointer', padding: '4px 8px' }}
                                    >✕</button>
                                    <DynamicChart chart={chart} onFilter={() => { }} />
                                </div>
                            ))}
                        </div>
                    </div>
                )}


            {/* Python Plotter Tab */}
            {activeTab === 'python' && (
                <div className="tab-content animate-slide-up">
                    <div className="python-editor card">
                        <h3>Python Code Editor</h3>
                        <p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '1rem' }}>
                            Write valid Python code using <code>df</code> (DataFrame), <code>plt</code> (Matplotlib), and <code>sns</code> (Seaborn).
                            The plot will be automatically captured.
                        </p>
                        <textarea
                            value={pythonCode}
                            onChange={(e) => setPythonCode(e.target.value)}
                            style={{
                                width: '100%',
                                height: '200px',
                                fontFamily: 'monospace',
                                padding: '1rem',
                                borderRadius: '0.5rem',
                                border: '1px solid #e2e8f0',
                                marginBottom: '1rem',
                                background: '#1e293b',
                                color: '#f8fafc'
                            }}
                        />
                        <button className="btn-primary" onClick={handleRunPython}>Run Code</button>
                    </div>

                    {plotResult && (
                        <div className="plot-result card" style={{ marginTop: '2rem', textAlign: 'center' }}>
                            <img src={plotResult} alt="Generated Plot" style={{ maxWidth: '100%', borderRadius: '0.5rem' }} />
                        </div>
                    )}
                </div>
            )}

            {/* Insights Tab */}
            {
                activeTab === 'insights' && (
                    <div className="tab-content animate-slide-up">
                        {/* AI Summary in Insights */}
                        {data.ai_summary && (
                            <div className="ai-summary-card card" style={{ marginBottom: '2rem' }}>
                                <div className="ai-summary-header">
                                    <span className="ai-badge">AI Analysis</span>
                                    <h3>Executive Summary</h3>
                                </div>
                                <div className="ai-summary-content">
                                    <p>{data.ai_summary}</p>
                                </div>
                            </div>
                        )}

                        {data.insights && data.insights.length > 0 ? (
                            <div className="insights-list">
                                {data.insights.map((insight, i) => (
                                    <div key={i} className={`insight-card card insight-${insight.type}`}>
                                        <div className="insight-icon">
                                            {getInsightIcon(insight.type)}
                                        </div>
                                        <div className="insight-content">
                                            <h4>{insight.title}</h4>
                                            <p>{insight.description}</p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className="empty-state card">
                                <p>No insigths generated.</p>
                            </div>
                        )}
                    </div>
                )
            }

            {/* Data Quality Tab - Kept simplified for brevity in this update */}
            {
                activeTab === 'data' && (
                    <div className="tab-content animate-slide-up">
                        <div className="quality-overview card">
                            <h3>Data Quality</h3>
                            <div className="score-circle">
                                <span className="score-value">{Math.round(data.summary?.data_quality || 0)}</span>
                            </div>
                        </div>
                    </div>
                )
            }
        </div >
    )
}

function DynamicChart({ chart, onFilter }) {
    if (!chart || !chart.data) return null

    return (
        <div className="chart-card card">
            <h3>{chart.title}</h3>

            {chart.type === 'bar' && <BarChart data={chart.data} xLabel={chart.xLabel} yLabel={chart.yLabel} onClick={(val) => onFilter(chart.xLabel, val)} />}
            {chart.type === 'line' && <LineChart data={chart.data} xLabel={chart.xLabel} yLabel={chart.yLabel} />}
            {chart.type === 'area' && <AreaChart data={chart.data} xLabel={chart.xLabel} yLabel={chart.yLabel} />}
            {chart.type === 'donut' && <DonutChart data={chart.data} onClick={(val) => onFilter(chart.xLabel, val)} />}
            {chart.type === 'histogram' && <HistogramChart data={chart.data} />}
            {chart.type === 'scatter' && <ScatterChart data={chart.data} xLabel={chart.xLabel} yLabel={chart.yLabel} />}
            {chart.type === 'heatmap' && <HeatmapChart data={chart.data} xLabel={chart.xLabel} yLabel={chart.yLabel} />}
            {chart.type === 'box' && <BoxPlotChart stats={chart.stats} />}
        </div>
    )
}

function ScatterChart({ data, xLabel, yLabel }) {
    const xMax = Math.max(...data.map(d => d.x))
    const yMax = Math.max(...data.map(d => d.y))

    return (
        <div className="scatter-chart">
            <svg viewBox="0 0 100 100" className="chart-svg">
                {/* Grid lines */}
                <line x1="0" y1="100" x2="100" y2="100" stroke="#e2e8f0" strokeWidth="2" />
                <line x1="0" y1="0" x2="0" y2="100" stroke="#e2e8f0" strokeWidth="2" />

                {data.map((point, i) => (
                    <circle
                        key={i}
                        cx={(point.x / xMax) * 90 + 5}
                        cy={100 - ((point.y / yMax) * 90 + 5)}
                        r="2.5"
                        className="scatter-point"
                    >
                        <title>{`${xLabel}: ${point.x}, ${yLabel}: ${point.y}`}</title>
                    </circle>
                ))}
            </svg>
            <div className="chart-labels">
                <span>{xLabel}</span>
                <span className="y-label-vertical">{yLabel}</span>
            </div>
        </div>
    )
}

function HeatmapChart({ data, xLabel, yLabel }) {
    // Get unique X and Y keys
    const xKeys = [...new Set(data.map(d => d.x))]
    const yKeys = [...new Set(data.map(d => d.y))]

    // Create grid
    return (
        <div className="heatmap-chart">
            <div className="heatmap-grid" style={{
                gridTemplateColumns: `repeat(${xKeys.length}, 1fr)`,
                gridTemplateRows: `repeat(${yKeys.length}, 1fr)`
            }}>
                {data.map((cell, i) => (
                    <div
                        key={i}
                        className="heatmap-cell"
                        style={{
                            backgroundColor: interpolateColor(cell.value),
                            opacity: Math.abs(cell.value)
                        }}
                        title={`${cell.y} vs ${cell.x}: ${cell.value.toFixed(2)}`}
                    />
                ))}
            </div>
            <div className="heatmap-labels">
                <small>Correlation Matrix</small>
            </div>
        </div>
    )
}

function interpolateColor(value) {
    // Red for negative, Blue for positive
    if (value < 0) return `rgba(239, 68, 68, ${Math.abs(value)})`
    return `rgba(59, 130, 246, ${Math.abs(value)})`
}

function BoxPlotChart({ stats }) {
    if (!stats) return null;
    const { min, q1, median, q3, max } = stats;
    // Normalize to 0-100 range based on min/max
    const range = max - min;
    const normalize = (val) => ((val - min) / range) * 100;

    return (
        <div className="boxplot-chart">
            <svg viewBox="0 0 100 60" className="chart-svg">
                {/* Whisker Line */}
                <line x1={normalize(min)} y1="30" x2={normalize(max)} y2="30" stroke="#64748b" strokeWidth="2" />

                {/* Box */}
                <rect
                    x={normalize(q1)}
                    y="15"
                    width={normalize(q3) - normalize(q1)}
                    height="30"
                    fill="#e0f2fe"
                    stroke="#0284c7"
                    strokeWidth="2"
                />

                {/* Median Line */}
                <line
                    x1={normalize(median)} y1="15"
                    x2={normalize(median)} y2="45"
                    stroke="#0284c7"
                    strokeWidth="3"
                />

                {/* End Caps */}
                <line x1={normalize(min)} y1="25" x2={normalize(min)} y2="35" stroke="#64748b" strokeWidth="2" />
                <line x1={normalize(max)} y1="25" x2={normalize(max)} y2="35" stroke="#64748b" strokeWidth="2" />
            </svg>
            <div className="boxplot-stats">
                <span>Min: {min.toFixed(1)}</span>
                <span>Med: {median.toFixed(1)}</span>
                <span>Max: {max.toFixed(1)}</span>
            </div>
        </div>
    )
}

function AreaChart({ data, xLabel, yLabel }) {
    const maxValue = Math.max(...data.map(d => d.y))
    const points = data.map((d, i) => ({
        x: (i / (data.length - 1)) * 100,
        y: 100 - (d.y / maxValue) * 100
    }))

    const pathD = points.map((p, i) =>
        `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`
    ).join(' ')

    // Closed path for area fill
    const areaPath = `${pathD} L 100 100 L 0 100 Z`

    return (
        <div className="area-chart">
            <svg viewBox="0 0 100 100" preserveAspectRatio="none">
                <defs>
                    <linearGradient id="areaGradient" x1="0" x2="0" y1="0" y2="1">
                        <stop offset="0%" stopColor="var(--primary-color)" stopOpacity="0.5" />
                        <stop offset="100%" stopColor="var(--primary-color)" stopOpacity="0.05" />
                    </linearGradient>
                </defs>
                <path d={areaPath} fill="url(#areaGradient)" stroke="none" />
                <path d={pathD} fill="none" stroke="var(--primary-color)" strokeWidth="2" />
            </svg>
            <div className="line-labels">
                {data.filter((_, i) => i % Math.ceil(data.length / 6) === 0).map((d, i) => (
                    <span key={i}>{d.x}</span>
                ))}
            </div>
        </div>
    )
}

function BarChart({ data, xLabel, yLabel, onClick }) {
    const maxValue = Math.max(...data.map(d => d.value))

    return (
        <div className="bar-chart">
            <div className="chart-bars">
                {data.map((item, i) => (
                    <div key={i} className="bar-group">
                        <div className="bar-container">
                            <div
                                className={`bar ${onClick ? 'interactive' : ''}`}
                                style={{ height: `${(item.value / maxValue) * 100}%` }}
                                onClick={() => onClick && onClick(item.label)}
                                title={`${item.label}: ${item.value.toLocaleString()}`}
                            />
                        </div>
                        <span className="bar-label" title={item.label}>
                            {truncateLabel(item.label)}
                        </span>
                    </div>
                ))}
            </div>
            {yLabel && <div className="chart-y-label">{yLabel}</div>}
            {onClick && <div className="chart-hint">Click bars to filter</div>}
        </div>
    )
}

function LineChart({ data, xLabel, yLabel }) {
    const maxValue = Math.max(...data.map(d => d.y))
    const points = data.map((d, i) => ({
        x: (i / (data.length - 1)) * 100,
        y: 100 - (d.y / maxValue) * 100
    }))

    const pathD = points.map((p, i) =>
        `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`
    ).join(' ')

    return (
        <div className="line-chart">
            <svg viewBox="0 0 100 100" preserveAspectRatio="none">
                <path d={pathD} fill="none" stroke="var(--primary-500)" strokeWidth="2" vectorEffect="non-scaling-stroke" />
                {points.length <= 20 && points.map((p, i) => (
                    <circle key={i} cx={p.x} cy={p.y} r="2" fill="var(--primary-500)" vectorEffect="non-scaling-stroke" />
                ))}
            </svg>
            <div className="line-labels">
                {data.filter((_, i) => i % Math.ceil(data.length / 6) === 0).map((d, i) => (
                    <span key={i}>{d.x}</span>
                ))}
            </div>
        </div>
    )
}

function DonutChart({ data, onClick }) {
    const total = data.reduce((sum, d) => sum + d.value, 0)
    const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']

    let currentAngle = 0
    const segments = data.map((d, i) => {
        const angle = (d.value / total) * 360
        const segment = { ...d, startAngle: currentAngle, angle, color: colors[i % colors.length] }
        currentAngle += angle
        return segment
    })

    return (
        <div className="donut-chart">
            <div className="donut-visual">
                <svg viewBox="0 0 100 100">
                    {segments.map((seg, i) => {
                        const startAngle = (seg.startAngle - 90) * Math.PI / 180
                        const endAngle = (seg.startAngle + seg.angle - 90) * Math.PI / 180
                        const x1 = 50 + 40 * Math.cos(startAngle)
                        const y1 = 50 + 40 * Math.sin(startAngle)
                        const x2 = 50 + 40 * Math.cos(endAngle)
                        const y2 = 50 + 40 * Math.sin(endAngle)
                        const largeArc = seg.angle > 180 ? 1 : 0

                        return (
                            <path
                                key={i}
                                d={`M 50 50 L ${x1} ${y1} A 40 40 0 ${largeArc} 1 ${x2} ${y2} Z`}
                                fill={seg.color}
                                className={onClick ? 'interactive' : ''}
                                onClick={() => onClick && onClick(seg.label)}
                            />
                        )
                    })}
                    <circle cx="50" cy="50" r="25" fill="white" />
                </svg>
            </div>
            <div className="donut-legend">
                {segments.map((seg, i) => (
                    <div key={i} className="legend-item">
                        <span className="legend-dot" style={{ background: seg.color }}></span>
                        <span className="legend-label">{truncateLabel(seg.label)}</span>
                        <span className="legend-value">{seg.value.toLocaleString()}</span>
                    </div>
                ))}
            </div>
        </div>
    )
}

function HistogramChart({ data }) {
    const maxCount = Math.max(...data.map(d => d.count))

    return (
        <div className="histogram-chart">
            <div className="histogram-bars">
                {data.map((item, i) => (
                    <div key={i} className="histogram-bar-group">
                        <div
                            className="histogram-bar"
                            style={{ height: `${(item.count / maxCount) * 100}%` }}
                            title={`${item.range}: ${item.count}`}
                        />
                        <span className="histogram-label">{item.range}</span>
                    </div>
                ))}
            </div>
        </div>
    )
}

function formatKpiValue(value, format) {
    if (value === null || value === undefined) return '-'

    switch (format) {
        case 'currency':
            if (value >= 1000000) return `$${(value / 1000000).toFixed(2)}M`
            if (value >= 1000) return `$${(value / 1000).toFixed(1)}K`
            return `$${value.toFixed(2)}`
        case 'percent':
            return `${value.toFixed(1)}%`
        case 'number':
        default:
            return value.toLocaleString()
    }
}

function truncateLabel(label, maxLen = 12) {
    if (!label) return ''
    const str = String(label)
    return str.length > maxLen ? str.substring(0, maxLen) + '...' : str
}

function getInsightIcon(type) {
    switch (type) {
        case 'warning':
            return (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                    <line x1="12" y1="9" x2="12" y2="13" />
                    <line x1="12" y1="17" x2="12.01" y2="17" />
                </svg>
            )
        case 'tip':
            return (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" />
                    <line x1="12" y1="16" x2="12" y2="12" />
                    <line x1="12" y1="8" x2="12.01" y2="8" />
                </svg>
            )
        default:
            return (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" />
                    <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
                    <line x1="12" y1="17" x2="12.01" y2="17" />
                </svg>
            )
    }
}

export default DashboardPage
