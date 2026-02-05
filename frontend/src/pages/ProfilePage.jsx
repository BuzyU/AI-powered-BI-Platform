import { useState, useEffect } from 'react'
import { useSession } from '../contexts/SessionContext'
import '../styles/ProfilePage.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

// Helper for formatting large numbers
const formatNumber = (num) => {
    return new Intl.NumberFormat('en-US').format(num)
}

function ProfilePage({ dataset, onBack, onProfileUpdate, onContinue }) {
    const { sessionId } = useSession()
    const [activeTab, setActiveTab] = useState('overview')
    const [cleaningPreview, setCleaningPreview] = useState(null)
    const [pendingOperations, setPendingOperations] = useState([])
    const [applying, setApplying] = useState(false)
    const [profile, setProfile] = useState(null)
    const [loading, setLoading] = useState(true)
    const [editedFormula, setEditedFormula] = useState(null)
    const [error, setError] = useState(null)

    // Fetch profile data on mount
    useEffect(() => {
        const fetchProfile = async () => {
            if (!dataset?.id || !sessionId) return
            
            setLoading(true)
            setError(null)
            
            try {
                const res = await fetch(`${API_BASE}/datasets/${dataset.id}/profile`, {
                    headers: { 'x-session-id': sessionId }
                })
                
                if (!res.ok) {
                    throw new Error('Failed to load profile')
                }
                
                const data = await res.json()
                setProfile(data)
            } catch (err) {
                console.error('Profile fetch error:', err)
                setError(err.message)
            } finally {
                setLoading(false)
            }
        }
        
        fetchProfile()
    }, [dataset?.id, sessionId])

    if (loading) return (
        <div className="profile-page">
            <header className="page-header">
                <button onClick={onBack} className="btn-secondary" style={{ marginBottom: '1rem' }}>← Back</button>
                <h1>Loading Dataset Profile...</h1>
                <div className="loading-spinner" style={{ margin: '2rem auto' }}></div>
            </header>
        </div>
    )

    if (error) return (
        <div className="profile-page">
            <header className="page-header">
                <button onClick={onBack} className="btn-secondary" style={{ marginBottom: '1rem' }}>← Back</button>
                <h1>Error Loading Profile</h1>
                <p style={{ color: '#ef4444' }}>{error}</p>
            </header>
        </div>
    )

    if (!profile) return (
        <div className="profile-page">
            <header className="page-header">
                <button onClick={onBack} className="btn-secondary" style={{ marginBottom: '1rem' }}>← Back</button>
                <h1>No Profile Data Available</h1>
            </header>
        </div>
    )

    const getSeverityColor = (severity) => {
        switch (severity) {
            case 'critical': return '#ef4444'
            case 'warning': return '#f59e0b'
            default: return '#3b82f6'
        }
    }

    const refreshPreview = async () => {
        if (!cleaningPreview || !editedFormula) return
        setError(null)

        const newOperation = { ...cleaningPreview.operation, formula: editedFormula, method: 'custom_formula' }

        try {
            const res = await fetch(`${API_BASE}/datasets/${dataset.id}/clean/preview`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'x-session-id': sessionId
                },
                body: JSON.stringify([newOperation])
            })
            const preview = await res.json()
            setCleaningPreview({ operation: newOperation, preview })
            // Don't reset editedFormula, keep it in sync
        } catch (err) {
            console.error("Preview refresh failed", err)
            setError(`Preview refresh failed: ${err.message}`)
        }
    }

    const handleCleanAction = async (issue, option) => {
        setEditedFormula(null) // Reset edit state
        setError(null)
        const operation = {
            type: issue.type || issue.issue_type,
            column: issue.column,
            method: option.method,
            ...option // unique params for the method
        }

        // Get preview
        try {
            const res = await fetch(`${API_BASE}/datasets/${dataset.id}/clean/preview`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'x-session-id': sessionId
                },
                body: JSON.stringify([operation])
            })
            const preview = await res.json()

            // Set initial formula from preview if available
            const generatedFormula = preview.changes[0]?.formula || ''
            setCleaningPreview({ operation: { ...operation, formula: generatedFormula }, preview })
            setEditedFormula(generatedFormula)
        } catch (err) {
            console.error("Preview failed", err)
            setError(`Preview failed: ${err.message}`)
        }
    }

    const applyClean = async () => {
        if (!cleaningPreview) return
        setApplying(true)
        setError(null)

        try {
            const res = await fetch(`${API_BASE}/datasets/${dataset.id}/clean/apply`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'x-session-id': sessionId
                },
                body: JSON.stringify([cleaningPreview.operation])
            })

            const result = await res.json()
            if (result.success) {
                // Update local profile
                const newProfile = await fetch(`${API_BASE}/datasets/${dataset.id}/profile`, {
                    headers: { 'x-session-id': sessionId }
                }).then(r => r.json())
                setProfile(newProfile)
                onProfileUpdate(dataset.id, newProfile)
                setCleaningPreview(null)
            }
        } catch (err) {
            console.error("Apply failed", err)
            setError(`Apply failed: ${err.message}`)
        } finally {
            setApplying(false)
        }
    }

    const renderOverview = () => (
        <div className="profile-overview">
            <div className="stats-grid">
                <div className="stat-card">
                    <h4>Rows</h4>
                    <div className="value">{formatNumber(profile.shape.rows)}</div>
                </div>
                <div className="stat-card">
                    <h4>Columns</h4>
                    <div className="value">{formatNumber(profile.shape.columns)}</div>
                </div>
                <div className="stat-card quality">
                    <h4>Data Quality</h4>
                    <div className="value" style={{
                        color: profile.overall_quality > 80 ? '#10b981' : profile.overall_quality > 50 ? '#f59e0b' : '#ef4444'
                    }}>
                        {profile.overall_quality}%
                    </div>
                </div>
                <div className="stat-card">
                    <h4>Issues</h4>
                    <div className="value">{profile.issues.length}</div>
                </div>
            </div>

            <div className="section-header">
                <h3>Critical Issues</h3>
            </div>
            <div className="issues-list">
                {profile.issues.filter(i => i.severity === 'critical').length === 0 ? (
                    <div className="no-issues">No critical issues found! 🎉</div>
                ) : (
                    profile.issues.filter(i => i.severity === 'critical').map((issue, idx) => (
                        <div key={idx} className="issue-card critical">
                            <div className="issue-header">
                                <span className="badge critical">CRITICAL</span>
                                <span className="col-name">{issue.column}</span>
                            </div>
                            <p>{issue.description}</p>
                            <div className="actions">
                                <button onClick={() => setActiveTab('cleaning')}>Fix in Cleaning Tab</button>
                            </div>
                        </div>
                    ))
                )}
            </div>

            <div className="section-header">
                <h3>Correlations</h3>
            </div>
            <div className="correlations-list">
                {profile.correlations && profile.correlations.length > 0 ? (
                    profile.correlations.slice(0, 5).map((corr, idx) => (
                        <div key={idx} className="correlation-item">
                            <span>{corr.column1}</span>
                            <span className="connector">↔</span>
                            <span>{corr.column2}</span>
                            <span className="corr-value" style={{ opacity: Math.abs(corr.correlation) }}>
                                {corr.correlation > 0 ? '+' : ''}{corr.correlation}
                            </span>
                            <span className="tag">{corr.strength}</span>
                        </div>
                    ))
                ) : (
                    <div className="empty-state">No significant correlations found</div>
                )}
            </div>
        </div>
    )

    const renderColumns = () => (
        <div className="columns-table-container">
            <table className="columns-table">
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Type</th>
                        <th>Missing</th>
                        <th>Unique</th>
                        <th>Stats / Examples</th>
                    </tr>
                </thead>
                <tbody>
                    {profile.columns.map(col => (
                        <tr key={col.name}>
                            <td className="font-medium">{col.name}</td>
                            <td><span className={`type-badge ${col.semantic_type}`}>{col.semantic_type}</span></td>
                            <td>
                                {col.missing.count > 0 ? (
                                    <span className="missing-warning">
                                        {col.missing.count} ({col.missing.percentage}%)
                                    </span>
                                ) : (
                                    <span className="text-gray-400">-</span>
                                )}
                            </td>
                            <td>{formatNumber(col.unique.count)}</td>
                            <td className="small-text">
                                {col.statistics ? (
                                    <>
                                        <div>Mean: {col.statistics.mean}</div>
                                        <div>Min: {col.statistics.min} | Max: {col.statistics.max}</div>
                                    </>
                                ) : (
                                    <div className="samples">{col.samples.slice(0, 3).join(", ")}</div>
                                )}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    )

    const renderCleaning = () => (
        <div className="cleaning-view">
            <div className="cleaning-sidebar">
                <h4>Suggestions</h4>
                {profile.cleaning_suggestions.length === 0 ? (
                    <div className="empty-state">No suggestions available</div>
                ) : (
                    profile.cleaning_suggestions.map((sugg, idx) => (
                        <div key={idx} className={`suggestion-card priority-${sugg.priority}`}>
                            <div className="suggestion-header">
                                <span className="col-target">{sugg.column}</span>
                                <span className="issue-type">{sugg.issue_type}</span>
                            </div>
                            <p>{sugg.description}</p>

                            <div className="fix-options">
                                <label>Recommended:</label>
                                <button
                                    className="primary-action"
                                    onClick={() => handleCleanAction(sugg, sugg.recommended_action)}
                                >
                                    {sugg.recommended_action?.description || "Fix"}
                                </button>

                                {sugg.all_options.length > 1 && (
                                    <div className="more-options">
                                        <label>Alternatives:</label>
                                        <select
                                            onChange={(e) => {
                                                const opt = sugg.all_options.find(o => o.method === e.target.value)
                                                if (opt) handleCleanAction(sugg, opt)
                                            }}
                                            value=""
                                        >
                                            <option value="" disabled>Select option...</option>
                                            {sugg.all_options.filter(o => o.method !== sugg.recommended_action?.method).map(opt => (
                                                <option key={opt.method} value={opt.method}>{opt.description}</option>
                                            ))}
                                        </select>
                                    </div>
                                )}
                            </div>
                        </div>
                    ))
                )}
            </div>

            <div className="cleaning-preview-area">
                {error && (
                    <div className="error-alert" style={{
                        padding: '1rem',
                        background: '#fee2e2',
                        color: '#b91c1c',
                        marginBottom: '1rem',
                        borderRadius: '0.5rem',
                        display: 'flex',
                        justifyContent: 'space-between'
                    }}>
                        <span>{error}</span>
                        <button onClick={() => setError(null)} style={{ background: 'none', border: 'none', cursor: 'pointer' }}>✕</button>
                    </div>
                )}
                {cleaningPreview ? (
                    <div className="preview-panel">
                        <div className="preview-header">
                            <h3>Preview Changes</h3>
                            <div className="formula-editor">
                                <label>Cleaning Formula (Editable)</label>
                                <div className="formula-input-group">
                                    <input
                                        type="text"
                                        value={editedFormula !== null ? editedFormula : cleaningPreview.operation.formula || ''}
                                        onChange={(e) => setEditedFormula(e.target.value)}
                                        className="formula-input"
                                    />
                                    <button
                                        className="btn-refresh"
                                        onClick={refreshPreview}
                                        disabled={!editedFormula || editedFormula === cleaningPreview.operation.formula}
                                    >
                                        ↻ Update Preview
                                    </button>
                                </div>
                            </div>
                        </div>

                        <div className="comparison-table">
                            <div className="before">
                                <h4>Before</h4>
                                <pre>{JSON.stringify(cleaningPreview.preview.changes[0].sample_before, null, 2)}</pre>
                            </div>
                            <div className="arrow">→</div>
                            <div className="after">
                                <h4>After</h4>
                                <pre>{JSON.stringify(cleaningPreview.preview.changes[0].sample_after, null, 2)}</pre>
                            </div>
                        </div>

                        <div className="impact-summary">
                            Rows affected: <strong>{cleaningPreview.preview.changes[0].rows_affected}</strong>
                        </div>

                        <div className="preview-actions">
                            <button className="btn-cancel" onClick={() => {
                                setCleaningPreview(null)
                                setEditedFormula(null)
                            }}>Cancel</button>
                            <button
                                className="btn-apply"
                                onClick={applyClean}
                                disabled={applying}
                                style={{
                                    backgroundColor: applying ? '#93c5fd' : '#10b981',
                                    cursor: applying ? 'not-allowed' : 'pointer'
                                }}
                            >
                                {applying ? "Applying..." : "✅ Apply Transformation"}
                            </button>
                        </div>
                    </div>
                ) : (
                    <div className="preview-placeholder">
                        <div className="icon">🪄</div>
                        <h3>Select a suggestion to preview fixes</h3>
                        <p>You can see exactly what will happen to your data before applying changes.</p>
                    </div>
                )}
            </div>
        </div >
    )

    return (
        <div className="profile-page fade-in">
            <header className="page-header">
                <div className="header-left">
                    <button className="btn-back" onClick={onBack}>← Back</button>
                    <div className="header-title">
                        <h1>{dataset.filename}</h1>
                        <span className="subtitle">Data Profile & Quality Report</span>
                    </div>
                </div>
                <div className="header-actions">
                    <button className="btn-secondary" onClick={() => window.print()}>Export Report</button>
                </div>
            </header>

            <div className="tabs">
                <button
                    className={activeTab === 'overview' ? 'active' : ''}
                    onClick={() => setActiveTab('overview')}
                >
                    Overview
                </button>
                <button
                    className={activeTab === 'columns' ? 'active' : ''}
                    onClick={() => setActiveTab('columns')}
                >
                    Column Details
                </button>
                <button
                    className={activeTab === 'cleaning' ? 'active' : ''}
                    onClick={() => setActiveTab('cleaning')}
                >
                    Cleaning & Fixes
                    {profile.cleaning_suggestions.length > 0 && (
                        <span className="badge-count">{profile.cleaning_suggestions.length}</span>
                    )}
                </button>
            </div>

            <div className="tab-content">
                {activeTab === 'overview' && renderOverview()}
                {activeTab === 'columns' && renderColumns()}
                {activeTab === 'cleaning' && renderCleaning()}
            </div>
        </div>
    )
}

export default ProfilePage
