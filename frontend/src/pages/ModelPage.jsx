import { useState, useEffect } from 'react'
import { useSession } from '../contexts/SessionContext'
import './ModelPage.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

function ModelPage({ model, onBack }) {
    const { sessionId } = useSession()
    const [modelInfo, setModelInfo] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [predictions, setPredictions] = useState(null)
    const [evaluation, setEvaluation] = useState(null)
    const [datasets, setDatasets] = useState([])
    const [selectedDataset, setSelectedDataset] = useState('')
    const [targetColumn, setTargetColumn] = useState('')
    const [activeTab, setActiveTab] = useState('info')
    const [customInput, setCustomInput] = useState({})

    // Helper to safely parse errors
    const parseError = (err) => {
        if (!err) return 'Unknown error'
        if (typeof err === 'string') return err

        // Handle Pydantic validation errors (array)
        if (Array.isArray(err.detail)) {
            return err.detail.map(e => e.msg).join('\n')
        }

        // Handle simple detail string or object
        if (err.detail) {
            return typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail)
        }

        return err.message || JSON.stringify(err)
    }

    // Fetch datasets for evaluation
    useEffect(() => {
        const fetchDatasets = async () => {
            try {
                const res = await fetch(`${API_BASE}/datasets`, {
                    headers: { 'x-session-id': sessionId }
                })
                const data = await res.json()
                const dataFiles = (data.datasets || []).filter(d => !d.metadata?.is_model)
                setDatasets(dataFiles)
            } catch (err) {
                console.error('Failed to fetch datasets:', err)
            }
        }
        fetchDatasets()
    }, [sessionId])

    // Load model
    const loadModel = async () => {
        setLoading(true)
        setError(null)

        try {
            const res = await fetch(`${API_BASE}/models/${model.id}/load`, {
                method: 'POST',
                headers: { 'x-session-id': sessionId }
            })

            if (!res.ok) {
                const err = await res.json()
                throw new Error(parseError(err))
            }

            const data = await res.json()
            setModelInfo(data)

            // Initialize custom input fields
            if (data.features) {
                const initialInput = {}
                data.features.forEach(f => { initialInput[f] = '' })
                setCustomInput(initialInput)
            }
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    // Get model info on mount, auto-load if needed
    useEffect(() => {
        const initModel = async () => {
            try {
                // Try to get info first
                const res = await fetch(`${API_BASE}/models/${model.id}`, {
                    headers: { 'x-session-id': sessionId }
                })

                if (res.ok) {
                    const data = await res.json()
                    setModelInfo(data)

                    // If status is 'not_loaded', allow manual load or auto-load
                    if (data.status === 'not_loaded') {
                        await loadModel()
                    } else if (data.features) {
                        const initialInput = {}
                        data.features.forEach(f => { initialInput[f] = '' })
                        setCustomInput(initialInput)
                    }
                } else {
                    // Not found or error -> Try loading it
                    await loadModel()
                }
            } catch (err) {
                console.error('Failed to init model:', err)
                // Fallback attempt to load
                await loadModel()
            }
        }
        initModel()
    }, [model.id, sessionId])

    // Evaluate model
    const evaluateModel = async () => {
        if (!selectedDataset || !targetColumn) {
            setError('Please select a dataset and target column')
            return
        }

        setLoading(true)
        setError(null)

        try {
            const res = await fetch(`${API_BASE}/models/${model.id}/evaluate-from-dataset`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'x-session-id': sessionId
                },
                body: JSON.stringify({
                    dataset_id: selectedDataset,
                    target_column: targetColumn,
                    test_split: 0.2
                })
            })

            if (!res.ok) {
                const err = await res.json()
                throw new Error(parseError(err))
            }

            const data = await res.json()
            setEvaluation(data)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    // Predict from dataset
    const predictFromDataset = async () => {
        if (!selectedDataset) {
            setError('Please select a dataset')
            return
        }

        setLoading(true)
        setError(null)

        try {
            const res = await fetch(`${API_BASE}/models/${model.id}/predict-from-dataset`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'x-session-id': sessionId
                },
                body: JSON.stringify({
                    dataset_id: selectedDataset,
                    limit: 100
                })
            })

            if (!res.ok) {
                const err = await res.json()
                throw new Error(parseError(err))
            }

            const data = await res.json()
            setPredictions(data)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    // Single prediction with custom input
    const predictSingle = async () => {
        setLoading(true)
        setError(null)

        try {
            const inputData = Object.entries(customInput).map(([key, value]) => {
                const num = parseFloat(value)
                return isNaN(num) ? value : num
            })

            const res = await fetch(`${API_BASE}/models/${model.id}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'x-session-id': sessionId
                },
                body: JSON.stringify({ input_data: [inputData] })
            })

            if (!res.ok) {
                const err = await res.json()
                throw new Error(parseError(err))
            }

            const data = await res.json()
            setPredictions(data)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    const formatMetricValue = (value, isPercentage = true) => {
        if (value === undefined || value === null) return 'N/A'
        if (isPercentage) return `${(value * 100).toFixed(2)}%`
        return value.toFixed(4)
    }

    const renderModelHeader = () => (
        <div className="model-header">
            <div className="model-header-content">
                <h1>
                    <span>🤖</span>
                    {model.filename}
                    <span className="model-type-badge">
                        {modelInfo?.framework || model.metadata?.model_type || 'ML Model'}
                    </span>
                </h1>
                <p className="model-file">
                    {model.metadata?.file_size_mb || modelInfo?.file_size_mb || '?'} MB •
                    {modelInfo?.task || 'Unknown Task'}
                </p>

                <div className="model-quick-stats">
                    {modelInfo?.n_features && (
                        <div className="quick-stat">
                            <div className="quick-stat-value">{modelInfo.n_features}</div>
                            <div className="quick-stat-label">Features</div>
                        </div>
                    )}
                    {modelInfo?.n_classes && (
                        <div className="quick-stat">
                            <div className="quick-stat-value">{modelInfo.n_classes}</div>
                            <div className="quick-stat-label">Classes</div>
                        </div>
                    )}
                    {modelInfo?.n_layers && (
                        <div className="quick-stat">
                            <div className="quick-stat-value">{modelInfo.n_layers}</div>
                            <div className="quick-stat-label">Layers</div>
                        </div>
                    )}
                    {modelInfo?.trainable_params && (
                        <div className="quick-stat">
                            <div className="quick-stat-value">
                                {(modelInfo.trainable_params / 1000000).toFixed(2)}M
                            </div>
                            <div className="quick-stat-label">Parameters</div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )

    const renderModelInfo = () => {
        if (!modelInfo) {
            return (
                <div className="model-not-loaded">
                    <h3>Model Not Loaded</h3>
                    <p>Load the model to see detailed information and run predictions.</p>
                    <button className="btn btn-primary btn-lg" onClick={loadModel} disabled={loading}>
                        {loading ? 'Loading...' : '📥 Load Model'}
                    </button>
                </div>
            )
        }

        return (
            <div className="tab-content">
                <div className="model-info-section">
                    <h3>📋 Model Architecture</h3>
                    <div className="info-grid">
                        <div className="info-card">
                            <h4>Framework</h4>
                            <span className="value highlight">{modelInfo.framework || 'Unknown'}</span>
                        </div>
                        <div className="info-card">
                            <h4>Model Type</h4>
                            <span className="value">{modelInfo.class || modelInfo.type || 'Unknown'}</span>
                        </div>
                        <div className="info-card">
                            <h4>Task</h4>
                            <span className="value">{modelInfo.task || 'Unknown'}</span>
                        </div>
                        <div className="info-card">
                            <h4>File Size</h4>
                            <span className="value">{modelInfo.file_size_mb} MB</span>
                        </div>
                        {modelInfo.input_shape && (
                            <div className="info-card">
                                <h4>Input Shape</h4>
                                <span className="value">{modelInfo.input_shape}</span>
                            </div>
                        )}
                        {modelInfo.output_shape && (
                            <div className="info-card">
                                <h4>Output Shape</h4>
                                <span className="value">{modelInfo.output_shape}</span>
                            </div>
                        )}
                    </div>
                </div>

                {modelInfo.features && modelInfo.features.length > 0 && (
                    <div className="model-info-section">
                        <h3>📊 Input Features ({modelInfo.features.length})</h3>
                        <div className="features-list">
                            {modelInfo.features.map((f, i) => (
                                <span key={i} className="feature-tag">{f}</span>
                            ))}
                        </div>
                    </div>
                )}

                {modelInfo.classes && modelInfo.classes.length > 0 && (
                    <div className="model-info-section">
                        <h3>🏷️ Output Classes ({modelInfo.n_classes})</h3>
                        <div className="classes-list">
                            {modelInfo.classes.map((c, i) => (
                                <span key={i} className="class-tag">{c}</span>
                            ))}
                        </div>
                    </div>
                )}

                {modelInfo.layers && modelInfo.layers.length > 0 && (
                    <div className="model-info-section">
                        <h3>🧱 Network Layers ({modelInfo.n_layers})</h3>
                        <table className="layers-table">
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Name</th>
                                    <th>Type</th>
                                    <th>Output Shape</th>
                                </tr>
                            </thead>
                            <tbody>
                                {modelInfo.layers.map((l, i) => (
                                    <tr key={i}>
                                        <td>{i + 1}</td>
                                        <td>{l.name}</td>
                                        <td>{l.type}</td>
                                        <td>{l.output_shape || 'N/A'}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        )
    }

    const renderEvaluation = () => (
        <div className="tab-content">
            <div className="evaluation-section">
                <h3>📊 Model Evaluation</h3>
                <p style={{ color: '#64748b', marginBottom: '1.5rem' }}>
                    Test your model's performance against a dataset with known labels.
                </p>

                <div className="eval-form">
                    <div className="form-group">
                        <label>Select Dataset</label>
                        <select
                            value={selectedDataset}
                            onChange={(e) => setSelectedDataset(e.target.value)}
                        >
                            <option value="">Choose a dataset...</option>
                            {datasets.map(d => (
                                <option key={d.id} value={d.id}>{d.filename}</option>
                            ))}
                        </select>
                    </div>

                    <div className="form-group">
                        <label>Target Column (y)</label>
                        <input
                            type="text"
                            value={targetColumn}
                            onChange={(e) => setTargetColumn(e.target.value)}
                            placeholder="e.g., label, target, class"
                        />
                    </div>

                    <div className="form-group" style={{ justifyContent: 'flex-end' }}>
                        <button
                            className="btn btn-primary"
                            onClick={evaluateModel}
                            disabled={loading || !selectedDataset || !targetColumn}
                        >
                            {loading ? '⏳ Evaluating...' : '🎯 Run Evaluation'}
                        </button>
                    </div>
                </div>

                {evaluation && (
                    <div className="eval-results">
                        <h4>📈 Results</h4>

                        {/* Classification Metrics */}
                        {evaluation.accuracy !== undefined && (
                            <div className="metrics-grid">
                                <div className="metric-card primary">
                                    <div className="metric-value">
                                        {formatMetricValue(evaluation.accuracy)}
                                    </div>
                                    <div className="metric-label">Accuracy</div>
                                </div>
                                <div className="metric-card">
                                    <div className="metric-value">
                                        {formatMetricValue(evaluation.precision)}
                                    </div>
                                    <div className="metric-label">Precision</div>
                                </div>
                                <div className="metric-card">
                                    <div className="metric-value">
                                        {formatMetricValue(evaluation.recall)}
                                    </div>
                                    <div className="metric-label">Recall</div>
                                </div>
                                <div className="metric-card">
                                    <div className="metric-value">
                                        {formatMetricValue(evaluation.f1_score)}
                                    </div>
                                    <div className="metric-label">F1 Score</div>
                                </div>
                            </div>
                        )}

                        {/* Regression Metrics */}
                        {evaluation.r2 !== undefined && (
                            <div className="metrics-grid">
                                <div className="metric-card primary">
                                    <div className="metric-value">
                                        {formatMetricValue(evaluation.r2, false)}
                                    </div>
                                    <div className="metric-label">R² Score</div>
                                </div>
                                <div className="metric-card">
                                    <div className="metric-value">
                                        {formatMetricValue(evaluation.mse, false)}
                                    </div>
                                    <div className="metric-label">MSE</div>
                                </div>
                                <div className="metric-card">
                                    <div className="metric-value">
                                        {formatMetricValue(evaluation.rmse, false)}
                                    </div>
                                    <div className="metric-label">RMSE</div>
                                </div>
                                <div className="metric-card">
                                    <div className="metric-value">
                                        {formatMetricValue(evaluation.mae, false)}
                                    </div>
                                    <div className="metric-label">MAE</div>
                                </div>
                            </div>
                        )}

                        {/* Confusion Matrix */}
                        {evaluation.confusion_matrix && (
                            <div className="confusion-matrix-container">
                                <h4>Confusion Matrix</h4>
                                <div className="confusion-matrix">
                                    <table>
                                        <tbody>
                                            {evaluation.confusion_matrix.map((row, i) => (
                                                <tr key={i}>
                                                    {row.map((cell, j) => (
                                                        <td
                                                            key={j}
                                                            className={i === j ? 'cell-correct' : 'cell-incorrect'}
                                                        >
                                                            {cell}
                                                        </td>
                                                    ))}
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    )

    const renderPredictions = () => (
        <div className="tab-content">
            <div className="predictions-section">
                <h3>🔮 Make Predictions</h3>
                <p style={{ color: '#64748b', marginBottom: '1.5rem' }}>
                    Use your model to make predictions on new data.
                </p>

                {/* Custom Input Form */}
                {modelInfo?.features && modelInfo.features.length > 0 && (
                    <div className="custom-predict-form">
                        <h4>Single Prediction</h4>
                        <div className="input-grid">
                            {modelInfo.features.map((feature, i) => (
                                <div key={i} className="input-field">
                                    <label>{feature}</label>
                                    <input
                                        type="text"
                                        value={customInput[feature] || ''}
                                        onChange={(e) => setCustomInput({
                                            ...customInput,
                                            [feature]: e.target.value
                                        })}
                                        placeholder="Enter value"
                                    />
                                </div>
                            ))}
                        </div>
                        <button
                            className="btn btn-primary"
                            onClick={predictSingle}
                            disabled={loading}
                        >
                            {loading ? '⏳ Predicting...' : '🎯 Predict'}
                        </button>
                    </div>
                )}

                {/* Batch Prediction from Dataset */}
                <div className="eval-form" style={{ marginTop: '1.5rem' }}>
                    <div className="form-group">
                        <label>Batch Predict from Dataset</label>
                        <select
                            value={selectedDataset}
                            onChange={(e) => setSelectedDataset(e.target.value)}
                        >
                            <option value="">Choose a dataset...</option>
                            {datasets.map(d => (
                                <option key={d.id} value={d.id}>{d.filename}</option>
                            ))}
                        </select>
                    </div>

                    <div className="form-group" style={{ justifyContent: 'flex-end' }}>
                        <button
                            className="btn btn-primary"
                            onClick={predictFromDataset}
                            disabled={loading || !selectedDataset}
                        >
                            {loading ? '⏳ Predicting...' : '🚀 Run Batch Predictions'}
                        </button>
                    </div>
                </div>

                {/* Predictions Results */}
                {predictions && (
                    <div className="eval-results" style={{ marginTop: '1.5rem' }}>
                        <h4>📊 Prediction Results</h4>

                        {predictions.n_samples && (
                            <p style={{ color: '#64748b', marginBottom: '1rem' }}>
                                Generated {predictions.n_samples} predictions
                                {predictions.confidence && (
                                    <> • Average confidence: <strong>{formatMetricValue(predictions.confidence)}</strong></>
                                )}
                            </p>
                        )}

                        <div className="predictions-table-container">
                            <table className="predictions-table">
                                <thead>
                                    <tr>
                                        <th>#</th>
                                        <th>Prediction</th>
                                        {predictions.probabilities && <th>Confidence</th>}
                                    </tr>
                                </thead>
                                <tbody>
                                    {predictions.predictions?.slice(0, 25).map((pred, i) => (
                                        <tr key={i}>
                                            <td>{i + 1}</td>
                                            <td>
                                                <span className={`prediction-badge class-${i % 4}`}>
                                                    {String(pred)}
                                                </span>
                                            </td>
                                            {predictions.probabilities && predictions.probabilities[i] && (
                                                <td>
                                                    <div className="probability-bar">
                                                        <div
                                                            className="prob-fill"
                                                            style={{
                                                                width: `${Math.max(...predictions.probabilities[i]) * 100}%`
                                                            }}
                                                        />
                                                        <span className="prob-value">
                                                            {(Math.max(...predictions.probabilities[i]) * 100).toFixed(1)}%
                                                        </span>
                                                    </div>
                                                </td>
                                            )}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                            {predictions.predictions?.length > 25 && (
                                <p style={{
                                    textAlign: 'center',
                                    color: '#94a3b8',
                                    marginTop: '1rem',
                                    fontSize: '0.9rem'
                                }}>
                                    Showing 25 of {predictions.predictions.length} predictions
                                </p>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    )

    return (
        <div className="model-page">
            <button onClick={onBack} className="back-btn">
                ← Back to Upload
            </button>

            {renderModelHeader()}

            {error && (
                <div className="error-banner">
                    ⚠️ {error}
                    <button onClick={() => setError(null)}>×</button>
                </div>
            )}

            <div className="model-actions">
                {modelInfo?.status !== 'loaded' && (
                    <button
                        className="btn btn-primary btn-lg"
                        onClick={loadModel}
                        disabled={loading}
                    >
                        {loading ? '⏳ Loading...' : '📥 Load Model into Memory'}
                    </button>
                )}
                {modelInfo?.status === 'loaded' && (
                    <div className="loaded-indicator">
                        <span className="dot"></span>
                        Model Ready
                    </div>
                )}
            </div>

            <div className="model-tabs">
                <button
                    className={activeTab === 'info' ? 'active' : ''}
                    onClick={() => setActiveTab('info')}
                >
                    📋 Model Info
                </button>
                <button
                    className={activeTab === 'evaluate' ? 'active' : ''}
                    onClick={() => setActiveTab('evaluate')}
                    disabled={modelInfo?.status !== 'loaded'}
                >
                    📊 Evaluate
                </button>
                <button
                    className={activeTab === 'predict' ? 'active' : ''}
                    onClick={() => setActiveTab('predict')}
                    disabled={modelInfo?.status !== 'loaded'}
                >
                    🔮 Predict
                </button>
            </div>

            {loading && activeTab !== 'info' && (
                <div className="loading-overlay">
                    <div className="spinner"></div>
                    <p>Processing...</p>
                </div>
            )}

            {activeTab === 'info' && renderModelInfo()}
            {activeTab === 'evaluate' && renderEvaluation()}
            {activeTab === 'predict' && renderPredictions()}
        </div>
    )
}

export default ModelPage
