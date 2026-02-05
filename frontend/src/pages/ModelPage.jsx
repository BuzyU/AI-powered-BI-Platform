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
    const [featureColumns, setFeatureColumns] = useState([])
    const [activeTab, setActiveTab] = useState('info')

    // Fetch datasets for evaluation
    useEffect(() => {
        const fetchDatasets = async () => {
            try {
                const res = await fetch(`${API_BASE}/datasets`, {
                    headers: { 'x-session-id': sessionId }
                })
                const data = await res.json()
                // Filter out model files
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
                throw new Error(err.detail || 'Failed to load model')
            }
            
            const data = await res.json()
            setModelInfo(data)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    // Get model info
    const getModelInfo = async () => {
        try {
            const res = await fetch(`${API_BASE}/models/${model.id}`, {
                headers: { 'x-session-id': sessionId }
            })
            const data = await res.json()
            setModelInfo(data)
        } catch (err) {
            console.error('Failed to get model info:', err)
        }
    }

    useEffect(() => {
        getModelInfo()
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
                    feature_columns: featureColumns.length > 0 ? featureColumns : undefined,
                    test_split: 0.2
                })
            })

            if (!res.ok) {
                const err = await res.json()
                throw new Error(err.detail || 'Evaluation failed')
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
                    feature_columns: featureColumns.length > 0 ? featureColumns : undefined,
                    limit: 100
                })
            })

            if (!res.ok) {
                const err = await res.json()
                throw new Error(err.detail || 'Prediction failed')
            }

            const data = await res.json()
            setPredictions(data)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    const renderModelInfo = () => {
        if (!modelInfo) return <p>Model not loaded. Click "Load Model" to begin.</p>
        
        return (
            <div className="model-info">
                <div className="info-grid">
                    <div className="info-card">
                        <h4>Framework</h4>
                        <span className="value">{modelInfo.framework || 'Unknown'}</span>
                    </div>
                    <div className="info-card">
                        <h4>Type</h4>
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
                </div>

                {modelInfo.features && (
                    <div className="features-section">
                        <h4>Features ({modelInfo.n_features || modelInfo.features.length})</h4>
                        <div className="features-list">
                            {modelInfo.features.map((f, i) => (
                                <span key={i} className="feature-tag">{f}</span>
                            ))}
                        </div>
                    </div>
                )}

                {modelInfo.classes && (
                    <div className="classes-section">
                        <h4>Classes ({modelInfo.n_classes})</h4>
                        <div className="classes-list">
                            {modelInfo.classes.map((c, i) => (
                                <span key={i} className="class-tag">{c}</span>
                            ))}
                        </div>
                    </div>
                )}

                {modelInfo.layers && (
                    <div className="layers-section">
                        <h4>Layers ({modelInfo.n_layers})</h4>
                        <div className="layers-list">
                            {modelInfo.layers.map((l, i) => (
                                <div key={i} className="layer-item">
                                    <span className="layer-name">{l.name}</span>
                                    <span className="layer-type">{l.type}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {modelInfo.params && (
                    <div className="params-section">
                        <h4>Parameters</h4>
                        <pre className="params-json">{JSON.stringify(modelInfo.params, null, 2)}</pre>
                    </div>
                )}
            </div>
        )
    }

    const renderEvaluation = () => (
        <div className="evaluation-section">
            <h3>📊 Model Evaluation</h3>
            
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

                <button 
                    className="btn-primary"
                    onClick={evaluateModel}
                    disabled={loading || !selectedDataset || !targetColumn}
                >
                    {loading ? 'Evaluating...' : '🎯 Evaluate Model'}
                </button>
            </div>

            {evaluation && (
                <div className="eval-results">
                    <h4>Evaluation Results</h4>
                    
                    {evaluation.accuracy !== undefined && (
                        <div className="metrics-grid">
                            <div className="metric-card accuracy">
                                <h5>Accuracy</h5>
                                <span className="metric-value">{(evaluation.accuracy * 100).toFixed(2)}%</span>
                            </div>
                            <div className="metric-card">
                                <h5>Precision</h5>
                                <span className="metric-value">{(evaluation.precision * 100).toFixed(2)}%</span>
                            </div>
                            <div className="metric-card">
                                <h5>Recall</h5>
                                <span className="metric-value">{(evaluation.recall * 100).toFixed(2)}%</span>
                            </div>
                            <div className="metric-card">
                                <h5>F1 Score</h5>
                                <span className="metric-value">{(evaluation.f1_score * 100).toFixed(2)}%</span>
                            </div>
                        </div>
                    )}

                    {evaluation.r2 !== undefined && (
                        <div className="metrics-grid">
                            <div className="metric-card">
                                <h5>R² Score</h5>
                                <span className="metric-value">{evaluation.r2.toFixed(4)}</span>
                            </div>
                            <div className="metric-card">
                                <h5>MSE</h5>
                                <span className="metric-value">{evaluation.mse.toFixed(4)}</span>
                            </div>
                            <div className="metric-card">
                                <h5>RMSE</h5>
                                <span className="metric-value">{evaluation.rmse.toFixed(4)}</span>
                            </div>
                            <div className="metric-card">
                                <h5>MAE</h5>
                                <span className="metric-value">{evaluation.mae.toFixed(4)}</span>
                            </div>
                        </div>
                    )}

                    {evaluation.confusion_matrix && (
                        <div className="confusion-matrix">
                            <h5>Confusion Matrix</h5>
                            <table>
                                <tbody>
                                    {evaluation.confusion_matrix.map((row, i) => (
                                        <tr key={i}>
                                            {row.map((cell, j) => (
                                                <td key={j} className={i === j ? 'diagonal' : ''}>
                                                    {cell}
                                                </td>
                                            ))}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
            )}
        </div>
    )

    const renderPredictions = () => (
        <div className="predictions-section">
            <h3>🔮 Run Predictions</h3>
            
            <div className="pred-form">
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

                <button 
                    className="btn-primary"
                    onClick={predictFromDataset}
                    disabled={loading || !selectedDataset}
                >
                    {loading ? 'Predicting...' : '🚀 Run Predictions'}
                </button>
            </div>

            {predictions && (
                <div className="pred-results">
                    <h4>Prediction Results ({predictions.n_samples} samples)</h4>
                    
                    {predictions.confidence && (
                        <p className="confidence">
                            Average Confidence: <strong>{(predictions.confidence * 100).toFixed(2)}%</strong>
                        </p>
                    )}
                    
                    <div className="predictions-table-wrapper">
                        <table className="predictions-table">
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Prediction</th>
                                    {predictions.probabilities && <th>Probability</th>}
                                </tr>
                            </thead>
                            <tbody>
                                {predictions.predictions.slice(0, 20).map((pred, i) => (
                                    <tr key={i}>
                                        <td>{i + 1}</td>
                                        <td>{String(pred)}</td>
                                        {predictions.probabilities && (
                                            <td>
                                                {(Math.max(...predictions.probabilities[i]) * 100).toFixed(1)}%
                                            </td>
                                        )}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                        {predictions.predictions.length > 20 && (
                            <p className="more-results">...and {predictions.predictions.length - 20} more</p>
                        )}
                    </div>
                </div>
            )}
        </div>
    )

    return (
        <div className="model-page">
            <header className="page-header">
                <button onClick={onBack} className="btn-secondary">← Back</button>
                <div className="header-content">
                    <h1>🤖 {model.filename}</h1>
                    <span className="model-type">{model.metadata?.model_type || 'ML Model'}</span>
                </div>
            </header>

            {error && (
                <div className="error-banner">
                    ⚠️ {error}
                    <button onClick={() => setError(null)}>×</button>
                </div>
            )}

            <div className="model-actions">
                {modelInfo?.status !== 'loaded' && (
                    <button 
                        className="btn-primary"
                        onClick={loadModel}
                        disabled={loading}
                    >
                        {loading ? 'Loading...' : '📥 Load Model'}
                    </button>
                )}
                {modelInfo?.status === 'loaded' && (
                    <span className="loaded-badge">✅ Model Loaded</span>
                )}
            </div>

            <div className="tabs">
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

            <div className="tab-content">
                {activeTab === 'info' && renderModelInfo()}
                {activeTab === 'evaluate' && renderEvaluation()}
                {activeTab === 'predict' && renderPredictions()}
            </div>
        </div>
    )
}

export default ModelPage
