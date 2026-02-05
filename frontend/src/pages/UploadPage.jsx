import { useState, useCallback } from 'react'
import { useSession } from '../contexts/SessionContext'
import './UploadPage.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

function UploadPage({ datasets, onUploadComplete, onAnalyzeStart, onAnalyzeComplete, onViewProfile, onDeleteDataset }) {
    const { sessionId, ensureSession, loading: sessionLoading } = useSession()
    const [dragActive, setDragActive] = useState(false)
    const [uploading, setUploading] = useState(false)
    const [analyzing, setAnalyzing] = useState(false)
    const [error, setError] = useState(null)
    const [deletingId, setDeletingId] = useState(null)

    const handleDrag = useCallback((e) => {
        e.preventDefault()
        e.stopPropagation()
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true)
        } else if (e.type === 'dragleave') {
            setDragActive(false)
        }
    }, [])

    const handleDrop = useCallback((e) => {
        e.preventDefault()
        e.stopPropagation()
        setDragActive(false)
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFiles(e.dataTransfer.files)
        }
    }, [])

    const handleFiles = async (files) => {
        setUploading(true)
        setError(null)
        const formData = new FormData()

        Array.from(files).forEach(file => {
            formData.append('files', file)
        })

        try {
            // Ensure we have a session before uploading
            const currentSession = await ensureSession()
            
            if (!currentSession) {
                throw new Error("Failed to create session")
            }
            
            console.log('Uploading with session:', currentSession)
            
            const res = await fetch(`${API_BASE}/upload`, {
                method: 'POST',
                headers: {
                    'x-session-id': currentSession
                },
                body: formData
            })

            if (!res.ok) {
                const errorData = await res.json().catch(() => ({}))
                throw new Error(errorData.detail || "Upload failed")
            }

            const data = await res.json()
            console.log('Upload response:', data)
            onUploadComplete([...datasets, ...data.datasets])
        } catch (err) {
            console.error('Upload error:', err)
            setError(err.message)
        } finally {
            setUploading(false)
        }
    }

    const runAnalysis = async () => {
        setAnalyzing(true)
        onAnalyzeStart()

        try {
            const res = await fetch(`${API_BASE}/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'x-session-id': sessionId
                },
                body: JSON.stringify({})
            })

            const data = await res.json()
            onAnalyzeComplete(data)
        } catch (err) {
            setError("Analysis failed: " + err.message)
            setAnalyzing(false)
        }
    }

    const handleDelete = async (datasetId, e) => {
        e.stopPropagation()
        if (deletingId) return
        
        setDeletingId(datasetId)
        try {
            const res = await fetch(`${API_BASE}/datasets/${datasetId}`, {
                method: 'DELETE',
                headers: {
                    'x-session-id': sessionId
                }
            })

            if (res.ok) {
                onDeleteDataset(datasetId)
            } else {
                setError("Failed to delete dataset")
            }
        } catch (err) {
            setError("Delete failed: " + err.message)
        } finally {
            setDeletingId(null)
        }
    }

    const getQualityStatus = (score) => {
        if (score >= 80) return { class: 'excellent', label: 'Excellent' }
        if (score >= 60) return { class: 'good', label: 'Good' }
        if (score >= 40) return { class: 'fair', label: 'Fair' }
        return { class: 'poor', label: 'Poor' }
    }

    const getFileTypeIcon = (fileType, isModel) => {
        if (isModel) {
            return (
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="3"/>
                    <path d="M12 2v4"/>
                    <path d="M12 18v4"/>
                    <path d="m4.93 4.93 2.83 2.83"/>
                    <path d="m16.24 16.24 2.83 2.83"/>
                    <path d="M2 12h4"/>
                    <path d="M18 12h4"/>
                    <path d="m4.93 19.07 2.83-2.83"/>
                    <path d="m16.24 7.76 2.83-2.83"/>
                </svg>
            )
        }
        
        const type = fileType?.toLowerCase()
        if (type === 'csv') {
            return (
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                    <polyline points="14,2 14,8 20,8"/>
                    <line x1="8" y1="13" x2="16" y2="13"/>
                    <line x1="8" y1="17" x2="16" y2="17"/>
                </svg>
            )
        }
        if (type === 'xlsx' || type === 'xls') {
            return (
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="3" y="3" width="18" height="18" rx="2"/>
                    <line x1="3" y1="9" x2="21" y2="9"/>
                    <line x1="3" y1="15" x2="21" y2="15"/>
                    <line x1="9" y1="3" x2="9" y2="21"/>
                    <line x1="15" y1="3" x2="15" y2="21"/>
                </svg>
            )
        }
        if (type === 'json') {
            return (
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                    <polyline points="14,2 14,8 20,8"/>
                    <path d="M8 13h2"/>
                    <path d="M8 17h2"/>
                </svg>
            )
        }
        if (type === 'pkl' || type === 'pt' || type === 'onnx') {
            return (
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="3"/>
                    <path d="M12 2v4"/>
                    <path d="M12 18v4"/>
                    <path d="m4.93 4.93 2.83 2.83"/>
                    <path d="m16.24 16.24 2.83 2.83"/>
                    <path d="M2 12h4"/>
                    <path d="M18 12h4"/>
                    <path d="m4.93 19.07 2.83-2.83"/>
                    <path d="m16.24 7.76 2.83-2.83"/>
                </svg>
            )
        }
        return (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14,2 14,8 20,8"/>
            </svg>
        )
    }

    return (
        <div className="upload-page">
            <header className="upload-header">
                <div className="header-content">
                    <h1>Data & Models</h1>
                    <p>Upload your datasets and models for intelligent analysis</p>
                </div>
            </header>

            {/* Upload Zone */}
            <div
                className={`upload-dropzone ${dragActive ? 'active' : ''} ${uploading ? 'uploading' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
            >
                <input
                    type="file"
                    id="file-upload"
                    multiple
                    onChange={(e) => handleFiles(e.target.files)}
                    hidden
                />

                {uploading ? (
                    <div className="upload-loading">
                        <div className="loading-spinner"></div>
                        <span>Processing files...</span>
                    </div>
                ) : (
                    <label htmlFor="file-upload" className="dropzone-content">
                        <div className="dropzone-icon">
                            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                                <polyline points="17,8 12,3 7,8"/>
                                <line x1="12" y1="3" x2="12" y2="15"/>
                            </svg>
                        </div>
                        <div className="dropzone-text">
                            <span className="dropzone-title">Drop files here or click to browse</span>
                            <span className="dropzone-formats">CSV, Excel (.xlsx, .xls), JSON, Models (.pkl, .h5, .pt, .onnx)</span>
                        </div>
                    </label>
                )}
            </div>

            {/* Error Alert */}
            {error && (
                <div className="error-alert">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="12" y1="8" x2="12" y2="12"/>
                        <line x1="12" y1="16" x2="12.01" y2="16"/>
                    </svg>
                    <span>{error}</span>
                    <button onClick={() => setError(null)} className="error-close">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <line x1="18" y1="6" x2="6" y2="18"/>
                            <line x1="6" y1="6" x2="18" y2="18"/>
                        </svg>
                    </button>
                </div>
            )}

            {/* Assets Section */}
            {datasets.length > 0 && (
                <div className="assets-section">
                    <div className="assets-header">
                        <div className="assets-title">
                            <h2>Uploaded Assets</h2>
                            <span className="assets-count">{datasets.length} file{datasets.length !== 1 ? 's' : ''}</span>
                        </div>
                        <button
                            className="btn-analyze"
                            onClick={runAnalysis}
                            disabled={analyzing}
                        >
                            {analyzing ? (
                                <>
                                    <div className="btn-spinner"></div>
                                    Analyzing...
                                </>
                            ) : (
                                <>
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <polygon points="5,3 19,12 5,21"/>
                                    </svg>
                                    Run Analysis
                                </>
                            )}
                        </button>
                    </div>

                    <div className="assets-table-wrapper">
                        <table className="assets-table">
                            <thead>
                                <tr>
                                    <th>File</th>
                                    <th>Type</th>
                                    <th>Detection</th>
                                    <th>Size</th>
                                    <th>Quality</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {datasets.map(d => {
                                    const quality = getQualityStatus(d.qa_score || 0)
                                    return (
                                        <tr key={d.id} className="asset-row">
                                            <td className="cell-file">
                                                <div className={`file-icon ${d.is_model ? 'model' : 'data'}`}>
                                                    {getFileTypeIcon(d.file_type, d.is_model)}
                                                </div>
                                                <div className="file-details">
                                                    <span className="file-name">{d.filename}</span>
                                                    <span className="file-role">{d.detected_role || 'Unknown role'}</span>
                                                </div>
                                            </td>
                                            <td>
                                                <span className={`type-badge ${d.is_model ? 'model' : 'data'}`}>
                                                    {d.file_type?.toUpperCase() || 'N/A'}
                                                </span>
                                            </td>
                                            <td>
                                                <div className="detection-info">
                                                    <span className="detected-type">{d.detected_type || 'Processing...'}</span>
                                                    {d.confidence && (
                                                        <span className="confidence-score">{d.confidence}% confidence</span>
                                                    )}
                                                </div>
                                            </td>
                                            <td className="cell-size">
                                                {d.shape ? (
                                                    <span>{d.shape.rows?.toLocaleString() || '-'} rows</span>
                                                ) : d.is_model ? (
                                                    <span className="model-label">Model</span>
                                                ) : (
                                                    <span>-</span>
                                                )}
                                            </td>
                                            <td>
                                                {!d.is_model && d.qa_score !== undefined ? (
                                                    <div className={`quality-indicator ${quality.class}`}>
                                                        <div className="quality-bar">
                                                            <div 
                                                                className="quality-fill" 
                                                                style={{ width: `${d.qa_score}%` }}
                                                            ></div>
                                                        </div>
                                                        <span>{Math.round(d.qa_score)}%</span>
                                                    </div>
                                                ) : (
                                                    <span className="na-value">-</span>
                                                )}
                                            </td>
                                            <td className="cell-actions">
                                                <button 
                                                    className="btn-action view"
                                                    onClick={() => onViewProfile(d.id)}
                                                    title={d.is_model ? "View Model" : "View Profile"}
                                                >
                                                    {d.is_model ? (
                                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                            <circle cx="12" cy="12" r="3"/>
                                                            <path d="M12 2v4"/>
                                                            <path d="M12 18v4"/>
                                                            <path d="m4.93 4.93 2.83 2.83"/>
                                                            <path d="m16.24 16.24 2.83 2.83"/>
                                                            <path d="M2 12h4"/>
                                                            <path d="M18 12h4"/>
                                                        </svg>
                                                    ) : (
                                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
                                                            <circle cx="12" cy="12" r="3"/>
                                                        </svg>
                                                    )}
                                                </button>
                                                <button 
                                                    className="btn-action delete"
                                                    onClick={(e) => handleDelete(d.id, e)}
                                                    disabled={deletingId === d.id}
                                                    title="Delete"
                                                >
                                                    {deletingId === d.id ? (
                                                        <div className="mini-spinner"></div>
                                                    ) : (
                                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                            <polyline points="3,6 5,6 21,6"/>
                                                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                                                            <line x1="10" y1="11" x2="10" y2="17"/>
                                                            <line x1="14" y1="11" x2="14" y2="17"/>
                                                        </svg>
                                                    )}
                                                </button>
                                            </td>
                                        </tr>
                                    )
                                })}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* Empty State */}
            {datasets.length === 0 && !uploading && (
                <div className="empty-state">
                    <div className="empty-icon">
                        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                            <polyline points="14,2 14,8 20,8"/>
                            <line x1="12" y1="18" x2="12" y2="12"/>
                            <line x1="9" y1="15" x2="15" y2="15"/>
                        </svg>
                    </div>
                    <h3>No files uploaded yet</h3>
                    <p>Upload your data files or models to get started with the analysis</p>
                </div>
            )}
        </div>
    )
}

export default UploadPage
