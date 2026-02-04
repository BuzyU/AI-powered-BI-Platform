import { useState, useCallback } from 'react'
import './UploadPage.css'

const API_BASE = 'http://localhost:8000/api'

function UploadPage({ onUpload, datasets, onSelectDataset, onDelete, onAnalysisComplete }) {
    const [isDragging, setIsDragging] = useState(false)
    const [uploading, setUploading] = useState(false)
    const [processing, setProcessing] = useState(false)
    const [currentStep, setCurrentStep] = useState(datasets.length > 0 ? 1 : 0)
    const [uploadProgress, setUploadProgress] = useState(0)
    const [processingStatus, setProcessingStatus] = useState('')

    const steps = [
        { id: 1, name: 'Upload', description: 'Upload datasets' },
        { id: 2, name: 'Clean', description: 'Fix data issues' },
        { id: 3, name: 'Analyze', description: 'Generate insights' },
        { id: 4, name: 'Complete', description: 'View dashboard' },
    ]

    const handleDragOver = useCallback((e) => {
        e.preventDefault()
        setIsDragging(true)
    }, [])

    const handleDragLeave = useCallback((e) => {
        e.preventDefault()
        setIsDragging(false)
    }, [])

    const handleDrop = useCallback(async (e) => {
        e.preventDefault()
        setIsDragging(false)
        const files = Array.from(e.dataTransfer.files)
        await uploadFiles(files)
    }, [])

    const handleFileSelect = async (e) => {
        const files = Array.from(e.target.files)
        await uploadFiles(files)
    }

    const uploadFiles = async (files) => {
        setUploading(true)
        setCurrentStep(1)

        try {
            const formData = new FormData()
            files.forEach(file => formData.append('files', file))

            const progressInterval = setInterval(() => {
                setUploadProgress(prev => Math.min(prev + 10, 90))
            }, 200)

            const response = await fetch(`${API_BASE}/upload`, {
                method: 'POST',
                body: formData,
            })

            clearInterval(progressInterval)
            setUploadProgress(100)

            if (response.ok) {
                const data = await response.json()
                onUpload(data.datasets)
                setCurrentStep(2)
            } else {
                throw new Error('Upload failed')
            }
        } catch (error) {
            console.error('Upload error:', error)
        } finally {
            setUploading(false)
            setUploadProgress(0)
        }
    }

    const runAnalysis = async () => {
        setProcessing(true)
        setCurrentStep(3)

        try {
            setProcessingStatus('Analyzing columns and data types...')
            const response = await fetch(`${API_BASE}/analyze`, { method: 'POST' })

            if (response.ok) {
                const analysisData = await response.json()
                setProcessingStatus('Generating AI summary...')

                // Wait a bit if AI is processing
                if (!analysisData.ai_summary) {
                    await new Promise(resolve => setTimeout(resolve, 1500))
                }

                setCurrentStep(4)
                setProcessingStatus('Analysis complete!')
                await new Promise(resolve => setTimeout(resolve, 500))

                // Fetch full analysis again to ensure we have everything
                const fullAnalysis = await fetch(`${API_BASE}/analysis`).then(r => r.json())
                onAnalysisComplete(fullAnalysis)
            } else {
                throw new Error('Analysis failed')
            }
        } catch (error) {
            console.error('Analysis error:', error)
            setProcessingStatus('Analysis failed. Please try again.')
        } finally {
            setProcessing(false)
        }
    }

    return (
        <div className="upload-page animate-fade-in">
            <header className="page-header">
                <h1>Data Upload & Cleaning</h1>
                <p>Upload files, check quality, and clean data before analysis</p>
            </header>

            {/* Progress Steps */}
            <div className="steps-container card">
                {steps.map((step, index) => (
                    <div
                        key={step.id}
                        className={`step ${currentStep >= step.id ? 'active' : ''} ${currentStep > step.id ? 'completed' : ''}`}
                    >
                        <div className="step-indicator">
                            {currentStep > step.id ? (
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                                    <polyline points="20 6 9 17 4 12" />
                                </svg>
                            ) : step.id}
                        </div>
                        <div className="step-content">
                            <span className="step-name">{step.name}</span>
                            <span className="step-description">{step.description}</span>
                        </div>
                        {index < steps.length - 1 && <div className="step-connector" />}
                    </div>
                ))}
            </div>

            {/* Upload Zone */}
            {datasets.length === 0 ? (
                <div
                    className={`upload-zone card ${isDragging ? 'dragging' : ''}`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                >
                    <div className="upload-icon">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                            <polyline points="17 8 12 3 7 8" />
                            <line x1="12" y1="3" x2="12" y2="15" />
                        </svg>
                    </div>
                    <h3>Drag and drop your files here</h3>
                    <p className="upload-hint">Supports CSV, Excel (XLS, XLSX), and JSON files</p>
                    <div className="upload-divider">
                        <span>or</span>
                    </div>
                    <label className="btn btn-primary">
                        <input
                            type="file"
                            multiple
                            accept=".csv,.xls,.xlsx,.json"
                            onChange={handleFileSelect}
                            hidden
                        />
                        Browse Files
                    </label>
                </div>
            ) : (
                <div className="datasets-section">
                    <div className="section-header">
                        <h2>Your Datasets ({datasets.length})</h2>
                        <label className="btn btn-secondary">
                            <input
                                type="file"
                                multiple
                                accept=".csv,.xls,.xlsx,.json"
                                onChange={handleFileSelect}
                                hidden
                            />
                            + Add More
                        </label>
                    </div>

                    <div className="datasets-grid">
                        {datasets.map((dataset, index) => (
                            <div key={dataset.id || index} className="dataset-health-card card">
                                <div className="card-header">
                                    <h3 className="file-name" title={dataset.filename}>{truncateFilename(dataset.filename)}</h3>
                                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                                        <span className="file-type-badge">{dataset.file_type}</span>
                                        <button
                                            className="btn-icon delete-btn"
                                            onClick={(e) => { e.stopPropagation(); onDelete && onDelete(dataset.id); }}
                                            title="Delete Dataset"
                                            style={{ background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer', padding: '4px', display: 'flex' }}
                                        >
                                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                                <polyline points="3 6 5 6 21 6"></polyline>
                                                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                                            </svg>
                                        </button>
                                    </div>
                                </div>

                                <div className="health-score-section">
                                    {dataset.shape?.rows > 0 ? (
                                        <>
                                            <div className="score-circle-small" style={{
                                                borderColor: getQualityColor(dataset.overall_quality)
                                            }}>
                                                <span className="score-value-small" style={{ color: getQualityColor(dataset.overall_quality) }}>
                                                    {dataset.overall_quality ? Math.round(dataset.overall_quality) : '-'}
                                                </span>
                                            </div>
                                            <span className="health-label">Quality Score</span>
                                        </>
                                    ) : (
                                        <div style={{ color: 'var(--error-color)', fontWeight: 'bold' }}>Load Error</div>
                                    )}
                                </div>

                                <div className="issues-summary">
                                    <div className="issue-item">
                                        <span className="issue-label">Rows</span>
                                        <span className="issue-val">{dataset.shape?.rows?.toLocaleString() || 0}</span>
                                    </div>
                                    <div className="issue-item warning">
                                        <span className="issue-label">Missing</span>
                                        <span className="issue-val">{dataset.issues_summary?.missing || 0}</span>
                                    </div>
                                    <div className="issue-item info">
                                        <span className="issue-label">Outliers</span>
                                        <span className="issue-val">{dataset.issues_summary?.outliers || 0}</span>
                                    </div>
                                    <div className="issue-item error" title={dataset.issues_summary?.skew_details || 'No skew detected'}>
                                        <span className="issue-label">Skewed ⓘ</span>
                                        <span className="issue-val">{dataset.issues_summary?.skew || 0}</span>
                                    </div>
                                </div>

                                <div className="card-actions">
                                    <button
                                        className="btn btn-primary btn-sm"
                                        style={{ width: '100%' }}
                                        onClick={() => onSelectDataset(dataset)}
                                    >
                                        Clean & Review
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>

                    {!processing && currentStep < 4 && (
                        <div className="action-section">
                            <div className="action-card">
                                <h3>Analyze All Datasets</h3>
                                <p>Generate specialized dashboards and AI insights. (Cleaning recommended first)</p>
                                <button
                                    className="btn btn-secondary btn-lg"
                                    onClick={runAnalysis}
                                    disabled={processing}
                                >
                                    Skip Cleaning & Generate Dashboard
                                </button>
                            </div>
                        </div>
                    )}

                    {processing && (
                        <div className="processing-overlay">
                            <div className="processing-card card">
                                <div className="spinner spinner-lg"></div>
                                <h3>{processingStatus}</h3>
                                <p>This may take a few moments depending on data size.</p>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {uploading && (
                <div className="upload-progress card">
                    <div className="progress-bar">
                        <div className="progress-fill" style={{ width: `${uploadProgress}%` }} />
                    </div>
                    <span className="progress-text">Uploading... {uploadProgress}%</span>
                </div>
            )}
        </div>
    )
}

function getStatusColor(status) {
    const colors = {
        profiled: 'info',
        cleaned: 'success',
        processed: 'primary',
        error: 'error',
    }
    return colors[status] || 'info'
}

function truncateFilename(name) {
    if (!name) return ''
    return name.length > 20 ? name.substring(0, 20) + '...' : name
}

function getQualityColor(score) {
    if (!score) return '#d1d5db'
    if (score > 80) return '#10b981'
    if (score > 50) return '#f59e0b'
    return '#ef4444'
}

export default UploadPage
