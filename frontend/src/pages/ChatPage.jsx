import { useState, useRef, useEffect } from 'react'
import './ChatPage.css'

const API_BASE = 'http://localhost:8000/api'

function ChatPage({ analysisData }) {
    const [messages, setMessages] = useState([
        {
            type: 'system',
            content: "Ask questions about your data. I'll analyze the uploaded datasets to answer."
        }
    ])
    const [input, setInput] = useState('')
    const [isLoading, setIsLoading] = useState(false)
    const messagesEndRef = useRef(null)

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }

    useEffect(() => {
        scrollToBottom()
    }, [messages])

    // Generate suggestions based on actual data
    const getSuggestions = () => {
        if (!analysisData) return defaultSuggestions

        const suggestions = []

        if (analysisData.kpis?.length > 0) {
            const kpi = analysisData.kpis[0]
            suggestions.push(`What is the total ${kpi.label?.toLowerCase()}?`)
        }

        if (analysisData.charts?.length > 0) {
            suggestions.push("Show me the distribution breakdown")
        }

        if (analysisData.summary) {
            suggestions.push(`How many records are in the data?`)
            suggestions.push(`What is the data quality score?`)
        }

        if (analysisData.insights?.length > 0) {
            suggestions.push("What issues were found in the data?")
        }

        suggestions.push("Summarize the key findings")

        return suggestions.slice(0, 6)
    }

    const handleSend = async () => {
        if (!input.trim() || isLoading) return

        const userMessage = input.trim()
        setInput('')

        setMessages(prev => [...prev, { type: 'user', content: userMessage }])
        setIsLoading(true)

        try {
            // Try API first
            const response = await fetch(`${API_BASE}/ask`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: userMessage })
            })

            if (response.ok) {
                const data = await response.json()

                // If AI service failed, throw to trigger local fallback
                if (!data.success) {
                    throw new Error(data.answer || 'AI Service Unavailable')
                }

                setMessages(prev => [...prev, {
                    type: 'assistant',
                    content: data.answer,
                    data: null, // AI responses are text-heavy, data cards used for local fallback mostly
                    meta: {
                        model: data.model,
                        tokens: data.tokens,
                        success: data.success
                    }
                }])
            } else {
                throw new Error('API failed')
            }
        } catch (error) {
            console.error("AI Error", error)
            // Use local analysis data to answer
            const answer = generateLocalAnswer(userMessage, analysisData)
            setMessages(prev => [...prev, {
                type: 'assistant',
                content: answer.summary,
                data: answer,
                meta: { source: 'local_rule_engine' }
            }])
        } finally {
            setIsLoading(false)
        }
    }

    const handleSuggestion = (question) => {
        setInput(question)
    }

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSend()
        }
    }

    const suggestions = getSuggestions()

    return (
        <div className="chat-page">
            <header className="page-header">
                <h1>Data Assistant</h1>
                <p>Ask questions about your analyzed data</p>
            </header>

            <div className="chat-container card">
                <div className="messages-area">
                    {messages.map((message, index) => (
                        <div key={index} className={`message message-${message.type}`}>
                            <div className="message-avatar">
                                {message.type === 'user' ? (
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                                        <circle cx="12" cy="7" r="4" />
                                    </svg>
                                ) : (
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M12 2L2 7l10 5 10-5-10-5z" />
                                        <path d="M2 17l10 5 10-5" />
                                        <path d="M2 12l10 5 10-5" />
                                    </svg>
                                )}
                            </div>
                            <div className="message-content">
                                <div className="message-label">
                                    {message.type === 'user' ? 'You' : message.type === 'system' ? 'System' : 'Assistant'}
                                </div>
                                <div className="message-text">{message.content}</div>
                                {message.data && <DataCard data={message.data} />}
                                {message.meta && (
                                    <div className="message-meta">
                                        {message.meta.model && <span className="meta-tag model">{message.meta.model}</span>}
                                        {message.meta.tokens && <span className="meta-tag tokens">{message.meta.tokens} tokens</span>}
                                        {message.meta.source === 'local_rule_engine' && <span className="meta-tag local">Local Rules</span>}
                                    </div>
                                )}
                            </div>
                        </div>
                    ))}

                    {isLoading && (
                        <div className="message message-assistant">
                            <div className="message-avatar">
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M12 2L2 7l10 5 10-5-10-5z" />
                                    <path d="M2 17l10 5 10-5" />
                                    <path d="M2 12l10 5 10-5" />
                                </svg>
                            </div>
                            <div className="message-content">
                                <div className="message-label">Assistant</div>
                                <div className="typing-indicator">
                                    <span></span>
                                    <span></span>
                                    <span></span>
                                </div>
                            </div>
                        </div>
                    )}

                    <div ref={messagesEndRef} />
                </div>

                {messages.length === 1 && (
                    <div className="suggestions">
                        <span className="suggestions-label">Suggested questions</span>
                        <div className="suggestions-grid">
                            {suggestions.map((q, i) => (
                                <button
                                    key={i}
                                    className="suggestion-btn"
                                    onClick={() => handleSuggestion(q)}
                                >
                                    {q}
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                <div className="input-area">
                    <div className="input-container">
                        <input
                            type="text"
                            placeholder="Ask a question about your data..."
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyPress={handleKeyPress}
                            disabled={isLoading}
                        />
                        <button
                            className="send-btn"
                            onClick={handleSend}
                            disabled={!input.trim() || isLoading}
                        >
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <line x1="22" y1="2" x2="11" y2="13" />
                                <polygon points="22 2 15 22 11 13 2 9 22 2" />
                            </svg>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    )
}

function DataCard({ data }) {
    if (!data || typeof data === 'string') return null

    return (
        <div className="data-card">
            {data.metrics && (
                <div className="data-metrics">
                    {Object.entries(data.metrics).map(([key, value], i) => (
                        <div key={i} className="metric-item">
                            <span className="metric-label">{formatLabel(key)}</span>
                            <span className="metric-value">{formatMetricValue(value)}</span>
                        </div>
                    ))}
                </div>
            )}

            {data.items && data.items.length > 0 && (
                <div className="data-list">
                    <table>
                        <tbody>
                            {data.items.slice(0, 5).map((item, i) => (
                                <tr key={i}>
                                    <td>{item.label || item.name}</td>
                                    <td className="text-right">{formatMetricValue(item.value)}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    )
}

function formatLabel(key) {
    return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
}

function formatMetricValue(value) {
    if (value === null || value === undefined) return '-'
    if (typeof value === 'number') {
        if (value >= 1000000) return `${(value / 1000000).toFixed(2)}M`
        if (value >= 1000) return `${(value / 1000).toFixed(1)}K`
        return value.toLocaleString()
    }
    return String(value)
}

const defaultSuggestions = [
    "What is the total number of records?",
    "Show me the data summary",
    "What patterns exist in the data?",
    "Are there any data quality issues?",
    "What are the key metrics?",
    "Summarize the findings",
]

function generateLocalAnswer(question, analysisData) {
    const q = question.toLowerCase()

    if (!analysisData) {
        return {
            summary: "No analysis data available. Please upload and analyze data first."
        }
    }

    // Total / count questions
    if (q.includes('total') || q.includes('count') || q.includes('how many') || q.includes('records')) {
        return {
            summary: `The data contains ${analysisData.summary?.total_rows?.toLocaleString() || 0} total records across ${analysisData.summary?.datasets || 0} dataset(s) with ${analysisData.summary?.total_columns || 0} columns.`,
            metrics: {
                total_records: analysisData.summary?.total_rows || 0,
                datasets: analysisData.summary?.datasets || 0,
                columns: analysisData.summary?.total_columns || 0,
            }
        }
    }

    // Quality questions
    if (q.includes('quality')) {
        return {
            summary: `Data quality score is ${Math.round(analysisData.summary?.data_quality || 0)}%. This measures completeness and consistency of the uploaded data.`,
            metrics: {
                quality_score: analysisData.summary?.data_quality || 0,
            }
        }
    }

    // Issues / problems
    if (q.includes('issue') || q.includes('problem') || q.includes('warning')) {
        const warnings = analysisData.insights?.filter(i => i.type === 'warning') || []
        if (warnings.length === 0) {
            return { summary: "No significant issues were detected in the data." }
        }
        return {
            summary: `Found ${warnings.length} issue(s) in the data:\n${warnings.map(w => `• ${w.title}: ${w.description}`).join('\n')}`,
            items: warnings.map(w => ({ label: w.title, value: w.column || '' }))
        }
    }

    // Distribution / breakdown
    if (q.includes('distribution') || q.includes('breakdown')) {
        const donutChart = analysisData.charts?.find(c => c.type === 'donut')
        if (donutChart) {
            return {
                summary: `${donutChart.title}:`,
                items: donutChart.data.slice(0, 5).map(d => ({ label: d.label, value: d.value }))
            }
        }
    }

    // KPI questions
    if (q.includes('metric') || q.includes('kpi')) {
        if (analysisData.kpis?.length > 0) {
            const metrics = {}
            analysisData.kpis.forEach(k => {
                metrics[k.label?.toLowerCase().replace(/\s+/g, '_')] = k.value
            })
            return {
                summary: `Key metrics from your data:`,
                metrics
            }
        }
    }

    // Summary / findings
    if (q.includes('summary') || q.includes('finding') || q.includes('summarize')) {
        let summary = `Analysis Summary:\n`
        summary += `• ${analysisData.summary?.total_rows?.toLocaleString() || 0} records analyzed\n`
        summary += `• ${analysisData.summary?.datasets || 0} dataset(s)\n`
        summary += `• Data quality: ${Math.round(analysisData.summary?.data_quality || 0)}%\n`

        if (analysisData.kpis?.length > 0) {
            summary += `\nKey Metrics:\n`
            analysisData.kpis.slice(0, 4).forEach(k => {
                summary += `• ${k.label}: ${formatMetricValue(k.value)}\n`
            })
        }

        if (analysisData.insights?.length > 0) {
            summary += `\n${analysisData.insights.length} insight(s) generated.`
        }

        return { summary }
    }

    // Default response
    return {
        summary: `Based on your data: ${analysisData.summary?.total_rows?.toLocaleString() || 0} records were analyzed with a data quality score of ${Math.round(analysisData.summary?.data_quality || 0)}%. Try asking about specific metrics, distributions, or data quality issues.`,
        metrics: {
            records: analysisData.summary?.total_rows || 0,
            quality: analysisData.summary?.data_quality || 0,
        }
    }
}

export default ChatPage
