import { useState } from 'react'
import { useSession } from '../contexts/SessionContext'
import './Sidebar.css'

function Sidebar({ currentPage, onNavigate, analysisComplete }) {
    const { sessions, currentSessionId, createSession, switchSession, deleteSession } = useSession();
    const [showSessions, setShowSessions] = useState(true);

    const navItems = [
        {
            id: 'upload',
            label: 'Data & Cleaning',
            icon: (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
            )
        },
        {
            id: 'dashboard',
            label: 'Dashboard',
            icon: (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="3" y="3" width="7" height="9" />
                    <rect x="14" y="3" width="7" height="5" />
                    <rect x="14" y="12" width="7" height="9" />
                    <rect x="3" y="16" width="7" height="5" />
                </svg>
            ),
            disabled: !analysisComplete
        },
        {
            id: 'chat',
            label: 'Ask Questions',
            icon: (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                </svg>
            ),
            disabled: !analysisComplete
        },
    ]

    const handleNewSession = async () => {
        await createSession();
        onNavigate('upload');
    };

    const handleSessionClick = async (sessionId) => {
        await switchSession(sessionId);
        onNavigate('upload');
    };

    const handleDeleteSession = async (e, sessionId) => {
        e.stopPropagation();
        if (confirm('Delete this session?')) {
            await deleteSession(sessionId);
        }
    };

    const formatSessionDate = (dateStr) => {
        if (!dateStr) return '';
        const date = new Date(dateStr);
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    };

    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <div className="logo">
                    <div className="logo-icon">
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z" />
                            <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12" />
                        </svg>
                    </div>
                    <div className="logo-text">
                        <span className="logo-title">Sustainable</span>
                        <span className="logo-subtitle">Pronto</span>
                    </div>
                </div>
            </div>

            <nav className="sidebar-nav">
                {/* Sessions Section */}
                <div className="nav-section">
                    <div className="nav-section-header" onClick={() => setShowSessions(!showSessions)}>
                        <span className="nav-section-label">Sessions</span>
                        <button className="new-session-btn" onClick={(e) => { e.stopPropagation(); handleNewSession(); }}>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <line x1="12" y1="5" x2="12" y2="19" />
                                <line x1="5" y1="12" x2="19" y2="12" />
                            </svg>
                        </button>
                    </div>
                    
                    {showSessions && (
                        <div className="sessions-list">
                            {sessions.map(session => (
                                <div 
                                    key={session.id}
                                    className={`session-item ${session.id === currentSessionId ? 'active' : ''}`}
                                    onClick={() => handleSessionClick(session.id)}
                                >
                                    <div className="session-icon">
                                        {session.has_analysis ? (
                                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                <polyline points="20 6 9 17 4 12" />
                                            </svg>
                                        ) : (
                                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                <circle cx="12" cy="12" r="10" />
                                            </svg>
                                        )}
                                    </div>
                                    <div className="session-info">
                                        <span className="session-name">{session.name || `Session`}</span>
                                        <span className="session-meta">
                                            {session.datasets_count || 0} files • {formatSessionDate(session.updated_at)}
                                        </span>
                                    </div>
                                    <button 
                                        className="session-delete"
                                        onClick={(e) => handleDeleteSession(e, session.id)}
                                        title="Delete session"
                                    >
                                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                            <polyline points="3 6 5 6 21 6" />
                                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                                        </svg>
                                    </button>
                                </div>
                            ))}
                            {sessions.length === 0 && (
                                <div className="sessions-empty">No sessions yet</div>
                            )}
                        </div>
                    )}
                </div>

                {/* Navigation Section */}
                <div className="nav-section">
                    <div className="nav-section-label">Navigation</div>
                    {navItems.map(item => (
                        <button
                            key={item.id}
                            className={`nav-item ${(currentPage === item.id || (item.id === 'upload' && currentPage === 'profile')) ? 'active' : ''} ${item.disabled ? 'disabled' : ''}`}
                            onClick={() => !item.disabled && onNavigate(item.id)}
                            disabled={item.disabled}
                        >
                            <span className="nav-icon">{item.icon}</span>
                            <span className="nav-label">{item.label}</span>
                            {item.disabled && (
                                <svg className="nav-lock" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                                    <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                                </svg>
                            )}
                        </button>
                    ))}
                </div>
            </nav>

            <div className="sidebar-footer">
                <div className="status-indicator">
                    <span className="status-dot"></span>
                    <span className="status-text">System Ready</span>
                </div>
            </div>
        </aside>
    )
}

export default Sidebar
