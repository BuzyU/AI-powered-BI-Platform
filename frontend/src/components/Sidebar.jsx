import './Sidebar.css'

function Sidebar({ currentPage, onNavigate, analysisComplete }) {
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

    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <div className="logo">
                    <div className="logo-icon">
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M12 2L2 7l10 5 10-5-10-5z" />
                            <path d="M2 17l10 5 10-5" />
                            <path d="M2 12l10 5 10-5" />
                        </svg>
                    </div>
                    <div className="logo-text">
                        <span className="logo-title">Analytics</span>
                        <span className="logo-subtitle">Intelligence Platform</span>
                    </div>
                </div>
            </div>

            <nav className="sidebar-nav">
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
