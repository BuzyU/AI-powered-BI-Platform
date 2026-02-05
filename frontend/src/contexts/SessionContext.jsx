import { createContext, useContext, useState, useEffect, useCallback } from 'react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

// Use native crypto API instead of uuid package
const generateId = () => crypto.randomUUID();

const SessionContext = createContext();

export function SessionProvider({ children }) {
    const [currentSessionId, setCurrentSessionId] = useState(null);
    const [sessions, setSessions] = useState([]);
    const [currentSession, setCurrentSession] = useState(null);
    const [loading, setLoading] = useState(true);

    // Fetch all sessions from backend
    const fetchSessions = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/sessions`);
            if (res.ok) {
                const data = await res.json();
                setSessions(data.sessions || []);
                return data.sessions || [];
            }
        } catch (e) {
            console.error("Failed to fetch sessions", e);
        }
        return [];
    }, []);

    // Create a new session
    const createSession = useCallback(async (name = null) => {
        const newId = generateId();
        try {
            await fetch(`${API_BASE}/sessions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id: newId, name })
            });
            
            setCurrentSessionId(newId);
            localStorage.setItem('bi_session_id', newId);
            
            // Refresh session list
            await fetchSessions();
            
            return newId;
        } catch (e) {
            console.error("Failed to create session", e);
            // Still set locally
            setCurrentSessionId(newId);
            localStorage.setItem('bi_session_id', newId);
            return newId;
        }
    }, [fetchSessions]);

    // Switch to an existing session
    const switchSession = useCallback(async (sessionId) => {
        setCurrentSessionId(sessionId);
        localStorage.setItem('bi_session_id', sessionId);
        
        // Load session info
        try {
            const res = await fetch(`${API_BASE}/sessions/${sessionId}/info`, {
                headers: { 'x-session-id': sessionId }
            });
            if (res.ok) {
                const data = await res.json();
                setCurrentSession(data);
            }
        } catch (e) {
            console.error("Failed to load session info", e);
        }
    }, []);

    // Delete a session
    const deleteSession = useCallback(async (sessionId) => {
        try {
            await fetch(`${API_BASE}/sessions/${sessionId}`, {
                method: 'DELETE'
            });
            
            // If deleting current session, switch to another or create new
            if (sessionId === currentSessionId) {
                const remaining = sessions.filter(s => s.id !== sessionId);
                if (remaining.length > 0) {
                    await switchSession(remaining[0].id);
                } else {
                    await createSession();
                }
            }
            
            await fetchSessions();
        } catch (e) {
            console.error("Failed to delete session", e);
        }
    }, [currentSessionId, sessions, switchSession, createSession, fetchSessions]);

    // Get current session ID (ensure one exists)
    const ensureSession = useCallback(async () => {
        let sid = currentSessionId || localStorage.getItem('bi_session_id');
        
        if (!sid) {
            sid = await createSession();
        } else if (!currentSessionId) {
            setCurrentSessionId(sid);
        }
        
        return sid;
    }, [currentSessionId, createSession]);

    // Reset/start fresh
    const resetSession = useCallback(async () => {
        const newId = await createSession();
        window.location.href = '/upload';
        return newId;
    }, [createSession]);

    // Update session name
    const updateSessionName = useCallback(async (sessionId, name) => {
        try {
            await fetch(`${API_BASE}/sessions/${sessionId}/name`, {
                method: 'PUT',
                headers: { 
                    'Content-Type': 'application/json',
                    'x-session-id': sessionId 
                },
                body: JSON.stringify({ name })
            });
            await fetchSessions();
        } catch (e) {
            console.error("Failed to update session name", e);
        }
    }, [fetchSessions]);

    // Initialize
    useEffect(() => {
        let isMounted = true;
        
        const init = async () => {
            setLoading(true);
            
            // Load sessions from backend
            const existingSessions = await fetchSessions();
            
            if (!isMounted) return;
            
            // Check for stored session ID
            const storedId = localStorage.getItem('bi_session_id');
            
            if (storedId && existingSessions.some(s => s.id === storedId)) {
                // Resume existing session
                setCurrentSessionId(storedId);
            } else if (existingSessions.length > 0) {
                // Use most recent session
                const latest = existingSessions[0];
                setCurrentSessionId(latest.id);
                localStorage.setItem('bi_session_id', latest.id);
            } else {
                // Create new session
                await createSession();
            }
            
            if (isMounted) setLoading(false);
        };
        
        init();
        
        return () => { isMounted = false; };
    }, [fetchSessions, createSession]);

    // Backwards compatibility: sessionId alias
    const sessionId = currentSessionId;

    return (
        <SessionContext.Provider value={{ 
            // Current session
            sessionId,
            currentSessionId,
            currentSession,
            
            // All sessions
            sessions,
            
            // Actions
            ensureSession,
            createSession,
            switchSession,
            deleteSession,
            resetSession,
            updateSessionName,
            fetchSessions,
            
            // State
            loading
        }}>
            {children}
        </SessionContext.Provider>
    );
}

export function useSession() {
    const context = useContext(SessionContext);
    if (context === undefined) {
        throw new Error('useSession must be used within a SessionProvider');
    }
    return context;}