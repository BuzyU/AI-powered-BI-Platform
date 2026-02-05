import { useState, useEffect, useCallback } from 'react'
import { SessionProvider, useSession } from './contexts/SessionContext'
import Sidebar from './components/Sidebar'
import './App.css'

// Pages
import UploadPage from './pages/UploadPage'
import EnhancedDashboard from './pages/EnhancedDashboard'
import ProfilePage from './pages/ProfilePage'
import ChatPage from './pages/ChatPage'
import ModelPage from './pages/ModelPage'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

function App() {
  return (
    <SessionProvider>
      <MainApp />
    </SessionProvider>
  )
}

function MainApp() {
  const { sessionId, fetchSessions } = useSession();

  const [datasets, setDatasets] = useState([])
  const [activePage, setActivePage] = useState('upload')
  const [analysisData, setAnalysisData] = useState(null)
  const [selectedDataset, setSelectedDataset] = useState(null)

  // Fetch initial data when session is ready
  useEffect(() => {
    if (sessionId) {
      fetchDatasets();
      fetchAnalysis();
    }
  }, [sessionId]);

  const fetchDatasets = async () => {
    if (!sessionId) return;
    try {
      const res = await fetch(`${API_BASE}/datasets`, {
        headers: { 'x-session-id': sessionId }
      })
      const data = await res.json()
      if (data.datasets) setDatasets(data.datasets)
    } catch (err) {
      console.error("Failed to load datasets", err)
    }
  }

  const fetchAnalysis = async () => {
    if (!sessionId) return;
    try {
      const res = await fetch(`${API_BASE}/analysis`, {
        headers: { 'x-session-id': sessionId }
      })
      if (res.ok) {
        const data = await res.json()
        setAnalysisData(data)
      }
    } catch (err) { }
  }

  const handleUploadComplete = (newDatasets) => {
    setDatasets(newDatasets)
    fetchDatasets()
    // Refresh sessions list to update counts
    fetchSessions()
  }

  const handleDeleteDataset = (datasetId) => {
    setDatasets(prev => prev.filter(d => d.id !== datasetId))
    fetchSessions()
  }

  const handleAnalysisComplete = (data) => {
    setAnalysisData(data)
    setActivePage('dashboard')
    fetchSessions()
  }

  const handleViewProfile = (datasetId) => {
    const dataset = datasets.find(d => d.id === datasetId)
    setSelectedDataset(dataset)
    
    // Check if it's a model file
    if (dataset?.metadata?.is_model) {
      setActivePage('model')
    } else {
      setActivePage('profile')
    }
  }

  const handleNavigate = (page) => {
    setActivePage(page)
  }

  const renderPage = () => {
    switch (activePage) {
      case 'upload':
        return (
          <UploadPage
            datasets={datasets}
            onUploadComplete={handleUploadComplete}
            onDeleteDataset={handleDeleteDataset}
            onAnalyzeStart={() => { }}
            onAnalyzeComplete={handleAnalysisComplete}
            onViewProfile={handleViewProfile}
          />
        )
      case 'profile':
        return (
          <ProfilePage
            dataset={selectedDataset}
            onBack={() => setActivePage('upload')}
            onProfileUpdate={(id, p) => fetchDatasets()}
            onContinue={() => setActivePage('dashboard')}
          />
        )
      case 'model':
        return (
          <ModelPage
            model={selectedDataset}
            onBack={() => setActivePage('upload')}
          />
        )
      case 'dashboard':
        return (
          <EnhancedDashboard />
        )
      case 'chat':
        return <ChatPage />
      default:
        return <UploadPage />
    }
  }

  return (
    <div className="app-container">
      {/* Sidebar with Sessions */}
      <Sidebar 
        currentPage={activePage}
        onNavigate={handleNavigate}
        analysisComplete={!!analysisData}
      />

      {/* Main Content */}
      <main className="main-content">
        {renderPage()}
      </main>
    </div>
  )
}

export default App
