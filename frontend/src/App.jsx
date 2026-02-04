import { useState, useEffect } from 'react'
import './App.css'
import Sidebar from './components/Sidebar'
import UploadPage from './pages/UploadPage'
import ProfilePage from './pages/ProfilePage'
import DashboardPage from './pages/DashboardPage'
import ChatPage from './pages/ChatPage'

const API_BASE = 'http://localhost:8000/api'

function App() {
  const [currentPage, setCurrentPage] = useState('upload')
  const [datasets, setDatasets] = useState([])
  const [profiles, setProfiles] = useState([])
  const [selectedDataset, setSelectedDataset] = useState(null)
  const [analysisComplete, setAnalysisComplete] = useState(false)
  const [analysisData, setAnalysisData] = useState(null)

  useEffect(() => {
    fetchDatasets()
  }, [])

  const fetchDatasets = async () => {
    try {
      const response = await fetch(`${API_BASE}/datasets`)
      if (response.ok) {
        const data = await response.json()
        setDatasets(data.datasets || [])
      }
    } catch (error) {
      console.log('Backend not available')
    }
  }

  const handleDatasetUpload = (uploadedDatasets) => {
    setDatasets(prev => [...prev, ...uploadedDatasets])
  }

  const handleSelectDataset = async (dataset) => {
    setSelectedDataset(dataset)
    setCurrentPage('profile')

    // Fetch full profile in background
    try {
      const response = await fetch(`${API_BASE}/datasets/${dataset.id}/profile`)
      if (response.ok) {
        const profile = await response.json()
        setSelectedDataset(prev => ({ ...prev, profile }))
      }
    } catch (error) {
      console.error('Failed to fetch profile')
    }
  }

  const handleAnalysisComplete = async (analysis) => {
    setAnalysisData(analysis)
    setAnalysisComplete(true)
    setCurrentPage('dashboard')
  }

  const handleProfileUpdate = (datasetId, newProfile) => {
    // Update profiles
    setProfiles(prev => prev.map(p =>
      p.dataset_name === newProfile.dataset_name ? newProfile : p
    ))

    // Update selected dataset
    if (selectedDataset?.id === datasetId) {
      setSelectedDataset(prev => ({ ...prev, profile: newProfile }))
    }
  }

  const handleDeleteDataset = async (datasetId) => {
    try {
      const response = await fetch(`${API_BASE}/datasets/${datasetId}`, { method: 'DELETE' })
      if (response.ok) {
        setDatasets(prev => prev.filter(d => d.id !== datasetId))
        if (selectedDataset?.id === datasetId) {
          setSelectedDataset(null)
        }
      }
    } catch (error) {
      console.error('Delete failed', error)
    }
  }

  const renderPage = () => {
    switch (currentPage) {
      case 'upload':
        return (
          <UploadPage
            onUpload={handleDatasetUpload}
            datasets={datasets}
            onSelectDataset={handleSelectDataset}
            onDelete={handleDeleteDataset}
            onAnalysisComplete={handleAnalysisComplete}
          />
        )
      case 'profile':
        return (
          <ProfilePage
            dataset={selectedDataset}
            onBack={() => setCurrentPage('upload')}
            onProfileUpdate={handleProfileUpdate}
            onContinue={() => setCurrentPage('upload')}
          />
        )
      case 'dashboard':
        return <DashboardPage analysisData={analysisData} />
      case 'chat':
        return <ChatPage analysisData={analysisData} />
      default:
        return <UploadPage />
    }
  }

  const navItems = [
    { id: 'upload', label: 'Upload & Profile', icon: 'upload' },
    { id: 'dashboard', label: 'Dashboard', icon: 'chart', disabled: !analysisComplete },
    { id: 'chat', label: 'Ask AI', icon: 'chat', disabled: !analysisComplete },
  ]

  return (
    <div className="app">
      <Sidebar
        currentPage={currentPage}
        onNavigate={setCurrentPage}
        analysisComplete={analysisComplete}
        navItems={navItems}
      />
      <main className="main-content">
        {renderPage()}
      </main>
    </div>
  )
}

export default App
