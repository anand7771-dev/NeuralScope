import { Routes, Route, Navigate } from 'react-router-dom'
import { TopNav } from '@/app/components/TopNav'
import { Dashboard } from '@/features/dashboard/Dashboard'
import { InferenceWorkspace } from '@/app/pages/InferenceWorkspace'
import { ModelsPage } from '@/app/pages/ModelsPage'

export default function App() {
  return (
    <div className="min-h-screen bg-ns-bg text-ns-textPrimary flex flex-col font-sans selection:bg-ns-blue/30 selection:text-white">
      <TopNav />
      <main className="flex-1 flex flex-col overflow-auto">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/inference/:sessionId" element={<InferenceWorkspace />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  )
}

