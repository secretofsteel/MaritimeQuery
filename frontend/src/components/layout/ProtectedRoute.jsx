// frontend/src/components/layout/ProtectedRoute.jsx
import { Navigate } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext'

export default function ProtectedRoute({ children, requireSuperuser = false }) {
  const { isAuthenticated, isSuperuser, loading } = useAuth()

  // Still checking auth status — show nothing (avoids flash)
  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-gray-500 animate-pulse">Loading...</div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />
  }

  if (requireSuperuser && !isSuperuser) {
    return <Navigate to="/chat" replace />
  }

  return children
}
