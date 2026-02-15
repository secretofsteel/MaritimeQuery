// frontend/src/context/AuthContext.jsx
import { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { api } from '../api/client'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)       // { username, display_name, tenant_id, role }
  const [loading, setLoading] = useState(true)  // True during initial auth check

  // Check if we have a valid session (cookie) on app startup
  const checkAuth = useCallback(async () => {
    try {
      const data = await api.get('/api/v1/auth/me')
      setUser(data)
    } catch {
      setUser(null)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    checkAuth()
  }, [checkAuth])

  const login = async (username, password) => {
    // POST /login sets the httpOnly cookie automatically
    await api.post('/api/v1/auth/login', { username, password })
    // Now fetch full user info from /me
    const me = await api.get('/api/v1/auth/me')
    setUser(me)
    return me
  }

  const logout = async () => {
    try {
      await api.post('/api/v1/auth/logout')
    } catch {
      // Even if the server call fails, clear local state
    }
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{
      user,
      loading,
      login,
      logout,
      isAuthenticated: !!user,
      isSuperuser: user?.role === 'superuser',
    }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider')
  }
  return context
}
