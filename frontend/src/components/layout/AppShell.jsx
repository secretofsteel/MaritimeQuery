// frontend/src/components/layout/AppShell.jsx
import { Outlet, NavLink as RouterNavLink, useLocation } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext'
import { MessageSquare, Settings as SettingsIcon, LogOut, Shield } from 'lucide-react'

// Helper for NavLink styling
function NavLink({ to, icon: Icon, label }) {
  const location = useLocation()
  // Active if path starts with 'to', but handle root '/' if needed
  const isActive = location.pathname.startsWith(to)
  
  return (
    <RouterNavLink
      to={to}
      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm transition-colors ${
        isActive
          ? 'bg-gray-800 text-white'
          : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
      }`}
    >
      <Icon size={16} />
      {label}
    </RouterNavLink>
  )
}

export default function AppShell() {
  const { user, logout, isSuperuser } = useAuth()

  const handleLogout = async () => {
    await logout()
    window.location.href = '/login' // Hard reload to clear state
  }

  return (
    <div className="h-screen bg-gray-950 flex flex-col overflow-hidden">
      {/* Top navigation bar */}
      <header className="bg-gray-900 border-b border-gray-800 px-4 h-14 flex items-center justify-between shrink-0">
        {/* Left: App name + nav */}
        <div className="flex items-center gap-6">
          <span className="text-white font-bold text-lg tracking-tight flex items-center gap-2">
            ⚓ MA.D.ASS
          </span>

          <nav className="flex items-center gap-1">
            <NavLink to="/chat" icon={MessageSquare} label="Chat" />
            <NavLink to="/settings" icon={SettingsIcon} label="Settings" />
            
            {isSuperuser && (
              <NavLink to="/admin" icon={Shield} label="Admin" />
            )}
          </nav>
        </div>

        {/* Right: User info + logout */}
        <div className="flex items-center gap-4">
          <div className="text-right hidden sm:block">
            <p className="text-sm font-medium text-gray-200">
              {user?.display_name}
            </p>
            <p className="text-xs text-gray-500">
              {user?.role === 'superuser' ? 'Administrator' : 'User'}
              <span className="mx-1">•</span>
              {user?.tenant_id}
            </p>
          </div>

          <button
            onClick={handleLogout}
            className="text-gray-400 hover:text-gray-200 p-2 rounded-lg hover:bg-gray-800 transition-colors"
            title="Sign out"
          >
            <LogOut size={18} />
          </button>
        </div>
      </header>

      {/* Main content area — pages render here via <Outlet /> */}
      <main className="flex-1 flex overflow-hidden">
        <Outlet />
      </main>
    </div>
  )
}
