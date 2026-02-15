// frontend/src/components/layout/AppShell.jsx
import { Outlet, NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext'
import { MessageSquare, Settings, LogOut } from 'lucide-react'

export default function AppShell() {
  const { user, logout, isSuperuser } = useAuth()
  const navigate = useNavigate()

  const handleLogout = async () => {
    await logout()
    navigate('/login', { replace: true })
  }

  // Helper for NavLink styling
  const navLinkClass = ({ isActive }) =>
    `flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
      isActive
        ? 'bg-gray-800 text-white'
        : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
    }`

  return (
    <div className="min-h-screen bg-gray-950 flex flex-col">
      {/* Top navigation bar */}
      <header className="bg-gray-900 border-b border-gray-800 px-4 h-14 flex items-center justify-between shrink-0">
        {/* Left: App name + nav */}
        <div className="flex items-center gap-6">
          <span className="text-white font-bold text-lg">⚓ MA.D.ASS</span>

          <nav className="flex items-center gap-1">
            <NavLink to="/chat" className={navLinkClass}>
              <MessageSquare size={16} />
              Chat
            </NavLink>

            {isSuperuser && (
              <NavLink to="/admin" className={navLinkClass}>
                <Settings size={16} />
                Admin
              </NavLink>
            )}
          </nav>
        </div>

        {/* Right: User info + logout */}
        <div className="flex items-center gap-4">
          <div className="text-right">
            <p className="text-sm font-medium text-gray-200">
              {user?.display_name}
            </p>
            <p className="text-xs text-gray-500">
              {user?.tenant_id}
              {isSuperuser && ' • superuser'}
            </p>
          </div>

          <button
            onClick={handleLogout}
            className="text-gray-400 hover:text-gray-200 p-2 rounded-lg
                       hover:bg-gray-800 transition-colors"
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
