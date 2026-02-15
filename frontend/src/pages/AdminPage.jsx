// frontend/src/pages/AdminPage.jsx
import { useAuth } from '../context/AuthContext'

export default function AdminPage() {
  const { user } = useAuth()

  return (
    <div className="flex items-center justify-center h-full">
      <div className="text-center">
        <h2 className="text-xl font-semibold text-gray-200">
          Admin Panel
        </h2>
        <p className="text-gray-400 mt-2">
          Coming in Step 5.3 — tenant: {user?.tenant_id}
        </p>
      </div>
    </div>
  )
}
