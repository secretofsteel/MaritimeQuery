// frontend/src/pages/ChatPage.jsx
import { useAuth } from '../context/AuthContext'

export default function ChatPage() {
  const { user } = useAuth()

  return (
    <div className="flex items-center justify-center h-full">
      <div className="text-center">
        <h2 className="text-xl font-semibold text-gray-200">
          Chat Interface
        </h2>
        <p className="text-gray-400 mt-2">
          Coming in Step 5.2 — logged in as {user?.display_name}
        </p>
      </div>
    </div>
  )
}
