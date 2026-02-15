// frontend/src/components/sessions/SessionList.jsx
import { Plus } from 'lucide-react'
import SessionItem from './SessionItem'

export default function SessionList({ 
  sessions, 
  activeSessionId, 
  isLoading, 
  onSelectSession, 
  onNewChat, 
  onDeleteSession, 
  onRenameSession 
}) {
  return (
    <div className="flex flex-col h-full">
      {/* New Chat Button */}
      <div className="p-3">
        <button
          onClick={onNewChat}
          className="w-full flex items-center gap-2 px-4 py-3 bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-colors font-medium text-sm shadow-lg shadow-blue-900/20"
        >
          <Plus size={18} />
          New Chat
        </button>
      </div>

      {/* Session List */}
      <div className="flex-1 overflow-y-auto space-y-1 py-2 custom-scrollbar">
        {isLoading && sessions.length === 0 ? (
          // Skeleton loader
          Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="mx-2 p-3 rounded-lg flex items-center gap-3 animate-pulse">
              <div className="w-4 h-4 bg-gray-800 rounded" />
              <div className="flex-1 space-y-2">
                <div className="h-3 bg-gray-800 rounded w-3/4" />
                <div className="h-2 bg-gray-800 rounded w-1/3" />
              </div>
            </div>
          ))
        ) : sessions.length === 0 ? (
          <div className="text-center text-gray-500 text-sm mt-10 px-4">
            No conversations yet.
            <br />
            Start a new chat!
          </div>
        ) : (
          sessions.map(session => (
            <SessionItem
              key={session.session_id}
              session={session}
              isActive={session.session_id === activeSessionId}
              onSelect={onSelectSession}
              onDelete={onDeleteSession}
              onRename={onRenameSession}
            />
          ))
        )}
      </div>
    </div>
  )
}
