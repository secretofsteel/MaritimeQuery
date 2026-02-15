// frontend/src/components/sessions/SessionItem.jsx
import { useState } from 'react'
import { MessageSquare, Trash2, Edit2, Check, X } from 'lucide-react'

export default function SessionItem({ session, isActive, onSelect, onDelete, onRename }) {
  const [isEditing, setIsEditing] = useState(false)
  const [editTitle, setEditTitle] = useState(session.title)

  const handleRename = (e) => {
    e.stopPropagation()
    if (!editTitle.trim()) {
      setEditTitle(session.title)
      setIsEditing(false)
      return
    }
    onRename(session.session_id, editTitle)
    setIsEditing(false)
  }

  const handleDelete = (e) => {
    e.stopPropagation()
    onDelete(session.session_id)
  }

  // Simple relative time helper
  const getTimeString = (isoString) => {
    const date = new Date(isoString)
    const now = new Date()
    const diffDays = Math.floor((now - date) / (1000 * 60 * 60 * 24))
    
    if (diffDays === 0) return 'Today'
    if (diffDays === 1) return 'Yesterday'
    if (diffDays < 7) return `${diffDays}d ago`
    return date.toLocaleDateString()
  }

  if (isEditing) {
    return (
      <div className="flex items-center gap-1 p-2 bg-gray-800 rounded mx-2 border border-blue-500/50">
        <input
          type="text"
          value={editTitle}
          onChange={(e) => setEditTitle(e.target.value)}
          className="flex-1 bg-transparent text-sm text-gray-200 outline-none min-w-0"
          autoFocus
          onKeyDown={(e) => {
            if (e.key === 'Enter') handleRename(e)
            if (e.key === 'Escape') {
              setEditTitle(session.title)
              setIsEditing(false)
            }
          }}
          onClick={(e) => e.stopPropagation()}
        />
        <button onClick={handleRename} className="text-green-400 hover:text-green-300">
          <Check size={14} />
        </button>
        <button 
          onClick={(e) => {
            e.stopPropagation()
            setEditTitle(session.title)
            setIsEditing(false)
          }} 
          className="text-red-400 hover:text-red-300"
        >
          <X size={14} />
        </button>
      </div>
    )
  }

  return (
    <div
      onClick={() => onSelect(session.session_id)}
      className={`group flex items-center gap-3 p-3 mx-2 rounded-lg cursor-pointer transition-colors relative ${
        isActive 
          ? 'bg-gray-800 text-gray-100' 
          : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
      }`}
    >
      <MessageSquare size={16} className="shrink-0" />
      
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium truncate pr-14">
          {session.title || 'Untitled Chat'}
        </div>
        <div className="text-[10px] text-gray-500 mt-0.5">
          {getTimeString(session.last_active_at)} • {session.message_count} msgs
        </div>
      </div>

      {/* Action buttons (visible on hover or active) */}
      <div className={`absolute right-2 flex items-center gap-1 ${
        isActive ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
      } transition-opacity`}>
        
        <button
          onClick={(e) => {
            e.stopPropagation()
            setEditTitle(session.title)
            setIsEditing(true)
          }}
          className="p-1 hover:bg-gray-700 rounded text-gray-400 hover:text-blue-400"
          title="Rename"
        >
          <Edit2 size={12} />
        </button>

        <button
          onClick={handleDelete}
          className="p-1 hover:bg-gray-700 rounded text-gray-400 hover:text-red-400"
          title="Delete"
        >
          <Trash2 size={12} />
        </button>
      </div>
    </div>
  )
}
