// frontend/src/components/chat/ChatInput.jsx
import { useState, useRef, useEffect } from 'react'
import { Send, Paperclip, X } from 'lucide-react'
import { api } from '../../api/client'

export default function ChatInput({ onSubmit, disabled, sessionId }) {
  const [query, setQuery] = useState('')
  const [files, setFiles] = useState([])
  const [uploading, setUploading] = useState(false)
  const textareaRef = useRef(null)
  const fileInputRef = useRef(null)

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
    }
  }, [query])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleSubmit = async () => {
    if (!query.trim() || disabled || uploading) return
    onSubmit(query)
    setQuery('')
    // Files are already uploaded, just clear visually
    setFiles([])
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleFileSelect = async (e) => {
    const selected = Array.from(e.target.files)
    if (!selected.length || !sessionId) return

    setUploading(true)
    try {
      // Limit to 3 files total (existing + new)
      const slots = 3 - files.length
      const toUpload = selected.slice(0, slots)

      for (const file of toUpload) {
        const formData = new FormData()
        formData.append('file', file)
        // Note: Backend might not support session_scope param yet, but we'll send it
        // If not supported, standard upload endpoint works for now (global scope)
        // Ideally: /api/v1/documents/upload?session_id={sessionId}
        await api.upload(`/api/v1/documents/upload?session_id=${sessionId}`, formData)
        
        setFiles(prev => [...prev, file])
      }
    } catch (err) {
      console.error('Upload failed:', err)
      // Could show toast or error state here
    } finally {
      setUploading(false)
      // Reset input so same file can be selected again if removed
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const removeFile = (index) => {
    setFiles(prev => prev.filter((_, i) => i !== index))
    // Note: We don't delete from server here, just from the "to send" context visual
    // Truly session-scoped files would be managed differently on backend.
  }

  return (
    <div className="border-t border-gray-800 bg-gray-900 p-4">
      <div className="max-w-4xl mx-auto space-y-3">
        
        {/* File chips */}
        {files.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {files.map((file, i) => (
              <div key={i} className="flex items-center gap-1.5 bg-gray-800 text-gray-300 text-xs px-2 py-1 rounded-md border border-gray-700">
                <span className="truncate max-w-[150px]">{file.name}</span>
                <button 
                  onClick={() => removeFile(i)}
                  className="hover:text-white hover:bg-gray-700 rounded-full p-0.5"
                >
                  <X size={12} />
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="relative flex items-end gap-2 bg-gray-800 border border-gray-700 rounded-xl p-2 focus-within:ring-2 focus-within:ring-blue-500/50 focus-within:border-blue-500 transition-all">
          
          {/* File attachment button */}
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled || uploading || files.length >= 3 || !sessionId}
            className="p-2 text-gray-400 hover:text-gray-200 hover:bg-gray-700 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title={sessionId ? "Attach file (max 3)" : "Start chat to attach files"}
          >
            <Paperclip size={20} />
            <input 
              type="file" 
              ref={fileInputRef} 
              className="hidden" 
              onChange={handleFileSelect}
              multiple 
            />
          </button>

          {/* Text input */}
          <textarea
            ref={textareaRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about maritime regulations..."
            className="flex-1 bg-transparent border-none text-gray-100 placeholder-gray-500 resize-none py-2 px-1 focus:ring-0 max-h-[200px]"
            rows={1}
            disabled={disabled}
          />

          {/* Submit button */}
          <button
            onClick={handleSubmit}
            disabled={!query.trim() || disabled || uploading}
            className="p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-500 transition-colors disabled:bg-gray-700 disabled:text-gray-500 disabled:cursor-not-allowed mb-0.5"
          >
            <Send size={18} />
          </button>
        </div>
        
        <p className="text-center text-xs text-gray-500">
          AI can make mistakes. Verify important information.
        </p>
      </div>
    </div>
  )
}
