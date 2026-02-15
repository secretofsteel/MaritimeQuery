// frontend/src/pages/ChatPage.jsx
import { useState, useEffect, useCallback } from 'react'
import { Menu, X } from 'lucide-react'
import { api } from '../api/client'
import { useStreamingQuery } from '../hooks/useStreamingQuery'
import SessionList from '../components/sessions/SessionList'
import MessageList from '../components/chat/MessageList'
import ChatInput from '../components/chat/ChatInput'

export default function ChatPage() {
  // Session state
  const [sessions, setSessions] = useState([])
  const [activeSessionId, setActiveSessionId] = useState(null)
  const [isLoadingSessions, setIsLoadingSessions] = useState(true)
  const [showSidebar, setShowSidebar] = useState(false)

  // Message state
  const [messages, setMessages] = useState([])
  const [isLoadingMessages, setIsLoadingMessages] = useState(false)

  // Streaming
  const {
    streamQuery, isStreaming, streamingText, streamingMetadata, cancel
  } = useStreamingQuery()

  // --- Data fetching ---

  const fetchSessions = useCallback(async () => {
    try {
      const data = await api.get('/api/v1/sessions')
      const sessionList = data.sessions || []
      setSessions(sessionList)
      return sessionList
    } catch (err) {
      console.error('Failed to fetch sessions:', err)
      return []
    } finally {
      setIsLoadingSessions(false)
    }
  }, [])

  const fetchMessages = useCallback(async (sessionId) => {
    if (!sessionId) return
    setIsLoadingMessages(true)
    try {
      const data = await api.get(`/api/v1/sessions/${sessionId}/messages`)
      setMessages(data.messages || [])
    } catch (err) {
      console.error('Failed to fetch messages:', err)
      setMessages([])
    } finally {
      setIsLoadingMessages(false)
    }
  }, [])

  // Initial load
  useEffect(() => {
    fetchSessions().then(sessionList => {
      if (sessionList.length > 0) {
        const mostRecent = sessionList[0] // Already sorted by last_active desc
        setActiveSessionId(mostRecent.session_id)
      }
    })
  }, [fetchSessions])

  // Fetch messages when active session changes
  useEffect(() => {
    if (activeSessionId) {
      fetchMessages(activeSessionId)
    } else {
      setMessages([])
    }
  }, [activeSessionId, fetchMessages])

  // --- Session operations ---

  const handleNewChat = useCallback(async () => {
    if (isStreaming) cancel()
    try {
      const data = await api.post('/api/v1/sessions', { title: 'New Chat' })
      await fetchSessions()
      setActiveSessionId(data.session_id)
      setShowSidebar(false) // Close sidebar on mobile
    } catch (err) {
      console.error('Failed to create session:', err)
    }
  }, [isStreaming, cancel, fetchSessions])

  const handleSelectSession = useCallback((sessionId) => {
    if (isStreaming) cancel()
    setActiveSessionId(sessionId)
    setShowSidebar(false) // Close sidebar on mobile
  }, [isStreaming, cancel])

  const handleDeleteSession = useCallback(async (sessionId) => {
    try {
      await api.delete(`/api/v1/sessions/${sessionId}`)
      const updated = sessions.filter(s => s.session_id !== sessionId)
      setSessions(updated)
      if (activeSessionId === sessionId) {
        setActiveSessionId(updated.length > 0 ? updated[0].session_id : null)
      }
    } catch (err) {
      console.error('Failed to delete session:', err)
    }
  }, [sessions, activeSessionId])

  const handleRenameSession = useCallback(async (sessionId, newTitle) => {
    try {
      await api.patch(`/api/v1/sessions/${sessionId}`, { title: newTitle })
      setSessions(prev =>
        prev.map(s => s.session_id === sessionId ? { ...s, title: newTitle } : s)
      )
    } catch (err) {
      console.error('Failed to rename session:', err)
    }
  }, [])

  // --- Query submission ---

  const handleSubmit = useCallback(async (query) => {
    if (!query.trim() || isStreaming) return

    let sessionId = activeSessionId

    // If no session, create one
    if (!sessionId) {
      try {
        const data = await api.post('/api/v1/sessions', { title: 'New Chat' })
        sessionId = data.session_id
        setActiveSessionId(sessionId)
      } catch (err) {
        console.error('Failed to create session:', err)
        return
      }
    }

    // Optimistic: add user message locally
    const userMsg = {
      role: 'user',
      content: query,
      timestamp: new Date().toISOString(),
      metadata: {},
    }
    setMessages(prev => [...prev, userMsg])

    try {
      const result = await streamQuery(query, sessionId)

      if (result) {
        // Add the completed assistant message
        const assistantMsg = {
          role: 'assistant',
          content: result.text,
          timestamp: new Date().toISOString(),
          metadata: result.metadata || {},
        }
        setMessages(prev => [...prev, assistantMsg])
      }

      // Refresh sessions (title may have been auto-generated)
      fetchSessions()

    } catch (err) {
      // Add error as an assistant message
      const errorMsg = {
        role: 'assistant',
        content: `Sorry, an error occurred: ${err.message}`,
        timestamp: new Date().toISOString(),
        metadata: {},
      }
      setMessages(prev => [...prev, errorMsg])
    }
  }, [activeSessionId, isStreaming, streamQuery, fetchSessions])

  // --- Feedback ---

  const handleFeedback = useCallback(async (messageIndex, feedbackType, correction) => {
    const msg = messages[messageIndex]
    if (!msg || msg.role !== 'assistant') return

    try {
      await api.post('/api/v1/feedback', {
        feedback_type: feedbackType,
        correction: correction || '',
        query: messages[messageIndex - 1]?.content || '', // Previous user message
        answer: msg.content,
        confidence_pct: msg.metadata?.confidence_pct || 0,
        confidence_level: msg.metadata?.confidence_level || '',
        num_sources: msg.metadata?.num_sources || 0,
        sources: msg.metadata?.sources || [],
        retrieval_strategy: msg.metadata?.retrieval_strategy || '',
      })
    } catch (err) {
      console.error('Failed to submit feedback:', err)
    }
  }, [messages])

  // --- Render ---

  return (
    <div className="flex h-full w-full relative">
      {/* Sidebar */}
      <div className={`
        flex-col border-r border-gray-800 bg-gray-800 shrink-0 transition-all duration-300
        ${showSidebar ? 'flex w-full absolute inset-0 z-20' : 'hidden md:flex md:w-80 md:static'}
      `}>
        {/* Mobile Sidebar Header */}
        <div className="md:hidden flex items-center justify-between p-4 border-b border-gray-700">
          <span className="font-semibold text-gray-200">Chats</span>
          <button 
            onClick={() => setShowSidebar(false)}
            className="p-2 text-gray-400 hover:text-white rounded-lg hover:bg-gray-700"
          >
            <X size={20} />
          </button>
        </div>

        <SessionList
          sessions={sessions}
          activeSessionId={activeSessionId}
          isLoading={isLoadingSessions}
          onSelectSession={handleSelectSession}
          onNewChat={handleNewChat}
          onDeleteSession={handleDeleteSession}
          onRenameSession={handleRenameSession}
        />
      </div>

      {/* Main chat area */}
      <div className={`
        flex-col min-w-0 bg-gray-900 flex-1
        ${showSidebar ? 'hidden md:flex' : 'flex'}
      `}>
        {/* Mobile Header with Hamburger */}
        <div className="md:hidden flex items-center p-4 border-b border-gray-800 shrink-0">
          <button
            onClick={() => setShowSidebar(true)}
            className="p-2 -ml-2 text-gray-400 hover:text-white rounded-lg hover:bg-gray-800 mr-3"
          >
            <Menu size={20} />
          </button>
          <span className="font-semibold text-gray-200 truncate">
            {sessions.find(s => s.session_id === activeSessionId)?.title || 'Chat'}
          </span>
        </div>

        <MessageList
          messages={messages}
          streamingText={streamingText}
          streamingMetadata={streamingMetadata}
          isStreaming={isStreaming}
          isLoading={isLoadingMessages}
          onFeedback={handleFeedback}
        />
        <ChatInput
          onSubmit={handleSubmit}
          disabled={isStreaming}
          sessionId={activeSessionId}
        />
      </div>
    </div>
  )
}
