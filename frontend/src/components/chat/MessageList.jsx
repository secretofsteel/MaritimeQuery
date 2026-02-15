// frontend/src/components/chat/MessageList.jsx
import { useEffect, useRef } from 'react'
import { Anchor } from 'lucide-react'
import MessageBubble from './MessageBubble'

export default function MessageList({ 
  messages, 
  streamingText, 
  streamingMetadata, 
  isStreaming, 
  isLoading,
  onFeedback 
}) {
  const endRef = useRef(null)
  
  // Auto-scroll to bottom
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingText, isStreaming])

  if (isLoading && messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-gray-500 animate-pulse">Loading messages...</div>
      </div>
    )
  }

  if (messages.length === 0 && !isStreaming) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-gray-500 p-8">
        <div className="w-16 h-16 bg-gray-800 rounded-full flex items-center justify-center mb-4">
          <Anchor size={32} className="text-blue-500" />
        </div>
        <h3 className="text-lg font-medium text-gray-300">Welcome Aboard</h3>
        <p className="mt-2 text-center max-w-sm">
          Ask questions about maritime regulations, safety codes, and operational procedures.
        </p>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto p-4 md:p-6 custom-scrollbar">
      <div className="max-w-4xl mx-auto">
        {messages.map((msg, i) => (
          <MessageBubble 
            key={i} 
            message={msg} 
            isStreaming={false}
            onFeedback={(type, corr) => onFeedback(i, type, corr)}
          />
        ))}

        {isStreaming && (
          <MessageBubble 
            message={{ 
              role: 'assistant', 
              content: streamingText, 
              timestamp: new Date().toISOString(),
              metadata: streamingMetadata
            }} 
            isStreaming={true}
          />
        )}
        
        <div ref={endRef} className="h-4" />
      </div>
    </div>
  )
}
