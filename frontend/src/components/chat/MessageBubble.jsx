import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { User, Bot, Copy, Check } from 'lucide-react'
import ConfidenceBadge from './ConfidenceBadge'
import SourceCitations from './SourceCitations'
import FeedbackControls from './FeedbackControls'

export default function MessageBubble({ message, isStreaming, onFeedback }) {
  const isUser = message.role === 'user'
  const [copied, setCopied] = useState(false)
  
  // Format timestamp (HH:MM if today, Date if older)
  const timestamp = new Date(message.timestamp).toLocaleString(undefined, {
    hour: 'numeric', minute: 'numeric'
  })

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy', err)
    }
  }

  return (
    <div className={`flex w-full mb-6 ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`flex max-w-[85%] md:max-w-[75%] gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
        
        {/* Avatar */}
        <div className={`shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser ? 'bg-blue-600' : 'bg-emerald-600'
        }`}>
          {isUser ? <User size={16} className="text-white" /> : <Bot size={16} className="text-white" />}
        </div>

        {/* Message Content */}
        <div className={`flex flex-col min-w-0 ${isUser ? 'items-end' : 'items-start'}`}>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-medium text-gray-400">
              {isUser ? 'You' : 'Maritime Assistant'}
            </span>
            <span className="text-[10px] text-gray-600">{timestamp}</span>
          </div>

          <div className={`rounded-2xl px-4 py-3 text-sm leading-relaxed overflow-hidden ${
            isUser 
              ? 'bg-blue-900/20 border border-blue-800/30 text-gray-100 rounded-tr-sm' 
              : 'bg-transparent text-gray-100 -ml-4 pl-4' // Assistant minimal style
          }`}>
            {isUser ? (
              <div className="whitespace-pre-wrap">{message.content}</div>
            ) : (
              <div className="markdown-content">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    table: ({ children }) => (
                      <div className="overflow-x-auto my-2">
                        <table className="min-w-full text-sm border-collapse border border-gray-700">
                          {children}
                        </table>
                      </div>
                    ),
                    th: ({ children }) => (
                      <th className="border border-gray-700 bg-gray-800 px-3 py-1.5 text-left font-medium text-gray-300">
                        {children}
                      </th>
                    ),
                    td: ({ children }) => (
                      <td className="border border-gray-700 px-3 py-1.5 text-gray-300">
                        {children}
                      </td>
                    ),
                    code: ({ inline, children, ...props }) => (
                      inline
                        ? <code className="bg-gray-800 text-blue-300 px-1.5 py-0.5 rounded text-sm" {...props}>{children}</code>
                        : <pre className="bg-gray-800 border border-gray-700 rounded-lg p-3 overflow-x-auto my-2 block">
                            <code className="text-sm text-gray-300" {...props}>{children}</code>
                          </pre>
                    ),
                    p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
                    ul: ({ children }) => <ul className="list-disc pl-4 mb-3 space-y-1">{children}</ul>,
                    ol: ({ children }) => <ol className="list-decimal pl-4 mb-3 space-y-1">{children}</ol>,
                    a: ({ children, href }) => (
                      <a href={href} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">
                        {children}
                      </a>
                    ),
                    blockquote: ({ children }) => (
                      <blockquote className="border-l-4 border-gray-700 pl-3 italic text-gray-400 my-2">
                        {children}
                      </blockquote>
                    )
                  }}
                >
                  {message.content + (isStreaming ? ' ' : '')}
                </ReactMarkdown>
                {isStreaming && (
                  <span className="inline-block w-2 h-4 align-middle bg-blue-500 animate-pulse ml-0.5" />
                )}
              </div>
            )}
          </div>

          {!isUser && !isStreaming && (
            <div className="ml-1 mt-1 space-y-1 w-full max-w-2xl">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <ConfidenceBadge 
                    level={message.metadata?.confidence_level}
                    percentage={message.metadata?.confidence_pct}
                  />
                  <FeedbackControls onFeedback={(type, correction) => onFeedback(type, correction)} disabled={false} />
                  
                  {/* Copy Button */}
                  <button 
                    onClick={handleCopy}
                    className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-300 transition-colors"
                    title="Copy response"
                  >
                    {copied ? <Check size={14} className="text-green-500" /> : <Copy size={14} />}
                    {copied ? 'Copied' : 'Copy'}
                  </button>
                </div>
              </div>
              <SourceCitations sources={message.metadata?.sources} />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
