// frontend/src/components/chat/SourceCitations.jsx
import { useState } from 'react'
import { ChevronRight, FileText } from 'lucide-react'

export default function SourceCitations({ sources }) {
  const [expanded, setExpanded] = useState(false)

  if (!sources || sources.length === 0) return null

  return (
    <div className="mt-3 border-t border-gray-800 pt-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-gray-200 transition-colors select-none"
      >
        <ChevronRight
          size={14}
          className={`transition-transform duration-200 ${expanded ? 'rotate-90' : ''}`}
        />
        <span className="font-medium">Sources ({sources.length})</span>
      </button>

      {expanded && (
        <div className="mt-2 space-y-2 animate-in fade-in slide-in-from-top-1 duration-200">
          {sources.map((src, i) => (
            <div key={i} className="bg-gray-800/50 border border-gray-800 rounded p-2 text-sm text-gray-300">
              <div className="flex items-start gap-2">
                <FileText size={14} className="mt-0.5 text-blue-400 shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="font-medium truncate" title={src.title}>
                    {src.title || src.source}
                  </div>
                  {src.section && (
                    <div className="text-xs text-gray-500 truncate">
                      Section: {src.section}
                    </div>
                  )}
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-[10px] uppercase bg-gray-700 text-gray-400 px-1.5 py-0.5 rounded">
                      {src.doc_type || 'Document'}
                    </span>
                    {src.relevance_score > 0 && (
                      <div className="flex items-center gap-1 text-[10px] text-gray-500">
                        <div className="w-12 h-1 bg-gray-700 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-blue-500/50" 
                            style={{ width: `${Math.min(src.relevance_score * 100, 100)}%` }}
                          />
                        </div>
                        {Math.round(src.relevance_score * 100)}%
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
