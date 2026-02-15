// frontend/src/components/chat/FeedbackControls.jsx
import { useState } from 'react'
import { ThumbsUp, ThumbsDown } from 'lucide-react'

export default function FeedbackControls({ onFeedback, disabled }) {
  const [submitted, setSubmitted] = useState(null) // null, 'positive', 'negative'
  const [showCorrection, setShowCorrection] = useState(false)
  const [correction, setCorrection] = useState('')

  const handlePositive = () => {
    if (submitted || disabled) return
    setSubmitted('positive')
    onFeedback('positive', '')
  }

  const handleNegativeClick = () => {
    if (submitted || disabled) return
    setShowCorrection(true)
  }

  const submitCorrection = () => {
    setSubmitted('negative')
    setShowCorrection(false)
    onFeedback('negative', correction)
  }

  return (
    <div className="flex flex-col gap-2 mt-2">
      <div className="flex items-center gap-2">
        <button
          onClick={handlePositive}
          disabled={!!submitted || disabled}
          className={`p-1.5 rounded transition-colors ${
            submitted === 'positive'
              ? 'text-green-400 bg-green-900/20'
              : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800'
          }`}
          title="Good answer"
        >
          <ThumbsUp size={14} />
        </button>

        <button
          onClick={handleNegativeClick}
          disabled={!!submitted || disabled}
          className={`p-1.5 rounded transition-colors ${
            submitted === 'negative'
              ? 'text-red-400 bg-red-900/20'
              : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800'
          }`}
          title="Bad answer"
        >
          <ThumbsDown size={14} />
        </button>
      </div>

      {showCorrection && (
        <div className="animate-in fade-in slide-in-from-top-2 duration-200">
          <div className="flex gap-2">
            <input
              type="text"
              value={correction}
              onChange={(e) => setCorrection(e.target.value)}
              placeholder="How should this be answered?"
              className="flex-1 bg-gray-800 border border-gray-700 rounded text-sm text-gray-200 px-2 py-1 focus:outline-none focus:border-blue-500 placeholder-gray-600"
              autoFocus
              onKeyDown={(e) => e.key === 'Enter' && submitCorrection()}
            />
            <button
              onClick={submitCorrection}
              disabled={!correction.trim()}
              className="bg-blue-600 text-white px-2 py-1 rounded text-xs font-medium hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Submit
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
