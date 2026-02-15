// frontend/src/components/chat/ConfidenceBadge.jsx

export default function ConfidenceBadge({ level, percentage, note }) {
  if (!level) return null

  const getColors = () => {
    switch (level?.toUpperCase()) {
      case 'HIGH':
        return 'bg-green-900/40 text-green-400 border-green-800'
      case 'MEDIUM':
        return 'bg-amber-900/40 text-amber-400 border-amber-800'
      case 'LOW':
        return 'bg-red-900/40 text-red-400 border-red-800'
      default:
        return 'bg-gray-800 text-gray-400 border-gray-700'
    }
  }

  return (
    <div className="inline-flex items-center gap-1.5 mt-2 select-none group relative">
      <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${getColors()} uppercase tracking-wide`}>
        ● {level} {percentage ? `${percentage}%` : ''}
      </span>
      
      {note && (
        <span className="text-xs text-gray-500 group-hover:text-gray-300 transition-colors cursor-help">
          ⓘ
          <span className="absolute bottom-full left-0 mb-2 w-48 p-2 bg-gray-900 border border-gray-700 rounded shadow-xl text-xs text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
            {note}
          </span>
        </span>
      )}
    </div>
  )
}
