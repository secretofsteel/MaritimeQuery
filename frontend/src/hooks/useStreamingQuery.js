// frontend/src/hooks/useStreamingQuery.js
import { useState, useCallback, useRef } from 'react'

/**
 * Hook for streaming queries via SSE over fetch.
 *
 * Uses fetch() with ReadableStream to consume SSE events from
 * POST /api/v1/query/stream. This avoids the EventSource limitation
 * (GET-only) while still processing server-sent events.
 *
 * Returns:
 *   streamQuery(query, sessionId) - trigger a streaming query
 *   isStreaming - whether a stream is currently active
 *   streamingText - accumulated answer text (updates on each token)
 *   streamingMetadata - metadata object (populated when stream completes)
 *   error - error message if stream failed
 *   cancel - abort the current stream
 */
export function useStreamingQuery() {
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamingText, setStreamingText] = useState('')
  const [streamingMetadata, setStreamingMetadata] = useState(null)
  const [error, setError] = useState(null)
  const abortRef = useRef(null)

  const cancel = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort()
      abortRef.current = null
    }
    setIsStreaming(false)
  }, [])

  const streamQuery = useCallback(async (query, sessionId, useContext = true) => {
    // Reset state
    setStreamingText('')
    setStreamingMetadata(null)
    setError(null)
    setIsStreaming(true)

    // Create abort controller for cancellation
    const controller = new AbortController()
    abortRef.current = controller

    try {
      const res = await fetch('/api/v1/query/stream', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          session_id: sessionId,
          use_conversation_context: useContext,
        }),
        signal: controller.signal,
      })

      if (!res.ok) {
        const detail = await res.text()
        throw new Error(detail || `HTTP ${res.status}`)
      }

      // Read the SSE stream
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let accumulatedText = ''
      let metadata = null

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        // Parse SSE events from buffer
        // Format: "event: <type>\ndata: <json>\n\n"
        const events = buffer.split('\n\n')
        buffer = events.pop() // Keep incomplete event in buffer

        for (const eventStr of events) {
          if (!eventStr.trim()) continue

          const lines = eventStr.split('\n')
          let eventType = ''
          let eventData = ''

          for (const line of lines) {
            if (line.startsWith('event: ')) {
              eventType = line.slice(7).trim()
            } else if (line.startsWith('data: ')) {
              eventData = line.slice(6)
            }
          }

          if (!eventType || !eventData) continue

          try {
            const parsed = JSON.parse(eventData)

            switch (eventType) {
              case 'token':
                accumulatedText += parsed.text || ''
                setStreamingText(accumulatedText)
                break

              case 'metadata':
                metadata = parsed
                setStreamingMetadata(parsed)
                break

              case 'done':
                // Stream complete
                break

              case 'error':
                throw new Error(parsed.detail || 'Stream error')
            }
          } catch (parseErr) {
            if (parseErr.message !== 'Stream error' &&
                !parseErr.message.startsWith('Stream error')) {
              console.warn('Failed to parse SSE event:', eventData, parseErr)
            } else {
              throw parseErr
            }
          }
        }
      }

      // Return the final result for the caller to use
      return { text: accumulatedText, metadata }

    } catch (err) {
      if (err.name === 'AbortError') {
        // User cancelled — not an error
        return null
      }
      setError(err.message)
      throw err
    } finally {
      setIsStreaming(false)
      abortRef.current = null
    }
  }, [])

  return {
    streamQuery,
    isStreaming,
    streamingText,
    streamingMetadata,
    error,
    cancel,
  }
}
