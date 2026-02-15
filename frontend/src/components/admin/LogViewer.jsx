import React, { useState, useEffect, useRef } from 'react';

export default function LogViewer({ token }) {
  const [lines, setLines] = useState([]);
  const [autoScroll, setAutoScroll] = useState(true);
  const [connected, setConnected] = useState(false);
  const containerRef = useRef(null);

  useEffect(() => {
    // Note: EventSource doesn't support custom headers (Auth), so we use fetch + ReadableStream
    if (!token) return;

    const controller = new AbortController();

    async function connect() {
      try {
        const res = await fetch('/api/v1/system/logs/stream', {
          headers: { 
            'Authorization': `Bearer ${token}` 
          },
          signal: controller.signal,
          credentials: 'include',
        });

        if (!res.ok) {
            console.error("Log stream failed:", res.status, res.statusText);
            setConnected(false);
            return;
        }

        setConnected(true);
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // Parse SSE lines
          const parts = buffer.split('\n\n');
          buffer = parts.pop() || '';  // last incomplete chunk stays in buffer

          for (const part of parts) {
            if (part.startsWith('data: ')) {
              const line = part.slice(6);
              if (line !== '---' && !line.startsWith(': keepalive')) {  // skip catchup separator and keepalives
                setLines(prev => {
                  const next = [...prev, line];
                  // Cap at 1000 lines in the UI to prevent memory bloat
                  return next.length > 1000 ? next.slice(-1000) : next;
                });
              }
            }
          }
        }
      } catch (err) {
        if (err.name !== 'AbortError') {
          console.error('Log stream error:', err);
          setConnected(false);
        }
      } finally {
          setConnected(false);
      }
    }

    connect();

    return () => {
      controller.abort();
      setConnected(false);
    };
  }, [token]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (autoScroll && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [lines, autoScroll]);

  return (
    <div className="mt-6">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
          Server Logs
          <span className={`inline-block w-2 h-2 rounded-full ${
            connected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
          }`} />
        </h3>
        <div className="flex items-center gap-3 text-xs text-gray-500">
          <label className="flex items-center gap-1 cursor-pointer hover:text-gray-300 transition-colors">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={e => setAutoScroll(e.target.checked)}
              className="rounded border-gray-700 bg-gray-800 text-blue-500 focus:ring-0 focus:ring-offset-0"
            />
            Auto-scroll
          </label>
          <button
            onClick={() => setLines([])}
            className="text-gray-500 hover:text-gray-300 transition-colors"
          >
            Clear
          </button>
          <span>{lines.length} lines</span>
        </div>
      </div>

      <div
        ref={containerRef}
        className="bg-black/50 border border-gray-800 rounded-lg p-3
                   h-64 overflow-y-auto font-mono text-xs leading-5
                   text-gray-400 select-text scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-transparent"
      >
        {lines.length === 0 ? (
          <div className="h-full flex items-center justify-center text-gray-600 italic">
            {connected ? 'Waiting for log entries...' : 'Connecting to log stream...'}
          </div>
        ) : (
          lines.map((line, i) => (
            <div
              key={i}
              className={`whitespace-pre-wrap break-all ${
                line.includes('ERROR') ? 'text-red-400' :
                line.includes('WARNING') ? 'text-amber-400' :
                line.includes('INFO') ? 'text-gray-400' :
                'text-gray-500'
              }`}
            >
              {line}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
