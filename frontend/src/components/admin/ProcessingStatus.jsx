import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Loader2, CheckCircle, AlertTriangle, XCircle, ChevronDown, ChevronRight } from 'lucide-react';

const ProcessingStatus = ({ tenantId, isActive, onComplete }) => {
  const [status, setStatus] = useState(null);
  const [report, setReport] = useState(null);
  const [isReportExpanded, setIsReportExpanded] = useState(false);
  const pollIntervalRef = useRef(null);

  const fetchStatus = useCallback(async () => {
    try {
      const url = new URL('/api/v1/documents/process/status', window.location.origin);
      if (tenantId) url.searchParams.set('target_tenant_id', tenantId);

      const res = await fetch(url.toString(), { credentials: 'include' });
      if (res.ok) {
        const data = await res.json();
        setStatus(data);

        // Check for completion
        if (data.status === 'completed' || data.status === 'failed') {
          if (onComplete) onComplete();
        }
      } else if (res.status === 404) {
        // Job gone/finished
        if (onComplete) onComplete();
      }
    } catch (err) {
      console.error('Status poll failed:', err);
    }
  }, [tenantId, onComplete]);

  const fetchReport = useCallback(async () => {
    try {
      const url = new URL('/api/v1/documents/process/report', window.location.origin);
      if (tenantId) url.searchParams.set('target_tenant_id', tenantId);

      const res = await fetch(url.toString(), { credentials: 'include' });
      if (res.ok) {
        const data = await res.json();
        setReport(prev => {
          // Auto-expand if new report
          if (data && (!prev || prev.timestamp !== data.timestamp)) {
             setIsReportExpanded(true);
          }
          return data;
        });
      } else {
        setReport(null);
      }
    } catch (err) {
      console.error('Report fetch failed:', err);
    }
  }, [tenantId]);

  const stopPolling = useCallback(() => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  }, []);

  const startPolling = useCallback(() => {
    // Initial fetch
    fetchStatus();
    // Poll every 1s
    pollIntervalRef.current = setInterval(fetchStatus, 1000);
  }, [fetchStatus]);

  // Poll for status while active
  useEffect(() => {
    if (isActive) {
      startPolling();
    } else {
      stopPolling();
      // If we just stopped (and were active), fetch report
      if (!isActive) fetchReport();
    }
    return () => stopPolling();
  }, [isActive, startPolling, stopPolling, fetchReport]);

  // Fetch report on mount
  useEffect(() => {
    fetchReport();
  }, [fetchReport]);

  // Render helpers
  const getProgressWidth = () => {
    if (!status || !status.progress) return '0%';
    const { current, total } = status.progress;
    if (!total) return '0%';
    return `${Math.min(100, (current / total) * 100)}%`;
  };

  return (
    <div className="space-y-4">
      {/* Active Processing Card */}
      {isActive && status && (
        <div className="bg-gray-800 border border-blue-900/50 rounded-lg p-5 shadow-lg animate-pulse-subtle">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-blue-400 font-medium flex items-center gap-2">
              <Loader2 className="animate-spin" size={18} />
              Processing Library...
            </h3>
            <span className="text-xs bg-blue-900/30 text-blue-300 px-2 py-1 rounded-full font-mono">
              {status.elapsed_time || '0s'}
            </span>
          </div>

          {/* Progress Bar */}
          <div className="h-2 bg-gray-700 rounded-full overflow-hidden mb-2">
            <div 
              className="h-full bg-blue-500 transition-all duration-300 ease-out"
              style={{ width: getProgressWidth() }}
            />
          </div>
          
          <div className="flex justify-between text-xs text-gray-400 font-mono">
            <span>{status.current_step || 'Initializing...'}</span>
            <span>
              {status.progress?.current || 0} / {status.progress?.total || 0}
            </span>
          </div>

          <div className="mt-4 p-3 bg-gray-900/50 rounded text-xs text-gray-300 font-mono overflow-hidden whitespace-nowrap text-ellipsis">
            {status.current_file ? `> ${status.current_file}` : '> waiting...'}
          </div>
        </div>
      )}

      {/* Completion Report */}
      {!isActive && report && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg overflow-hidden">
          <div 
            className="p-4 flex items-center justify-between cursor-pointer hover:bg-gray-750 transition-colors"
            onClick={() => setIsReportExpanded(!isReportExpanded)}
          >
            <div className="flex items-center gap-3">
              {report.failed > 0 ? (
                <XCircle className="text-red-500" size={20} />
              ) : report.warnings > 0 ? (
                <AlertTriangle className="text-amber-500" size={20} />
              ) : (
                <CheckCircle className="text-green-500" size={20} />
              )}
              <div>
                <h3 className="text-sm font-medium text-gray-200">
                  Last Processing Report
                </h3>
                <p className="text-xs text-gray-400">
                  {new Date(report.timestamp).toLocaleString()} • {report.total} files
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <div className="flex gap-2 text-xs">
                <span className="text-green-400">{report.successful} success</span>
                <span className="text-gray-600">|</span>
                <span className={report.warnings > 0 ? 'text-amber-400' : 'text-gray-500'}>
                  {report.warnings} warning
                </span>
                <span className="text-gray-600">|</span>
                <span className={report.failed > 0 ? 'text-red-400' : 'text-gray-500'}>
                  {report.failed} failed
                </span>
              </div>
              {isReportExpanded ? <ChevronDown size={18} className="text-gray-500" /> : <ChevronRight size={18} className="text-gray-500" />}
            </div>
          </div>

          {isReportExpanded && (
            <div className="border-t border-gray-700 bg-gray-900/30 max-h-96 overflow-y-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-gray-800/50 text-gray-400 uppercase tracking-wider sticky top-0">
                  <tr>
                    <th className="px-4 py-2 font-medium">File</th>
                    <th className="px-4 py-2 font-medium w-24">Status</th>
                    <th className="px-4 py-2 font-medium w-20 text-right">Time</th>
                    <th className="px-4 py-2 font-medium">Details</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800">
                  {report.details && report.details.map((item, idx) => (
                    <tr key={idx} className="hover:bg-gray-800/30">
                      <td className="px-4 py-2 text-gray-300 font-medium truncate max-w-xs" title={item.filename}>
                        {item.filename}
                      </td>
                      <td className="px-4 py-2">
                        <span className={`px-2 py-0.5 rounded-full text-[10px] uppercase font-bold ${
                          item.status === 'success' ? 'bg-green-900/30 text-green-400' :
                          item.status === 'warning' ? 'bg-amber-900/30 text-amber-400' :
                          'bg-red-900/30 text-red-400'
                        }`}>
                          {item.status}
                        </span>
                      </td>
                      <td className="px-4 py-2 text-right text-gray-500 font-mono">
                        {item.duration?.toFixed(1)}s
                      </td>
                      <td className="px-4 py-2 text-gray-500 italic truncate max-w-xs" title={item.error || item.warning}>
                        {item.error || item.warning || '-'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {!isActive && !report && (
        <div className="text-center p-8 border border-gray-800 rounded-lg border-dashed text-gray-500 text-sm">
          No processing history available.
        </div>
      )}
    </div>
  );
};

export default ProcessingStatus;
