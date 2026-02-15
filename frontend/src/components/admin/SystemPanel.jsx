import React, { useState, useEffect } from 'react';
import {
  Activity,
  Server,
  Database,
  HardDrive,
  CheckCircle,
  XCircle,
  AlertTriangle,
  RefreshCw,
  Cpu
} from 'lucide-react';
import LogViewer from './LogViewer';

const SystemPanel = ({ tenantId, tenantList = [] }) => {
  const [status, setStatus] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  const fetchStatus = React.useCallback(async () => {
    setIsLoading(true);
    try {
      const url = new URL('/api/v1/system/status', window.location.origin);
      if (tenantId) url.searchParams.set('target_tenant_id', tenantId);

      const res = await fetch(url.toString(), {
        credentials: 'include'
      });
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  }, [tenantId]);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000); // Poll every 30s
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const getStatusColor = (s) => {
    switch (s) {
      case 'ok': return 'text-green-400';
      case 'degraded': return 'text-amber-400';
      case 'empty': return 'text-gray-400';
      default: return 'text-red-400';
    }
  };

  const StatusIcon = ({ status }) => {
    switch (status) {
      case 'ok': return <CheckCircle size={24} className="text-green-400" />;
      case 'degraded': return <AlertTriangle size={24} className="text-amber-400" />;
      case 'empty': return <Database size={24} className="text-gray-400" />;
      default: return <XCircle size={24} className="text-red-400" />;
    }
  };

  // Helper for boolean status cards
  const ConnectionStatus = ({ active, label, details }) => (
    <div className={`p-4 rounded-lg border ${
      active ? 'bg-green-900/10 border-green-900/30' : 'bg-red-900/10 border-red-900/30'
    }`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-300">{label}</span>
        {active ? (
          <CheckCircle size={16} className="text-green-400" />
        ) : (
          <XCircle size={16} className="text-red-400" />
        )}
      </div>
      <div className={`text-xl font-mono ${active ? 'text-green-400' : 'text-red-400'}`}>
        {active ? 'Connected' : 'Disconnected'}
      </div>
      {details && <div className="text-xs text-gray-500 mt-1">{details}</div>}
    </div>
  );

  if (!status) {
    return (
      <div className="h-full flex items-center justify-center text-gray-500">
        <RefreshCw size={24} className="animate-spin mb-2" />
        <span className="ml-2">Loading system status...</span>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-gray-900 text-gray-100 overflow-hidden p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-semibold text-white flex items-center gap-2">
          <Activity size={20} /> System Status
        </h2>
        <button 
          onClick={fetchStatus}
          className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
          title="Refresh"
        >
          <RefreshCw size={18} className={isLoading ? 'animate-spin' : ''} />
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {/* Overall Status */}
        <div className="p-4 rounded-lg bg-gray-800 border border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-400">System Status</span>
            <StatusIcon status={status.status} />
          </div>
          <div className={`text-2xl font-bold capitalize ${getStatusColor(status.status)}`}>
            {status.status}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            Global Health Check
          </div>
        </div>

        {/* Vector Index */}
        <div className={`p-4 rounded-lg border ${
          status.index_loaded ? 'bg-green-900/10 border-green-900/30' : 'bg-amber-900/10 border-amber-900/30'
        }`}>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-300">Vector Index</span>
            {status.index_loaded ? (
              <CheckCircle size={16} className="text-green-400" />
            ) : (
              <AlertTriangle size={16} className="text-amber-400" />
            )}
          </div>
          <div className={`text-xl font-mono ${status.index_loaded ? 'text-green-400' : 'text-amber-400'}`}>
            {status.index_loaded ? 'Loaded' : 'Not Loaded'}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {status.total_nodes} nodes / {status.total_documents} docs
          </div>
        </div>

        {/* Qdrant */}
        <ConnectionStatus
          active={status.qdrant_connected}
          label="Qdrant Vector DB"
          details={`${status.qdrant_vectors} vectors`}
        />

        {/* PostgreSQL */}
        <ConnectionStatus
          active={status.pg_connected}
          label="PostgreSQL"
          details={`${status.pg_tables} tables`}
        />
      </div>

      {/* Tenant Stats */}
      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700 mb-6">
        <h3 className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
          <Database size={16} className="text-blue-400" />
          Tenant Statistics ({tenantList.find(t => t.tenant_id === status.tenant_id)?.display_name || status.tenant_id})
        </h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-900/50 p-3 rounded border border-gray-700">
            <div className="text-xs text-gray-500 uppercase tracking-wider cursor-help" title="Tenant Documents / Total System Documents">
              Documents (Tenant / Total)
            </div>
            <div className="text-xl font-mono text-gray-200">
              {status.tenant_documents} <span className="text-gray-500 text-base">/ {status.total_documents}</span>
            </div>
          </div>
          <div className="bg-gray-900/50 p-3 rounded border border-gray-700">
            <div className="text-xs text-gray-500 uppercase tracking-wider cursor-help" title="Tenant Chunks / Total System Chunks">
              Text Chunks (Tenant / Total)
            </div>
            <div className="text-xl font-mono text-gray-200">
              {status.tenant_nodes} <span className="text-gray-500 text-base">/ {status.total_nodes}</span>
            </div>
          </div>
        </div>
      </div>

      <LogViewer />
    </div>
  );
};

export default SystemPanel;
