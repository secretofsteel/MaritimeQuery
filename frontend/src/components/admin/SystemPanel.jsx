import React, { useState, useEffect } from 'react';
import { Activity, Database, Server, HardDrive, RefreshCw, CheckCircle, AlertTriangle, XCircle } from 'lucide-react';

const SystemPanel = ({ tenantId }) => {
  const [status, setStatus] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000); // Poll every 30s
    return () => clearInterval(interval);
  }, [tenantId]);

  const fetchStatus = async () => {
    setIsLoading(true);
    try {
      const res = await fetch('/api/v1/system/status');
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const getStatusColor = (s) => {
    switch (s) {
      case 'healthy': return 'text-green-400';
      case 'degraded': return 'text-amber-400';
      case 'empty': return 'text-gray-400';
      default: return 'text-red-400';
    }
  };

  const getStatusIcon = (s) => {
    switch (s) {
      case 'healthy': return <CheckCircle size={20} className="text-green-400" />;
      case 'degraded': return <AlertTriangle size={20} className="text-amber-400" />;
      case 'empty': return <Database size={20} className="text-gray-400" />;
      default: return <XCircle size={20} className="text-red-400" />;
    }
  };

  return (
    <div className="h-full flex flex-col bg-gray-900 text-gray-100 overflow-y-auto">
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

      {!status ? (
        <div className="text-center p-8 text-gray-500">Loading system status...</div>
      ) : (
        <div className="space-y-6">
          {/* Overall Status Badge */}
          <div className="bg-gray-800 p-4 rounded-lg border border-gray-700 flex items-center gap-4">
            <div className={`p-3 rounded-full bg-gray-900 ${getStatusColor(status.status)}`}>
              {getStatusIcon(status.status)}
            </div>
            <div>
              <div className="text-sm text-gray-400 uppercase font-medium">System Health</div>
              <div className={`text-2xl font-bold capitalize ${getStatusColor(status.status)}`}>
                {status.status}
              </div>
            </div>
          </div>

          {/* Grid of Components */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Index Status */}
            <StatusCard 
              title="Vector Index" 
              icon={Database}
              metrics={[
                { label: 'Status', value: status.index?.loaded ? 'Loaded' : 'Not Loaded', hl: status.index?.loaded },
                { label: 'Total Nodes', value: status.index?.total_nodes || 0 },
                { label: 'Total Docs', value: status.index?.total_documents || 0 },
              ]}
            />

            {/* Qdrant Status */}
            <StatusCard 
              title="Vector Database" 
              icon={Server}
              metrics={[
                { label: 'Connection', value: status.qdrant?.connected ? 'Connected' : 'Error', hl: status.qdrant?.connected },
                { label: 'Collection', value: status.qdrant?.collection_name || '-' },
                { label: 'Vectors', value: status.qdrant?.points_count || 0 },
              ]}
            />

            {/* Postgres Status */}
            <StatusCard 
              title="Relational DB" 
              icon={HardDrive}
              metrics={[
                { label: 'Connection', value: status.database?.connected ? 'Connected' : 'Error', hl: status.database?.connected },
                { label: 'Tenant Docs', value: status.database?.tenant_doc_count || 0 },
              ]}
            />
          </div>
        </div>
      )}
    </div>
  );
};

const StatusCard = ({ title, icon: Icon, metrics }) => (
  <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
    <div className="flex items-center gap-2 mb-4 text-blue-400">
      <Icon size={18} />
      <h3 className="font-medium">{title}</h3>
    </div>
    <div className="space-y-3">
      {metrics.map((m, i) => (
        <div key={i} className="flex justify-between text-sm border-b border-gray-700/50 pb-2 last:border-0 last:pb-0">
          <span className="text-gray-400">{m.label}</span>
          <span className={`font-mono ${m.hl === true ? 'text-green-400' : m.hl === false ? 'text-red-400' : 'text-gray-200'}`}>
            {m.value}
          </span>
        </div>
      ))}
    </div>
  </div>
);

export default SystemPanel;
