import React, { useState, useEffect, useCallback } from 'react';
import { Trash2, RefreshCw, FileText, CheckCircle, AlertTriangle, XCircle, Search } from 'lucide-react';

const DocumentsPanel = ({ tenantId }) => {
  const [documents, setDocuments] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedFiles, setSelectedFiles] = useState(new Set());
  const [sortField, setSortField] = useState('filename');
  const [sortDirection, setSortDirection] = useState('asc');
  const [searchQuery, setSearchQuery] = useState('');

  const fetchDocuments = useCallback(async () => {
    setIsLoading(true);
    try {
      const url = new URL('/api/v1/documents', window.location.origin);
      if (tenantId) url.searchParams.set('target_tenant_id', tenantId);

      const res = await fetch(url.toString());
      if (!res.ok) throw new Error('Failed to load documents');
      
      const data = await res.json();
      setDocuments(data.documents || []);
      setSelectedFiles(new Set()); // Clear selection on refresh
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  }, [tenantId]);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  const handleDelete = async (filename) => {
    if (!window.confirm(`Are you sure you want to delete "${filename}"?`)) return;

    try {
      const url = new URL(`/api/v1/documents/${filename}`, window.location.origin);
      if (tenantId) url.searchParams.set('target_tenant_id', tenantId);

      const res = await fetch(url.toString(), { method: 'DELETE' });
      if (!res.ok) throw new Error('Delete failed');

      fetchDocuments();
    } catch (err) {
      alert('Failed to delete document');
    }
  };

  const handleBatchDelete = async () => {
    const count = selectedFiles.size;
    if (!count) return;
    if (!window.confirm(`Are you sure you want to delete ${count} selected documents?`)) return;

    try {
      const url = new URL('/api/v1/documents/batch-delete', window.location.origin);
      if (tenantId) url.searchParams.set('target_tenant_id', tenantId);

      const res = await fetch(url.toString(), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filenames: Array.from(selectedFiles) }),
      });

      if (!res.ok) throw new Error('Batch delete failed');

      fetchDocuments();
    } catch (err) {
      alert('Failed to delete documents');
    }
  };

  // Move filtering logic before it's used
  const filteredDocs = documents.filter(doc => 
    doc.filename.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const sortedDocs = [...filteredDocs].sort((a, b) => {
    let aVal = a[sortField];
    let bVal = b[sortField];

    // Specialized sorting for specific fields
    if (sortField === 'file_size_bytes') {
      aVal = Number(aVal) || 0;
      bVal = Number(bVal) || 0;
    } else if (sortField === 'chunk_count') {
      aVal = Number(aVal) || 0;
      bVal = Number(bVal) || 0;
    } else {
      aVal = String(aVal || '').toLowerCase();
      bVal = String(bVal || '').toLowerCase();
    }

    if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1;
    if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1;
    return 0;
  });

  const toggleSelect = (filename) => {
    const newSelected = new Set(selectedFiles);
    if (newSelected.has(filename)) {
      newSelected.delete(filename);
    } else {
      newSelected.add(filename);
    }
    setSelectedFiles(newSelected);
  };

  const toggleSelectAll = () => {
    if (selectedFiles.size === sortedDocs.length) {
      setSelectedFiles(new Set());
    } else {
      setSelectedFiles(new Set(sortedDocs.map(d => d.filename)));
    }
  };

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  const formatBytes = (bytes, decimals = 2) => {
    if (!bytes) return '0 B';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'success': return <CheckCircle size={14} className="text-green-500" />;
      case 'warning': return <AlertTriangle size={14} className="text-amber-500" />;
      case 'failed': return <XCircle size={14} className="text-red-500" />;
      default: return <div className="w-2 h-2 rounded-full bg-gray-500" />;
    }
  };

  return (
    <div className="h-full flex flex-col bg-gray-900 text-gray-100">
      {/* Header / Toolbar */}
      <div className="flex flex-col gap-4 mb-4">
        <div className="flex justify-between items-center">
          <h2 className="text-xl font-semibold text-white flex items-center gap-2">
            <FileText size={20} /> Documents
            <span className="text-sm font-normal text-gray-400 ml-2">({documents.length})</span>
          </h2>
          <button 
            onClick={fetchDocuments}
            className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
            title="Refresh"
          >
            <RefreshCw size={18} className={isLoading ? 'animate-spin' : ''} />
          </button>
        </div>

        <div className="flex gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={16} />
            <input
              type="text"
              placeholder="Search documents..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-gray-800 text-white text-sm rounded-md pl-9 pr-3 py-2 border border-gray-700 focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          </div>
          {selectedFiles.size > 0 && (
            <button
              onClick={handleBatchDelete}
              className="flex items-center gap-2 px-4 py-2 bg-red-900/30 text-red-400 border border-red-800 rounded-md hover:bg-red-900/50 transition-colors text-sm"
            >
              <Trash2 size={16} />
              Delete {selectedFiles.size} selected
            </button>
          )}
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-hidden border border-gray-800 rounded-lg flex flex-col">
        <div className="overflow-y-auto flex-1">
          <table className="w-full text-left border-collapse">
            <thead className="bg-gray-800/50 text-gray-400 text-xs uppercase tracking-wider sticky top-0 z-10">
              <tr>
                <th className="px-4 py-3 w-10 border-b border-gray-800">
                  <input
                    type="checkbox"
                    checked={documents.length > 0 && selectedFiles.size === sortedDocs.length}
                    onChange={toggleSelectAll}
                    className="rounded border-gray-600 bg-gray-700 text-blue-500 focus:ring-offset-gray-900"
                  />
                </th>
                <th 
                  className="px-4 py-3 font-medium cursor-pointer hover:text-white border-b border-gray-800 group"
                  onClick={() => handleSort('filename')}
                >
                  <div className="flex items-center gap-1">
                    Filename
                    {sortField === 'filename' && <span className="text-[10px]">{sortDirection === 'asc' ? '▲' : '▼'}</span>}
                  </div>
                </th>
                <th 
                  className="px-4 py-3 font-medium cursor-pointer hover:text-white border-b border-gray-800 w-24"
                  onClick={() => handleSort('file_size_bytes')}
                >
                  <div className="flex items-center gap-1">
                    Size
                    {sortField === 'file_size_bytes' && <span className="text-[10px]">{sortDirection === 'asc' ? '▲' : '▼'}</span>}
                  </div>
                </th>
                <th 
                  className="px-4 py-3 font-medium cursor-pointer hover:text-white border-b border-gray-800 w-24 text-center"
                  onClick={() => handleSort('chunk_count')}
                >
                  <div className="flex items-center gap-1 justify-center">
                    Chunks
                    {sortField === 'chunk_count' && <span className="text-[10px]">{sortDirection === 'asc' ? '▲' : '▼'}</span>}
                  </div>
                </th>
                <th className="px-4 py-3 font-medium w-20 border-b border-gray-800 text-center">Status</th>
                <th className="px-4 py-3 font-medium w-20 border-b border-gray-800 text-center">Cached</th>
                <th className="px-4 py-3 font-medium w-16 border-b border-gray-800 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {isLoading && documents.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-4 py-8 text-center text-gray-500">
                    <RefreshCw size={24} className="animate-spin mx-auto mb-2" />
                    Loading documents...
                  </td>
                </tr>
              ) : sortedDocs.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-4 py-8 text-center text-gray-500 italic">
                    {searchQuery ? 'No documents match your search.' : 'No documents uploaded yet.'}
                  </td>
                </tr>
              ) : (
                sortedDocs.map((doc) => (
                  <tr key={doc.filename} className="hover:bg-gray-800/30 transition-colors group">
                    <td className="px-4 py-3">
                      <input
                        type="checkbox"
                        checked={selectedFiles.has(doc.filename)}
                        onChange={() => toggleSelect(doc.filename)}
                        className="rounded border-gray-600 bg-gray-700 text-blue-500 focus:ring-offset-gray-900"
                      />
                    </td>
                    <td className="px-4 py-3 text-sm font-medium text-gray-200 truncate max-w-xs" title={doc.filename}>
                      {doc.filename}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-400 font-mono">
                      {formatBytes(doc.file_size_bytes)}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-400 text-center">
                      {doc.chunk_count || '-'}
                    </td>
                    <td className="px-4 py-3 text-center">
                      <div className="flex justify-center" title={doc.extraction_status}>
                        {getStatusIcon(doc.extraction_status)}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-center text-xs">
                      {doc.is_cached ? (
                        <span className="text-blue-400 bg-blue-900/20 px-1.5 py-0.5 rounded">YES</span>
                      ) : (
                        <span className="text-gray-600">-</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <button
                        onClick={() => handleDelete(doc.filename)}
                        className="text-gray-600 hover:text-red-400 p-1 opacity-0 group-hover:opacity-100 transition-opacity"
                        title="Delete"
                      >
                        <Trash2 size={16} />
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default DocumentsPanel;
