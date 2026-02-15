import React, { useState, useEffect, useCallback } from 'react';
import { Trash2, RefreshCw, FileText, CheckCircle, AlertTriangle, XCircle, Search, Edit2, X, Save } from 'lucide-react';
import { useDocTypes } from '../../hooks/useDocTypes';

const DocumentsPanel = ({ tenantId, token }) => {
  const [documents, setDocuments] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedFiles, setSelectedFiles] = useState(new Set());
  const [sortField, setSortField] = useState('filename');
  const [sortDirection, setSortDirection] = useState('asc');
  const [searchQuery, setSearchQuery] = useState('');
  const [docTypeFilter, setDocTypeFilter] = useState(null);
  
  // Dynamic doc types
  const docTypes = useDocTypes(token);

  // Edit state
  const [editingFile, setEditingFile] = useState(null);
  const [editForm, setEditForm] = useState({});
  const [isSaving, setIsSaving] = useState(false);

  const fetchDocuments = useCallback(async () => {
    setIsLoading(true);
    try {
      const url = new URL('/api/v1/documents', window.location.origin);
      if (tenantId) url.searchParams.set('target_tenant_id', tenantId);
      if (docTypeFilter) url.searchParams.set('doc_type', docTypeFilter);

      const res = await fetch(url.toString(), { 
        headers: { 'Authorization': `Bearer ${token}` },
        credentials: 'include' 
      });
      if (!res.ok) throw new Error('Failed to load documents');
      
      const data = await res.json();
      setDocuments(data.documents || []);
      setSelectedFiles(new Set());
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  }, [tenantId, docTypeFilter, token]);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  const handleDelete = async (filename) => {
    if (!window.confirm(`Are you sure you want to delete "${filename}"?`)) return;

    try {
      const url = new URL(`/api/v1/documents/${filename}`, window.location.origin);
      if (tenantId) url.searchParams.set('target_tenant_id', tenantId);

      const res = await fetch(url.toString(), { 
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` },
        credentials: 'include' 
      });
      if (!res.ok) throw new Error('Delete failed');

      fetchDocuments();
    } catch {
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
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        credentials: 'include',
        body: JSON.stringify({ filenames: Array.from(selectedFiles) }),
      });

      if (!res.ok) throw new Error('Batch delete failed');

      fetchDocuments();
    } catch {
      alert('Failed to delete documents');
    }
  };

  // Edit Handlers
  const startEditing = async (doc) => {
    // If we have full metadata in the table, use it. Otherwise fetch fresh.
    // The list endpoint returns 'title', 'doc_type', 'topics'.
    // We might need 'form_number' or 'category' which aren't in list view yet?
    // Let's fetch clean metadata to be safe and get corrections info.
    
    setEditingFile(doc.filename);
    setEditForm({
      title: doc.title || '',
      doc_type: doc.doc_type || 'FORM',
      form_number: '', // Will populate from fetch
      form_category_name: '', // Will populate from fetch
    });

    try {
      const url = new URL(`/api/v1/documents/${doc.filename}/metadata`, window.location.origin);
      if (tenantId) url.searchParams.set('target_tenant_id', tenantId);
      
      const res = await fetch(url.toString(), {
        headers: { 'Authorization': `Bearer ${token}` },
        credentials: 'include' 
      });
      
      if (res.ok) {
        const data = await res.json();
        setEditForm({
          title: data.title || '',
          doc_type: data.doc_type || 'FORM',
          form_number: data.form_number || '',
          form_category_name: data.form_category_name || '',
          corrections_applied: data.corrections_applied,
        });
      }
    } catch (err) {
      console.error("Failed to fetch metadata details:", err);
    }
  };

  const saveEditing = async () => {
    setIsSaving(true);
    try {
        const url = new URL(`/api/v1/documents/${editingFile}/metadata`, window.location.origin);
        if (tenantId) url.searchParams.set('target_tenant_id', tenantId);

        // Send only fields that actually have values or are relevant
        const payload = {
            title: editForm.title,
            doc_type: editForm.doc_type,
            // Only send these if not empty/null
            ...(editForm.form_number ? { form_number: editForm.form_number } : {}),
            ...(editForm.form_category_name ? { form_category_name: editForm.form_category_name } : {}),
        };

        const res = await fetch(url.toString(), {
            method: 'PATCH',
            headers: { 
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            credentials: 'include',
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            const err = await res.json();
            alert(`Update failed: ${err.detail?.message || 'Unknown error'}`);
            return;
        }

        setEditingFile(null);
        fetchDocuments(); // Refresh list to show changes
    } catch (err) {
        console.error(err);
        alert('Failed to save changes');
    } finally {
        setIsSaving(false);
    }
  };

  const cancelEditing = () => {
    setEditingFile(null);
    setEditForm({});
  };

  // Sorting & Filtering
  const filteredDocs = documents.filter(doc => 
    doc.filename.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const sortedDocs = [...filteredDocs].sort((a, b) => {
    let aVal = a[sortField];
    let bVal = b[sortField];

    if (sortField === 'file_size_bytes' || sortField === 'chunk_count') {
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
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(decimals)) + ' ' + sizes[i];
  };

  const getDocTypeBadge = (type) => {
      if (!type) return null;
      const colors = {
          'FORM': 'bg-blue-900/40 text-blue-400 border-blue-900',
          'PROCEDURE': 'bg-green-900/40 text-green-400 border-green-900',
          'CHECKLIST': 'bg-purple-900/40 text-purple-400 border-purple-900',
          'REGULATION': 'bg-amber-900/40 text-amber-400 border-amber-900',
          'VETTING': 'bg-red-900/40 text-red-400 border-red-900',
          'CIRCULAR': 'bg-gray-700/40 text-gray-400 border-gray-700',
      };
      const className = colors[type] || 'bg-gray-800 text-gray-400 border-gray-700';
      return (
          <span className={`text-[10px] px-2 py-0.5 rounded border ${className} font-medium`}>
              {type}
          </span>
      );
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
            {/* Filter Bar - Dynamic */}
            <div className="flex bg-gray-800 rounded-md p-0.5 border border-gray-700 overflow-x-auto">
                <button
                    onClick={() => setDocTypeFilter(null)}
                    className={`px-3 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap ${
                        !docTypeFilter ? 'bg-blue-600 text-white shadow-sm' : 'text-gray-400 hover:text-white hover:bg-gray-700'
                    }`}
                >
                    ALL
                </button>
                {docTypes.map(type => (
                    <button
                        key={type}
                        onClick={() => setDocTypeFilter(type)}
                        className={`px-3 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap ${
                            docTypeFilter === type
                                ? 'bg-blue-600 text-white shadow-sm'
                                : 'text-gray-400 hover:text-white hover:bg-gray-700'
                        }`}
                    >
                        {type}
                    </button>
                ))}
            </div>

          <div className="relative flex-1 min-w-[200px]">
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
              className="flex items-center gap-2 px-4 py-2 bg-red-900/30 text-red-400 border border-red-800 rounded-md hover:bg-red-900/50 transition-colors text-sm whitespace-nowrap"
            >
              <Trash2 size={16} />
              Delete {selectedFiles.size}
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
                <th className="px-4 py-3 font-medium cursor-pointer hover:text-white border-b border-gray-800">
                    Document
                </th>
                <th className="px-4 py-3 font-medium border-b border-gray-800 w-28">Type</th>
                <th className="px-4 py-3 font-medium border-b border-gray-800 w-24 text-right">Size</th>
                <th className="px-4 py-3 font-medium border-b border-gray-800 w-20 text-center">Charts</th>
                <th className="px-4 py-3 font-medium w-20 border-b border-gray-800 text-center">Status</th>
                <th className="px-4 py-3 font-medium w-20 border-b border-gray-800 text-center">Cached</th>
                <th className="px-4 py-3 font-medium w-24 border-b border-gray-800 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {isLoading && documents.length === 0 ? (
                <tr>
                  <td colSpan={8} className="px-4 py-8 text-center text-gray-500">
                    <RefreshCw size={24} className="animate-spin mx-auto mb-2" />
                    Loading documents...
                  </td>
                </tr>
              ) : sortedDocs.length === 0 ? (
                <tr>
                  <td colSpan={8} className="px-4 py-8 text-center text-gray-500 italic">
                    {searchQuery ? 'No documents match your search.' : 'No documents uploaded yet.'}
                  </td>
                </tr>
              ) : (
                sortedDocs.map((doc) => (
                    <React.Fragment key={doc.filename}>
                      <tr className={`${editingFile === doc.filename ? 'bg-gray-800/50' : 'hover:bg-gray-800/30'} transition-colors group`}>
                        <td className="px-4 py-3">
                        <input
                            type="checkbox"
                            checked={selectedFiles.has(doc.filename)}
                            onChange={() => toggleSelect(doc.filename)}
                            className="rounded border-gray-600 bg-gray-700 text-blue-500 focus:ring-offset-gray-900"
                        />
                        </td>
                        <td className="px-4 py-3">
                        <div className="flex flex-col max-w-xs">
                            <span className="text-sm font-medium text-gray-200 truncate" title={doc.title || doc.filename}>
                                {doc.title || doc.filename}
                            </span>
                            {doc.title && doc.title !== doc.filename && (
                                <span className="text-xs text-gray-500 truncate">{doc.filename}</span>
                            )}
                        </div>
                        </td>
                        <td className="px-4 py-3">
                            {getDocTypeBadge(doc.doc_type)}
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-400 font-mono text-right">
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
                        {doc.extraction_cached ? (
                            <span className="text-blue-400 bg-blue-900/20 px-1.5 py-0.5 rounded">YES</span>
                        ) : (
                            <span className="text-gray-600">-</span>
                        )}
                        </td>
                        <td className="px-4 py-3 text-right">
                          <div className="flex items-center justify-end gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                            <button
                                onClick={() => startEditing(doc)}
                                className="text-gray-400 hover:text-blue-400 p-1"
                                title="Edit Metadata"
                            >
                                <Edit2 size={16} />
                            </button>
                            <button
                                onClick={() => handleDelete(doc.filename)}
                                className="text-gray-400 hover:text-red-400 p-1"
                                title="Delete"
                            >
                                <Trash2 size={16} />
                            </button>
                          </div>
                        </td>
                    </tr>
                    
                    {/* Inline Edit Panel */}
                    {editingFile === doc.filename && (
                        <tr className="bg-gray-800/30 border-b border-gray-700">
                            <td colSpan={8} className="px-4 py-4">
                                <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 shadow-lg animate-in fade-in slide-in-from-top-2">
                                    <div className="flex items-center justify-between mb-4">
                                        <h3 className="text-sm font-semibold text-gray-200 flex items-center gap-2">
                                            <Edit2 size={16} className="text-blue-400" />
                                            Edit Metadata: {doc.filename}
                                        </h3>
                                        <button onClick={cancelEditing} className="text-gray-500 hover:text-gray-300">
                                            <X size={16} />
                                        </button>
                                    </div>
                                    
                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <label className="block text-xs text-gray-500 mb-1">Title</label>
                                            <input
                                                type="text"
                                                value={editForm.title}
                                                onChange={e => setEditForm(prev => ({...prev, title: e.target.value}))}
                                                className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-200 focus:border-blue-500 outline-none"
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-xs text-gray-500 mb-1">Document Type</label>
                                            <select
                                                value={editForm.doc_type}
                                                onChange={e => setEditForm(prev => ({...prev, doc_type: e.target.value}))}
                                                className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-200 focus:border-blue-500 outline-none"
                                            >
                                                {docTypes.map(dt => (
                                                    <option key={dt} value={dt}>{dt}</option>
                                                ))}
                                            </select>
                                        </div>
                                        
                                        {/* Conditional Fields based on Type */}
                                        {(editForm.doc_type === 'FORM' || editForm.doc_type === 'CHECKLIST') && (
                                            <>
                                                <div>
                                                    <label className="block text-xs text-gray-500 mb-1">Form Number</label>
                                                    <input
                                                        type="text"
                                                        value={editForm.form_number}
                                                        onChange={e => setEditForm(prev => ({...prev, form_number: e.target.value}))}
                                                        placeholder="e.g. C-04"
                                                        className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-200 focus:border-blue-500 outline-none"
                                                    />
                                                </div>
                                                <div>
                                                    <label className="block text-xs text-gray-500 mb-1">Category Code / Name</label>
                                                    <input
                                                        type="text"
                                                        value={editForm.form_category_name}
                                                        onChange={e => setEditForm(prev => ({...prev, form_category_name: e.target.value}))}
                                                        placeholder="e.g. Crew or C"
                                                        className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-200 focus:border-blue-500 outline-none"
                                                    />
                                                </div>
                                            </>
                                        )}
                                    </div>

                                    {editForm.corrections_applied && Object.keys(editForm.corrections_applied).length > 0 && (
                                        <div className="mt-3 bg-blue-900/20 border border-blue-900/40 rounded px-3 py-2 text-xs text-blue-300">
                                            <span className="font-semibold">Previously Corrected: </span>
                                            {Object.keys(editForm.corrections_applied).join(', ')}
                                        </div>
                                    )}

                                    <div className="flex justify-end gap-3 mt-4">
                                        <button 
                                            onClick={cancelEditing}
                                            className="px-3 py-1.5 text-sm text-gray-400 hover:text-gray-200 transition-colors"
                                        >
                                            Cancel
                                        </button>
                                        <button
                                            onClick={saveEditing}
                                            disabled={isSaving}
                                            className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-sm rounded transition-colors flex items-center gap-2 disabled:opacity-50"
                                        >
                                            <Save size={14} />
                                            {isSaving ? 'Saving...' : 'Save Changes'}
                                        </button>
                                    </div>
                                </div>
                            </td>
                        </tr>
                    )}
                  </React.Fragment>
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
