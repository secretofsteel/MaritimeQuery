import React, { useState, useEffect } from 'react';
import { Plus, Trash2, Save, X, Edit2, Check, AlertTriangle } from 'lucide-react';

const FormSchemaPanel = ({ apiBasePath, tenantId, readOnly = false }) => {
  const [categories, setCategories] = useState({});
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState(null);
  const [successMsg, setSuccessMsg] = useState('');

  // Editing state
  const [editingCode, setEditingCode] = useState(null); // Code of the item being edited
  const [editDescription, setEditDescription] = useState('');

  // New item state
  const [newCode, setNewCode] = useState('');
  const [newDescription, setNewDescription] = useState('');

  useEffect(() => {
    fetchSchema();
  }, [apiBasePath, tenantId]);

  const fetchSchema = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const url = new URL(apiBasePath, window.location.origin);
      if (tenantId) url.searchParams.set('target_tenant_id', tenantId);

      const res = await fetch(url.toString());
      if (!res.ok) throw new Error(`Failed to load schema: ${res.statusText}`);
      
      const data = await res.json();
      setCategories(data.categories || {});
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const saveSchema = async (newCategories) => {
    setIsSaving(true);
    setError(null);
    try {
      const url = new URL(apiBasePath, window.location.origin);
      if (tenantId) url.searchParams.set('target_tenant_id', tenantId);

      const res = await fetch(url.toString(), {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ categories: newCategories }),
      });

      if (!res.ok) throw new Error('Failed to save changes');
      
      const data = await res.json();
      setCategories(data.categories);
      showSuccess('Schema updated successfully');
      return true;
    } catch (err) {
      setError(err.message);
      return false;
    } finally {
      setIsSaving(false);
    }
  };

  const showSuccess = (msg) => {
    setSuccessMsg(msg);
    setTimeout(() => setSuccessMsg(''), 3000);
  };

  const handleAdd = async () => {
    if (!newCode.trim() || !newDescription.trim()) return;
    if (categories[newCode.toUpperCase()]) {
      setError(`Code ${newCode.toUpperCase()} already exists`);
      return;
    }

    const updated = {
      ...categories,
      [newCode.toUpperCase()]: newDescription.trim(),
    };

    const success = await saveSchema(updated);
    if (success) {
      setNewCode('');
      setNewDescription('');
    }
  };

  const handleDelete = async (code) => {
    if (!window.confirm(`Are you sure you want to delete code "${code}"?`)) return;

    const updated = { ...categories };
    delete updated[code];
    await saveSchema(updated);
  };

  const startEditing = (code, currentDesc) => {
    setEditingCode(code);
    setEditDescription(currentDesc);
  };

  const saveEdit = async () => {
    if (!editingCode) return;
    
    const updated = {
      ...categories,
      [editingCode]: editDescription.trim(),
    };

    const success = await saveSchema(updated);
    if (success) {
      setEditingCode(null);
    }
  };

  const handleClearAll = async () => {
     if (!window.confirm("Are you sure? This will remove ALL codes.")) return;
     if (!window.confirm("This action cannot be undone. Confirm clear all?")) return;
     
     await saveSchema({});
  };

  const sortedCodes = Object.keys(categories).sort();

  if (isLoading) return <div className="p-8 text-center text-gray-500">Loading schema...</div>;

  return (
    <div className="h-full flex flex-col bg-gray-900 text-gray-100">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-white">Form Schema Configuration</h2>
          <p className="text-sm text-gray-400 mt-1">
            Manage document categories and codes ({sortedCodes.length} codes configured)
          </p>
        </div>
        {successMsg && (
          <div className="text-sm text-green-400 bg-green-900/30 px-3 py-1 rounded-md animate-fade-in flex items-center gap-2">
            <Check size={14} /> {successMsg}
          </div>
        )}
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-900/30 border border-red-800 rounded-md text-red-300 text-sm flex items-center gap-2">
          <AlertTriangle size={16} />
          {error}
        </div>
      )}

      {/* Add New Row */}
      {!readOnly && (
        <div className="bg-gray-800/50 p-4 rounded-lg border border-gray-700 mb-6">
          <h3 className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
            <Plus size={16} /> Add New Category
          </h3>
          <div className="flex gap-4">
            <input
              type="text"
              placeholder="CODE (e.g. HR)"
              value={newCode}
              onChange={(e) => setNewCode(e.target.value.toUpperCase())}
              className="w-24 bg-gray-900 border border-gray-600 rounded px-3 py-2 text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none uppercase"
              maxLength={10}
            />
            <input
              type="text"
              placeholder="Description (e.g. Human Resources)"
              value={newDescription}
              onChange={(e) => setNewDescription(e.target.value)}
              className="flex-1 bg-gray-900 border border-gray-600 rounded px-3 py-2 text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
              onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
            />
            <button
              onClick={handleAdd}
              disabled={!newCode || !newDescription || isSaving}
              className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Add
            </button>
          </div>
        </div>
      )}

      {/* List */}
      <div className="flex-1 overflow-y-auto border border-gray-800 rounded-lg">
        <table className="w-full text-left border-collapse">
          <thead className="bg-gray-800/50 text-gray-400 text-xs uppercase tracking-wider sticky top-0">
            <tr>
              <th className="px-4 py-3 font-medium w-32 border-b border-gray-800">Code</th>
              <th className="px-4 py-3 font-medium border-b border-gray-800">Description</th>
              {!readOnly && <th className="px-4 py-3 font-medium w-24 border-b border-gray-800 text-right">Actions</th>}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {sortedCodes.length === 0 ? (
              <tr>
                <td colSpan={3} className="px-4 py-8 text-center text-gray-500 italic">
                  No form codes configured.
                </td>
              </tr>
            ) : (
              sortedCodes.map((code) => (
                <tr key={code} className="hover:bg-gray-800/30 group transition-colors">
                  <td className="px-4 py-3 font-mono text-sm text-blue-400 font-medium">
                    {code}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-300">
                    {editingCode === code ? (
                      <div className="flex items-center gap-2">
                        <input
                          type="text"
                          value={editDescription}
                          onChange={(e) => setEditDescription(e.target.value)}
                          className="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm focus:border-blue-500 focus:outline-none"
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') saveEdit();
                            if (e.key === 'Escape') setEditingCode(null);
                          }}
                          autoFocus
                        />
                        <button onClick={saveEdit} className="text-green-400 hover:text-green-300 p-1">
                          <Check size={16} />
                        </button>
                        <button onClick={() => setEditingCode(null)} className="text-red-400 hover:text-red-300 p-1">
                          <X size={16} />
                        </button>
                      </div>
                    ) : (
                      <div 
                        className="cursor-pointer group-hover:text-white flex items-center gap-2"
                        onClick={() => !readOnly && startEditing(code, categories[code])}
                      >
                        {categories[code]}
                        {!readOnly && <Edit2 size={12} className="opacity-0 group-hover:opacity-30" />}
                      </div>
                    )}
                  </td>
                  {!readOnly && (
                    <td className="px-4 py-3 text-right">
                      <button
                        onClick={() => handleDelete(code)}
                        className="text-gray-600 hover:text-red-400 p-1 transition-colors"
                        title="Delete"
                      >
                        <Trash2 size={16} />
                      </button>
                    </td>
                  )}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {!readOnly && sortedCodes.length > 0 && (
        <div className="mt-6 pt-4 border-t border-gray-800 flex justify-end">
          <button
            onClick={handleClearAll}
            className="text-red-400 hover:text-red-300 text-sm hover:underline"
          >
            Clear All Codes
          </button>
        </div>
      )}
    </div>
  );
};

export default FormSchemaPanel;
