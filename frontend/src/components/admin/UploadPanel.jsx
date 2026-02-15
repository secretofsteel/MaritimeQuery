import React, { useState, useCallback, useEffect } from 'react';
import { UploadCloud, File, X, AlertCircle, CheckCircle, Zap } from 'lucide-react';
import { useDocTypes } from '../../hooks/useDocTypes';

const UploadPanel = ({ tenantId, onUploadComplete, token }) => {
  const [dragActive, setDragActive] = useState(false);
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({});
  const [uploadResults, setUploadResults] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [docTypeOverride, setDocTypeOverride] = useState(''); // "" = Auto-detect

  // Dynamic doc types
  const docTypes = useDocTypes(token);

  const fetchExistingDocs = useCallback(async () => {
    // Just a trigger for parent to refresh, logic handled in DocumentsPanel usually
    if (onUploadComplete) onUploadComplete(); 
  }, [onUploadComplete]);

  const checkActiveProcessing = useCallback(async () => {
    try {
      const url = new URL('/api/v1/documents/process/status', window.location.origin);
      if (tenantId) url.searchParams.set('target_tenant_id', tenantId);

      const res = await fetch(url.toString(), {
          headers: { 'Authorization': `Bearer ${token}` },
          credentials: 'include' 
      });
      if (res.ok) {
        const data = await res.json();
        if (data.status === 'processing' || data.status === 'starting') {
          setIsProcessing(true);
        }
      }
    } catch {
      // Ignore errors during background status check
    }
  }, [tenantId, token]);

  useEffect(() => {
    fetchExistingDocs();
    checkActiveProcessing();
  }, [fetchExistingDocs, checkActiveProcessing]);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFiles(e.target.files);
    }
  };

  const handleFiles = (fileList) => {
    const newFiles = Array.from(fileList);
    setFiles((prev) => [...prev, ...newFiles]);
    setUploadResults(null);
  };

  const removeFile = (idx) => {
    setFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  const handleUpload = async (autoProcess = false) => {
    if (files.length === 0) return;

    setUploading(true);
    setUploadProgress({});
    setUploadResults(null);

    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });

    try {
      const url = new URL('/api/v1/documents/upload', window.location.origin);
      if (tenantId) url.searchParams.set('target_tenant_id', tenantId);
      url.searchParams.set('overwrite', 'true'); // Always overwrite to avoid 409s

      const res = await fetch(url.toString(), {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: formData, 
        credentials: 'include', 
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Upload failed');
      }

      const data = await res.json();
      setUploadResults(data);
      setFiles([]); // Clear queue on success

      if (onUploadComplete) onUploadComplete();

      if (autoProcess) {
        triggerProcessing(false); // Normal sync
      }

    } catch (err) {
      setUploadResults({ error: err.message });
    } finally {
      setUploading(false);
    }
  };

  const triggerProcessing = async (forceRebuild = false) => {
    setIsProcessing(true);
    try {
      const url = new URL('/api/v1/documents/process', window.location.origin);
      if (tenantId) url.searchParams.set('target_tenant_id', tenantId);

      const payload = {
          force_rebuild: forceRebuild,
          doc_type_override: docTypeOverride || null // Pass logic from UI state
      };

      const res = await fetch(url.toString(), {
        method: 'POST',
        headers: { 
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        },
        credentials: 'include',
        body: JSON.stringify(payload)
      });

      if (!res.ok) throw new Error('Failed to start processing');

    } catch (err) {
      console.error(err);
      setIsProcessing(false);
      alert('Failed to start processing: ' + err.message);
    }
  };

  return (
    <div className="h-full flex flex-col bg-gray-900 text-gray-100 p-6 overflow-y-auto">
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-white flex items-center gap-2 mb-2">
          <UploadCloud size={20} /> Upload Documents
        </h2>
        <p className="text-sm text-gray-400">
          Drag & drop PDF, DOCX, or TXT files here to add them to the knowledge base.
        </p>
      </div>

      {/* Drop Zone */}
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          dragActive
            ? 'border-blue-500 bg-blue-500/10'
            : 'border-gray-700 hover:border-gray-600 bg-gray-800/50'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
            type="file"
            multiple
            className="hidden"
            id="file-upload"
            onChange={handleChange}
        />
        <label
            htmlFor="file-upload"
            className="cursor-pointer flex flex-col items-center justify-center gap-2"
        >
            <UploadCloud size={40} className="text-gray-500 mb-2" />
            <span className="text-lg font-medium text-gray-300">
                Click to upload or drag and drop
            </span>
            <span className="text-sm text-gray-500">
                PDF, DOCX, TXT (Max 50MB per file)
            </span>
        </label>
      </div>

      {/* Classification */}
      <div className="mt-6 bg-gray-800 p-4 rounded-lg border border-gray-700">
          <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-gray-300">Document Classification</label>
              <span className="text-xs text-gray-500">Optional</span>
          </div>
          <div className="flex gap-4 items-center">
              <div className="flex-1">
                <select
                    value={docTypeOverride}
                    onChange={(e) => setDocTypeOverride(e.target.value)}
                    className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-sm text-gray-200 focus:border-blue-500 outline-none"
                >
                    <option value="">Auto-detect (Recommended)</option>
                    {docTypes.map(dt => (
                        <option key={dt} value={dt}>{dt}</option>
                    ))}
                </select>
              </div>
              <div className="flex-1 text-xs text-gray-500 leading-relaxed">
                  <p>
                      <strong>Auto-detect:</strong> AI analyzes each file to determine type.
                  </p>
                  <p>
                      <strong>Select Type:</strong> Forces ALL files in this batch to be classified as the selected type.
                  </p>
              </div>
          </div>
      </div>

      {/* File List */}
      {files.length > 0 && (
        <div className="mt-6">
          <h3 className="text-sm font-medium text-gray-400 mb-3">Selected Files ({files.length})</h3>
          <div className="space-y-2 bg-gray-800 rounded-lg p-2 max-h-60 overflow-y-auto">
            {files.map((file, idx) => (
              <div key={idx} className="flex items-center justify-between p-2 bg-gray-900/50 rounded border border-gray-700">
                <div className="flex items-center gap-3 overflow-hidden">
                  <File size={16} className="text-blue-400 shrink-0" />
                  <span className="text-sm text-gray-300 truncate max-w-[200px]">{file.name}</span>
                  <span className="text-xs text-gray-500">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                </div>
                {!uploading && (
                  <button onClick={() => removeFile(idx)} className="text-gray-500 hover:text-red-400">
                    <X size={16} />
                  </button>
                )}
              </div>
            ))}
          </div>

          <div className="mt-4 flex justify-end gap-3">
            <button
              onClick={() => handleUpload(false)}
              disabled={uploading}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-md text-sm font-medium transition-colors disabled:opacity-50"
            >
              {uploading ? 'Uploading...' : 'Upload Only'}
            </button>
            <button
              onClick={() => handleUpload(true)}
              disabled={uploading}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-md text-sm font-medium transition-colors disabled:opacity-50"
            >
              {uploading ? 'Uploading...' : 'Upload & Process'}
            </button>
          </div>
        </div>
      )}

      {/* Upload Results */}
      {uploadResults && (
        <div className={`mt-6 p-4 rounded-lg border ${uploadResults.error ? 'bg-red-900/10 border-red-900/30' : 'bg-green-900/10 border-green-900/30'}`}>
          <div className="flex items-start gap-3">
            {uploadResults.error ? (
              <AlertCircle className="text-red-400 mt-0.5" size={20} />
            ) : (
              <CheckCircle className="text-green-400 mt-0.5" size={20} />
            )}
            <div>
              <h4 className={`text-sm font-medium ${uploadResults.error ? 'text-red-400' : 'text-green-400'}`}>
                {uploadResults.error ? 'Upload Failed' : 'Upload Complete'}
              </h4>
              {uploadResults.error ? (
                <p className="text-xs text-red-300/80 mt-1">{uploadResults.error}</p>
              ) : (
                <div className="mt-2 text-xs text-green-300/80">
                  <p>Saved {uploadResults.saved_count} files.</p>
                  {uploadResults.results?.map((r, i) => (
                    <div key={i} className="flex gap-2 mt-1">
                      <span>• {r.filename}</span>
                      {r.overwritten && <span className="opacity-70">(Updated)</span>}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Manual Processing Triggers */}
      <div className="mt-auto border-t border-gray-800 pt-6">
        <h3 className="text-sm font-medium text-gray-300 mb-4 flex items-center gap-2">
            <Zap size={16} className="text-amber-400" />
            Processing Controls
        </h3>
        
        <div className="flex gap-4">
            <button
                onClick={() => triggerProcessing(false)}
                disabled={isProcessing || uploading}
                className="flex-1 py-3 bg-gray-800 hover:bg-gray-700 border border-gray-600 text-gray-200 rounded-lg text-sm font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
                {isProcessing ? 'Processing...' : 'Sync Library'}
                <span className="text-[10px] bg-gray-900 px-1.5 py-0.5 rounded text-gray-500">Incremental</span>
            </button>

            <button
                onClick={() => {
                    if(window.confirm('Rebuild will re-process ALL documents. This is slow and expensive. Continue?')) {
                        triggerProcessing(true);
                    }
                }}
                disabled={isProcessing || uploading}
                className="flex-1 py-3 bg-red-900/20 hover:bg-red-900/30 border border-red-900/50 text-red-400 rounded-lg text-sm font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
                Force Rebuild
                <span className="text-[10px] bg-red-900/40 px-1.5 py-0.5 rounded text-red-200">Full</span>
            </button>
        </div>
        <p className="text-xs text-gray-500 mt-2 text-center">
            Sync processes only new/modified files. Rebuild re-runs AI extraction on everything.
        </p>
      </div>
    </div>
  );
};

export default UploadPanel;
