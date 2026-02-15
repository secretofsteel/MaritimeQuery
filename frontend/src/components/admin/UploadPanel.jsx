import React, { useState, useEffect, useCallback } from 'react';
import { Upload, File, X, AlertTriangle, ShieldAlert, Play, RefreshCw, CheckCircle } from 'lucide-react';
import ProcessingStatus from './ProcessingStatus';

const UploadPanel = ({ tenantId }) => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [existingDocs, setExistingDocs] = useState(new Set());
  const [isUploading, setIsUploading] = useState(false);
  const [uploadResults, setUploadResults] = useState(null);
  
  // Processing state
  const [isProcessing, setIsProcessing] = useState(false);
  const [forceRebuild, setForceRebuild] = useState(false);
  
  // Drag state
  const [isDragging, setIsDragging] = useState(false);

  const fetchExistingDocs = useCallback(async () => {
    try {
      const url = new URL('/api/v1/documents', window.location.origin);
      if (tenantId) url.searchParams.set('target_tenant_id', tenantId);

      const res = await fetch(url.toString());
      if (res.ok) {
        const data = await res.json();
        setExistingDocs(new Set(data.documents.map(d => d.filename)));
      }
    } catch (err) {
      console.error('Failed to load existing docs for validation', err);
    }
  }, [tenantId]);

  const checkActiveProcessing = useCallback(async () => {
    try {
      const url = new URL('/api/v1/documents/process/status', window.location.origin);
      if (tenantId) url.searchParams.set('target_tenant_id', tenantId);

      const res = await fetch(url.toString());
      if (res.ok) {
        const data = await res.json();
        if (data.status === 'processing' || data.status === 'starting') {
          setIsProcessing(true);
        }
      }
    } catch {
      // Ignore errors during background status check
    }
  }, [tenantId]);

  useEffect(() => {
    fetchExistingDocs();
    checkActiveProcessing();
  }, [fetchExistingDocs, checkActiveProcessing]);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      addFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      addFiles(Array.from(e.target.files));
    }
  };

  const addFiles = (files) => {
    const validExtensions = ['.pdf', '.docx', '.doc', '.xlsx', '.xls', '.txt'];
    
    const newFiles = files.filter(file => {
      // Check extension
      if (!validExtensions.some(ext => file.name.toLowerCase().endsWith(ext))) {
        alert(`File type not supported: ${file.name}`);
        return false;
      }
      // Check duplicate in selection
      if (selectedFiles.some(f => f.name === file.name)) {
        return false;
      }
      return true;
    });

    setSelectedFiles(prev => [...prev, ...newFiles]);
  };

  const removeFile = (index) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async (autoProcess = false) => {
    if (selectedFiles.length === 0) return;

    setIsUploading(true);
    setUploadResults(null);

    const formData = new FormData();
    selectedFiles.forEach(file => {
      formData.append('files', file);
    });

    try {
      const url = new URL('/api/v1/documents/upload', window.location.origin);
      if (tenantId) url.searchParams.set('target_tenant_id', tenantId);
      url.searchParams.set('overwrite', 'true'); // Fix P1/P4: Always overwrite to avoid 409s

      const res = await fetch(url.toString(), {
        method: 'POST',
        body: formData, // Browser sets Content-Type automatically
        credentials: 'include',
      });


      if (!res.ok) throw new Error('Upload failed');

      const data = await res.json();
      setUploadResults(data);
      setSelectedFiles([]); // Clear selection on success
      fetchExistingDocs(); // Refresh cache

      if (autoProcess) {
        handleProcess();
      }
    } catch (err) {
      setUploadResults({ error: err.message });
    } finally {
      setIsUploading(false);
    }
  };

  const handleProcess = async () => {
    if (forceRebuild && !window.confirm("This will delete the index and re-process ALL documents. This cannot be undone. Continue?")) {
      return;
    }

    try {
      const url = new URL('/api/v1/documents/process', window.location.origin);
      if (tenantId) url.searchParams.set('target_tenant_id', tenantId);

      const res = await fetch(url.toString(), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ force_rebuild: forceRebuild }),
      });

      if (!res.ok) throw new Error('Processing start failed');
      
      setIsProcessing(true);
    } catch (err) {
      alert(`Failed to start processing: ${err.message}`);
    }
  };

  return (
    <div className="h-full flex flex-col bg-gray-900 text-gray-100 overflow-y-auto">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-full">
        {/* Left Column: Upload */}
        <div className="flex flex-col gap-4">
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <Upload size={20} /> Upload Files
          </h2>

          <div 
            className={`border-2 border-dashed rounded-lg p-8 flex flex-col items-center justify-center text-center transition-colors cursor-pointer min-h-[200px]
              ${isDragging 
                ? 'border-blue-500 bg-blue-900/10' 
                : 'border-gray-700 hover:border-gray-500 hover:bg-gray-800/30'}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => document.getElementById('fileInput').click()}
          >
            <Upload className={`mb-4 ${isDragging ? 'text-blue-500' : 'text-gray-500'}`} size={48} />
            <p className="text-lg font-medium text-gray-200">
              Drag files here or click to browse
            </p>
            <p className="text-sm text-gray-500 mt-2">
              Supported: PDF, DOCX, XLSX, TXT
            </p>
            <input 
              id="fileInput" 
              type="file" 
              multiple 
              className="hidden" 
              onChange={handleFileSelect} 
            />
          </div>

          <div className="flex-1 overflow-y-auto bg-gray-800/30 rounded-lg border border-gray-800 p-4">
            <h3 className="text-sm font-medium text-gray-400 mb-2">Selected Files ({selectedFiles.length})</h3>
            
            {selectedFiles.length === 0 ? (
              <p className="text-sm text-gray-600 italic">No files selected</p>
            ) : (
              <div className="space-y-2">
                {selectedFiles.map((file, idx) => (
                  <div key={idx} className="flex items-center justify-between bg-gray-800 p-2 rounded border border-gray-700">
                    <div className="flex items-center gap-3 overflow-hidden">
                      <File size={16} className="text-blue-400 shrink-0" />
                      <div className="flex flex-col min-w-0">
                        <span className="text-sm truncate" title={file.name}>{file.name}</span>
                        <span className="text-xs text-gray-500">{(file.size / 1024).toFixed(1)} KB</span>
                      </div>
                      {existingDocs.has(file.name) && (
                        <div className="flex items-center gap-1 text-amber-500 text-xs shrink-0 bg-amber-900/20 px-2 py-0.5 rounded">
                          <AlertTriangle size={12} /> Overwrite
                        </div>
                      )}
                    </div>
                    <button onClick={(e) => { e.stopPropagation(); removeFile(idx); }} className="text-gray-500 hover:text-red-400">
                      <X size={16} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="flex gap-4">
            <button
              onClick={() => handleUpload(false)}
              disabled={selectedFiles.length === 0 || isUploading}
              className="flex-1 bg-gray-800 hover:bg-gray-700 text-white py-2 rounded-md font-medium disabled:opacity-50 transition-colors"
            >
              {isUploading ? 'Uploading...' : 'Upload Only'}
            </button>
            <button
              onClick={() => handleUpload(true)}
              disabled={selectedFiles.length === 0 || isUploading || isProcessing}
              className="flex-1 bg-blue-600 hover:bg-blue-500 text-white py-2 rounded-md font-medium disabled:opacity-50 transition-colors"
            >
              Upload & Process
            </button>
          </div>

          {uploadResults && (
            <div className={`p-3 rounded-md text-sm ${
              uploadResults.error ? 'bg-red-900/30 text-red-300 border border-red-800' : 'bg-green-900/30 text-green-300 border border-green-800'
            }`}>
              {uploadResults.error ? (
                <div className="flex items-center gap-2"><ShieldAlert size={16} /> {uploadResults.error}</div>
              ) : (
                <div className="flex items-center gap-2"><CheckCircle size={16} /> Upload successful!</div>
              )}
            </div>
          )}
        </div>

        {/* Right Column: Processing */}
        <div className="flex flex-col gap-4 border-l border-gray-800 pl-6">
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <Play size={20} /> Processing
          </h2>

          <div className="bg-gray-800 p-4 rounded-lg border border-gray-700 space-y-4">
            <label className="flex items-center gap-3 p-3 bg-gray-900/50 rounded border border-gray-800 cursor-pointer hover:bg-gray-900 transition-colors">
              <input 
                type="checkbox" 
                checked={forceRebuild} 
                onChange={(e) => setForceRebuild(e.target.checked)}
                className="w-4 h-4 rounded border-gray-600 bg-gray-700 text-red-500 focus:ring-offset-gray-900"
                disabled={isProcessing}
              />
              <div className="flex-1">
                <div className="text-sm font-medium text-red-300">Force Full Rebuild</div>
                <div className="text-xs text-gray-500">Deletes entire index and re-processes all files. Use only if index is corrupted.</div>
              </div>
            </label>

            <div className="flex gap-4">
              <button
                onClick={handleProcess}
                disabled={isProcessing}
                className="flex-1 flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-500 text-white py-2 rounded-md font-medium disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {forceRebuild ? <AlertTriangle size={16} /> : <RefreshCw size={16} />}
                {forceRebuild ? 'Full Rebuild' : 'Sync Library'}
              </button>
            </div>
          </div>

          <div className="flex-1">
            <ProcessingStatus 
              tenantId={tenantId} 
              isActive={isProcessing} 
              onComplete={() => setIsProcessing(false)} 
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default UploadPanel;
