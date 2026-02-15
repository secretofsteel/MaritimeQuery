import React, { useState, useEffect } from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import AdminSidebar from '../components/admin/AdminSidebar';
import DocumentsPanel from '../components/admin/DocumentsPanel';
import UploadPanel from '../components/admin/UploadPanel';
import FeedbackPanel from '../components/admin/FeedbackPanel';
import FormSchemaPanel from '../components/admin/FormSchemaPanel';
import SystemPanel from '../components/admin/SystemPanel';

const AdminPage = () => {
  const { user, isSuperuser } = useAuth();
  
  // State
  const [activeTab, setActiveTab] = useState('documents');
  const [selectedTenant, setSelectedTenant] = useState('');
  const [tenantList, setTenantList] = useState([]);

  useEffect(() => {
    if (user?.tenant_id) {
      setSelectedTenant(user.tenant_id);
    }
    fetchTenants();
  }, [user]);

  const fetchTenants = async () => {
    try {
      const res = await fetch('/api/v1/auth/tenants');
      if (res.ok) {
        const data = await res.json();
        setTenantList(data.tenants || []);
      }
    } catch (err) {
      console.error('Failed to load tenants:', err);
    }
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'documents':
        return <DocumentsPanel tenantId={selectedTenant} />;
      case 'upload':
        return <UploadPanel tenantId={selectedTenant} />;
      case 'feedback':
        return <FeedbackPanel tenantId={selectedTenant} />;
      case 'schema':
        return (
          <FormSchemaPanel 
            apiBasePath="/api/v1/admin/form-schema"
            tenantId={selectedTenant}
            readOnly={false}
          />
        );
      case 'system':
        return <SystemPanel tenantId={selectedTenant} />;
      default:
        return <div className="p-8 text-center text-gray-500">Unknown tab</div>;
    }
  };

  return (
    <div className="flex h-full w-full overflow-hidden bg-gray-950">
      {/* Sidebar */}
      <AdminSidebar 
        activeTab={activeTab}
        onTabChange={setActiveTab}
        selectedTenant={selectedTenant}
        tenantList={tenantList}
        onTenantChange={setSelectedTenant}
      />

      {/* Main Content Area */}
      <main className="flex-1 overflow-hidden bg-gray-900 m-2 ml-0 rounded-lg border border-gray-800 shadow-xl relative">
        <div className="absolute inset-0 overflow-hidden">
          {/* Header for mobile/context (optional, sidebar covers most) */}
          <div className="h-full p-4 overflow-y-auto custom-scrollbar">
            {renderContent()}
          </div>
        </div>
      </main>
    </div>
  );
};

export default AdminPage;
