import React from 'react';
import { FileText, Upload, BarChart3, Settings, Activity, Building } from 'lucide-react';

const AdminSidebar = ({ 
  activeTab, 
  onTabChange, 
  selectedTenant, 
  tenantList = [], 
  onTenantChange 
}) => {
  const tabs = [
    { id: 'documents', label: 'Documents', icon: FileText },
    { id: 'upload', label: 'Upload & Process', icon: Upload },
    { id: 'feedback', label: 'Feedback', icon: BarChart3 },
    { id: 'schema', label: 'Form Schema', icon: Settings },
    { id: 'system', label: 'System', icon: Activity },
  ];

  return (
    <aside className="w-64 bg-gray-900 border-r border-gray-800 flex flex-col h-full shrink-0">
      {/* Tenant Selector */}
      <div className="p-4 border-b border-gray-800">
        <label className="block text-xs uppercase text-gray-500 font-semibold mb-2">
          Managing Tenant
        </label>
        <div className="relative">
          <Building className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={16} />
          <select
            value={selectedTenant}
            onChange={(e) => onTenantChange(e.target.value)}
            className="w-full bg-gray-800 text-white text-sm rounded-md pl-9 pr-3 py-2 border border-gray-700 focus:outline-none focus:ring-1 focus:ring-blue-500 appearance-none cursor-pointer"
          >
            {tenantList.map((tenant) => (
              <option key={tenant.tenant_id} value={tenant.tenant_id}>
                {tenant.display_name || tenant.tenant_id}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Navigation Tabs */}
      <nav className="flex-1 p-2 space-y-1 overflow-y-auto">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          
          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors ${
                isActive
                  ? 'bg-gray-800 text-white border-l-2 border-blue-500'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
              }`}
            >
              <Icon size={18} />
              {tab.label}
            </button>
          );
        })}
      </nav>
      
      <div className="p-4 border-t border-gray-800 text-xs text-gray-600 text-center">
        Admin Panel v1.0
      </div>
    </aside>
  );
};

export default AdminSidebar;
