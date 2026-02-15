import React from 'react';
import FormSchemaPanel from '../components/admin/FormSchemaPanel';

const SettingsPage = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white">Settings</h1>
        <p className="text-gray-400 mt-1">Manage your preferences and configuration</p>
      </div>

      {/* Form Schema Section */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
        <div className="p-6 h-[600px]">
          <FormSchemaPanel 
            apiBasePath="/api/v1/settings/form-schema"
            readOnly={false}
          />
        </div>
      </div>
    </div>
  );
};

export default SettingsPage;
