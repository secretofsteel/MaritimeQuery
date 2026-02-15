import { useState, useEffect } from 'react';

let cachedDocTypes = null; // module-level cache — fetch once per session

export function useDocTypes() {
  const [docTypes, setDocTypes] = useState(cachedDocTypes || []);

  useEffect(() => {
    if (cachedDocTypes && cachedDocTypes.length > 0) {
      return;
    }

    fetch('/api/v1/system/config/doc-types', {
      credentials: 'include',
    })
      .then(res => {
        if (!res.ok) throw new Error('Failed to fetch doc types');
        return res.json();
      })
      .then(data => {
        if (data.doc_types) {
          cachedDocTypes = data.doc_types;
          setDocTypes(data.doc_types);
        }
      })
      .catch(err => {
        console.error('Failed to fetch doc types:', err);
        // Fallback to known types if API fails
        const fallback = ['FORM', 'CHECKLIST', 'PROCEDURE', 'REGULATION', 'VETTING', 'CIRCULAR'];
        setDocTypes(fallback);
      });
  }, []);

  return docTypes;
}
