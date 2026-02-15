// frontend/src/api/client.js

const BASE = ''  // Same origin — Vite proxy handles /api/* in dev

/**
 * Make an authenticated API request.
 * 
 * Automatically includes cookies (for JWT auth) and handles
 * common error patterns (401 → redirect, non-OK → throw).
 */
async function request(path, options = {}) {
  const config = {
    credentials: 'include',  // Send cookies
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  }

  // Don't set Content-Type for FormData (browser sets it with boundary)
  if (options.body instanceof FormData) {
    delete config.headers['Content-Type']
  }

  const res = await fetch(`${BASE}${path}`, config)

  if (res.status === 401) {
    // Token expired or invalid — redirect to login
    // (unless we're already on the login page)
    if (!window.location.pathname.startsWith('/login')) {
      window.location.href = '/login'
    }
    throw new Error('Unauthorized')
  }

  if (!res.ok) {
    // Try to extract error detail from FastAPI's error format
    let detail = `HTTP ${res.status}`
    try {
      const body = await res.json()
      detail = body.detail || detail
    } catch {
      // Response wasn't JSON
    }
    throw new Error(detail)
  }

  // Some endpoints return empty responses (204, etc.)
  if (res.status === 204 || res.headers.get('content-length') === '0') {
    return null
  }

  return res.json()
}


// Convenience methods
export const api = {
  get: (path) => request(path, { method: 'GET' }),

  post: (path, body) => request(path, {
    method: 'POST',
    body: JSON.stringify(body),
  }),

  put: (path, body) => request(path, {
    method: 'PUT',
    body: JSON.stringify(body),
  }),

  delete: (path) => request(path, { method: 'DELETE' }),

  upload: (path, formData) => request(path, {
    method: 'POST',
    body: formData,  // FormData — Content-Type set automatically
  }),

  patch: (path, body) => request(path, {
    method: 'PATCH',
    body: JSON.stringify(body),
  }),
}
