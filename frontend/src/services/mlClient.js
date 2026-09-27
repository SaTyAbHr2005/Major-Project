/**
 * Client for the hospital-local ML service (python -m hospital_client.local_api).
 *
 * The service runs the Member 1 pipeline (Modules 4-8, 16) on THIS machine.
 * The browser only ever sends a dataset folder path, JSON options, or a single
 * image for local inference to 127.0.0.1 - nothing here talks to a remote server.
 * Auth: the backend-issued JWT is forwarded; the service verifies it with the backend.
 */
import authService from './authService';

export const ML_API_URL = import.meta.env.VITE_ML_API_URL || 'http://127.0.0.1:8765/api/ml';

async function request(path, { method = 'GET', body, raw, contentType } = {}) {
  const headers = { Accept: 'application/json' };
  const token = authService.getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  let payload;
  if (raw !== undefined) {
    headers['Content-Type'] = contentType || 'application/octet-stream';
    payload = raw;
  } else if (method !== 'GET') {
    headers['Content-Type'] = 'application/json';
    payload = JSON.stringify(body || {});
  }

  let res;
  try {
    res = await fetch(`${ML_API_URL}${path}`, { method, headers, body: payload });
  } catch {
    throw new Error(`Local ML service is not reachable at ${ML_API_URL}. Start it with: python -m hospital_client.local_api`);
  }
  if ((res.headers.get('Content-Type') || '').startsWith('image/')) {
    if (!res.ok) throw new Error(`Request failed (${res.status})`);
    return res.blob();
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export const ml = {
  get: (path) => request(path),
  post: (path, body) => request(path, { method: 'POST', body }),
  upload: (path, file) => request(path, { method: 'POST', raw: file }),

  /** Polls a background job until it finishes; onUpdate receives every snapshot. */
  async waitForJob(job, onUpdate, intervalMs = 1500) {
    let current = job;
    onUpdate?.(current);
    while (current.status === 'running') {
      await sleep(intervalMs);
      current = await request(`/jobs/${current.job_id}`);
      onUpdate?.(current);
    }
    if (current.status === 'failed') throw new Error(current.error || 'The ML job failed.');
    if (current.status === 'cancelled') throw new Error('The ML job was cancelled.');
    return current.result;
  },

  /** Plot PNGs need the auth header, so they are fetched and shown as object URLs. */
  async plotUrl(modelId, name) {
    const blob = await request(`/training-runs/${encodeURIComponent(modelId)}/plots/${name}`);
    return URL.createObjectURL(blob);
  },

  health: () => request('/health'),
};

export default ml;
