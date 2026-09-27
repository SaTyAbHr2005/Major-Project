/**
 * API Service for Researcher/Admin Desktop
 * Interfaces with Backend for Training Requests, Admin Operations, Model Reports, and Telemetry
 */

const API_BASE_URL = (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_URL)
  ? import.meta.env.VITE_API_URL
  : 'http://localhost:5000/api/v1';

class ApiService {
  getToken() {
    try {
      return localStorage.getItem('medfl_researcher_token') || sessionStorage.getItem('medfl_researcher_token');
    } catch {
      return null;
    }
  }

  getHeaders() {
    const headers = {
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    };
    const token = this.getToken();
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    return headers;
  }

  // --- Training Requests (Researcher & Admin) ---

  async createTrainingRequest({ disease, task, description, model_architecture, training_config }) {
    const res = await fetch(`${API_BASE_URL}/training-requests`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({
        disease,
        task,
        description,
        model_architecture,
        training_config
      })
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data?.error?.message || 'Failed to create training request');
    }
    return data.data.training_request;
  }

  async getAllTrainingRequests() {
    const res = await fetch(`${API_BASE_URL}/training-requests`, {
      headers: this.getHeaders()
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data?.error?.message || 'Failed to fetch training requests');
    }
    return data.data.training_requests;
  }

  async getMyTrainingRequests() {
    const res = await fetch(`${API_BASE_URL}/training-requests/my`, {
      headers: this.getHeaders()
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data?.error?.message || 'Failed to fetch your training requests');
    }
    return data.data.training_requests;
  }

  // --- Hospital Operator: participation & local-training progress ---

  async participateInTrainingRequest(requestId) {
    const res = await fetch(`${API_BASE_URL}/training-requests/${encodeURIComponent(requestId)}/participate`, {
      method: 'POST',
      headers: this.getHeaders()
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data?.error?.message || 'Failed to join training request');
    }
    return data.data;
  }

  /** Aggregate metrics only (epoch, loss, accuracy) - never images or patient data. */
  async reportTrainingProgress(requestId, { status, current_epoch, total_epochs, loss, accuracy }) {
    const res = await fetch(`${API_BASE_URL}/training-requests/${encodeURIComponent(requestId)}/progress`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({ status, current_epoch, total_epochs, loss, accuracy })
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data?.error?.message || 'Failed to report training progress');
    }
    return data.data;
  }

  // --- Admin Operations ---

  async getPendingResearchers() {
    const res = await fetch(`${API_BASE_URL}/admin/researchers/pending`, {
      headers: this.getHeaders()
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data?.error?.message || 'Failed to fetch pending researchers');
    }
    return data.data.researchers;
  }

  async getAllResearchers() {
    const res = await fetch(`${API_BASE_URL}/admin/researchers`, {
      headers: this.getHeaders()
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data?.error?.message || 'Failed to fetch researchers');
    }
    return data.data.researchers;
  }

  async approveResearcher(userId) {
    const res = await fetch(`${API_BASE_URL}/admin/users/${userId}/approve`, {
      method: 'POST',
      headers: this.getHeaders()
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data?.error?.message || 'Failed to approve researcher');
    }
    return data.data;
  }

  async rejectResearcher(userId) {
    const res = await fetch(`${API_BASE_URL}/admin/users/${userId}/reject`, {
      method: 'POST',
      headers: this.getHeaders()
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data?.error?.message || 'Failed to reject researcher');
    }
    return data.data;
  }

  async getDiseases() {
    const res = await fetch(`${API_BASE_URL}/admin/diseases`, {
      headers: this.getHeaders()
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data?.error?.message || 'Failed to fetch diseases');
    }
    return data.data.diseases;
  }

  async getModelReport(disease) {
    const url = disease
      ? `${API_BASE_URL}/admin/reports?disease=${encodeURIComponent(disease)}`
      : `${API_BASE_URL}/admin/reports`;
    const res = await fetch(url, {
      headers: this.getHeaders()
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data?.error?.message || 'Failed to fetch model report');
    }
    return data.data;
  }

  async getStoredModels(disease) {
    const url = disease
      ? `${API_BASE_URL}/admin/stored-models?disease=${encodeURIComponent(disease)}`
      : `${API_BASE_URL}/admin/stored-models`;
    const res = await fetch(url, {
      headers: this.getHeaders()
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data?.error?.message || 'Failed to fetch stored models');
    }
    return data.data.models;
  }

  async getTelemetry() {
    const res = await fetch(`${API_BASE_URL}/admin/telemetry`, {
      headers: this.getHeaders()
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data?.error?.message || 'Failed to fetch telemetry');
    }
    return data.data;
  }
}

export const apiService = new ApiService();
export default apiService;
