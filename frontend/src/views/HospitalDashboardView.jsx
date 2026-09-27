import React, { useEffect, useState } from 'react';
import { useApp } from '../context/AppContext';
import { useML } from '../context/MLContext';
import trainingService from '../services/trainingService';
import PrivacyNotice from '../components/common/PrivacyNotice';
import {
  Building2,
  Database,
  Cpu,
  Activity,
  PlayCircle,
  CheckCircle,
  Server,
  ShieldCheck,
  ArrowRight,
  Box
} from 'lucide-react';

const gb = (mb) => (mb == null ? '-' : `${(mb / 1024).toFixed(1)} GB`);

const DashboardView = () => {
  const { user, hospitalId, hospitalName, setActiveTab } = useApp();
  const { hardware, hardwareError, datasets, activeDataset, trainingJob, serviceStatus } = useML();
  const [runs, setRuns] = useState([]);

  useEffect(() => {
    if (serviceStatus === 'online') trainingService.getTrainingHistory().then(setRuns).catch(() => setRuns([]));
  }, [serviceStatus, trainingJob?.status]);

  const isTraining = trainingJob?.status === 'running';
  const lastEpoch = trainingJob?.epochs?.[trainingJob.epochs.length - 1];
  const gpu = hardware?.gpu;
  const profile = activeDataset?.profile;

  return (
    <div style={{ padding: '20px', overflowY: 'auto', height: '100%' }}>
      <PrivacyNotice />

      <div className="card" style={{ marginBottom: '16px', backgroundColor: '#0e1628', borderColor: 'rgba(6, 182, 212, 0.3)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
            <div style={{ width: '42px', height: '42px', borderRadius: 'var(--radius-lg)', backgroundColor: 'rgba(6, 182, 212, 0.15)',
              display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-teal)' }}>
              <Building2 size={24} />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <h2 style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)' }}>{hospitalName}</h2>
                <span className="badge badge-teal font-mono">{hospitalId}</span>
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                Operator: <strong className="text-primary">{user?.name}</strong> | Role: <span className="font-mono text-cyan">{user?.role}</span>
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span className={`badge ${serviceStatus === 'online' ? 'badge-healthy' : 'badge-danger'}`}>
              <ShieldCheck size={12} /> Local ML service: {serviceStatus}
            </span>
            <button onClick={() => setActiveTab('datasets')} className="btn btn-teal">
              <PlayCircle size={14} /> Start Training Workflow
            </button>
          </div>
        </div>
        {serviceStatus === 'offline' && (
          <div style={{ marginTop: '10px', fontSize: '11px', color: 'var(--status-danger)' }}>
            Start the local ML service on this workstation: <code className="font-mono">python -m hospital_client.local_api</code>
          </div>
        )}
      </div>

      {/* Measured hardware (Module 8 detect) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginBottom: '16px' }}>
        <div className="card" style={{ padding: '14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
            <span className="card-title-muted">GPU / CUDA</span><Server size={14} color="var(--accent-teal)" />
          </div>
          <div className="font-mono text-cyan" style={{ fontSize: '14px', fontWeight: 700 }}>
            {hardware ? (gpu?.cuda_available ? gpu.gpu_name.replace('NVIDIA GeForce ', '') : 'No CUDA device') : (hardwareError ? 'unavailable' : 'detecting...')}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
            {gpu?.cuda_available ? `Compute capability ${gpu.cuda_capability}` : gpu?.fallback_reason || ''}
          </div>
        </div>

        <div className="card" style={{ padding: '14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
            <span className="card-title-muted">VRAM</span><Cpu size={14} color="#818cf8" />
          </div>
          <div className="font-mono text-primary" style={{ fontSize: '16px', fontWeight: 700 }}>
            {gpu?.cuda_available ? `${gb(gpu.free_vram_mb)} free / ${gb(gpu.total_vram_mb)}` : '-'}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>Measured at {hardware ? new Date(hardware.timestamp).toLocaleTimeString() : '-'}</div>
        </div>

        <div className="card" style={{ padding: '14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
            <span className="card-title-muted">Host RAM & CPU</span><Activity size={14} color="var(--brand-blue)" />
          </div>
          <div className="font-mono text-primary" style={{ fontSize: '16px', fontWeight: 700 }}>
            {hardware ? `${gb(hardware.ram.available_mb)} free / ${gb(hardware.ram.total_mb)}` : '-'}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
            {hardware ? `${hardware.cpu.physical_cores} cores / ${hardware.cpu.logical_cores} threads` : ''}
          </div>
        </div>

        <div className="card" style={{ padding: '14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
            <span className="card-title-muted">Local Datasets</span><Database size={14} color="var(--status-healthy)" />
          </div>
          <div className="font-mono text-emerald" style={{ fontSize: '16px', fontWeight: 700 }}>{datasets.length} inspected</div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
            {datasets.filter((d) => d.preprocessed).length} preprocessed | storage free {gb(hardware?.storage?.free_mb)}
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px', marginBottom: '16px' }}>
        <div className="card">
          <div className="card-title" style={{ justifyContent: 'space-between' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><Database size={15} color="var(--accent-teal)" /> Active Local Dataset</span>
            <button onClick={() => setActiveTab('datasets')} className="btn btn-secondary" style={{ padding: '2px 8px', fontSize: '11px' }}>
              Inspect Details <ArrowRight size={12} />
            </button>
          </div>
          {profile ? (
            <div>
              <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '4px' }}>{activeDataset.name}</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '12px' }}>
                Local Path: <code className="font-mono text-code">{activeDataset.source_path}</code>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', padding: '10px', backgroundColor: '#090d16', borderRadius: 'var(--radius-md)', marginBottom: '12px', fontSize: '11px' }}>
                <div><span className="text-muted">Total Samples:</span><div className="font-mono text-primary" style={{ fontWeight: 600 }}>{profile.total_samples?.toLocaleString()}</div></div>
                <div><span className="text-muted">Type:</span><div className="font-mono text-cyan" style={{ fontWeight: 600 }}>{profile.dataset_type}</div></div>
                <div><span className="text-muted">Splits:</span><div className="font-mono text-primary" style={{ fontWeight: 600 }}>
                  {activeDataset.preprocessing ? Object.values(activeDataset.preprocessing.splits).join(' / ') : 'not preprocessed'}
                </div></div>
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                <strong>Classes:</strong> {Object.entries(profile.class_distribution || {}).map(([c, n]) => `${c} (${n.toLocaleString()})`).join(' | ')}
              </div>
            </div>
          ) : (
            <div className="text-muted" style={{ padding: '20px', textAlign: 'center' }}>No dataset located yet. Open Dataset Inspection to locate one.</div>
          )}
        </div>

        <div className="card">
          <div className="card-title" style={{ justifyContent: 'space-between' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Activity size={15} color={isTraining ? 'var(--status-warning)' : 'var(--status-healthy)'} /> Local Training & Federation State
            </span>
            <span className={`badge ${isTraining ? 'badge-warning' : 'badge-healthy'}`}>{isTraining ? 'TRAINING IN PROGRESS' : 'READY FOR TRAINING'}</span>
          </div>
          {isTraining ? (
            <div>
              <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '6px' }}>
                Run: <span className="font-mono text-cyan">{trainingJob.meta.model_id}</span>
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '10px' }}>
                {lastEpoch ? `Epoch ${lastEpoch.epoch}/${lastEpoch.total_epochs}` : 'Preparing data & model'} | elapsed {Math.round(trainingJob.elapsed_seconds)}s
              </div>
              <button onClick={() => setActiveTab('training_monitor')} className="btn btn-teal" style={{ width: '100%' }}>
                Open Live Training Monitor <ArrowRight size={13} />
              </button>
            </div>
          ) : (
            <div>
              <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '14px', lineHeight: 1.4 }}>
                Module 8 re-measures this workstation before every recommendation and training run.
                Federated rounds (Module 9) will start from the locally prepared FederationHandoff once that engine is available.
              </p>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button onClick={() => setActiveTab('start_training')} className="btn btn-teal" style={{ flex: 1 }}><PlayCircle size={14} /> Start Training Workflow</button>
                <button onClick={() => setActiveTab('models')} className="btn btn-secondary"><Box size={14} /> Local Models</button>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="card">
        <div className="card-title" style={{ justifyContent: 'space-between' }}>
          <span>Recent Local Training Sessions</span>
          <span className="badge badge-neutral" style={{ fontSize: '10px' }}>Module 7 results on disk</span>
        </div>
        <div className="fl-table-wrapper">
          <table className="fl-table">
            <thead>
              <tr><th>Run / Model ID</th><th>Architecture</th><th>Dataset</th><th>Epochs</th><th>Val Acc</th><th>Test Acc</th><th>Completed</th><th>Federation Handoff</th></tr>
            </thead>
            <tbody>
              {runs.length === 0 && <tr><td colSpan={8} className="text-muted" style={{ textAlign: 'center' }}>No local training runs yet.</td></tr>}
              {runs.slice(0, 5).map((h) => (
                <tr key={h.model_id}>
                  <td className="font-mono text-cyan">{h.model_id}</td>
                  <td style={{ fontWeight: 600 }}>{h.architecture}</td>
                  <td style={{ color: 'var(--text-secondary)' }}>{h.dataset_name || '-'}</td>
                  <td className="font-mono">{h.epochs_run}</td>
                  <td className="font-mono text-emerald" style={{ fontWeight: 600 }}>{h.validation_accuracy != null ? `${(h.validation_accuracy * 100).toFixed(1)}%` : '-'}</td>
                  <td className="font-mono">{h.test_accuracy != null ? `${(h.test_accuracy * 100).toFixed(1)}%` : '-'}</td>
                  <td style={{ color: 'var(--text-muted)' }}>{h.generated_at ? new Date(h.generated_at).toLocaleString() : '-'}</td>
                  <td>
                    <span className={`badge ${h.ready_for_federation ? 'badge-healthy' : 'badge-neutral'}`} style={{ fontSize: '10px' }}>
                      <CheckCircle size={10} /> {h.ready_for_federation ? 'PREPARED' : h.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default DashboardView;
