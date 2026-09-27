import React from 'react';
import { CheckCircle, Loader, AlertTriangle, Circle } from 'lucide-react';
import { useML } from '../../context/MLContext';
import { JobLog } from '../charts/MLCharts';

const STEPS = [
  { id: 'inspect', label: 'Module 4 - Inspect dataset' },
  { id: 'preprocess', label: 'Module 5 - Preprocess & split' },
  { id: 'recommend', label: 'Module 8 - Hardware-aware recommendations' },
];

/** Progress of the automatic local pipeline (locate -> M4 -> M5 -> M8). */
const PipelineStatus = () => {
  const { pipeline, activeDataset } = useML();
  if (!pipeline.running && !pipeline.error && pipeline.step !== 'done') return null;

  const current = STEPS.findIndex((s) => s.id === pipeline.step);
  const stateOf = (idx) => {
    if (pipeline.step === 'done') return 'done';
    if (idx < current) return 'done';
    if (idx === current) return pipeline.error ? 'error' : 'running';
    return 'pending';
  };
  const icon = { done: <CheckCircle size={14} color="var(--status-healthy)" />, running: <Loader size={14} className="spin" color="var(--accent-teal)" />,
    error: <AlertTriangle size={14} color="var(--status-danger)" />, pending: <Circle size={14} color="var(--text-muted)" /> };

  return (
    <div className="card" style={{ marginBottom: '16px' }}>
      <div className="card-title" style={{ justifyContent: 'space-between' }}>
        <span>Local ML Pipeline {activeDataset ? `- ${activeDataset.name}` : ''}</span>
        <span className={`badge ${pipeline.error ? 'badge-danger' : pipeline.running ? 'badge-warning' : 'badge-healthy'}`}>
          {pipeline.error ? 'FAILED' : pipeline.running ? 'RUNNING ON THIS MACHINE' : 'READY FOR TRAINING'}
        </span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px', marginBottom: '12px' }}>
        {STEPS.map((s, idx) => (
          <div key={s.id} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', padding: '8px 10px',
            backgroundColor: '#090d16', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
            {icon[stateOf(idx)]}
            <span style={{ color: stateOf(idx) === 'pending' ? 'var(--text-muted)' : 'var(--text-primary)' }}>{s.label}</span>
          </div>
        ))}
      </div>
      {pipeline.error && (
        <div style={{ padding: '8px 12px', marginBottom: '10px', backgroundColor: 'var(--status-danger-bg)', border: '1px solid var(--status-danger-border)',
          borderRadius: 'var(--radius-md)', color: 'var(--status-danger)', fontSize: '11px' }}>{pipeline.error}</div>
      )}
      {pipeline.job && <JobLog lines={pipeline.job.log} height={110} />}
    </div>
  );
};

export default PipelineStatus;
