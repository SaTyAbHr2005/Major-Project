import React from 'react';
import { CheckCircle, Loader, AlertTriangle, Radio } from 'lucide-react';

/** The steps a hospital goes through in one round of a Flower job (state names from the local ML service). */
export const FLOWER_STEPS = [
  { state: 'waiting', label: 'Connected' },
  { state: 'received_model', label: 'Model received' },
  { state: 'training', label: 'Training' },
  { state: 'sending', label: 'Sending update' },
  { state: 'waiting_for_others', label: 'Waiting for other hospitals' },
  { state: 'calibrating', label: 'Calibrating' },
  { state: 'done', label: 'Round done' },
];
const PROBLEM = { missed: '#eab308', failed: '#ef4444', disconnected: '#94a3b8' };

/** Hospital: which Flower step this machine is on for a job. `status` is the local connection status. */
export const FlowerSteps = ({ status, issue }) => {
  if (!status) {
    return (
      <div style={{ fontSize: '12px', color: issue ? '#eab308' : 'var(--text-muted)', display: 'flex', gap: '6px', alignItems: 'center' }}>
        <Radio size={13} /> {issue || 'Connecting to the federation server…'}
      </div>
    );
  }
  const current = status.state === 'connecting' ? -1 : FLOWER_STEPS.findIndex((s) => s.state === status.state);
  const problem = PROBLEM[status.state];
  return (
    <div role="status" aria-live="polite" style={{ padding: '12px', background: 'var(--bg-card)', borderRadius: '6px', border: '1px solid rgba(59,130,246,0.35)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '10px' }}>
        <span style={{ color: '#fff', fontWeight: 600 }}>Flower federation · {status.job_id}</span>
        {status.fl_round && <span style={{ color: '#3b82f6' }}>Round {status.fl_round}{status.total_rounds ? ` of ${status.total_rounds}` : ''}</span>}
      </div>
      <ol style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', listStyle: 'none', padding: 0, margin: 0 }}>
        {FLOWER_STEPS.map((s, i) => {
          const done = current > i || status.state === 'done';
          const now = i === current && status.state !== 'done';
          return (
            <li key={s.state} aria-current={now ? 'step' : undefined} style={{
              display: 'flex', alignItems: 'center', gap: '5px', fontSize: '11px', padding: '4px 8px', borderRadius: '999px',
              border: `1px solid ${now ? '#3b82f6' : 'var(--border-subtle)'}`,
              background: now ? 'rgba(59,130,246,0.15)' : 'transparent',
              color: done ? '#10b981' : now ? '#93c5fd' : 'var(--text-muted)',
            }}>
              {done ? <CheckCircle size={11} /> : now ? <Loader size={11} style={{ animation: 'spin 1.2s linear infinite' }} /> : null}
              {s.label}
            </li>
          );
        })}
      </ol>
      {(status.message || problem) && (
        <div style={{ marginTop: '8px', fontSize: '11px', color: problem || 'var(--text-secondary)', display: 'flex', gap: '6px', alignItems: 'center' }}>
          {problem && <AlertTriangle size={12} />} {status.message}
        </div>
      )}
    </div>
  );
};

export default FlowerSteps;
