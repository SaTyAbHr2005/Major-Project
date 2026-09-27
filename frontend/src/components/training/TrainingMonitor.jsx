import React from 'react';
import { Activity, CheckCircle, PauseCircle } from 'lucide-react';
import { EpochLineChart, JobLog } from '../charts/MLCharts';

const formatTime = (secs) => {
  const s = Math.max(0, Math.round(secs || 0));
  return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s`;
};

const STATUS_BADGE = { running: 'badge-warning', succeeded: 'badge-healthy', failed: 'badge-danger', cancelled: 'badge-neutral' };

/** Live view of a local training job (resource_training train ... run by the local ML service). */
const TrainingMonitor = ({ job, onHalt, onViewResults }) => {
  if (!job) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '40px' }}>
        <Activity size={32} color="var(--text-muted)" style={{ margin: '0 auto 12px' }} />
        <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '4px' }}>No Active Local Training Session</div>
        <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Select a dataset and launch a resource-aware recommendation in the Start Training view.</p>
      </div>
    );
  }

  const { status, meta, epochs, log, elapsed_seconds } = job;
  const last = epochs[epochs.length - 1];
  const total = last?.total_epochs;
  const isRunning = status === 'running';
  const epochProgress = total ? Math.round((100 * last.epoch) / total) : 0;
  const perEpoch = last ? last.elapsed_seconds / last.epoch : null;
  const eta = perEpoch && total ? perEpoch * (total - last.epoch) : null;
  const pct = (v) => (typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : '-');
  const num = (v) => (typeof v === 'number' ? v.toFixed(4) : '-');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div className="card" style={{ border: isRunning ? '1px solid rgba(6, 182, 212, 0.4)' : '1px solid var(--border-subtle)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span className={`pulse-dot ${isRunning ? 'warning' : status === 'succeeded' ? 'healthy' : 'danger'}`} />
            <div>
              <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)' }}>
                Model: <span className="font-mono text-cyan">{meta?.model_id}</span>
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                Dataset: <strong className="text-primary">{meta?.dataset_name}</strong> | Recommendation: <strong className="text-primary">{meta?.choice}</strong>
                {meta?.request_id && <> | Consortium request: <strong className="text-primary">{meta.request_id}</strong></>}
              </div>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span className={`badge ${STATUS_BADGE[status] || 'badge-neutral'}`}>{status.toUpperCase()}</span>
            {isRunning && (
              <button onClick={onHalt} className="btn btn-danger" style={{ padding: '4px 8px', fontSize: '11px' }}>
                <PauseCircle size={13} /> Terminate
              </button>
            )}
            {status === 'succeeded' && onViewResults && (
              <button onClick={onViewResults} className="btn btn-teal" style={{ padding: '4px 10px', fontSize: '11px' }}>
                <CheckCircle size={13} /> View Full TrainingResult
              </button>
            )}
          </div>
        </div>

        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: '4px' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Epoch Progress:</span>
            <span className="font-mono text-primary" style={{ fontWeight: 600 }}>
              {last ? `${last.epoch} / ${total} (${epochProgress}%)` : isRunning ? 'preparing data & model...' : '-'}
            </span>
          </div>
          <div style={{ height: '8px', backgroundColor: '#090d16', borderRadius: '4px', overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${epochProgress}%`, background: 'linear-gradient(90deg, #0891b2 0%, #10b981 100%)', transition: 'width 0.3s ease' }} />
          </div>
        </div>

        <div style={{ display: 'flex', gap: '24px', marginTop: '14px', paddingTop: '12px', borderTop: '1px solid var(--border-subtle)', fontSize: '11px' }}>
          <div><span className="text-muted">Elapsed Time:</span>{' '}<span className="font-mono text-primary" style={{ fontWeight: 600 }}>{formatTime(elapsed_seconds)}</span></div>
          <div><span className="text-muted">Estimated Remaining:</span>{' '}
            <span className="font-mono text-cyan" style={{ fontWeight: 600 }}>{isRunning ? (eta != null ? `~${formatTime(eta)} + test evaluation` : 'after first epoch') : '0s'}</span>
          </div>
          <div><span className="text-muted">Avg epoch time:</span>{' '}<span className="font-mono text-primary" style={{ fontWeight: 600 }}>{perEpoch ? formatTime(perEpoch) : '-'}</span></div>
        </div>
        {job.error && <div className="text-danger" style={{ fontSize: '11px', marginTop: '10px' }}>{job.error}</div>}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
        {[
          ['Train Loss', num(last?.train_loss), 'text-cyan', `Train Acc: ${pct(last?.train_acc)}`],
          ['Validation Loss', num(last?.val_loss), 'text-primary', 'Lowest val loss is checkpointed'],
          ['Validation Accuracy', pct(last?.val_acc), 'text-emerald', `Epoch ${last?.epoch ?? '-'}`],
          ['Checkpoints', `${epochs.length} epoch(s)`, 'text-primary', 'Resource stats recorded at run end'],
        ].map(([k, v, cls, note]) => (
          <div className="card" style={{ padding: '12px' }} key={k}>
            <div className="card-title-muted">{k}</div>
            <div className={`font-mono ${cls}`} style={{ fontSize: '20px', fontWeight: 700, marginTop: '4px' }}>{v}</div>
            <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px' }}>{note}</div>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px' }}>
        <div className="card">
          <div className="card-title"><span>Loss per Epoch</span></div>
          <EpochLineChart epochs={epochs} series={[{ key: 'train_loss', label: 'train loss', color: '#38bdf8' }, { key: 'val_loss', label: 'val loss', color: '#f59e0b' }]} />
        </div>
        <div className="card">
          <div className="card-title"><span>Accuracy per Epoch</span></div>
          <EpochLineChart epochs={epochs} yMax={1} series={[{ key: 'train_acc', label: 'train acc', color: '#10b981' }, { key: 'val_acc', label: 'val acc', color: '#a78bfa' }]} />
        </div>
      </div>

      <div className="card">
        <div className="card-title" style={{ justifyContent: 'space-between' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><Activity size={14} color="var(--accent-teal)" /> Module 8 / Module 7 Execution Stream</span>
          <span className="badge badge-teal font-mono" style={{ fontSize: '10px' }}>STDOUT</span>
        </div>
        <JobLog lines={log} height={180} />
      </div>
    </div>
  );
};

export default TrainingMonitor;
