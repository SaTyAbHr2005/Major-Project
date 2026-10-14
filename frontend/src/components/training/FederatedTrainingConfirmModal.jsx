import React from 'react';
import Modal from '../common/Modal';
import { Network, PlayCircle, ShieldCheck, AlertTriangle } from 'lucide-react';

/** Confirms joining a federation round; training then runs locally like any Module 7 run. */
const FederatedTrainingConfirmModal = ({ isOpen, onClose, onConfirm, job, round, datasetName, eligibility, starting, error }) => {
  if (!job) return null;
  const epochs = job.training_requirements?.epochs;

  return (
    <Modal isOpen={isOpen} onClose={starting ? () => {} : onClose} title="Confirm Federated Training Participation" maxWidth="560px">
      <div>
        <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '16px' }}>
          Your hospital trains the job's model on its local dataset, starting from the round's canonical base model.
          When training finishes, only the resulting model parameters are submitted to the federation round.
        </p>

        <div style={{ backgroundColor: '#0a101d', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-lg)', padding: '14px', marginBottom: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600, marginBottom: '10px' }}>
            <Network size={12} /> Federation Run
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px', fontSize: '12px' }}>
            {[
              ['Job', job.federation_job_id], ['Round', round?.round_id || 'no open round'],
              ['Task', job.task], ['Model', job.architecture],
              ['Classes', job.num_classes], ['Epochs', epochs ?? "Module 8's setting"],
              ['Strategy', job.aggregation_configuration?.strategy === 'fedprox' ? `FedProx (μ ${job.training_requirements?.proximal_mu})` : 'FedAvg'],
              ['Planned rounds', job.plan?.total_rounds && round ? `${round.round_number} of ${job.plan.total_rounds}` : (round?.round_number ?? '-')],
              ['Dataset', datasetName], ['Training samples', eligibility?.train_sample_count?.toLocaleString() ?? '-'],
            ].map(([k, v]) => (
              <div key={k}>
                <span className="text-muted">{k}:</span>
                <div className="font-mono text-primary" style={{ fontWeight: 600, wordBreak: 'break-all' }}>{String(v)}</div>
              </div>
            ))}
          </div>
        </div>

        <div style={{
          padding: '10px 12px', backgroundColor: 'rgba(16, 185, 129, 0.08)', border: '1px solid rgba(16, 185, 129, 0.25)',
          borderRadius: 'var(--radius-md)', marginBottom: error ? '12px' : '20px', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '8px'
        }}>
          <ShieldCheck size={16} color="var(--status-healthy)" />
          <span style={{ color: 'var(--text-secondary)' }}>
            Images and patient data stay on this machine. You can follow the run live in the Training Monitor.
          </span>
        </div>

        {error && (
          <div style={{
            padding: '10px 12px', backgroundColor: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: 'var(--radius-md)', marginBottom: '20px', fontSize: '12px', color: '#f87171',
            display: 'flex', alignItems: 'center', gap: '8px'
          }}>
            <AlertTriangle size={16} /> <span>{error}</span>
          </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
          <button onClick={onClose} className="btn btn-secondary" disabled={starting}>Cancel</button>
          <button onClick={onConfirm} className="btn btn-teal" disabled={starting || !round}>
            <PlayCircle size={14} /> {starting ? 'Starting...' : 'Start Federated Training'}
          </button>
        </div>
      </div>
    </Modal>
  );
};

export default FederatedTrainingConfirmModal;
