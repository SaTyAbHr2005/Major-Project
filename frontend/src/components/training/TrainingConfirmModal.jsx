import React from 'react';
import Modal from '../common/Modal';
import { PlayCircle, ShieldCheck } from 'lucide-react';

const TrainingConfirmModal = ({ isOpen, onClose, onConfirm, recommendation, datasetName, requestId }) => {
  if (!recommendation) return null;
  const r = recommendation;

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Confirm Module 7 Local Training Launch" maxWidth="560px">
      <div>
        <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '16px' }}>
          <strong>Module 8</strong> resolves this recommendation into a Module 7 TrainingConfig and runs
          <strong> Module 7</strong> training on this workstation, followed by test-set evaluation and evaluation plots.
        </p>

        <div style={{ backgroundColor: '#0a101d', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-lg)', padding: '14px', marginBottom: '16px' }}>
          <div style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600, marginBottom: '10px' }}>Target Training Parameters</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px', fontSize: '12px' }}>
            {[
              ['Dataset', datasetName], ['Model', `${r.display_name} (${r.model})`], ['Device', r.device], ['Precision', r.precision],
              ['Batch Size', r.batch_size], ['Epochs', r.epochs], ['DataLoader Workers', r.num_workers], ['Estimated Duration', r.estimated_training_time_display],
              ['Consortium request', requestId || 'not linked (local run only)'],
            ].map(([k, v]) => (
              <div key={k}>
                <span className="text-muted">{k}:</span>
                <div className="font-mono text-primary" style={{ fontWeight: 600 }}>{String(v)}</div>
              </div>
            ))}
          </div>
        </div>

        <div style={{
          padding: '10px 12px', backgroundColor: 'rgba(16, 185, 129, 0.08)', border: '1px solid rgba(16, 185, 129, 0.25)',
          borderRadius: 'var(--radius-md)', marginBottom: '20px', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '8px'
        }}>
          <ShieldCheck size={16} color="var(--status-healthy)" />
          <span style={{ color: 'var(--text-secondary)' }}>
            Training reads the dataset on this machine only. If a consortium request is linked, only aggregate
            progress (epoch, loss, accuracy) is reported to the backend - never images or patient data.
          </span>
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
          <button onClick={onClose} className="btn btn-secondary">Cancel</button>
          <button onClick={onConfirm} className="btn btn-teal"><PlayCircle size={14} /> Start Training</button>
        </div>
      </div>
    </Modal>
  );
};

export default TrainingConfirmModal;
