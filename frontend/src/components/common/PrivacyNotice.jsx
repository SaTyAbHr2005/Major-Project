import React from 'react';
import { ShieldAlert, ShieldCheck } from 'lucide-react';

const PrivacyNotice = ({ compact = false }) => {
  if (compact) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        padding: '6px 12px',
        backgroundColor: 'rgba(16, 185, 129, 0.08)',
        border: '1px solid rgba(16, 185, 129, 0.25)',
        borderRadius: 'var(--radius-md)',
        fontSize: '11px',
        color: 'var(--text-secondary)'
      }}>
        <ShieldCheck size={14} color="var(--status-healthy)" />
        <span>
          <strong style={{ color: 'var(--status-healthy)' }}>Privacy Guarantee:</strong> Raw medical images, DICOMs, and patient identifiers strictly remain local to this hospital node.
        </span>
      </div>
    );
  }

  return (
    <div style={{
      display: 'flex',
      alignItems: 'flex-start',
      gap: '12px',
      padding: '12px 16px',
      backgroundColor: '#0a121e',
      border: '1px solid rgba(16, 185, 129, 0.3)',
      borderRadius: 'var(--radius-lg)',
      marginBottom: '16px'
    }}>
      <div style={{
        padding: '6px',
        borderRadius: 'var(--radius-md)',
        backgroundColor: 'rgba(16, 185, 129, 0.15)',
        color: 'var(--status-healthy)'
      }}>
        <ShieldCheck size={20} />
      </div>
      <div>
        <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--status-healthy)', marginBottom: '2px' }}>
          Hospital data boundary
        </div>
        <p style={{ fontSize: '11px', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
          Raw scans stay on this machine: Module 4 inspection, Module 5 preprocessing and local training run here
          without uploading images. Federated learning (Module 9) sends model parameter updates and summary metrics.
          Differential privacy (Module 10) is chosen per job: in a DP job this machine trains with DP-SGD (clipped,
          noised gradients) and its privacy cost ε is tracked; in other jobs updates are sent without noise.
        </p>
      </div>
    </div>
  );
};

export default PrivacyNotice;
