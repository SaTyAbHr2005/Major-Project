import React from 'react';
import { CheckCircle, Clock } from 'lucide-react';

const BADGES = {
  recommended: ['Balanced Trade-Off', 'badge-healthy'],
  high_capacity: ['More Capacity / More Time', 'badge-purple'],
  fast: ['Fastest / Lower Resource', 'badge-teal'],
};

/** One Module 8 RecommendedConfig, exactly as `resource_training recommend` produced it. */
const RecommendationCard = ({ recommendation, isSelected, onSelect, isAlternative = false }) => {
  const r = recommendation;
  const [badge, badgeColor] = BADGES[r.recommendation_type] || [r.label, 'badge-neutral'];

  return (
    <div
      className="card"
      style={{
        border: isSelected ? '2px solid var(--accent-teal)' : isAlternative ? '1px dashed var(--border-subtle)'
          : r.recommendation_type === 'recommended' ? '1px solid rgba(6, 182, 212, 0.4)' : '1px solid var(--border-subtle)',
        backgroundColor: isSelected ? 'rgba(8, 145, 178, 0.08)' : 'var(--bg-card)',
        display: 'flex', flexDirection: 'column', justifyContent: 'space-between', transition: 'all 0.2s ease'
      }}
    >
      <div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
          <span className={`badge ${isAlternative ? 'badge-neutral' : badgeColor}`} style={{ fontSize: '10px' }}>
            {isAlternative ? 'Same-Tier Alternative' : badge}
          </span>
          <span className={`badge ${r.safe ? 'badge-healthy' : 'badge-danger'}`} style={{ fontSize: '10px' }}>{r.safe ? 'SAFE FOR THIS HARDWARE' : 'NOT SAFE'}</span>
        </div>

        <div style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px' }}>{r.label}</div>
        <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--accent-teal)', marginBottom: '12px' }}>
          {r.display_name} <span className="text-muted">({r.model})</span>
        </div>

        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px', padding: '10px', backgroundColor: '#090d16',
          borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)', fontSize: '11px', marginBottom: '12px'
        }}>
          {[['Device', r.device], ['Precision', r.precision], ['Batch Size', r.batch_size], ['Epochs', r.epochs], ['DataLoader Workers', r.num_workers], ['Estimate basis', r.estimation_method]].map(([k, v]) => (
            <div key={k}>
              <span className="text-muted">{k}:</span>
              <div className="font-mono text-primary" style={{ fontWeight: 600 }}>{String(v)}</div>
            </div>
          ))}
        </div>

        <div style={{ padding: '8px 10px', backgroundColor: '#0f172a', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)', marginBottom: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '3px' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '11px', color: 'var(--text-secondary)' }}>
              <Clock size={13} color="var(--accent-teal)" /> Estimated Duration:
            </span>
            <span className="font-mono text-emerald" style={{ fontWeight: 700, fontSize: '12px' }}>{r.estimated_training_time_display}</span>
          </div>
          <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Estimate confidence: {r.estimation_confidence}</div>
        </div>

        <div style={{ fontSize: '11px', marginBottom: '12px', color: 'var(--text-secondary)' }}>
          <div style={{ fontSize: '10px', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600, marginBottom: '4px' }}>Resource Expectation</div>
          {r.resource_summary}
        </div>

        <div style={{ fontSize: '11px', marginBottom: '14px' }}>
          <div style={{ color: 'var(--text-primary)', marginBottom: '4px' }}>
            <strong>Rationale:</strong> <span style={{ color: 'var(--text-secondary)' }}>{r.reason}</span>
          </div>
          <div style={{ color: 'var(--text-primary)' }}>
            <strong>Trade-offs:</strong> <span style={{ color: 'var(--text-secondary)' }}>{r.tradeoff}</span>
          </div>
        </div>
      </div>

      <button onClick={() => onSelect(r)} className={isSelected ? 'btn btn-teal' : 'btn btn-secondary'} style={{ width: '100%', marginTop: '8px' }}>
        {isSelected ? <><CheckCircle size={14} /> Selected Configuration</> : 'Select Configuration'}
      </button>
    </div>
  );
};

export default RecommendationCard;
