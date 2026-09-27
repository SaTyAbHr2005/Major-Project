import React from 'react';

/** Per-epoch curves from the trainer's "[epoch i/n] ..." lines (real values only). */
export const EpochLineChart = ({ epochs, series, height = 180, yMax }) => {
  const width = 520;
  const pad = { l: 40, r: 12, t: 10, b: 24 };
  const points = epochs || [];
  if (points.length === 0) {
    return <div className="text-muted" style={{ fontSize: '11px', padding: '20px', textAlign: 'center' }}>Waiting for the first completed epoch...</div>;
  }
  const values = series.flatMap((s) => points.map((p) => p[s.key]).filter((v) => typeof v === 'number'));
  const top = yMax ?? Math.max(...values, 0.0001) * 1.1;
  const maxEpoch = Math.max(points[0].total_epochs || 1, points[points.length - 1].epoch);
  const x = (e) => pad.l + ((e - 1) / Math.max(maxEpoch - 1, 1)) * (width - pad.l - pad.r);
  const y = (v) => pad.t + (1 - v / top) * (height - pad.t - pad.b);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * top);

  return (
    <div>
      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height }}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={width - pad.r} y1={y(t)} y2={y(t)} stroke="rgba(255,255,255,0.06)" />
            <text x={pad.l - 6} y={y(t) + 3} textAnchor="end" fontSize="9" fill="var(--text-muted)">{t.toFixed(2)}</text>
          </g>
        ))}
        {points.map((p) => (
          <text key={p.epoch} x={x(p.epoch)} y={height - 8} textAnchor="middle" fontSize="9" fill="var(--text-muted)">{p.epoch}</text>
        ))}
        {series.map((s) => {
          const pts = points.filter((p) => typeof p[s.key] === 'number');
          if (pts.length === 0) return null;
          return (
            <g key={s.key}>
              <polyline fill="none" stroke={s.color} strokeWidth="2" points={pts.map((p) => `${x(p.epoch)},${y(p[s.key])}`).join(' ')} />
              {pts.map((p) => <circle key={p.epoch} cx={x(p.epoch)} cy={y(p[s.key])} r="2.5" fill={s.color} />)}
            </g>
          );
        })}
      </svg>
      <div style={{ display: 'flex', gap: '14px', justifyContent: 'center', fontSize: '10px', color: 'var(--text-secondary)' }}>
        {series.map((s) => (
          <span key={s.key} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span style={{ width: '10px', height: '3px', backgroundColor: s.color, display: 'inline-block' }} /> {s.label}
          </span>
        ))}
        <span className="text-muted">x: epoch</span>
      </div>
    </div>
  );
};

/** Confusion matrix straight from TrainingResult.*_metrics.confusion_matrix. */
export const ConfusionMatrixGrid = ({ matrix, classOrder }) => {
  if (!matrix || !classOrder) return null;
  const max = Math.max(...matrix.flat(), 1);
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ borderCollapse: 'collapse', fontSize: '11px', fontFamily: 'var(--font-mono)' }}>
        <thead>
          <tr>
            <th style={{ padding: '4px 8px', color: 'var(--text-muted)', fontWeight: 400, textAlign: 'left' }}>true \ predicted</th>
            {classOrder.map((c) => <th key={c} style={{ padding: '4px 8px', color: 'var(--text-secondary)' }}>{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={classOrder[i]}>
              <th style={{ padding: '4px 8px', color: 'var(--text-secondary)', textAlign: 'left' }}>{classOrder[i]}</th>
              {row.map((v, j) => (
                <td key={j} style={{
                  padding: '8px 12px', textAlign: 'center', minWidth: '56px',
                  backgroundColor: `rgba(${i === j ? '16,185,129' : '239,68,68'}, ${0.08 + 0.6 * (v / max)})`,
                  color: 'var(--text-primary)', border: '1px solid var(--border-subtle)'
                }}>{v}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

/** Horizontal bars for a {label: count} distribution. */
export const DistributionBars = ({ distribution, colors = ['#10b981', '#f59e0b', '#8b5cf6', '#3b82f6', '#ef4444', '#06b6d4'] }) => {
  const entries = Object.entries(distribution || {});
  const total = entries.reduce((s, [, v]) => s + v, 0) || 1;
  if (entries.length === 0) return <div className="text-muted" style={{ fontSize: '11px' }}>No data.</div>;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
      {entries.map(([label, count], idx) => {
        const pct = (100 * count) / total;
        return (
          <div key={label}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: '3px' }}>
              <span style={{ fontWeight: 600 }}>{label}</span>
              <span className="font-mono text-primary">{count.toLocaleString()} ({pct.toFixed(1)}%)</span>
            </div>
            <div style={{ height: '6px', backgroundColor: '#090d16', borderRadius: '3px', overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${pct}%`, backgroundColor: colors[idx % colors.length] }} />
            </div>
          </div>
        );
      })}
    </div>
  );
};

/** Renders a CLI-generated Markdown report as preformatted text (no HTML injection). */
export const ReportText = ({ text }) => (
  <pre style={{
    whiteSpace: 'pre-wrap', fontFamily: 'var(--font-mono)', fontSize: '11px', lineHeight: 1.5,
    color: 'var(--text-secondary)', backgroundColor: '#070b12', border: '1px solid var(--border-subtle)',
    borderRadius: 'var(--radius-md)', padding: '10px', maxHeight: '260px', overflowY: 'auto', margin: 0
  }}>{text}</pre>
);

/** Live log tail of a local ML job. */
export const JobLog = ({ lines, height = 140 }) => (
  <div style={{
    height, overflowY: 'auto', backgroundColor: '#070b12', padding: '10px', borderRadius: 'var(--radius-md)',
    border: '1px solid var(--border-subtle)', fontFamily: 'var(--font-mono)', fontSize: '11px', lineHeight: 1.6,
    color: 'var(--text-secondary)'
  }}>
    {lines && lines.length > 0
      ? lines.map((l, i) => <div key={i} style={{ color: l.startsWith('[epoch') ? '#38bdf8' : 'inherit', whiteSpace: 'pre-wrap' }}>{l}</div>)
      : <div className="text-muted">Waiting for output...</div>}
  </div>
);
