import React, { useEffect, useMemo, useState } from 'react';
import { Sliders, Gauge, CheckCircle } from 'lucide-react';

const EPOCH_MAX = 40; // Module 8 policy.epoch_max (the CLI re-validates)

const inputStyle = {
  width: '100%', padding: '7px 10px', backgroundColor: '#090d16', border: '1px solid var(--border-subtle)',
  borderRadius: 'var(--radius-md)', color: 'var(--text-primary)', fontSize: '12px', outline: 'none'
};

const level = (pct) => {
  if (pct == null) return { text: 'n/a', color: 'var(--text-muted)' };
  if (pct < 50) return { text: 'Low', color: 'var(--status-healthy)' };
  if (pct < 75) return { text: 'Moderate', color: 'var(--status-warning)' };
  if (pct <= 100) return { text: 'High', color: '#f97316' };
  return { text: 'Over budget', color: 'var(--status-danger)' };
};

const Meter = ({ label, pct, detail, note }) => {
  const l = level(pct);
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: '4px' }}>
        <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{label}</span>
        <span className="font-mono" style={{ color: l.color, fontWeight: 700 }}>{pct == null ? '-' : `${Math.round(pct)}%`} · {l.text}</span>
      </div>
      <div style={{ height: '10px', backgroundColor: '#090d16', borderRadius: '5px', overflow: 'hidden', border: '1px solid var(--border-subtle)' }}>
        <div style={{ height: '100%', width: `${Math.min(100, pct || 0)}%`, backgroundColor: l.color, transition: 'width 0.25s ease, background-color 0.25s ease' }} />
      </div>
      <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '3px' }}>{detail}{note ? ` - ${note}` : ''}</div>
    </div>
  );
};

/**
 * Operator-chosen model / batch size / epochs with a live load meter. Only models Module 8
 * assessed as feasible are offered, and only batch sizes up to the largest one Module 8
 * measured as safe for that model on this machine.
 */
const ManualTrainingPanel = ({ assessments, isSelected, onSelect }) => {
  const feasible = useMemo(() => Object.values(assessments || {}).filter((a) => a.feasible && a.batch_size), [assessments]);
  // Start from Module 8's Recommended model; the operator can change everything.
  const [arch, setArch] = useState((feasible.find((m) => m.role === 'recommended') || feasible[0])?.architecture || '');
  const a = feasible.find((x) => x.architecture === arch) || feasible[0];
  const batches = Object.keys(a?.batch_memory_mb || {}).map(Number).sort((x, y) => x - y);
  const [batch, setBatch] = useState(a?.batch_size || 1);
  const [epochs, setEpochs] = useState(a?.epochs || 10);

  // New model -> start from Module 8's own settings for it.
  useEffect(() => {
    if (a) { setBatch(a.batch_size); setEpochs(a.epochs); }
  }, [a?.architecture]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!a) return null;

  // Editing a selected manual configuration un-selects it, so a stale choice is never launched.
  const changed = () => { if (isSelected) onSelect(null); };

  const memMb = a.batch_memory_mb?.[String(batch)];
  const memPct = memMb != null && a.available_memory_mb ? (100 * memMb) / a.available_memory_mb : null;
  const measuredMem = batch === a.batch_size && a.memory_estimate?.measured;
  const perEpoch = (a.time_estimate?.estimated_training_time_seconds || 0) / (a.epochs || 1);
  const totalSec = perEpoch * epochs;
  const timePct = a.time_budget_seconds ? (100 * totalSec) / a.time_budget_seconds : null;
  const cpuPct = a.cpu_logical_cores ? (100 * (a.num_workers + 1)) / a.cpu_logical_cores : null;
  const overall = level(Math.max(memPct || 0, timePct || 0, cpuPct || 0));
  const epochsValid = Number.isInteger(epochs) && epochs >= 1 && epochs <= EPOCH_MAX;
  const kind = a.device === 'cuda' ? 'GPU memory (VRAM)' : 'System memory (RAM)';

  const select = () => onSelect({
    key: 'manual', architecture: a.architecture, batch_size: batch, epochs,
    config: {
      label: 'Manual Configuration', display_name: a.display_name, model: a.architecture, device: a.device,
      precision: a.precision, batch_size: batch, epochs, num_workers: a.num_workers,
      estimated_training_time_display: `~${(totalSec / 60).toFixed(1)} minutes (estimate)`,
    },
  });

  return (
    <div className="card" style={{ marginBottom: '20px', border: isSelected ? '2px solid var(--accent-teal)' : '1px solid var(--border-subtle)' }}>
      <div className="card-title" style={{ justifyContent: 'space-between' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><Sliders size={15} color="var(--brand-blue)" /> Manual Configuration</span>
        <span className="badge badge-neutral" style={{ fontSize: '10px' }}>Limits measured by Module 8 on this machine</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: '20px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '11px' }}>
          <label>
            <span className="text-muted">Model (feasible on this machine)</span>
            <select value={a.architecture} onChange={(e) => { changed(); setArch(e.target.value); }} style={inputStyle}>
              {feasible.map((m) => (
                <option key={m.architecture} value={m.architecture}>
                  {m.display_name} ({m.resource_tier.replace('_', ' ')}, {(m.param_count / 1e6).toFixed(1)}M params) - {m.status_label}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span className="text-muted">Batch size (max safe for this model: {a.batch_size})</span>
            <select value={batch} onChange={(e) => { changed(); setBatch(Number(e.target.value)); }} style={inputStyle}>
              {batches.map((b) => <option key={b} value={b}>{b}{b === a.batch_size ? ' (Module 8 choice)' : ''}</option>)}
            </select>
          </label>
          <label>
            <span className="text-muted">Epochs (1-{EPOCH_MAX}; Module 8 suggests {a.epochs})</span>
            <input type="number" min={1} max={EPOCH_MAX} value={epochs}
              onChange={(e) => { changed(); setEpochs(Math.round(Number(e.target.value))); }} style={{ ...inputStyle, fontFamily: 'var(--font-mono)' }} />
          </label>
          <div style={{ fontSize: '10px', color: 'var(--text-muted)', lineHeight: 1.4 }}>
            Device {a.device} / {a.precision}, {a.num_workers} DataLoader workers. Module 8 re-checks everything before training
            and refuses anything unsafe.
          </div>
          <button onClick={select} disabled={!epochsValid} className={isSelected ? 'btn btn-teal' : 'btn btn-secondary'}>
            {isSelected ? <><CheckCircle size={14} /> Manual Configuration Selected</> : 'Use This Manual Configuration'}
          </button>
        </div>

        <div style={{ padding: '12px', backgroundColor: '#0a101d', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)' }}>
              <Gauge size={15} color="var(--accent-teal)" /> Expected Load on This Machine
            </span>
            <span className="font-mono" style={{ fontSize: '12px', fontWeight: 700, color: overall.color }}>{overall.text.toUpperCase()}</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <Meter label={kind} pct={memPct}
              detail={memMb != null ? `~${(memMb / 1024).toFixed(2)} GB of ${(a.available_memory_mb / 1024).toFixed(2)} GB available` : 'unknown'}
              note={measuredMem ? 'measured (real dry run)' : 'estimated from the measured dry run'} />
            <Meter label="Training time vs time budget" pct={timePct}
              detail={`~${(totalSec / 60).toFixed(1)} min for ${epochs} epochs (budget ${(a.time_budget_seconds / 60).toFixed(0)} min)`}
              note={batch !== a.batch_size ? `estimated at batch ${a.batch_size}; smaller batches are usually slower` : 'estimate'} />
            <Meter label="CPU (data loading)" pct={cpuPct}
              detail={`${a.num_workers} workers + main process of ${a.cpu_logical_cores ?? '?'} CPU threads`} />
          </div>
          <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '10px' }}>
            Load estimates from Module 8 for this machine; actual usage is recorded in the run's resource statistics.
          </div>
        </div>
      </div>
    </div>
  );
};

export default ManualTrainingPanel;
