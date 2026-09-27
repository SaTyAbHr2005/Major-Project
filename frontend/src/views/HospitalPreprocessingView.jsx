import React, { useState } from 'react';
import { useApp } from '../context/AppContext';
import { useML } from '../context/MLContext';
import preprocessingService from '../services/preprocessingService';
import PrivacyNotice from '../components/common/PrivacyNotice';
import PipelineStatus from '../components/training/PipelineStatus';
import { DistributionBars, ReportText } from '../components/charts/MLCharts';
import {
  Cpu,
  Play,
  CheckCircle,
  ShieldCheck,
  Sliders,
  AlertTriangle,
  ArrowRight
} from 'lucide-react';

const inputStyle = {
  width: '100%', padding: '7px 10px', backgroundColor: '#090d16', border: '1px solid var(--border-subtle)',
  borderRadius: 'var(--radius-md)', color: 'var(--text-primary)', fontSize: '12px', outline: 'none', fontFamily: 'var(--font-mono)'
};

const PreprocessingView = () => {
  const { setActiveTab } = useApp();
  const { activeDataset, pipeline, rerunPreprocessing } = useML();
  const [config, setConfig] = useState(preprocessingService.getDefaultConfig());
  const report = activeDataset?.preprocessing;
  const sv = report?.split_validation;
  const leakage = [
    ...(sv?.duplicate_leakage || []).map((x) => `Duplicate across splits: ${JSON.stringify(x)}`),
    ...Object.entries(sv?.group_leakage || {}).map(([k, v]) => `Group leakage ${k}: ${JSON.stringify(v)}`),
  ];
  const set = (k) => (e) => setConfig({ ...config, [k]: e.target.value });

  return (
    <div style={{ padding: '20px', overflowY: 'auto', height: '100%' }}>
      {/* Privacy Notice */}
      <PrivacyNotice />

      {/* Header */}
      <div className="card" style={{ marginBottom: '16px' }}>
        <div className="card-title" style={{ justifyContent: 'space-between' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Cpu size={16} color="var(--accent-teal)" />
            <span>Module 5: Automated Preprocessing Engine</span>
          </span>
          <span className="badge badge-teal font-mono">Runs Locally</span>
        </div>
        <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
          Validates every sample, quarantines unusable files, and builds reproducible, seed-controlled, class-aware
          train/validation/test splits with leakage checks. Runs automatically after inspection with the defaults below;
          re-run here with a patient/group-id map for patient-level splitting. Source files are never modified.
        </p>
      </div>

      <PipelineStatus />

      {/* Target Dataset & Pipeline Config */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px', marginBottom: '16px' }}>
        <div className="card">
          <div className="card-title-muted" style={{ marginBottom: '8px' }}>Target Dataset</div>
          {activeDataset ? (
            <div>
              <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '4px' }}>{activeDataset.name}</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '10px' }}>
                ID: <span className="font-mono text-cyan">{activeDataset.dataset_id}</span> | Type: {activeDataset.profile?.dataset_type}
              </div>
              <div style={{ padding: '8px 10px', backgroundColor: '#090d16', borderRadius: 'var(--radius-sm)', fontSize: '11px', color: 'var(--text-secondary)' }}>
                Input: <strong>{activeDataset.profile?.valid_samples?.toLocaleString()} valid samples</strong> across {(activeDataset.profile?.classes || []).length} classes
              </div>
            </div>
          ) : (
            <div className="text-muted" style={{ fontSize: '12px' }}>No dataset located yet. Go to Dataset Inspection.</div>
          )}
        </div>

        <div className="card">
          <div className="card-title">
            <Sliders size={14} color="var(--brand-blue)" />
            <span>Preprocessing Options (CLI flags)</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '11px' }}>
            <label>
              <span className="text-muted">Output mode (--mode)</span>
              <select value={config.mode} onChange={set('mode')} style={inputStyle}>
                <option value="lazy">lazy (manifests, no copy)</option>
                <option value="materialized">materialized (processed copy)</option>
              </select>
            </label>
            <label>
              <span className="text-muted">Invalid existing split</span>
              <select value={config.invalid_split_policy} onChange={set('invalid_split_policy')} style={inputStyle}>
                <option value="error">error (stop)</option>
                <option value="regenerate">regenerate</option>
              </select>
            </label>
            <label style={{ gridColumn: 'span 2' }}>
              <span className="text-muted">Patient/group id map JSON (--group-id-map, optional)</span>
              <input value={config.group_id_map_path} onChange={set('group_id_map_path')} placeholder="C:\path\group_id_map.json" style={inputStyle} />
            </label>
            <label style={{ gridColumn: 'span 2' }}>
              <span className="text-muted">Tabular target column (--target-column, optional)</span>
              <input value={config.target_column} onChange={set('target_column')} style={inputStyle} />
            </label>
          </div>
        </div>
      </div>

      {/* Execution Controls */}
      <div className="card" style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>Re-run Preprocessing Pipeline</div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              Replaces this dataset's previous preprocessing output, then refreshes the Module 8 recommendations.
            </div>
          </div>
          <button onClick={() => rerunPreprocessing(config)} disabled={pipeline.running || !activeDataset?.profile}
            className="btn btn-teal" style={{ padding: '8px 16px', fontSize: '12px' }}>
            <Play size={14} />
            {pipeline.running ? 'Pipeline Running...' : 'Run Preprocessing'}
          </button>
        </div>
      </div>

      {/* Preprocessing Summary Report */}
      {report && (
        <div className="card" style={{ borderColor: 'rgba(16, 185, 129, 0.4)' }}>
          <div className="card-title" style={{ justifyContent: 'space-between' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <CheckCircle size={16} color="var(--status-healthy)" />
              <span>Module 5 Preprocessing Report</span>
            </span>
            <span className={`badge font-mono ${report.reconciliation?.is_balanced ? 'badge-healthy' : 'badge-danger'}`}>
              ACCOUNTING: {report.reconciliation?.is_balanced ? 'BALANCED' : 'MISMATCH'}
            </span>
          </div>

          {/* Leakage Check Card */}
          <div style={{
            padding: '12px', backgroundColor: '#0a1622', borderRadius: 'var(--radius-md)', marginBottom: '14px',
            border: `1px solid ${leakage.length ? 'var(--status-danger-border)' : 'rgba(16, 185, 129, 0.3)'}`,
            display: 'flex', alignItems: 'flex-start', gap: '10px'
          }}>
            {leakage.length ? <AlertTriangle size={18} color="var(--status-danger)" /> : <ShieldCheck size={18} color="var(--status-healthy)" style={{ flexShrink: 0, marginTop: '2px' }} />}
            <div style={{ fontSize: '11px' }}>
              <div style={{ fontWeight: 600, color: leakage.length ? 'var(--status-danger)' : 'var(--status-healthy)', marginBottom: '2px' }}>
                Split leakage checks: {leakage.length ? `${leakage.length} issue(s)` : 'no duplicate or group leakage detected'}
              </div>
              <p style={{ color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                Split strategy: <strong>{report.config?.split_strategy}</strong>.{' '}
                {report.config?.split_strategy === 'grouped'
                  ? 'Groups (patients) never span two splits.'
                  : 'No patient/group ids were supplied, so patient-level leakage cannot be ruled out for this dataset.'}
              </p>
              {leakage.map((l, i) => <div key={i} className="font-mono text-danger">{l}</div>)}
            </div>
          </div>

          {/* Accounting Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px', marginBottom: '14px', fontSize: '11px' }}>
            {[
              ['Discovered', report.quarantine?.total_discovered, 'text-primary'],
              ['Accepted', report.quarantine?.accepted, 'text-emerald'],
              ['Rejected (quarantined)', report.quarantine?.rejected, 'text-danger'],
              ['Needs review', report.quarantine?.review, 'text-amber'],
            ].map(([k, v, cls]) => (
              <div key={k} style={{ padding: '10px', backgroundColor: '#090d16', borderRadius: 'var(--radius-sm)' }}>
                <span className="text-muted">{k}:</span>
                <div className={`font-mono ${cls}`} style={{ fontSize: '14px', fontWeight: 600, marginTop: '2px' }}>{(v ?? 0).toLocaleString()}</div>
              </div>
            ))}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px', marginBottom: '14px' }}>
            <div>
              <div className="card-title-muted" style={{ marginBottom: '8px' }}>Split sizes</div>
              <DistributionBars distribution={report.splits} colors={['#3b82f6', '#06b6d4', '#10b981']} />
            </div>
            <div>
              <div className="card-title-muted" style={{ marginBottom: '8px' }}>preprocessing_report.md</div>
              {activeDataset.preprocessing_report_md && <ReportText text={activeDataset.preprocessing_report_md} />}
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
            <button onClick={() => setActiveTab('start_training')} className="btn btn-teal">
              Proceed to Start Training (Module 8) <ArrowRight size={14} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default PreprocessingView;
