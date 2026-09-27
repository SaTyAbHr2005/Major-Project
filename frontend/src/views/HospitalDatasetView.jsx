import React, { useState } from 'react';
import { useApp } from '../context/AppContext';
import { useML } from '../context/MLContext';
import datasetService from '../services/datasetService';
import PrivacyNotice from '../components/common/PrivacyNotice';
import PipelineStatus from '../components/training/PipelineStatus';
import { DistributionBars, ReportText } from '../components/charts/MLCharts';
import {
  Search,
  FolderOpen,
  CheckCircle,
  AlertTriangle,
  BarChart2,
  Image,
  Users,
  Layers,
  ArrowRight,
  Database
} from 'lucide-react';

const DatasetView = () => {
  const { setActiveTab } = useApp();
  const { activeDataset, datasets, selectDataset, pipeline, runPipeline, serviceStatus } = useML();
  const [folderInput, setFolderInput] = useState(activeDataset?.source_path || '');
  const [browseError, setBrowseError] = useState(null);
  const [browsing, setBrowsing] = useState(false);

  const handleBrowse = async () => {
    setBrowseError(null);
    setBrowsing(true);
    try {
      const path = await datasetService.browseFolder();
      if (path) {
        setFolderInput(path);
        runPipeline(path);
      }
    } catch (err) {
      setBrowseError(err.message);
    } finally {
      setBrowsing(false);
    }
  };

  const handleScan = (e) => {
    e.preventDefault();
    runPipeline(folderInput);
  };

  const ds = activeDataset;
  const p = ds?.profile;
  const dims = p?.image_statistics?.dimensions;
  const busy = pipeline.running || browsing;

  return (
    <div style={{ padding: '20px', overflowY: 'auto', height: '100%' }}>
      {/* Privacy Notice Banner */}
      <PrivacyNotice />

      {/* Dataset Selection & Local Folder Scanner */}
      <div className="card" style={{ marginBottom: '16px' }}>
        <div className="card-title">
          <FolderOpen size={16} color="var(--accent-teal)" />
          <span>Locate Local Dataset (Module 4 Ingestion)</span>
        </div>
        <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '12px' }}>
          Locate an image folder (class sub-folders, optional train/val/test) or a CSV/XLS/XLSX file on this workstation.
          Inspection (Module 4), preprocessing (Module 5) and hardware-aware training recommendations (Module 8) then run
          automatically on this machine. The dataset is only read - never modified, copied off this machine, or uploaded.
        </p>

        {serviceStatus === 'offline' && (
          <div style={{ marginBottom: '10px', padding: '8px 12px', backgroundColor: 'var(--status-danger-bg)', border: '1px solid var(--status-danger-border)',
            borderRadius: 'var(--radius-md)', color: 'var(--status-danger)', fontSize: '11px' }}>
            Local ML service is offline. Start it on this workstation: <code className="font-mono">python -m hospital_client.local_api</code>
          </div>
        )}

        <form onSubmit={handleScan} style={{ display: 'flex', gap: '10px' }}>
          <button type="button" onClick={handleBrowse} disabled={busy} className="btn btn-secondary" style={{ padding: '0 14px' }}>
            <FolderOpen size={14} /> {browsing ? 'Waiting for folder...' : 'Browse...'}
          </button>
          <div style={{ flex: 1, position: 'relative' }}>
            <input
              type="text"
              value={folderInput}
              onChange={(e) => setFolderInput(e.target.value)}
              placeholder="C:\path\to\dataset"
              style={{
                width: '100%',
                padding: '9px 12px',
                backgroundColor: '#090d16',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md)',
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-mono)',
                fontSize: '12px',
                outline: 'none'
              }}
            />
          </div>
          <button
            type="submit"
            disabled={busy || !folderInput.trim()}
            className="btn btn-teal"
            style={{ padding: '0 16px' }}
          >
            <Search size={14} />
            {pipeline.running ? 'Pipeline Running...' : 'Run Local Pipeline'}
          </button>
        </form>

        {browseError && (
          <div style={{
            marginTop: '10px',
            padding: '8px 12px',
            backgroundColor: 'var(--status-danger-bg)',
            border: '1px solid var(--status-danger-border)',
            borderRadius: 'var(--radius-md)',
            color: 'var(--status-danger)',
            fontSize: '11px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}>
            <AlertTriangle size={14} />
            <span>{browseError}</span>
          </div>
        )}

        {datasets.length > 0 && (
          <div style={{ marginTop: '12px', display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Previously inspected:</span>
            {datasets.map((d) => (
              <button key={d.dataset_id} disabled={busy} onClick={() => selectDataset(d.dataset_id)}
                className={`btn ${ds?.dataset_id === d.dataset_id ? 'btn-teal' : 'btn-secondary'}`}
                style={{ padding: '2px 10px', fontSize: '11px' }} title={d.source_path}>
                <Database size={11} /> {d.name} {d.preprocessed ? '' : '(not preprocessed)'}
              </button>
            ))}
          </div>
        )}
      </div>

      <PipelineStatus />

      {ds && p && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {/* Dataset Header Card */}
          <div className="card" style={{ borderLeft: '4px solid var(--accent-teal)' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <h3 style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)' }}>
                    {ds.name}
                  </h3>
                  <span className="badge badge-teal font-mono">{ds.dataset_id}</span>
                  <span className={`badge ${p.errors?.length ? 'badge-warning' : 'badge-healthy'} font-mono`}>
                    <CheckCircle size={11} /> {ds.preprocessing ? 'INSPECTED + PREPROCESSED' : 'INSPECTED'}
                  </span>
                </div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                  Storage Path: <code className="font-mono text-code">{ds.source_path}</code> | Profiled: {new Date(p.generated_at).toLocaleString()}
                </div>
              </div>

              <div style={{ display: 'flex', gap: '8px' }}>
                <span className="badge badge-blue">Type: {p.dataset_type}</span>
                <span className="badge badge-purple">Formats: {(p.image_statistics?.formats_found || []).join(', ') || 'tabular'}</span>
              </div>
            </div>
          </div>

          {/* Sample Counts */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
            {[
              ['Total Samples', p.total_samples, 'text-primary', 'Files / rows discovered'],
              ['Valid Samples', p.valid_samples, 'text-emerald', 'Readable and usable'],
              ['Invalid / Corrupted', p.invalid_samples, 'text-danger', `Missing: ${p.missing_samples ?? 0}`],
              ['Exact Duplicates', p.duplicate_information?.exact_duplicates ?? 0, 'text-primary', 'SHA-256 / exact row matches'],
            ].map(([label, value, cls, note]) => (
              <div className="card" style={{ padding: '12px' }} key={label}>
                <div className="card-title-muted">{label}</div>
                <div className={`font-mono ${cls}`} style={{ fontSize: '20px', fontWeight: 700, marginTop: '4px' }}>
                  {(value ?? 0).toLocaleString()}
                </div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px' }}>{note}</div>
              </div>
            ))}
          </div>

          {/* Splits and Class Distribution */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px' }}>
            <div className="card">
              <div className="card-title">
                <Layers size={15} color="var(--brand-blue)" />
                <span>Partition Splits (Train / Validation / Test)</span>
              </div>
              {ds.preprocessing ? (
                <>
                  <DistributionBars distribution={ds.preprocessing.splits} colors={['#3b82f6', '#06b6d4', '#10b981']} />
                  <div style={{ marginTop: '12px', fontSize: '11px', color: 'var(--text-secondary)' }}>
                    Split strategy: <strong className="text-primary">{ds.preprocessing.config?.split_strategy}</strong>
                    {' '}| existing split: <strong className="text-primary">{ds.preprocessing.split_validation?.status}</strong>
                  </div>
                </>
              ) : (
                <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                  Detected in source: {(p.splits?.detected_splits || []).join(', ') || 'none'} - Module 5 creates the final splits.
                </div>
              )}
              <div style={{ marginTop: '14px', padding: '8px 10px', backgroundColor: '#0a101d', borderRadius: 'var(--radius-sm)', fontSize: '11px', color: 'var(--text-secondary)' }}>
                Patient-level isolation requires real patient/group ids: provide a group-id map in the Preprocessing Engine.
              </div>
            </div>

            <div className="card">
              <div className="card-title">
                <BarChart2 size={15} color="var(--accent-teal)" />
                <span>Class Distribution ({(p.classes || []).length} classes)</span>
              </div>
              <DistributionBars distribution={p.class_distribution} />
            </div>
          </div>

          {/* Image & Tabular Statistics */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px' }}>
            <div className="card">
              <div className="card-title">
                <Image size={15} color="#818cf8" />
                <span>Image Characteristics</span>
              </div>
              {dims ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '12px' }}>
                  {[
                    ['Width range', `${dims.min_width} - ${dims.max_width} px (avg ${dims.avg_width?.toFixed(1)})`],
                    ['Height range', `${dims.min_height} - ${dims.max_height} px (avg ${dims.avg_height?.toFixed(1)})`],
                    ['Color modes', Object.entries(p.image_statistics.color_modes || {}).map(([k, v]) => `${k}: ${v}`).join(', ')],
                    ['Formats', (p.image_statistics.formats_found || []).join(', ')],
                  ].map(([k, v]) => (
                    <div key={k} style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '6px' }}>
                      <span className="text-muted">{k}:</span>
                      <span className="font-mono text-primary">{v}</span>
                    </div>
                  ))}
                </div>
              ) : <div className="text-muted" style={{ fontSize: '12px' }}>Not an image dataset.</div>}
            </div>

            <div className="card">
              <div className="card-title">
                <Users size={15} color="var(--brand-blue)" />
                <span>Tabular / Missing-Data Statistics</span>
              </div>
              {Object.keys(p.tabular_statistics || {}).length || Object.keys(p.missing_data_information || {}).length ? (
                <ReportText text={JSON.stringify({ ...p.tabular_statistics, missing_data: p.missing_data_information }, null, 2)} />
              ) : (
                <div className="text-muted" style={{ fontSize: '12px' }}>No tabular columns in this dataset (image-only).</div>
              )}
            </div>
          </div>

          {/* Warnings & Diagnostics */}
          {(p.warnings?.length > 0 || p.errors?.length > 0) && (
            <div className="card" style={{ borderColor: 'rgba(245, 158, 11, 0.4)' }}>
              <div className="card-title" style={{ color: 'var(--status-warning)' }}>
                <AlertTriangle size={15} />
                <span>Module 4 Inspection Warnings & Errors</span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {[...(p.errors || []).map((e) => ['danger', e]), ...(p.warnings || []).map((w) => ['warning', w])].map(([kind, msg], idx) => (
                  <div key={idx} style={{ fontSize: '11px', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span className={`pulse-dot ${kind}`} style={{ width: '6px', height: '6px' }} />
                    <span>{msg}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {ds.profile_report_md && (
            <div className="card">
              <div className="card-title"><span>dataset_report.md (Module 4 output)</span></div>
              <ReportText text={ds.profile_report_md} />
            </div>
          )}

          {ds.recommendations && (
            <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
              <button onClick={() => setActiveTab('start_training')} className="btn btn-teal">
                Review Training Recommendations (Module 8) <ArrowRight size={14} />
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DatasetView;
