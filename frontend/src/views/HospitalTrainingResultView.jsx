import React, { useEffect, useState } from 'react';
import { useML } from '../context/MLContext';
import trainingService from '../services/trainingService';
import TrainingResultModal from '../components/training/TrainingResultModal';
import PrivacyNotice from '../components/common/PrivacyNotice';
import { ConfusionMatrixGrid } from '../components/charts/MLCharts';
import { Award, CheckCircle, Share2, FileText, Cpu, Layers, BarChart3 } from 'lucide-react';

const pct = (v) => (typeof v === 'number' ? `${(v * 100).toFixed(2)}%` : '-');
const PLOT_TITLES = { 'confusion_matrix.png': 'Confusion matrix (test set)', 'roc_curves.png': 'ROC curves (one-vs-rest)', 'pr_curves.png': 'Precision-recall curves' };

const TrainingResultView = () => {
  const { lastRunId, trainingJob } = useML();
  const [history, setHistory] = useState([]);
  const [selectedId, setSelectedId] = useState(lastRunId);
  const [run, setRun] = useState(null);
  const [plotUrls, setPlotUrls] = useState({});
  const [error, setError] = useState(null);
  const [showModal, setShowModal] = useState(false);

  useEffect(() => {
    trainingService.getTrainingHistory()
      .then((runs) => {
        setHistory(runs);
        setSelectedId((id) => id || runs[0]?.model_id || null);
      })
      .catch((e) => setError(e.message));
  }, [trainingJob?.status]);

  useEffect(() => {
    if (!selectedId) return undefined;
    let urls = {};
    let cancelled = false;
    setError(null);
    trainingService.getRun(selectedId)
      .then(async (r) => {
        if (cancelled) return;
        setRun(r);
        for (const name of r.plots) urls[name] = await trainingService.getPlotUrl(r.model_id, name);
        if (!cancelled) setPlotUrls(urls);
      })
      .catch((e) => setError(e.message));
    return () => {
      cancelled = true;
      Object.values(urls).forEach(URL.revokeObjectURL);
      setPlotUrls({});
    };
  }, [selectedId]);

  const res = run?.result;
  const test = res?.test_metrics;
  const val = res?.validation_metrics;
  const evalSet = test || val;
  const stats = run?.resource_statistics;
  const mon = stats?.monitor_summary;

  return (
    <div style={{ padding: '20px', overflowY: 'auto', height: '100%' }}>
      <PrivacyNotice />

      <div className="card" style={{ marginBottom: '16px' }}>
        <div className="card-title">
          <Award size={18} color="var(--accent-teal)" />
          <span>Module 7: Training Results & Module 9 Federation Handoff</span>
        </div>
        <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
          Metrics, evaluation plots and resource statistics written by the local pipeline. Completed runs also produce a
          FederationHandoff package (model parameters + metadata, no images) for the future Module 9 federated engine.
        </p>
        {error && <div className="text-danger" style={{ fontSize: '11px', marginTop: '6px' }}>{error}</div>}
      </div>

      {res ? (
        <div className="card" style={{ marginBottom: '20px', border: '1px solid rgba(16, 185, 129, 0.4)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px', flexWrap: 'wrap', gap: '10px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <CheckCircle size={22} color="var(--status-healthy)" />
              <div>
                <div style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)' }}>
                  Run: <span className="font-mono text-cyan">{run.model_id}</span>
                </div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                  Architecture: <strong className="text-primary">{res.architecture}</strong> | Dataset: {run.run?.dataset_name || '-'} |
                  Epochs: {res.epochs_run} (best {res.best_epoch}) | Duration: {(res.training_duration_seconds / 60).toFixed(1)} min |
                  Precision: {res.resolved_precision} | Seed: {res.random_seed}
                </div>
              </div>
            </div>
            <div style={{ display: 'flex', gap: '8px' }}>
              <span className="badge badge-healthy font-mono">{res.status}</span>
              {res.ready_for_federation && <span className="badge badge-blue font-mono"><Share2 size={11} /> M9 HANDOFF PREPARED</span>}
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginBottom: '16px' }}>
            {[
              ['Validation Accuracy', pct(val?.accuracy), 'text-emerald', `n = ${val?.sample_count ?? '-'} | F1 ${pct(val?.f1)}`],
              ['Validation Loss', val?.loss?.toFixed(4) ?? '-', 'text-cyan', `Train loss ${res.training_metrics?.loss?.toFixed(4) ?? '-'}`],
              ['Held-out Test Accuracy', pct(test?.accuracy), 'text-primary', test ? `n = ${test.sample_count} | F1 ${pct(test.f1)}` : 'No test split'],
              ['ROC-AUC (macro, test)', run.curve_metrics?.macro?.roc_auc?.toFixed(4) ?? '-', 'text-purple', `AP ${run.curve_metrics?.macro?.average_precision?.toFixed(4) ?? '-'}`],
            ].map(([k, v, cls, note]) => (
              <div className="card" style={{ padding: '12px', backgroundColor: '#090d16' }} key={k}>
                <div className="card-title-muted">{k}</div>
                <div className={`font-mono ${cls}`} style={{ fontSize: '20px', fontWeight: 700, marginTop: '4px' }}>{v}</div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px' }}>{note}</div>
              </div>
            ))}
          </div>

          {evalSet && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px', marginBottom: '16px' }}>
              <div>
                <div className="card-title-muted" style={{ marginBottom: '8px' }}>Per-class metrics ({test ? 'test' : 'validation'} set)</div>
                <table className="fl-table">
                  <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
                  <tbody>
                    {Object.entries(evalSet.per_class || {}).map(([c, m]) => (
                      <tr key={c}><td style={{ fontWeight: 600 }}>{c}</td><td className="font-mono">{pct(m.precision)}</td>
                        <td className="font-mono">{pct(m.recall)}</td><td className="font-mono">{pct(m.f1)}</td><td className="font-mono">{m.support}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div>
                <div className="card-title-muted" style={{ marginBottom: '8px' }}>Confusion matrix ({test ? 'test' : 'validation'} set)</div>
                <ConfusionMatrixGrid matrix={evalSet.confusion_matrix} classOrder={evalSet.class_order} />
              </div>
            </div>
          )}

          {run.plots.length > 0 && (
            <div style={{ marginBottom: '16px' }}>
              <div className="card-title-muted" style={{ marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <BarChart3 size={13} /> Evaluation plots (generated locally by Module 7)
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '12px' }}>
                {run.plots.map((name) => (
                  <figure key={name} style={{ margin: 0, backgroundColor: '#ffffff', borderRadius: 'var(--radius-md)', padding: '6px' }}>
                    {plotUrls[name] ? <img src={plotUrls[name]} alt={PLOT_TITLES[name]} style={{ width: '100%', display: 'block' }} /> : <div style={{ height: '200px' }} />}
                    <figcaption style={{ fontSize: '10px', color: '#334155', textAlign: 'center', marginTop: '4px' }}>{PLOT_TITLES[name]}</figcaption>
                  </figure>
                ))}
              </div>
            </div>
          )}

          {stats && (
            <div style={{ padding: '12px', backgroundColor: '#090d16', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', marginBottom: '14px' }}>
              <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
                <Cpu size={14} color="var(--accent-teal)" /> Hardware Execution Profile (Module 8 measured statistics)
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px', fontSize: '11px' }}>
                {[
                  ['Peak VRAM', mon?.peak_vram_mb != null ? `${mon.peak_vram_mb.toFixed(0)} MB` : 'n/a'],
                  ['Peak GPU util.', mon?.gpu_utilization_available ? `${mon.peak_gpu_utilization_percent}%` : 'not available'],
                  ['Peak RAM', mon?.peak_ram_mb != null ? `${mon.peak_ram_mb.toFixed(0)} MB` : 'n/a'],
                  ['Avg / peak CPU', mon ? `${mon.average_cpu_percent?.toFixed(0)}% / ${mon.peak_cpu_percent?.toFixed(0)}%` : 'n/a'],
                  ['Throughput', stats.measured_samples_per_second ? `${stats.measured_samples_per_second.toFixed(1)} samples/s` : 'n/a'],
                  ['Estimated vs actual', stats.estimated_training_time_seconds ? `${(stats.estimated_training_time_seconds / 60).toFixed(1)} vs ${(stats.actual_training_duration_seconds / 60).toFixed(1)} min` : 'n/a'],
                  ['CUDA OOM events', stats.cuda_oom_event_count ?? 0],
                  ['Batch adaptations', (stats.adaptations || []).length],
                ].map(([k, v]) => (
                  <div key={k}><span className="text-muted">{k}:</span><div className="font-mono text-primary">{v}</div></div>
                ))}
              </div>
            </div>
          )}

          {/* Module 9 Federation Handoff Card (engine itself is future work) */}
          <div style={{ padding: '14px', backgroundColor: '#0a1426', border: '1px solid rgba(37, 99, 235, 0.35)', borderRadius: 'var(--radius-md)', marginBottom: '14px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
              <div style={{ fontSize: '12px', fontWeight: 600, color: '#60a5fa', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Share2 size={14} /> Module 9 FederationHandoff {res.ready_for_federation ? 'Prepared' : 'Not Available'}
              </div>
              <span className="badge badge-neutral font-mono" style={{ fontSize: '10px' }}>MODULE 9 ENGINE: PENDING</span>
            </div>
            <p style={{ fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '8px' }}>
              The best checkpoint's parameters and metadata are packaged locally as <code className="font-mono">{run.model_id}_federation_handoff</code>.
              No raw images are included. Transmission and aggregation belong to the Flower-based Module 9, which is not implemented yet.
            </p>
            <div style={{ display: 'flex', gap: '16px', fontSize: '11px', color: 'var(--text-muted)' }}>
              <span>Training examples: <strong className="text-primary">{res.training_metrics?.sample_count ?? '-'}</strong></span>
              <span>Checkpoint: <strong className="text-primary">{res.checkpoint_model_id} v{res.checkpoint_version}</strong></span>
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
              Experimental research result on this local dataset - not a clinical validation.
            </span>
            <button onClick={() => setShowModal(true)} className="btn btn-secondary"><FileText size={13} /> View Full Audit Inspector</button>
          </div>
        </div>
      ) : (
        <div className="card" style={{ textAlign: 'center', padding: '30px', marginBottom: '20px' }}>
          <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>No completed local training run yet.</div>
        </div>
      )}

      <div className="card">
        <div className="card-title"><Layers size={15} color="var(--brand-blue)" /><span>Local Node Training Run History</span></div>
        <div className="fl-table-wrapper">
          <table className="fl-table">
            <thead>
              <tr><th>Run / Model ID</th><th>Model</th><th>Dataset</th><th>Epochs</th><th>Val Acc</th><th>Test Acc</th><th>Duration</th><th>Module 9 Handoff</th></tr>
            </thead>
            <tbody>
              {history.length === 0 && <tr><td colSpan={8} className="text-muted" style={{ textAlign: 'center' }}>No runs yet.</td></tr>}
              {history.map((item) => (
                <tr key={item.model_id} onClick={() => setSelectedId(item.model_id)} style={{ cursor: 'pointer',
                  backgroundColor: item.model_id === selectedId ? 'rgba(8, 145, 178, 0.1)' : undefined }}>
                  <td className="font-mono text-cyan">{item.model_id}</td>
                  <td style={{ fontWeight: 600 }}>{item.architecture}</td>
                  <td style={{ color: 'var(--text-secondary)' }}>{item.dataset_name || '-'}</td>
                  <td className="font-mono">{item.epochs_run}</td>
                  <td className="font-mono text-emerald" style={{ fontWeight: 600 }}>{pct(item.validation_accuracy)}</td>
                  <td className="font-mono text-primary">{pct(item.test_accuracy)}</td>
                  <td className="font-mono text-muted">{item.duration_seconds ? `${(item.duration_seconds / 60).toFixed(1)} min` : '-'}</td>
                  <td>
                    <span className={`badge ${item.ready_for_federation ? 'badge-healthy' : 'badge-neutral'}`} style={{ fontSize: '10px' }}>
                      {item.ready_for_federation ? 'PREPARED' : item.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <TrainingResultModal isOpen={showModal} onClose={() => setShowModal(false)} run={run} />
    </div>
  );
};

export default TrainingResultView;
