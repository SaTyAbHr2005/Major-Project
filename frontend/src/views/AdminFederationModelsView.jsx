import React, { useState, useEffect } from 'react';
import { useFederationStore } from '../stores/federationStore';
import { getEvaluationProgress } from '../services/federationService';
import {
  Cpu, Layers, ShieldCheck, RefreshCw, ChevronDown, ChevronRight,
  CheckCircle, XCircle, Award, ArrowUpCircle, Clock, AlertTriangle
} from 'lucide-react';

const formatDuration = (ms) => {
  const s = Math.max(0, Math.round(ms / 1000));
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s`;
};

/** Live progress of a running evaluation: stage, images checked, elapsed and estimated time left. */
const EvaluationProgress = ({ progress }) => {
  const { status, done = 0, total, clickedAt } = progress;
  const elapsed = Date.now() - clickedAt;
  const counting = status === 'running' && total > 0;
  const pct = counting ? Math.round((done / total) * 100) : 0;
  // Time per image so far, measured from when image checks began (model loading excluded).
  const imageTime = progress.updated_at && done > 0 ? Date.parse(progress.updated_at) - Date.parse(progress.images_started_at) : 0;
  const remaining = counting && done > 0 && done < total ? (imageTime / done) * (total - done) : null;
  const stage = status === 'starting' ? 'Starting the evaluation…'
    : status === 'preparing' ? 'Loading the global model…'
    : done >= total ? 'Calculating metrics…'
    : `Checking evaluation images: ${done.toLocaleString()} of ${total.toLocaleString()}`;

  return (
    <div role="status" aria-live="polite" style={{ marginBottom: '16px', padding: '14px', background: 'var(--bg-card)', borderRadius: '6px', border: '1px solid rgba(59,130,246,0.35)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px', fontSize: '13px' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-primary)' }}>
          <RefreshCw size={14} color="#3b82f6" style={{ animation: 'spin 1.2s linear infinite' }} />
          {stage}
        </span>
        {counting && <span style={{ color: '#3b82f6', fontWeight: 600 }}>{pct}%</span>}
      </div>
      <div style={{ height: '6px', borderRadius: '3px', background: 'rgba(148,163,184,0.15)', overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${counting ? pct : 4}%`, background: '#3b82f6', borderRadius: '3px', transition: 'width 0.6s ease' }} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '8px', fontSize: '11px', color: '#94a3b8' }}>
        <span>Elapsed {formatDuration(elapsed)}</span>
        <span>{remaining != null ? `About ${formatDuration(remaining)} left` : 'Estimating time left…'}</span>
      </div>
    </div>
  );
};

export const AdminFederationModelsView = () => {
  const { globalModels, loadGlobalModels: loadModels, evaluateModel, promoteModel } = useFederationStore();
  const models = globalModels.data;
  const loading = globalModels.status === 'idle' || globalModels.status === 'loading';
  const error = globalModels.error;
  const [expandedModel, setExpandedModel] = useState(null);
  const [actionLoading, setActionLoading] = useState(null);
  const [actionError, setActionError] = useState(null);
  const [evalProgress, setEvalProgress] = useState(null);

  useEffect(() => { loadModels(); }, []);

  const handleEvaluate = async (globalModelId) => {
    setActionLoading(globalModelId);
    setActionError(null);
    const clickedAt = Date.now();
    setEvalProgress({ globalModelId, clickedAt, status: 'starting' });
    // The evaluation request stays open until it finishes; meanwhile poll the progress file.
    const timer = setInterval(async () => {
      try {
        const { progress } = await getEvaluationProgress(globalModelId);
        // Ignore the file left by an earlier evaluation of the same model.
        if (progress && Date.parse(progress.started_at) >= clickedAt - 2000) {
          setEvalProgress({ globalModelId, clickedAt, ...progress });
        }
      } catch { /* a missed poll is fine; the next one retries */ }
    }, 1500);
    try {
      await evaluateModel(globalModelId);
    } catch (err) {
      setActionError(`Evaluation failed: ${err.message}`);
    } finally {
      clearInterval(timer);
      setEvalProgress(null);
      setActionLoading(null);
    }
  };

  const handlePromote = async (globalModelId, evalRecordId) => {
    setActionLoading(globalModelId);
    setActionError(null);
    try {
      await promoteModel(globalModelId, evalRecordId);
    } catch (err) {
      setActionError(`Promotion failed: ${err.message}`);
    } finally {
      setActionLoading(null);
    }
  };

  const statusBadge = (status) => {
    const map = {
      'MAIN': { bg: 'rgba(16,185,129,0.15)', color: '#10b981', border: 'rgba(16,185,129,0.4)' },
      'EVALUATED': { bg: 'rgba(59,130,246,0.15)', color: '#3b82f6', border: 'rgba(59,130,246,0.4)' },
      'EVALUATION_PENDING': { bg: 'rgba(234,179,8,0.15)', color: '#eab308', border: 'rgba(234,179,8,0.4)' },
      // Shown only on this screen while its evaluation request runs; the stored status is still EVALUATION_PENDING.
      'EVALUATION_IN_PROGRESS': { bg: 'rgba(59,130,246,0.15)', color: '#60a5fa', border: 'rgba(59,130,246,0.5)' },
      'HISTORICAL': { bg: 'rgba(148,163,184,0.1)', color: '#94a3b8', border: 'rgba(148,163,184,0.3)' },
      'EVALUATED_NOT_PROMOTED': { bg: 'rgba(239,68,68,0.1)', color: '#ef4444', border: 'rgba(239,68,68,0.3)' },
      'CREATED': { bg: 'rgba(168,85,247,0.1)', color: '#a855f7', border: 'rgba(168,85,247,0.3)' },
    };
    const s = map[status] || map['CREATED'];
    return (
      <span style={{
        display: 'inline-flex', alignItems: 'center', gap: '4px',
        fontSize: '10px', fontWeight: 600, padding: '2px 8px',
        borderRadius: '4px', background: s.bg, color: s.color,
        border: `1px solid ${s.border}`
      }}>
        {status === 'MAIN' && <CheckCircle size={10} />}
        {status === 'EVALUATED_NOT_PROMOTED' && <XCircle size={10} />}
        {status === 'EVALUATION_PENDING' && <Clock size={10} />}
        {status === 'EVALUATION_IN_PROGRESS' && <RefreshCw size={10} style={{ animation: 'spin 1.2s linear infinite' }} />}
        {status}
      </span>
    );
  };

  // Group models by task + architecture
  const grouped = {};
  models.forEach(m => {
    const key = `${m.task} / ${m.architecture}`;
    if (!grouped[key]) grouped[key] = [];
    grouped[key].push(m);
  });
  // Sort each group by version descending
  Object.values(grouped).forEach(arr => arr.sort((a, b) => b.version - a.version));

  const mainModel = (group) => group.find(m => m.status === 'MAIN');

  return (
    <div style={{ padding: '20px', height: '100%', overflowY: 'auto' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <div>
          <h1 style={{ fontSize: '20px', fontWeight: 700, color: '#f8fafc', margin: 0 }}>
            <Cpu size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
            Federation Global Models
          </h1>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
            Aggregated models from M9 federation rounds — evaluation, promotion, and version history
          </div>
        </div>
        <button className="btn btn-secondary" onClick={loadModels} disabled={loading}>
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      {error && (
        <div style={{ padding: '12px', background: 'rgba(239,68,68,0.1)', color: '#ef4444', border: '1px solid rgba(239,68,68,0.3)', borderRadius: '6px', marginBottom: '16px', fontSize: '12px' }}>
          <AlertTriangle size={14} style={{ verticalAlign: 'middle', marginRight: '6px' }} />{error}
        </div>
      )}
      {actionError && (
        <div style={{ padding: '12px', background: 'rgba(239,68,68,0.1)', color: '#ef4444', border: '1px solid rgba(239,68,68,0.3)', borderRadius: '6px', marginBottom: '16px', fontSize: '12px' }}>
          <AlertTriangle size={14} style={{ verticalAlign: 'middle', marginRight: '6px' }} />{actionError}
        </div>
      )}

      {loading ? (
        <div style={{ color: '#94a3b8', padding: '40px', textAlign: 'center' }}>Loading global models...</div>
      ) : Object.keys(grouped).length === 0 ? (
        <div className="card" style={{ padding: '40px', textAlign: 'center' }}>
          <Layers size={32} color="#475569" style={{ marginBottom: '12px' }} />
          <div style={{ color: '#94a3b8', fontSize: '14px' }}>No global models created yet.</div>
          <div style={{ color: '#64748b', fontSize: '12px', marginTop: '4px' }}>Complete a federation round aggregation to generate the first model.</div>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {Object.entries(grouped).map(([groupKey, groupModels]) => {
            const current = mainModel(groupModels);
            return (
              <div key={groupKey} className="card" style={{ background: 'var(--bg-nested)', padding: 0, overflow: 'hidden' }}>
                {/* Group Header */}
                <div style={{
                  padding: '16px 20px', borderBottom: '1px solid var(--border-subtle)',
                  background: 'linear-gradient(90deg, rgba(124,58,237,0.06) 0%, transparent 100%)'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <div style={{ fontSize: '10px', color: 'var(--accent-teal)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        {groupKey}
                      </div>
                      <div style={{ fontSize: '15px', fontWeight: 700, color: '#f8fafc', marginTop: '2px' }}>
                        {current ? `MAIN: v${current.version}` : 'No MAIN version'}
                      </div>
                    </div>
                    <div style={{ fontSize: '11px', color: '#94a3b8' }}>
                      {groupModels.length} version{groupModels.length !== 1 ? 's' : ''}
                    </div>
                  </div>
                </div>

                {/* Model versions */}
                {groupModels.map(model => {
                  const isExpanded = expandedModel === model.global_model_id;
                  const evals = model.evaluations || [];
                  const latestEval = evals.length > 0 ? evals[0] : null;
                  const isLoading = actionLoading === model.global_model_id;

                  return (
                    <div key={model.global_model_id} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                      {/* Version row */}
                      <div
                        onClick={() => setExpandedModel(isExpanded ? null : model.global_model_id)}
                        style={{
                          padding: '12px 20px', cursor: 'pointer', display: 'flex',
                          justifyContent: 'space-between', alignItems: 'center',
                          transition: 'background 0.15s',
                          background: isExpanded ? 'rgba(255,255,255,0.02)' : 'transparent'
                        }}
                        onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.03)'}
                        onMouseLeave={e => { if (!isExpanded) e.currentTarget.style.background = 'transparent'; }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                          {isExpanded ? <ChevronDown size={14} color="#94a3b8" /> : <ChevronRight size={14} color="#94a3b8" />}
                          <span style={{ fontWeight: 600, color: '#f8fafc', fontSize: '13px' }}>Version v{model.version}</span>
                          {statusBadge(evalProgress?.globalModelId === model.global_model_id ? 'EVALUATION_IN_PROGRESS' : model.status)}
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '11px', color: '#94a3b8' }}>
                          <span>Round: {model.round_id || 'N/A'}</span>
                          <span>{model.created_at ? new Date(model.created_at).toLocaleDateString() : ''}</span>
                        </div>
                      </div>

                      {/* Expanded details */}
                      {isExpanded && (
                        <div style={{ padding: '0 20px 16px 46px' }}>
                          {/* Model info grid */}
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', marginBottom: '16px' }}>
                            <div style={{ background: 'var(--bg-card)', padding: '10px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
                              <div style={{ fontSize: '10px', color: '#64748b', marginBottom: '4px' }}>Federation Job</div>
                              <div style={{ fontSize: '12px', color: '#f8fafc', fontWeight: 600 }}>{model.federation_job_id || 'N/A'}</div>
                            </div>
                            <div style={{ background: 'var(--bg-card)', padding: '10px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
                              <div style={{ fontSize: '10px', color: '#64748b', marginBottom: '4px' }}>Participating Hospitals</div>
                              <div style={{ fontSize: '12px', color: '#f8fafc', fontWeight: 600 }}>
                                {model.participating_hospitals ? (Array.isArray(model.participating_hospitals) ? model.participating_hospitals.length : 'N/A') : 'N/A'}
                              </div>
                            </div>
                            <div style={{ background: 'var(--bg-card)', padding: '10px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
                              <div style={{ fontSize: '10px', color: '#64748b', marginBottom: '4px' }}>Aggregation Method</div>
                              <div style={{ fontSize: '12px', color: '#f8fafc', fontWeight: 600 }}>{model.aggregation_method || 'FedAvg'}</div>
                            </div>
                          </div>

                          {/* Technical Details (collapsible) */}
                          <details style={{ marginBottom: '16px' }}>
                            <summary style={{ fontSize: '11px', color: '#94a3b8', cursor: 'pointer', marginBottom: '8px' }}>
                              Technical Details
                            </summary>
                            <div style={{ background: 'var(--bg-card)', padding: '10px', borderRadius: '6px', border: '1px solid var(--border-subtle)', fontSize: '11px', color: '#94a3b8' }}>
                              <div>Global Model ID: <span className="font-mono">{model.global_model_id}</span></div>
                              <div>Checksum: <span className="font-mono">{model.artifact_checksum || 'N/A'}</span></div>
                              <div>Parameters: {model.parameter_count ? model.parameter_count.toLocaleString() : 'N/A'}</div>
                              <div>Size: {model.artifact_size ? `${(model.artifact_size / 1024 / 1024).toFixed(2)} MB` : 'N/A'}</div>
                            </div>
                          </details>

                          {/* Evaluation results */}
                          {latestEval ? (
                            <div style={{ marginBottom: '16px' }}>
                              <div style={{ fontSize: '12px', fontWeight: 600, color: '#cbd5e1', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                <Award size={14} color="#3b82f6" /> Evaluation Results
                              </div>
                              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px' }}>
                                {[
                                  { label: 'Accuracy', value: latestEval.accuracy },
                                  { label: 'Precision', value: latestEval.precision },
                                  { label: 'Recall', value: latestEval.recall },
                                  { label: 'F1 Score', value: latestEval.f1 }
                                ].map(metric => (
                                  <div key={metric.label} style={{
                                    background: 'var(--bg-card)', padding: '12px', borderRadius: '6px',
                                    border: '1px solid var(--border-subtle)', textAlign: 'center'
                                  }}>
                                    <div style={{ fontSize: '10px', color: '#64748b', marginBottom: '4px' }}>{metric.label}</div>
                                    <div style={{ fontSize: '18px', fontWeight: 700, color: '#f8fafc' }}>
                                      {metric.value != null ? `${(metric.value * 100).toFixed(1)}%` : 'N/A'}
                                    </div>
                                  </div>
                                ))}
                              </div>

                              {/* Confusion matrix */}
                              {latestEval.confusion_matrix && latestEval.confusion_matrix.length > 0 && (
                                <details style={{ marginTop: '12px' }}>
                                  <summary style={{ fontSize: '11px', color: '#94a3b8', cursor: 'pointer' }}>Confusion Matrix</summary>
                                  <div style={{ overflowX: 'auto', marginTop: '8px' }}>
                                    <table style={{ borderCollapse: 'collapse', fontSize: '11px' }}>
                                      <tbody>
                                        {latestEval.confusion_matrix.map((row, ri) => (
                                          <tr key={ri}>
                                            {row.map((val, ci) => (
                                              <td key={ci} style={{
                                                padding: '6px 10px', border: '1px solid var(--border-subtle)',
                                                textAlign: 'center', color: ri === ci ? '#10b981' : '#94a3b8',
                                                fontWeight: ri === ci ? 600 : 400,
                                                background: ri === ci ? 'rgba(16,185,129,0.06)' : 'transparent'
                                              }}>{val}</td>
                                            ))}
                                          </tr>
                                        ))}
                                      </tbody>
                                    </table>
                                  </div>
                                </details>
                              )}

                              {/* Compare with MAIN */}
                              {current && current.global_model_id !== model.global_model_id && current.evaluations && current.evaluations.length > 0 && (
                                <div style={{ marginTop: '12px', background: 'var(--bg-card)', padding: '12px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
                                  <div style={{ fontSize: '11px', fontWeight: 600, color: '#cbd5e1', marginBottom: '8px' }}>
                                    Candidate v{model.version} vs MAIN v{current.version}
                                  </div>
                                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
                                    <thead>
                                      <tr>
                                        <th style={{ textAlign: 'left', padding: '6px', color: '#64748b', borderBottom: '1px solid var(--border-subtle)' }}>Metric</th>
                                        <th style={{ textAlign: 'center', padding: '6px', color: '#64748b', borderBottom: '1px solid var(--border-subtle)' }}>MAIN v{current.version}</th>
                                        <th style={{ textAlign: 'center', padding: '6px', color: '#64748b', borderBottom: '1px solid var(--border-subtle)' }}>Candidate v{model.version}</th>
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {['accuracy', 'precision', 'recall', 'f1'].map(m => {
                                        const mainVal = current.evaluations[0][m];
                                        const candVal = latestEval[m];
                                        const better = candVal != null && mainVal != null && candVal > mainVal;
                                        return (
                                          <tr key={m}>
                                            <td style={{ padding: '6px', color: '#94a3b8', textTransform: 'capitalize' }}>{m}</td>
                                            <td style={{ padding: '6px', textAlign: 'center', color: '#f8fafc' }}>{mainVal != null ? `${(mainVal * 100).toFixed(1)}%` : 'N/A'}</td>
                                            <td style={{ padding: '6px', textAlign: 'center', color: better ? '#10b981' : '#f8fafc', fontWeight: better ? 600 : 400 }}>
                                              {candVal != null ? `${(candVal * 100).toFixed(1)}%` : 'N/A'}
                                              {better && ' ↑'}
                                            </td>
                                          </tr>
                                        );
                                      })}
                                    </tbody>
                                  </table>
                                </div>
                              )}
                            </div>
                          ) : evalProgress?.globalModelId === model.global_model_id ? (
                            <EvaluationProgress progress={evalProgress} />
                          ) : (
                            <div style={{ marginBottom: '16px', padding: '12px', background: 'var(--bg-card)', borderRadius: '6px', border: '1px solid var(--border-subtle)', fontSize: '12px', color: '#94a3b8' }}>
                              Not evaluated
                            </div>
                          )}

                          {/* Action buttons */}
                          <div style={{ display: 'flex', gap: '8px' }}>
                            {(model.status === 'CREATED' || model.status === 'EVALUATION_PENDING') && (
                              <button
                                className="btn btn-primary"
                                onClick={() => handleEvaluate(model.global_model_id)}
                                disabled={isLoading}
                                style={{ fontSize: '12px' }}
                              >
                                {isLoading ? 'Evaluating...' : <><Award size={13} /> Start Evaluation</>}
                              </button>
                            )}
                            {latestEval && model.status !== 'MAIN' && model.status !== 'HISTORICAL' && model.status !== 'EVALUATED_NOT_PROMOTED' && (
                              <button
                                className="btn btn-primary"
                                onClick={() => handlePromote(model.global_model_id, latestEval.evaluation_id)}
                                disabled={isLoading}
                                style={{ fontSize: '12px', background: 'linear-gradient(90deg, #10b981, #059669)' }}
                              >
                                {isLoading ? 'Promoting...' : <><ArrowUpCircle size={13} /> Promote to MAIN</>}
                              </button>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default AdminFederationModelsView;
