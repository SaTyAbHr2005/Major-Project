import React, { useState, useEffect, useMemo } from 'react';
import { useUiStore } from '../stores/uiStore';
import { useMLStore } from '../stores/mlStore';
import { useFederationStore } from '../stores/federationStore';
import { useAuthStore, selectHospitalId } from '../stores/authStore';
import { FlowerSteps } from '../components/federation/FlowerSteps';
import { ml } from '../services/mlClient';
import FederatedTrainingConfirmModal from '../components/training/FederatedTrainingConfirmModal';
import {
  Network, CheckCircle, XCircle, Clock, AlertTriangle, RefreshCw,
  ShieldCheck, Upload, Ban, Tag, Database
} from 'lucide-react';

const HospitalFederationView = () => {
  const setActiveTab = useUiStore((s) => s.setActiveScreen);
  const { datasets, refreshDatasets, refreshRuns } = useMLStore();
  const runs = useMLStore((s) => s.runs.data);
  const {
    jobs, hospitalUpdates, taskVocabulary, loadJobs, loadHospitalUpdates, loadTaskVocabulary, participate,
    requestParticipation, flower, flowerIssues, flowerDataset, setFlowerDataset,
  } = useFederationStore();
  const hospitalId = useAuthStore(selectHospitalId);
  const updates = hospitalUpdates.data;
  const availableJobs = jobs.data;
  // Page-local UI state: { job, round } while the participation confirm modal is open, etc.
  const [pendingJoin, setPendingJoin] = useState(null);
  const [joining, setJoining] = useState(false);
  const [joinError, setJoinError] = useState(null);
  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [eligibilityResults, setEligibilityResults] = useState({});
  const [evaluating, setEvaluating] = useState(false);
  const [settingTask, setSettingTask] = useState(false);
  const [retry, setRetry] = useState(null);       // { id: model being resubmitted, error }
  const [taskError, setTaskError] = useState(null);
  const [requesting, setRequesting] = useState(null);   // job id whose join request is being sent
  const [requestError, setRequestError] = useState(null); // { jobId, text }

  /* ── Helpers ────────────────────────────────────────────── */

  const loading = [jobs, hospitalUpdates].some((r) => r.status === 'idle' || r.status === 'loading');
  const error = hospitalUpdates.error || jobs.error;
  const localDatasets = useMemo(() => datasets.filter((d) => d.preprocessed === true), [datasets]);
  const selectedDataset = localDatasets.find(d => d.dataset_id === selectedDatasetId);

  // Completed federation runs whose handoff never reached M9 (matched via the
  // update's training_configuration.model_id, the local run id).
  const localHandoffs = useMemo(() => runs.filter((run) =>
    run.status === 'TRAINING_COMPLETED_AWAITING_FEDERATION' && run.ready_for_federation
    && !updates.some((u) => u.training_configuration?.model_id === run.model_id)
  ), [runs, updates]);

  const loadStatus = () => Promise.all([
    loadHospitalUpdates(), loadJobs(), refreshDatasets(), loadTaskVocabulary(), refreshRuns(),
  ]);

  useEffect(() => { loadStatus(); }, []);

  useEffect(() => {
    if (!selectedDatasetId && localDatasets.length > 0) setSelectedDatasetId(localDatasets[0].dataset_id);
  }, [localDatasets, selectedDatasetId]);

  /* ── Eligibility evaluation (read-only, no mutations) ── */

  const evaluateEligibility = async () => {
    if (!selectedDatasetId) return;
    setEvaluating(true);
    const results = {};
    const jobsToEvaluate = availableJobs.filter(j => j.status === 'CREATED' || j.status === 'ACTIVE');
    for (const job of jobsToEvaluate) {
      try {
        const data = await ml.post('/federation/evaluate-eligibility', { job, dataset_id: selectedDatasetId });
        results[job.federation_job_id] = data;
      } catch (err) {
        console.error("Evaluation failed for job", job.federation_job_id, err);
        results[job.federation_job_id] = {
          eligible: false,
          eligibility_type: 'NOT_ELIGIBLE',
          reasons: [`Evaluation error: ${err.message}`]
        };
      }
    }
    setEligibilityResults(results);
    setEvaluating(false);
  };

  useEffect(() => {
    if (selectedDatasetId && availableJobs.length > 0) {
      evaluateEligibility();
    }
  }, [selectedDatasetId, availableJobs]);

  /* ── Task association ──────────────────────────────────── */

  const handleSetTask = async (task) => {
    if (!selectedDatasetId) return;
    setSettingTask(true);
    setTaskError(null);
    try {
      await ml.post(`/datasets/${selectedDatasetId}/set-task`, { declared_task: task });
      await refreshDatasets(); // picks up the new declared_task
      // Re-evaluate eligibility with updated task
      setTimeout(() => evaluateEligibility(), 200);
    } catch (err) {
      setTaskError(err.message);
    } finally {
      setSettingTask(false);
    }
  };

  /* ── Request to join (admin approves) ─────────────────── */

  // Sends only the summary of the local eligibility check (counts, classes, hardware tier), never images.
  const handleRequestJoin = async (job) => {
    setRequesting(job.federation_job_id);
    setRequestError(null);
    try {
      await requestParticipation(job.federation_job_id, eligibilityResults[job.federation_job_id]);
      // Flower jobs train automatically with the dataset whose eligibility was sent.
      if (job.plan?.protocol === 'flower') setFlowerDataset(hospitalId, job.federation_job_id, selectedDatasetId);
    } catch (err) {
      setRequestError({ jobId: job.federation_job_id, text: err.message });
    } finally {
      setRequesting(null);
    }
  };

  /* ── Participate ───────────────────────────────────────── */

  const handleParticipate = (job) => {
    setJoinError(null);
    setPendingJoin({ job, round: job.rounds?.find(r => r.status === 'OPEN' || r.status === 'RECEIVING') });
  };

  const confirmParticipate = async () => {
    const { job, round } = pendingJoin;
    setJoining(true);
    setJoinError(null);
    try {
      // Registers with M9, starts local training and hands the job to the Training Monitor.
      await participate(job, round, selectedDatasetId);
      setPendingJoin(null);
      setActiveTab('training_monitor');
    } catch (err) {
      setJoinError(err.message);  // e.g. another training run already in progress
    } finally {
      setJoining(false);
    }
  };

  // Resubmits a finished run to the job/round saved in its run.json (local ML API knows the handoff folder).
  const handleRetrySubmission = async (run) => {
    setRetry({ id: run.model_id, error: null });
    try {
      await ml.post(`/federation/submit/${encodeURIComponent(run.model_id)}`, {});
      setRetry(null);
      loadStatus();
    } catch (err) {
      setRetry({ id: null, error: `${run.model_id}: ${err.message}` });
    }
  };

  /* ── Status helpers ────────────────────────────────────── */

  const statusIcon = (status) => {
    switch (status) {
      case 'ACCEPTED':
      case 'USED_IN_AGGREGATION':
        return <CheckCircle size={14} color="#10b981" />;
      case 'REJECTED':
        return <XCircle size={14} color="#ef4444" />;
      case 'EXPIRED':
        return <Ban size={14} color="#f59e0b" />;
      case 'RECEIVED':
      case 'VALIDATING':
        return <Clock size={14} color="#3b82f6" />;
      case 'QUARANTINED':
        return <AlertTriangle size={14} color="#ef4444" />;
      default:
        return <Clock size={14} color="#94a3b8" />;
    }
  };

  const statusColor = (status) => {
    switch (status) {
      case 'ACCEPTED': case 'USED_IN_AGGREGATION': return '#10b981';
      case 'REJECTED': case 'QUARANTINED': return '#ef4444';
      case 'EXPIRED': return '#f59e0b';
      default: return '#3b82f6';
    }
  };

  const jobsForDisplay = availableJobs.filter(j => j.status === 'CREATED' || j.status === 'ACTIVE');

  /* ── Render ────────────────────────────────────────────── */

  return (
    <div style={{ padding: '20px', height: '100%', overflowY: 'auto' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <div>
          <h1 style={{ fontSize: '20px', fontWeight: 700, color: '#f8fafc', margin: 0 }}>
            <Network size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
            Federation Status
          </h1>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
            Your hospital's participation in federated learning rounds
          </div>
        </div>
        <button className="btn btn-secondary" onClick={loadStatus} disabled={loading}>
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      {/* Privacy reminder */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '8px',
        padding: '10px 14px', marginBottom: '20px',
        background: 'rgba(16,185,129,0.06)', border: '1px solid rgba(16,185,129,0.25)',
        borderRadius: '6px', fontSize: '11px', color: '#10b981'
      }}>
        <ShieldCheck size={14} />
        <span>Your medical images remain local. Only model parameter updates are submitted to the federation.</span>
      </div>

      {error && (
        <div style={{
          padding: '12px', background: 'rgba(239,68,68,0.1)', color: '#ef4444',
          border: '1px solid rgba(239,68,68,0.3)', borderRadius: '6px', marginBottom: '16px', fontSize: '12px'
        }}>
          <AlertTriangle size={14} style={{ verticalAlign: 'middle', marginRight: '6px' }} />{error}
        </div>
      )}

      {loading ? (
        <div style={{ color: '#94a3b8', padding: '40px', textAlign: 'center' }}>Loading federation status...</div>
      ) : (
        <div style={{ display: 'flex', gap: '20px', alignItems: 'flex-start' }}>
          
          {/* Left Column: Updates */}
          <div style={{ flex: 2, display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <h3 style={{ margin: 0, color: '#f8fafc', fontSize: '16px' }}>Your Submissions</h3>

            {localHandoffs.map((run, idx) => (
              <div key={`local-${idx}`} className="card" style={{ padding: '16px', borderLeft: '3px solid #f59e0b' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <div>
                    <h2 style={{ fontSize: '15px', color: '#f8fafc', margin: 0 }}>Unsubmitted Local Handoff</h2>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Run: {run.model_id}</div>
                  </div>
                  <div style={{
                    display: 'inline-flex', alignItems: 'center', gap: '6px',
                    padding: '4px 10px', borderRadius: '4px',
                    background: '#f59e0b15', border: '1px solid #f59e0b40',
                    fontSize: '11px', fontWeight: 600, color: '#f59e0b'
                  }}>
                    FEDERATION SUBMISSION FAILED
                  </div>
                </div>
                <div style={{ fontSize: '11px', color: '#cbd5e1', marginBottom: '12px' }}>
                  Network or submission failure. Retry available.
                </div>
                <button className="btn btn-primary" disabled={retry?.id === run.model_id} onClick={() => handleRetrySubmission(run)}>
                  {retry?.id === run.model_id ? 'Submitting...' : 'Retry Submission'}
                </button>
              </div>
            ))}

            {retry?.error && <div style={{ fontSize: '11px', color: 'var(--status-danger)', margin: '0 0 12px' }}>{retry.error}</div>}
            {taskError && <div style={{ fontSize: '11px', color: 'var(--status-danger)', margin: '0 0 12px' }}>Failed to set task: {taskError}</div>}

            {updates.length === 0 && localHandoffs.length === 0 ? (
              <div className="card" style={{ padding: '40px', textAlign: 'center' }}>
                <Upload size={32} color="#475569" style={{ marginBottom: '12px' }} />
                <div style={{ color: '#94a3b8', fontSize: '14px' }}>No federation submissions yet.</div>
                <div style={{ color: '#64748b', fontSize: '12px', marginTop: '4px' }}>
                  Complete local training and submit a federation handoff to participate.
                </div>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {updates.map((update, idx) => (
                  <div key={update.update_id || idx} className="card" style={{ background: 'var(--bg-nested)', padding: '16px' }}>
                    {/* Update header */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
                      <div>
                        <div style={{ fontSize: '10px', color: 'var(--accent-teal)', fontWeight: 600, textTransform: 'uppercase' }}>
                          {update.architecture || 'Unknown Architecture'}
                        </div>
                        <div style={{ fontSize: '14px', fontWeight: 600, color: '#f8fafc', marginTop: '2px' }}>
                          Round: {update.round_id || 'N/A'}
                        </div>
                      </div>
                      <div style={{
                        display: 'inline-flex', alignItems: 'center', gap: '6px',
                        padding: '4px 10px', borderRadius: '4px',
                        background: `${statusColor(update.status)}15`,
                        border: `1px solid ${statusColor(update.status)}40`,
                        fontSize: '11px', fontWeight: 600, color: statusColor(update.status)
                      }}>
                        {statusIcon(update.status)} {update.status}
                      </div>
                    </div>
      
                    {/* Verification checklist */}
                    <div style={{
                      display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px',
                      padding: '12px', background: 'var(--bg-card)', borderRadius: '6px',
                      border: '1px solid var(--border-subtle)', marginBottom: '12px'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px' }}>
                        <CheckCircle size={12} color="#10b981" />
                        <span style={{ color: '#cbd5e1' }}>Handoff prepared</span>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px' }}>
                        {update.status === 'REJECTED' ? (
                          <><XCircle size={12} color="#ef4444" /><span style={{ color: '#ef4444' }}>Authorization failed</span></>
                        ) : (
                          <><CheckCircle size={12} color="#10b981" /><span style={{ color: '#cbd5e1' }}>Participant authorized</span></>
                        )}
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px' }}>
                        {update.status === 'REJECTED' && update.rejection_reason?.includes('model') ? (
                          <><XCircle size={12} color="#ef4444" /><span style={{ color: '#ef4444' }}>Base model mismatch</span></>
                        ) : (
                          <><CheckCircle size={12} color="#10b981" /><span style={{ color: '#cbd5e1' }}>Base model verified</span></>
                        )}
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px' }}>
                        {update.status === 'EXPIRED' ? (
                          <><Ban size={12} color="#f59e0b" /><span style={{ color: '#f59e0b' }}>Deadline passed</span></>
                        ) : update.status === 'ACCEPTED' || update.status === 'USED_IN_AGGREGATION' ? (
                          <><CheckCircle size={12} color="#10b981" /><span style={{ color: '#cbd5e1' }}>Update submitted</span></>
                        ) : (
                          <><Clock size={12} color="#3b82f6" /><span style={{ color: '#3b82f6' }}>Pending</span></>
                        )}
                      </div>
                    </div>
      
                    {/* Details */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', fontSize: '11px' }}>
                      <div>
                        <span style={{ color: '#64748b' }}>Training Samples: </span>
                        <span style={{ color: '#f8fafc', fontWeight: 600 }}>
                          {update.num_train_samples ? update.num_train_samples.toLocaleString() : 'N/A'}
                        </span>
                      </div>
                      <div>
                        <span style={{ color: '#64748b' }}>Submitted: </span>
                        <span style={{ color: '#f8fafc' }}>
                          {update.received_at ? new Date(update.received_at).toLocaleString() : 'N/A'}
                        </span>
                      </div>
                      <div>
                        <span style={{ color: '#64748b' }}>Job: </span>
                        <span style={{ color: '#f8fafc' }}>{update.federation_job_id || 'N/A'}</span>
                      </div>
                    </div>
      
                    {/* Rejection / Expiration reason */}
                    {(update.status === 'REJECTED' || update.status === 'EXPIRED') && update.rejection_reason && (
                      <div style={{
                        marginTop: '12px', padding: '10px', borderRadius: '6px',
                        background: update.status === 'EXPIRED' ? 'rgba(245,158,11,0.08)' : 'rgba(239,68,68,0.08)',
                        border: `1px solid ${update.status === 'EXPIRED' ? 'rgba(245,158,11,0.3)' : 'rgba(239,68,68,0.3)'}`,
                        fontSize: '12px'
                      }}>
                        <div style={{ fontWeight: 600, color: update.status === 'EXPIRED' ? '#f59e0b' : '#ef4444', marginBottom: '4px' }}>
                          {update.status === 'EXPIRED' ? 'Update Expired' : 'Update Rejected'}
                        </div>
                        <div style={{ color: update.status === 'EXPIRED' ? '#d97706' : '#f87171' }}>
                          {update.rejection_reason}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
          
          {/* Right Column: Available Jobs & Eligibility */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <h3 style={{ margin: 0, color: '#f8fafc', fontSize: '16px' }}>Available Jobs</h3>
            
            {/* Dataset selector */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', padding: '12px', background: 'var(--bg-nested)', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
              <label style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Evaluate dataset:</label>
              <select style={{ padding: '8px', background: 'var(--bg-card)', color: '#fff', border: '1px solid var(--border-subtle)', borderRadius: '4px' }} 
                      value={selectedDatasetId} onChange={e => setSelectedDatasetId(e.target.value)}>
                {localDatasets.map(d => (
                  <option key={d.dataset_id} value={d.dataset_id}>{d.name} ({d.dataset_id})</option>
                ))}
              </select>
              {evaluating && <div style={{ fontSize: '11px', color: 'var(--accent-teal)' }}>Evaluating eligibility...</div>}
            </div>

            {/* Selected dataset profile */}
            {selectedDataset && (
              <div style={{ padding: '12px', background: 'var(--bg-nested)', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px', fontSize: '12px', fontWeight: 600, color: '#f8fafc' }}>
                  <Database size={14} />
                  Dataset Profile
                </div>
                <div style={{ display: 'grid', gap: '6px', fontSize: '11px' }}>
                  <div>
                    <span style={{ color: '#64748b' }}>Name: </span>
                    <span style={{ color: '#f8fafc' }}>{selectedDataset.name}</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ color: '#64748b' }}>Declared task: </span>
                    {selectedDataset.declared_task ? (
                      <span style={{ 
                        color: '#10b981', fontWeight: 600,
                        padding: '1px 6px', borderRadius: '3px',
                        background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.2)'
                      }}>
                        <Tag size={10} style={{ marginRight: '3px', verticalAlign: 'middle' }} />
                        {selectedDataset.declared_task}
                      </span>
                    ) : (
                      <span style={{ color: '#f59e0b', fontStyle: 'italic' }}>Not set</span>
                    )}
                  </div>
                  {selectedDataset.classes && (
                    <div>
                      <span style={{ color: '#64748b' }}>Classes: </span>
                      <span style={{ color: '#cbd5e1' }}>{selectedDataset.classes.join(', ')}</span>
                    </div>
                  )}
                  {selectedDataset.total_samples && (
                    <div>
                      <span style={{ color: '#64748b' }}>Total samples: </span>
                      <span style={{ color: '#cbd5e1' }}>{selectedDataset.total_samples.toLocaleString()}</span>
                    </div>
                  )}
                </div>

                {/* Task association control */}
                <div style={{ marginTop: '10px', paddingTop: '10px', borderTop: '1px solid var(--border-subtle)' }}>
                  <label style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>
                    {selectedDataset.declared_task ? 'Change task association:' : 'Associate with task:'}
                  </label>
                  <div style={{ display: 'flex', gap: '6px' }}>
                    <select 
                      id="task-select"
                      style={{ flex: 1, padding: '6px', background: 'var(--bg-card)', color: '#fff', border: '1px solid var(--border-subtle)', borderRadius: '4px', fontSize: '11px' }}
                      defaultValue=""
                    >
                      <option value="" disabled>Select task...</option>
                      {taskVocabulary.map(t => (
                        <option key={t} value={t}>{t}</option>
                      ))}
                    </select>
                    <button 
                      className="btn btn-primary"
                      style={{ padding: '4px 10px', fontSize: '11px' }}
                      disabled={settingTask}
                      onClick={() => {
                        const sel = document.getElementById('task-select');
                        if (sel && sel.value) handleSetTask(sel.value);
                      }}
                    >
                      {settingTask ? '...' : 'Set'}
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Jobs list — ALL available jobs shown, never hidden */}
            {jobsForDisplay.length === 0 && (
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', padding: '16px', textAlign: 'center', background: 'var(--bg-nested)', borderRadius: '6px' }}>
                No active federation jobs available.
              </div>
            )}

            {jobsForDisplay.map(job => {
              const elig = eligibilityResults[job.federation_job_id];
              return (
                <div key={job.federation_job_id} className="card" style={{ padding: '16px', background: 'var(--bg-nested)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '4px' }}>
                    <div style={{ fontSize: '10px', color: 'var(--accent-teal)' }}>{job.architecture}</div>
                    <div style={{
                      fontSize: '10px', padding: '2px 6px', borderRadius: '3px',
                      background: 'rgba(59,130,246,0.1)', border: '1px solid rgba(59,130,246,0.2)',
                      color: '#3b82f6'
                    }}>
                      {job.status}
                    </div>
                  </div>
                  <h4 style={{ margin: '0 0 8px 0', fontSize: '14px', color: '#fff' }}>{job.federation_job_id}</h4>
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '12px' }}>
                    Task: {job.task} | Classes: {job.num_classes}
                  </div>
                  
                  {/* Eligibility result */}
                  {elig && (
                    <div style={{ padding: '12px', borderRadius: '4px', border: '1px solid var(--border-subtle)', background: 'var(--bg-card)', marginBottom: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      
                      {/* Dataset Eligibility */}
                      <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', fontWeight: 600, color: '#f8fafc' }}>
                          Dataset Eligibility:
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', marginTop: '2px' }}>
                          {elig.dataset_eligible ? <CheckCircle size={12} color="#10b981" /> : <XCircle size={12} color="#ef4444" />}
                          <span style={{ color: elig.dataset_eligible ? '#10b981' : '#ef4444' }}>
                            {elig.dataset_eligibility_type === 'FULL_DATASET' ? 'Full Dataset' : 
                             elig.dataset_eligibility_type === 'CLASS_SUBSET' ? 'Class Subset' : 'Not Eligible'}
                          </span>
                        </div>
                        {elig.dataset_reasons && elig.dataset_reasons.length > 0 && (
                          <div style={{ fontSize: '10px', color: 'var(--text-secondary)', marginTop: '2px', marginLeft: '18px' }}>
                            {elig.dataset_reasons.map((r, i) => <div key={i}>• {r}</div>)}
                          </div>
                        )}
                        {elig.dataset_eligible && (
                          <div style={{ marginTop: '4px', fontSize: '10px', color: '#cbd5e1', marginLeft: '18px' }}>
                            Train: <strong>{elig.train_sample_count?.toLocaleString()}</strong>
                            {elig.validation_sample_count > 0 && <> | Val: <strong>{elig.validation_sample_count?.toLocaleString()}</strong></>}
                            {elig.test_sample_count > 0 && <> | Test: <strong>{elig.test_sample_count?.toLocaleString()}</strong></>}
                          </div>
                        )}
                      </div>

                      {/* Architecture Eligibility */}
                      <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', fontWeight: 600, color: '#f8fafc' }}>
                          Architecture Eligibility:
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', marginTop: '2px' }}>
                          {elig.architecture_tier === 'NOT_EVALUATED' ? (
                            <><span style={{ color: '#64748b', marginLeft: '18px' }}>— Not Evaluated</span></>
                          ) : (
                            <>
                              {elig.architecture_eligible ? <CheckCircle size={12} color="#10b981" /> : <XCircle size={12} color="#ef4444" />}
                              <span style={{ color: elig.architecture_eligible ? '#10b981' : '#ef4444' }}>
                                {elig.required_architecture || job.architecture} (Tier: {elig.architecture_tier})
                              </span>
                            </>
                          )}
                        </div>
                        {elig.architecture_reasons && elig.architecture_reasons.length > 0 && (
                          <div style={{ fontSize: '10px', color: 'var(--text-secondary)', marginTop: '2px', marginLeft: '18px' }}>
                            {elig.architecture_reasons.map((r, i) => <div key={i}>• {r}</div>)}
                          </div>
                        )}
                      </div>

                      {/* Final Eligibility */}
                      <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '8px', marginTop: '4px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', fontWeight: 600 }}>
                          <span style={{ color: '#f8fafc' }}>Final:</span>
                          {elig.eligible ? <CheckCircle size={14} color="#10b981" /> : <XCircle size={14} color="#ef4444" />}
                          <span style={{ color: elig.eligible ? '#10b981' : '#ef4444' }}>
                            {elig.eligible ? 'Eligible' : 'Not Eligible'}
                          </span>
                        </div>
                      </div>

                    </div>
                  )}
                  
                  {/* Join flow: request with the eligibility summary -> admin approves -> join open rounds */}
                  {(() => {
                    const mine = job.my_participation;
                    const state = mine?.authorization_status;
                    const openRound = job.rounds?.find(r => r.status === 'OPEN' || r.status === 'RECEIVING');
                    const eligible = !!elig?.eligible;
                    const note = (color, text) => (
                      <div style={{ fontSize: '11px', color, marginBottom: '8px' }}>{text}</div>
                    );
                    const wide = { width: '100%', justifyContent: 'center' };
                    return (
                      <>
                        {state === 'PENDING' && note('#3b82f6', 'Request sent. Waiting for the admin to approve your hospital for this job.')}
                        {state === 'APPROVED' && job.plan?.protocol !== 'flower' && note('#10b981', 'Approved for this job. You can join its open rounds.')}
                        {state === 'REJECTED' && note('#ef4444', `Not approved${mine.decision_reason ? `: ${mine.decision_reason}` : '.'} You can request again.`)}
                        {requestError?.jobId === job.federation_job_id && note('#ef4444', requestError.text)}
                        {state === 'APPROVED' && job.plan?.protocol === 'flower' ? (
                          <FlowerJobState job={job} conn={flower[job.federation_job_id]} issue={flowerIssues[job.federation_job_id]}
                            datasetId={flowerDataset(hospitalId, job.federation_job_id)} datasets={localDatasets}
                            selectedDatasetId={selectedDatasetId}
                            onUseDataset={() => setFlowerDataset(hospitalId, job.federation_job_id, selectedDatasetId)}
                            onOpenMonitor={() => setActiveTab('training_monitor')} />
                        ) : state === 'APPROVED' ? (
                          <button className="btn btn-primary" style={wide} disabled={!eligible || !openRound} onClick={() => handleParticipate(job)}>
                            {!eligible ? 'Not Eligible' : openRound ? `Participate in round ${openRound.round_number}` : 'Waiting for a round to open'}
                          </button>
                        ) : state === 'PENDING' ? (
                          <button className="btn btn-secondary" style={wide} disabled>Waiting for approval</button>
                        ) : (
                          <button className="btn btn-primary" style={wide} disabled={!eligible || requesting === job.federation_job_id}
                                  onClick={() => handleRequestJoin(job)}
                                  title="Sends the admin only the summary above: eligibility, classes, image counts and hardware tier. No images.">
                            {!selectedDatasetId ? 'Prepare a dataset to check eligibility'
                              : !elig ? (evaluating ? 'Checking eligibility...' : 'Eligibility not checked yet')
                              : !eligible ? 'Not Eligible'
                              : requesting === job.federation_job_id ? 'Sending request...' : state === 'REJECTED' ? 'Request again' : 'Request to join'}
                          </button>
                        )}
                      </>
                    );
                  })()}
                </div>
              );
            })}
          </div>
        </div>
      )}

      <FederatedTrainingConfirmModal
        isOpen={!!pendingJoin}
        onClose={() => setPendingJoin(null)}
        onConfirm={confirmParticipate}
        job={pendingJoin?.job}
        round={pendingJoin?.round}
        datasetName={selectedDataset?.name}
        eligibility={pendingJoin && eligibilityResults[pendingJoin.job.federation_job_id]}
        starting={joining}
        error={joinError}
      />
    </div>
  );
};

/** An approved Flower job on the hospital side: which dataset it trains with and which Flower step is running. */
const FlowerJobState = ({ job, conn, issue, datasetId, datasets, selectedDatasetId, onUseDataset, onOpenMonitor }) => {
  const dataset = datasets.find((d) => d.dataset_id === datasetId);
  const running = job.flower?.running && !['finished', 'failed', 'stopped'].includes(job.flower?.state);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
        Flower job: training starts automatically on this machine when a round begins, while this app is open.
        {' '}Dataset: <strong style={{ color: '#fff' }}>{dataset?.name || (datasetId ? datasetId : 'not chosen')}</strong>
        {selectedDatasetId && selectedDatasetId !== datasetId && (
          <button className="btn btn-secondary" style={{ marginLeft: '8px', fontSize: '10px', padding: '2px 8px' }} onClick={onUseDataset}>
            Use the selected dataset
          </button>
        )}
      </div>
      {running ? <FlowerSteps status={conn} issue={issue} />
        : <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
            {job.flower?.state === 'finished' ? 'This job has finished all its rounds.' : 'Waiting for the admin to start the federation.'}
          </div>}
      {conn?.state === 'training' && (
        <button className="btn btn-secondary" style={{ fontSize: '11px' }} onClick={onOpenMonitor}>Open the Training Monitor</button>
      )}
    </div>
  );
};

export default HospitalFederationView;
