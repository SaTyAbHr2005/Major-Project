import React, { useState, useEffect } from 'react';
import { useFederationStore, COLLECTING } from '../stores/federationStore';
import { Network, Plus, CheckCircle, RefreshCw, Play } from 'lucide-react';
import { useConfirm } from '../components/common/ConfirmDialog';
import { Notice } from '../components/common/Notice';
import { useAggregateRound } from '../components/federation/AggregateRoundDialog';
import { FlowerRunPanel } from '../components/federation/FlowerRunPanel';

const FIELD_STYLE = { padding: '8px', background: 'var(--bg-nested)', color: '#fff', border: '1px solid var(--border-subtle)', borderRadius: '4px' };

/** One line describing a job's round plan, e.g. "FedProx (μ 0.01) · 1 of 3 rounds done · next round opens automatically (24 h)". */
const planSummary = (job) => {
  const plan = job.plan || {};
  const agg = job.aggregation_configuration || {};
  const parts = [plan.protocol === 'flower' ? 'Flower (automatic rounds)' : 'Manual rounds',
    plan.strategy === 'fedprox' ? `FedProx (μ ${job.training_requirements?.proximal_mu})` : 'FedAvg'];
  parts.push(plan.total_rounds ? `${plan.rounds_completed} of ${plan.total_rounds} rounds done` : `${plan.rounds_completed ?? 0} round(s) done, open-ended`);
  if (plan.protocol === 'flower') parts.push(`${agg.round_duration_hours} h per round`);
  else if (plan.auto_continue) parts.push(`next round opens automatically (${agg.round_duration_hours} h)`);
  if (plan.early_stopping) parts.push(`early stopping on ${plan.early_stopping.metric || 'accuracy'} (+${plan.early_stopping.min_delta ?? 0}, ${plan.early_stopping.patience} round(s))`);
  return parts.join(' · ');
};

const STATUS_COLOR = { PENDING: '#eab308', APPROVED: 'var(--status-healthy)', REJECTED: '#ef4444' };

/** The hospital's own eligibility summary in one line (reported by the hospital; the server cannot verify it). */
const eligibilityLine = (e = {}) => {
  if (!Object.keys(e).length) return 'No summary: this hospital joined before join requests existed.';
  const parts = [];
  if (e.task_match === false) parts.push('task does not match');
  if (e.required_classes) {
    const have = e.required_classes.filter((c) => (e.available_classes || []).includes(c)).length;
    parts.push(`classes ${have} of ${e.required_classes.length}${e.missing_classes?.length ? ` (missing ${e.missing_classes.join(', ')})` : ''}`);
  }
  if (e.train_sample_count != null) {
    parts.push(`${e.train_sample_count.toLocaleString()} training / ${(e.validation_sample_count ?? 0).toLocaleString()} validation / `
      + `${(e.test_sample_count ?? 0).toLocaleString()} test images`);
  }
  if (e.architecture_tier && e.architecture_tier !== 'NOT_EVALUATED') parts.push(`hardware: ${e.architecture_tier.toLowerCase().replace(/_/g, ' ')}`);
  return parts.join(' · ') || (e.eligible ? 'eligible' : 'not eligible');
};

/** Hospitals that asked to join a job, with their eligibility summary and the admin's decision buttons. */
const ParticipantRequests = ({ job, hospitalNames, onDecide }) => {
  const list = job.participants || [];
  const pending = list.filter((p) => p.authorization_status === 'PENDING').length;
  return (
    <div style={{ marginBottom: '16px', padding: '10px', background: 'rgba(0,0,0,0.2)', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
      <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '6px' }}>
        Hospital requests{pending ? ` · ${pending} waiting for your decision` : ''}
      </div>
      {list.length === 0 && (
        <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
          No hospital has asked to join yet. Hospitals check eligibility on their own machine and request from Federation Status.
        </div>
      )}
      {list.map((p) => (
        <div key={p.hospital_identifier} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '12px', padding: '6px 0', borderTop: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '11px', minWidth: 0 }}>
            <span style={{ color: '#fff', fontWeight: 600 }}>{hospitalNames[p.hospital_identifier] || p.hospital_identifier}</span>
            <span style={{ color: 'var(--text-muted)' }}> ({p.hospital_identifier})</span>
            <span style={{ marginLeft: '8px', color: STATUS_COLOR[p.authorization_status], fontWeight: 600 }}>{p.authorization_status}</span>
            <div style={{ color: 'var(--text-secondary)', marginTop: '2px' }}>{eligibilityLine(p.eligibility)}</div>
            {p.decision_reason && p.expected_status !== 'LEGACY' && <div style={{ color: 'var(--text-muted)' }}>Note: {p.decision_reason}</div>}
          </div>
          {onDecide && <div style={{ display: 'flex', gap: '6px', flexShrink: 0 }}>
            {p.authorization_status !== 'APPROVED' && (
              <button className="btn btn-primary" style={{ fontSize: '11px', padding: '3px 10px' }} onClick={() => onDecide(job, p, 'APPROVED')}>Approve</button>
            )}
            {p.authorization_status !== 'REJECTED' && (
              <button className="btn btn-danger" style={{ fontSize: '11px', padding: '3px 10px' }} onClick={() => onDecide(job, p, 'REJECTED')}>
                {p.authorization_status === 'APPROVED' ? 'Remove' : 'Reject'}
              </button>
            )}
          </div>}
        </div>
      ))}
      <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '6px' }}>
        The summary is reported by the hospital's own eligibility check; the server never sees its images and cannot verify it.
      </div>
    </div>
  );
};

export const FederatedTrainingView = () => {

  const {
    jobs: jobsResource, architectures: archResource, loadJobs, loadArchitectures,
    createJob, createRound, deleteJob, deleteRound, hospitalNames, decideParticipation, startFlower,
  } = useFederationStore();
  const [aggregateDialog, startAggregate] = useAggregateRound();
  const jobs = jobsResource.data;
  const architectures = archResource.data;
  const loading = jobsResource.status === 'idle' || (jobsResource.status === 'loading' && jobs.length === 0);
  const error = jobsResource.error || archResource.error;
  const [confirmDialog, confirm] = useConfirm();
  const [message, setMessage] = useState(null);   // { tone: 'success' | 'error', text } from the last action
  
  // Create Job Form
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newJob, setNewJob] = useState({
    job_id: '',
    task: '',
    architecture: 'ResNet50', // Will default to first if available
    num_classes: 2,
    class_mapping_list: ['', ''], // dynamic list of class names
    minimum_participants: 2,
    protocol: 'flower',          // flower: the Flower server runs every planned round; manual: admin and hospitals act per round
    // Round plan (Module 9): planned rounds, automatic next round, early stopping. Jobs created here use
    // FedAvg; FedProx is a research setting (API and the Flower simulation bench), not an admin choice.
    total_rounds: 3,             // empty = open-ended (manual rounds only): the admin opens every round
    round_duration_hours: 24,
    early_stop: false,
    es_metric: 'accuracy',
    es_min_delta: 0.005,
    es_patience: 2,
  });

  // Open-round form: one job at a time; the deadline input is local time.
  const [roundForm, setRoundForm] = useState(null);
  const toLocalInput = (d) => new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
  const nextRoundNumber = (job) => Math.max(0, ...(job.rounds || []).map((r) => r.round_number)) + 1;
  const roundInProgress = (job) => (job.rounds || []).some((r) => [...COLLECTING, 'CREATED', 'AGGREGATING'].includes(r.status));

  const handleCreateRound = async (e, job) => {
    e.preventDefault();
    const n = nextRoundNumber(job);
    setMessage(null);
    try {
      const res = await createRound({
        job_id: job.federation_job_id,
        round_id: `${job.federation_job_id}_R${n}`,
        round_number: n,
        deadline: new Date(roundForm.deadline).toISOString(),
        minimum_participants: Number(roundForm.minimum_participants),
      });
      setRoundForm(null);
      const from = res.base_model_source === 'seed' ? 'a new seed model' : res.base_model_source === 'provided' ? 'the given base model' : `global model ${res.base_model_source}`;
      setMessage({ tone: 'success', text: `Round ${n} is open. Hospitals start from ${from}.` });
    } catch (err) {
      setMessage({ tone: 'error', text: `Error opening round: ${err.message}` });
    }
  };

  const loadJobsAndArchs = () => Promise.all([loadJobs(), loadArchitectures()]);
  // The list stays visible while it reloads, so the button itself shows that a refresh is running.
  const [refreshing, setRefreshing] = useState(false);
  const refresh = async () => {
    setRefreshing(true);
    try { await loadJobsAndArchs(); } finally { setRefreshing(false); }
  };

  useEffect(() => {
    loadJobsAndArchs();
  }, []);

  // Default the form's architecture to the first one the registry offers.
  useEffect(() => {
    if (architectures.length > 0 && !architectures.some((a) => a.name === newJob.architecture)) {
      setNewJob((prev) => ({ ...prev, architecture: architectures[0].name }));
    }
  }, [architectures]);

  const handleCreateJob = async (e) => {
    e.preventDefault();
    try {
      if (newJob.num_classes < 2) throw new Error("Number of classes must be >= 2");
      if (!newJob.architecture) throw new Error("Architecture is required");
      
      let class_mapping = {};
      const validClasses = newJob.class_mapping_list.map(s => s.trim()).filter(s => s);
      if (validClasses.length !== newJob.num_classes) {
        throw new Error(`Expected exactly ${newJob.num_classes} non-empty class names, got ${validClasses.length}`);
      }
      validClasses.forEach((name, idx) => { class_mapping[name] = idx; });

      const aggregation_configuration = { strategy: 'fedavg', protocol: newJob.protocol };
      if (newJob.protocol === 'flower' && !newJob.total_rounds) throw new Error('Flower jobs need a total number of rounds.');
      if (newJob.total_rounds) {
        aggregation_configuration.total_rounds = Number(newJob.total_rounds);
        aggregation_configuration.round_duration_hours = Number(newJob.round_duration_hours);
      }
      if (newJob.early_stop) {
        aggregation_configuration.early_stopping = {
          metric: newJob.es_metric, min_delta: Number(newJob.es_min_delta), patience: Number(newJob.es_patience),
        };
      }
      await createJob({
        job_id: newJob.job_id,
        task: newJob.task,
        architecture: newJob.architecture,
        num_classes: newJob.num_classes,
        minimum_participants: newJob.minimum_participants,
        class_mapping,
        aggregation_configuration,
      });
      setShowCreateForm(false);
      setMessage({ tone: 'success', text: `Federation job ${newJob.job_id} created.` });
    } catch (err) {
      setMessage({ tone: 'error', text: `Error creating job: ${err.message}` });
    }
  };

  const [startingJob, setStartingJob] = useState(null);
  const handleStartFlower = async (job) => {
    setMessage(null);
    setStartingJob(job.federation_job_id);
    try {
      await startFlower(job.federation_job_id);
      setMessage({ tone: 'success', text: `Federation started for ${job.federation_job_id}. Rounds start as soon as enough approved hospitals are connected.` });
    } catch (err) {
      setMessage({ tone: 'error', text: `Could not start the federation: ${err.message}` });
    } finally {
      setStartingJob(null);
    }
  };

  // Approve / reject a hospital's join request (rejecting an approved hospital removes its access).
  const handleDecision = async (job, p, decision) => {
    const name = hospitalNames[p.hospital_identifier] || p.hospital_identifier;
    if (decision === 'REJECTED') {
      const ok = await confirm({
        title: p.authorization_status === 'APPROVED' ? `Remove ${name} from ${job.federation_job_id}?` : `Reject ${name}'s request?`,
        message: 'The hospital will not be able to join this job\'s rounds. Updates it already sent stay in their rounds. It can request again later.',
        confirmLabel: p.authorization_status === 'APPROVED' ? 'Remove' : 'Reject',
        danger: true,
      });
      if (!ok) return;
    }
    setMessage(null);
    try {
      await decideParticipation(job.federation_job_id, p.hospital_identifier, decision);
      setMessage({ tone: 'success', text: `${name} ${decision === 'APPROVED' ? 'approved' : 'is no longer allowed'} for ${job.federation_job_id}. The hospital was notified.` });
    } catch (err) {
      setMessage({ tone: 'error', text: err.message });
    }
  };

  // Deleting something with federation history archives/cancels it instead (Module 9 keeps the record).
  const handleDelete = async ({ title, hasHistory, historyText, emptyText, archiveLabel, deleteLabel, run }) => {
    const ok = await confirm({
      title,
      message: hasHistory ? historyText : emptyText,
      confirmLabel: hasHistory ? archiveLabel : deleteLabel,
      danger: true,
    });
    if (!ok) return;
    setMessage(null);
    try { await run(); } catch (err) { setMessage({ tone: 'error', text: err.message }); }
  };

  const handleDeleteJob = (job) => handleDelete({
    title: `Delete federation job ${job.federation_job_id}?`,
    hasHistory: job.rounds && job.rounds.some(r => r.accepted_participants.length > 0 || !['CREATED', 'OPEN', 'RECEIVING'].includes(r.status)),
    historyText: 'This job has completed federation activity, so it cannot be permanently deleted; it will be archived/cancelled instead.',
    emptyText: 'This job has no federation history and will be permanently removed.',
    archiveLabel: 'Archive job',
    deleteLabel: 'Delete job',
    run: () => deleteJob(job.federation_job_id),
  });

  const handleDeleteRound = (round) => handleDelete({
    title: `Delete round ${round.round_number}?`,
    hasHistory: round.accepted_participants.length > 0 || !['CREATED', 'OPEN', 'RECEIVING'].includes(round.status),
    historyText: 'This round contains federation history and cannot be permanently deleted; it will be cancelled instead.',
    emptyText: 'This round has no submitted updates and will be permanently deleted.',
    archiveLabel: 'Cancel round',
    deleteLabel: 'Delete round',
    run: () => deleteRound(round.round_id),
  });

  // Archived jobs are read-only records (kept because they have history); they get their own section.
  const activeJobs = jobs.filter((j) => j.status !== 'ARCHIVED');
  const archivedJobs = jobs.filter((j) => j.status === 'ARCHIVED');
  const renderJob = (job) => {
    const readOnly = job.status === 'ARCHIVED';
    return (
            <div key={job.federation_job_id} className="card" style={{ background: 'var(--bg-nested)', padding: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '12px', marginBottom: '12px' }}>
                <div>
                  <div style={{ fontSize: '10px', color: 'var(--accent-teal)' }}>{job.task} / {job.architecture}</div>
                  <h2 style={{ fontSize: '16px', color: '#fff' }}>{job.federation_job_id}</h2>
                </div>
                <div style={{ textAlign: 'right', display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '6px' }}>
                  <div className="badge badge-primary">{job.status}</div>
                  {!readOnly && (
                    <button className="btn btn-danger" style={{ padding: '2px 8px', fontSize: '10px' }} onClick={() => handleDeleteJob(job)}>
                      Delete
                    </button>
                  )}
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>Min Participants: {job.minimum_participants}</div>
                </div>
              </div>
              
              <div style={{ marginBottom: '10px', fontSize: '12px', color: 'var(--text-secondary)' }}>
                <span style={{ color: 'var(--text-muted)' }}>Training plan: </span>{planSummary(job)}
              </div>
              {readOnly ? (
                <Notice tone="info" style={{ marginBottom: '12px' }}>
                  Archived: this job is read-only. Its rounds, hospitals and global models are kept for the record;
                  no new rounds can be opened.
                </Notice>
              ) : job.plan?.finished && (
                <Notice tone="success" style={{ marginBottom: '12px' }}>Finished: {job.plan.reason}</Notice>
              )}

              <div style={{ marginBottom: '16px', padding: '10px', background: 'rgba(0,0,0,0.2)', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '4px' }}>Canonical Class Mapping:</div>
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  {job.class_mapping && Object.entries(job.class_mapping).map(([name, idx]) => (
                    <span key={name} style={{ fontSize: '11px', padding: '2px 6px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
                      {idx}: <span style={{ color: '#fff' }}>{name}</span>
                    </span>
                  ))}
                </div>
              </div>

              <ParticipantRequests job={job} hospitalNames={hospitalNames} onDecide={readOnly ? null : handleDecision} />

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <h4 style={{ fontSize: '13px', color: '#cbd5e1' }}>Rounds</h4>
                  {readOnly ? null : job.plan?.protocol === 'flower' ? (
                    <button className="btn btn-primary" style={{ fontSize: '11px', padding: '4px 10px' }}
                      disabled={job.flower?.running || job.plan?.finished || startingJob === job.federation_job_id}
                      title={job.plan?.finished ? 'This job has finished its training plan' : job.flower?.running ? 'The Flower server is running' : undefined}
                      onClick={() => handleStartFlower(job)}>
                      <Play size={12} /> {startingJob === job.federation_job_id ? 'Starting…' : job.flower?.running ? 'Federation running' : 'Start federation'}
                    </button>
                  ) : (
                  <button className="btn btn-primary" style={{ fontSize: '11px', padding: '4px 10px' }}
                    disabled={roundInProgress(job) || roundForm?.jobId === job.federation_job_id || job.plan?.finished || (job.plan?.total_rounds && job.plan?.next_round_number == null)}
                    title={job.plan?.finished ? 'This job has finished its training plan'
                      : roundInProgress(job) ? 'Finish or delete the current round first'
                        : job.plan?.total_rounds && job.plan?.next_round_number == null ? 'Every planned round already exists' : undefined}
                    onClick={() => setRoundForm({ jobId: job.federation_job_id, deadline: toLocalInput(new Date(Date.now() + 24 * 3600 * 1000)), minimum_participants: job.minimum_participants })}>
                    <Plus size={12} /> Open round {nextRoundNumber(job)}
                  </button>
                  )}
                </div>
                {job.plan?.protocol === 'flower' && <FlowerRunPanel job={job} hospitalNames={hospitalNames} onMessage={setMessage} />}
                {roundForm?.jobId === job.federation_job_id && (
                  <form onSubmit={(e) => handleCreateRound(e, job)} style={{ display: 'flex', gap: '12px', alignItems: 'flex-end', flexWrap: 'wrap', marginBottom: '12px', padding: '12px', background: 'var(--bg-card)', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
                    <label style={{ display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '12px', color: 'var(--text-muted)' }}>
                      Deadline (updates after it are rejected)
                      <input type="datetime-local" required min={toLocalInput(new Date())} value={roundForm.deadline} onChange={(e) => setRoundForm({ ...roundForm, deadline: e.target.value })}
                        style={{ padding: '6px 8px', background: 'var(--bg-nested)', color: '#fff', border: '1px solid var(--border-subtle)', borderRadius: '4px' }} />
                    </label>
                    <label style={{ display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '12px', color: 'var(--text-muted)' }}>
                      Minimum participants
                      <input type="number" min="1" required value={roundForm.minimum_participants} onChange={(e) => setRoundForm({ ...roundForm, minimum_participants: e.target.value })}
                        style={{ width: '120px', padding: '6px 8px', background: 'var(--bg-nested)', color: '#fff', border: '1px solid var(--border-subtle)', borderRadius: '4px' }} />
                    </label>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', flex: 1, minWidth: '200px' }}>
                      Every hospital starts from this job's latest global model, or from one shared freshly initialised model if there is none yet.
                      They download it before training and check its checksum.
                    </div>
                    <div style={{ display: 'flex', gap: '6px' }}>
                      <button type="button" className="btn btn-secondary" onClick={() => setRoundForm(null)}>Cancel</button>
                      <button type="submit" className="btn btn-primary">Open round</button>
                    </div>
                  </form>
                )}
                {job.rounds && job.rounds.length > 0 ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {job.rounds.map(round => (
                      <div key={round.round_id} style={{ background: 'var(--bg-card)', padding: '12px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <div>
                            <span style={{ fontWeight: 'bold', color: '#fff' }}>Round {round.round_number} ({round.round_id})</span>
                            <span style={{ marginLeft: '8px', fontSize: '12px', color: 'var(--text-muted)' }}>Status: {round.status}</span>
                            <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                              Base model: {round.base_model_id ? `${round.base_model_id} v${round.base_model_version}` : 'none'}
                            </div>
                            <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                              Deadline: {round.deadline}
                            </div>
                          </div>
                          <div style={{ display: 'flex', gap: '6px' }}>
                            {!readOnly && job.plan?.protocol !== 'flower' && (
                              <button className="btn btn-secondary" onClick={() => { setMessage(null); startAggregate(job, round); }} disabled={!COLLECTING.includes(round.status)}>
                                Aggregate Round
                              </button>
                            )}
                            {!readOnly && (
                              <button className="btn btn-danger" onClick={() => handleDeleteRound(round)}>
                                Delete Round
                              </button>
                            )}
                          </div>
                        </div>

                        <div style={{ marginTop: '12px', borderTop: '1px solid var(--border-subtle)', paddingTop: '8px' }}>
                          <h5 style={{ fontSize: '11px', color: '#94a3b8', marginBottom: '6px' }}>Participants ({round.received_participants.length})</h5>
                          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                            {round.received_participants.map((pid, idx) => {
                              const isAccepted = round.accepted_participants.includes(pid);
                              const isRejected = round.rejected_participants.includes(pid);
                              return (
                                <div key={idx} style={{ padding: '6px', background: 'rgba(255,255,255,0.02)', borderRadius: '4px', fontSize: '11px', display: 'flex', justifyContent: 'space-between' }}>
                                  <span>{hospitalNames[pid] ? `${hospitalNames[pid]} (${pid})` : pid}</span>
                                  {isAccepted ? <span style={{ color: 'var(--status-healthy)' }}>ACCEPTED ✓</span> : 
                                   isRejected ? <span style={{ color: '#ef4444' }}>REJECTED / EXPIRED</span> : 
                                   <span style={{ color: '#eab308' }}>WAITING</span>}
                                </div>
                              );
                            })}
                            {round.received_participants.length === 0 && <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>No updates received yet.</div>}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>No rounds created yet.</div>
                )}
              </div>
            </div>
    );
  };

  return (
    <div style={{ padding: '16px 20px', height: '100%', overflowY: 'auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <div>
          <h1 style={{ fontSize: '20px', fontWeight: '700', color: '#f8fafc' }}>Federation Jobs & Rounds</h1>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Manage federated learning jobs, participants, and aggregations</div>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button className="btn btn-secondary" onClick={refresh} disabled={refreshing} aria-busy={refreshing}>
            <RefreshCw size={14} className={refreshing ? 'spin' : undefined} /> {refreshing ? 'Refreshing...' : 'Refresh'}
          </button>
          <button className="btn btn-primary" onClick={() => setShowCreateForm(!showCreateForm)}>
            <Plus size={14} /> New Federation Job
          </button>
        </div>
      </div>

      {confirmDialog}
      {aggregateDialog}
      {error && <Notice tone="error" style={{ marginBottom: '16px' }}>{error}</Notice>}
      {message && <Notice tone={message.tone} style={{ marginBottom: '16px' }}>{message.text}</Notice>}

      {showCreateForm && (
        <div className="card" style={{ padding: '16px', marginBottom: '20px' }}>
          <h3 style={{ marginBottom: '12px' }}>Create Federation Job</h3>
          <form onSubmit={handleCreateJob} style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', flex: 1, minWidth: '150px' }}>
                <label style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Job ID</label>
                <input required placeholder="e.g. JOB_01" style={{ padding: '8px', background: 'var(--bg-nested)', color: '#fff', border: '1px solid var(--border-subtle)', borderRadius: '4px' }} value={newJob.job_id} onChange={e => setNewJob({...newJob, job_id: e.target.value})} />
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', flex: 1, minWidth: '150px' }}>
                <label style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Task / Disease</label>
                <input required placeholder="e.g. Malaria" style={{ padding: '8px', background: 'var(--bg-nested)', color: '#fff', border: '1px solid var(--border-subtle)', borderRadius: '4px' }} value={newJob.task} onChange={e => setNewJob({...newJob, task: e.target.value})} />
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', flex: 1, minWidth: '150px' }}>
                <label style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Architecture</label>
                <select required style={{ padding: '8px', background: 'var(--bg-nested)', color: '#fff', border: '1px solid var(--border-subtle)', borderRadius: '4px' }} value={newJob.architecture} onChange={e => setNewJob({...newJob, architecture: e.target.value})}>
                  {architectures.map(arch => (
                    <option key={arch.name} value={arch.name}>{arch.display_name || arch.name}</option>
                  ))}
                </select>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', flex: 1, minWidth: '120px' }}>
                <label style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Number of Classes</label>
                <input type="number" min="2" required style={{ padding: '8px', background: 'var(--bg-nested)', color: '#fff', border: '1px solid var(--border-subtle)', borderRadius: '4px' }} value={newJob.num_classes} onChange={e => {
                  const num = parseInt(e.target.value);
                  let newList = [...newJob.class_mapping_list];
                  if (num > newList.length) {
                    while(newList.length < num) newList.push('');
                  } else if (num < newList.length) {
                    newList = newList.slice(0, num);
                  }
                  setNewJob({...newJob, num_classes: num, class_mapping_list: newList});
                }} />
              </div>
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', background: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
              <label style={{ fontSize: '12px', color: '#fff', fontWeight: 'bold' }}>Canonical Class Mapping (Required)</label>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Specify the exact class labels for the dataset in order.</div>
              {newJob.class_mapping_list.map((cls, idx) => (
                <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ fontSize: '12px', color: 'var(--text-muted)', width: '20px' }}>{idx}:</span>
                  <input required placeholder={`Class ${idx} (e.g. Normal)`} style={{ padding: '6px 8px', background: 'var(--bg-nested)', color: '#fff', border: '1px solid var(--border-subtle)', borderRadius: '4px', flex: 1 }} value={cls} onChange={e => {
                    const newList = [...newJob.class_mapping_list];
                    newList[idx] = e.target.value;
                    setNewJob({...newJob, class_mapping_list: newList});
                  }} />
                </div>
              ))}
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', background: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
              <label style={{ fontSize: '12px', color: '#fff', fontWeight: 'bold' }}>Training plan</label>
              <label style={{ display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '12px', color: 'var(--text-muted)' }}>
                How rounds run
                <select style={{ ...FIELD_STYLE, maxWidth: '320px' }} value={newJob.protocol}
                  onChange={e => setNewJob({ ...newJob, protocol: e.target.value, total_rounds: e.target.value === 'flower' && !newJob.total_rounds ? 3 : newJob.total_rounds })}>
                  <option value="flower">Flower (automatic): approved hospitals train when each round starts</option>
                  <option value="manual">Manual rounds: you open and aggregate each round</option>
                </select>
              </label>
              <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', alignItems: 'flex-end' }}>
                <label style={{ display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '12px', color: 'var(--text-muted)' }}>
                  {newJob.protocol === 'flower' ? 'Total rounds' : 'Total rounds (empty = open-ended)'}
                  <input type="number" min="1" placeholder="e.g. 5" required={newJob.protocol === 'flower'} style={{ ...FIELD_STYLE, width: '150px' }} value={newJob.total_rounds} onChange={e => setNewJob({...newJob, total_rounds: e.target.value})} />
                </label>
                {newJob.total_rounds && (
                  <label style={{ display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '12px', color: 'var(--text-muted)' }}>
                    {newJob.protocol === 'flower' ? 'Hours per round (deadline)' : 'Hours per automatic round'}
                    <input type="number" step="any" min="0.1" required style={{ ...FIELD_STYLE, width: '150px' }} value={newJob.round_duration_hours} onChange={e => setNewJob({...newJob, round_duration_hours: e.target.value})} />
                  </label>
                )}
              </div>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: 'var(--text-secondary)' }}>
                <input type="checkbox" checked={newJob.early_stop} onChange={e => setNewJob({...newJob, early_stop: e.target.checked})} />
                Stop early when the evaluated global model stops improving
              </label>
              {newJob.early_stop && (
                <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                  <label style={{ display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '12px', color: 'var(--text-muted)' }}>
                    Metric
                    <select style={FIELD_STYLE} value={newJob.es_metric} onChange={e => setNewJob({...newJob, es_metric: e.target.value})}>
                      <option value="accuracy">Accuracy</option>
                      <option value="f1">F1</option>
                    </select>
                  </label>
                  <label style={{ display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '12px', color: 'var(--text-muted)' }}>
                    Minimum improvement
                    <input type="number" step="any" min="0" required style={{ ...FIELD_STYLE, width: '130px' }} value={newJob.es_min_delta} onChange={e => setNewJob({...newJob, es_min_delta: e.target.value})} />
                  </label>
                  <label style={{ display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '12px', color: 'var(--text-muted)' }}>
                    Rounds without improvement
                    <input type="number" min="1" required style={{ ...FIELD_STYLE, width: '130px' }} value={newJob.es_patience} onChange={e => setNewJob({...newJob, es_patience: e.target.value})} />
                  </label>
                </div>
              )}
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                {newJob.protocol === 'flower' ? 'With Flower you press Start federation once: each round starts as soon as enough approved '
                  + 'hospitals are connected, their apps train automatically, and the round is aggregated when every hospital has sent '
                  + 'its update (or at the deadline, or when you choose Aggregate now). ' : ''}
                Without a total, you open every round yourself. With a total, each aggregation opens the next round automatically
                until the total is reached; hospitals get the number of hours you set (field shown once a total is entered) to send
                their update. Early stopping evaluates each new global model on the server's evaluation dataset.
                Updates are combined with FedAvg (average weighted by each hospital's number of training images).
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '4px' }}>
              <button type="submit" className="btn btn-primary">Create Job</button>
            </div>
          </form>
        </div>
      )}

      {loading ? (
        <div style={{ color: '#fff' }}>Loading jobs...</div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {activeJobs.map(renderJob)}
          {activeJobs.length === 0 && (
            <div style={{ color: 'var(--text-muted)' }}>{archivedJobs.length ? 'No active federation jobs.' : 'No federation jobs found.'}</div>
          )}
          {archivedJobs.length > 0 && (
            <details className="card" style={{ background: 'var(--bg-nested)', padding: '12px 16px' }}>
              <summary style={{ cursor: 'pointer', fontSize: '13px', fontWeight: 600, color: '#cbd5e1' }}>
                Archived jobs ({archivedJobs.length})
                <span style={{ marginLeft: '8px', fontSize: '11px', fontWeight: 400, color: 'var(--text-muted)' }}>
                  read-only: kept because they have rounds or global models
                </span>
              </summary>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginTop: '12px' }}>
                {archivedJobs.map(renderJob)}
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  );
};
