import { create } from 'zustand';
import * as federation from '../services/federationService';
import ml from '../services/mlClient';
import authService from '../services/authService';
import { emptyResource, loadResource, unwrap } from './resource';
import { useMLStore } from './mlStore';
import trainingService from '../services/trainingService';

const API_URL = import.meta.env?.VITE_API_URL || 'http://localhost:5000/api/v1';

/** Module 9 federation state: jobs/rounds, global models and this hospital's submissions. */
const initial = {
  jobs: emptyResource([]),
  architectures: emptyResource([]),
  globalModels: emptyResource([]),
  hospitalUpdates: emptyResource([]),   // the signed-in hospital's own submissions
  jobDetails: emptyResource([]),        // admin: every job with its rounds and submitted updates
  taskVocabulary: [],
  hospitalNames: {},                    // admin: hospital_id -> name for round participants
  flower: {},                           // hospital: job_id -> this machine's Flower connection status
  flowerIssues: {},                     // hospital: job_id -> why the app could not connect (dataset missing, ...)
};

// ── Flower jobs on the hospital side ──
// While the hospital app is open, every approved Flower job whose server runs gets a background connection
// in the local ML service. Training it starts is followed by the Training Monitor like any other run.
const LIVE = ['starting', 'waiting_for_hospitals', 'training', 'aggregating', 'calibrating'];
const DEAD = ['failed', 'disconnected'];
const RETRY_MS = 30000;
const datasetKey = (hospitalId, jobId) => `medfl_flower_dataset:${hospitalId}:${jobId}`;
const store = {
  get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* private mode: asked again next time */ } },
};
let flowerTimer = null;
const attempts = {};      // job_id -> last connect attempt (ms)
let followed = null;      // training job id the Training Monitor was pointed at


export const useFederationStore = create((set, get) => ({
  ...initial,
  session: 0,
  reset: () => { get().stopFlowerSync?.(); set((s) => ({ ...initial, session: s.session + 1 })); },

  /** The dataset this hospital uses for a Flower job (remembered from its join request). */
  flowerDataset: (hospitalId, jobId) => store.get(datasetKey(hospitalId, jobId)),
  setFlowerDataset: (hospitalId, jobId, datasetId) => {
    store.set(datasetKey(hospitalId, jobId), datasetId);
    delete attempts[jobId];
    set((s) => ({ flowerIssues: { ...s.flowerIssues, [jobId]: null } }));
  },

  startFlowerSync: (hospitalId) => {
    clearInterval(flowerTimer);
    const session = get().session;
    const tick = async () => {
      if (get().session !== session) return;
      let jobs;
      let local;
      try {
        jobs = unwrap(await federation.getJobs(), 'jobs') || [];
        local = (await ml.get('/federation/flower/status')).connections || [];
      } catch { return; }                       // backend or local service briefly unreachable
      if (get().session !== session) return;
      const byJob = Object.fromEntries(local.map((c) => [c.job_id, c]));
      set((s) => ({ flower: byJob, jobs: s.jobs.status === 'ready' ? { ...s.jobs, data: jobs } : s.jobs }));
      for (const job of jobs) {
        const id = job.federation_job_id;
        const approved = job.my_participation?.authorization_status === 'APPROVED';
        if (!approved || !job.flower?.running || !LIVE.includes(job.flower.state)) continue;
        const conn = byJob[id];
        if (conn && !DEAD.includes(conn.state)) continue;
        if (Date.now() - (attempts[id] || 0) < RETRY_MS) continue;
        attempts[id] = Date.now();
        const datasetId = store.get(datasetKey(hospitalId, id));
        if (!datasetId) {
          set((s) => ({ flowerIssues: { ...s.flowerIssues, [id]: 'Choose the dataset this hospital trains with for this job.' } }));
          continue;
        }
        try {
          await ml.post('/federation/flower/connect', { job, dataset_id: datasetId, address: job.flower.address });
          set((s) => ({ flowerIssues: { ...s.flowerIssues, [id]: null } }));
        } catch (e) {
          set((s) => ({ flowerIssues: { ...s.flowerIssues, [id]: e.message } }));
        }
      }
      // A round's training is an ordinary training job: point the Training Monitor at it.
      const training = local.find((c) => c.training_job_id && c.state === 'training');
      if (training && training.training_job_id !== followed
          && useMLStore.getState().trainingJob?.job_id !== training.training_job_id) {
        followed = training.training_job_id;
        try { useMLStore.getState().trackTrainingJob(await trainingService.getJob(followed)); } catch { followed = null; }
      }
    };
    tick();
    flowerTimer = setInterval(tick, 3000);
  },
  stopFlowerSync: () => { clearInterval(flowerTimer); flowerTimer = null; followed = null; },

  loadJobs: () => loadResource(set, get, 'jobs', async () => {
    const res = await federation.getJobs();
    set({ hospitalNames: res.hospital_names || {} });
    return unwrap(res, 'jobs');
  }),
  loadArchitectures: () => loadResource(set, get, 'architectures',
    async () => unwrap(await federation.getArchitectures(), 'architectures')),
  loadGlobalModels: () => loadResource(set, get, 'globalModels',
    async () => unwrap(await federation.getGlobalModels(), 'models') || []),
  loadHospitalUpdates: () => loadResource(set, get, 'hospitalUpdates',
    async () => unwrap(await federation.getHospitalFederationStatus(), 'updates') || []),

  /** One request per job (the list endpoint has no updates); fine for the handful of jobs a demo has. */
  loadJobDetails: () => loadResource(set, get, 'jobDetails', async () => {
    const jobs = unwrap(await federation.getJobs(), 'jobs') || [];
    return Promise.all(jobs.map(async (j) => unwrap(await federation.getJobDetails(j.federation_job_id), 'job')));
  }),

  loadTaskVocabulary: async () => {
    const session = get().session;
    const { tasks } = await ml.get('/federation/task-vocabulary').catch(() => ({ tasks: [] }));
    if (get().session === session) set({ taskVocabulary: tasks || [] });
  },

  // Admin actions: each refreshes the list it changed.
  createJob: async (job) => { await federation.createJob(job); await get().loadJobs(); },
  createRound: async (round) => { const res = await federation.createRound(round); await get().loadJobs(); return res; },
  aggregateRound: async (jobId, roundId, forceClose) => {
    const res = await federation.aggregateRound(jobId, roundId, forceClose);
    await get().loadJobs();
    return res;
  },
  deleteJob: async (jobId) => { await federation.deleteJob(jobId); await get().loadJobs(); },
  deleteRound: async (roundId) => { await federation.deleteRound(roundId); await get().loadJobs(); },
  requestParticipation: async (jobId, eligibility) => { await federation.requestParticipation(jobId, eligibility); await get().loadJobs(); },
  decideParticipation: async (jobId, hospitalId, decision, reason) => {
    await federation.decideParticipation(jobId, hospitalId, decision, reason);
    await get().loadJobs();
  },
  startFlower: async (jobId) => { const res = await federation.startFlower(jobId); await get().loadJobs(); return res; },
  stopFlower: async (jobId) => { await federation.stopFlower(jobId); await get().loadJobs(); },
  aggregateFlowerNow: (jobId) => federation.aggregateFlowerNow(jobId),
  evaluateModel: async (id) => { await federation.evaluateModel(id); await get().loadGlobalModels(); },
  promoteModel: async (id, evalId) => { await federation.promoteModel(id, evalId); await get().loadGlobalModels(); },

  /**
   * Hospital joins an open round: registers with M9, then the local ML service trains
   * (and submits the handoff when done). The job is followed by the Training Monitor.
   */
  participate: async (job, round, datasetId) => {
    if (!round) throw new Error('No active round available for participation.');
    await federation.registerParticipant(job.federation_job_id, round.round_id);
    const trainingJob = await ml.post('/federation/participate', {
      job,
      dataset_id: datasetId,
      round_id: round.round_id,
      auth_token: authService.getToken(),
      api_url: API_URL,
    });
    useMLStore.getState().trackTrainingJob(trainingJob);
    return trainingJob;
  },
}));

// ── Derived views of Module 9 data (pure functions, shared by the admin screens) ──

export const COLLECTING = ['OPEN', 'RECEIVING', 'READY_FOR_AGGREGATION'];

/** Every round of every job, newest first, each with its job attached. */
export const allRounds = (jobs) => jobs
  .flatMap((job) => (job.rounds || []).map((round) => ({ ...round, job })))
  .sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));

/** The newest round still collecting hospital updates, else the newest round. */
export const currentRound = (jobs) => {
  const rounds = allRounds(jobs);
  return rounds.find((r) => COLLECTING.includes(r.status)) || rounds[0] || null;
};

/** Latest completed evaluation of a global model, if any. */
export const latestEvaluation = (model) => (model.evaluations || [])
  .filter((e) => e.status === 'COMPLETED' || e.accuracy != null)
  .sort((a, b) => String(b.completed_at || b.started_at).localeCompare(String(a.completed_at || a.started_at)))[0] || null;

/** Lifecycle events (job created, round opened/closed/aggregated, model created/evaluated/promoted), newest first. */
export const federationTimeline = (jobs, models) => {
  const events = [];
  const add = (time, title, detail) => time && events.push({ time, title, detail });
  jobs.forEach((j) => {
    add(j.created_at, `Job created: ${j.task}`, `${j.architecture} · ${j.federation_job_id}`);
    (j.rounds || []).forEach((r) => {
      const label = `${j.task} round ${r.round_number}`;
      add(r.opened_at, `Round opened: ${label}`, `${r.expected_participants.length} expected hospital(s)`);
      add(r.aggregation_completed_at, `Round aggregated: ${label}`, `${r.accepted_participants.length} update(s) aggregated`);
      add(r.closed_at, `Round closed: ${label}`, `Status ${r.status}`);
    });
  });
  models.forEach((m) => {
    add(m.created_at, `Global model v${m.version} created`, `${m.task} · ${m.architecture} · ${m.aggregation_method}`);
    (m.evaluations || []).forEach((e) => add(e.completed_at, `Global model v${m.version} evaluated`,
      `accuracy ${e.accuracy != null ? (e.accuracy * 100).toFixed(1) + '%' : '-'} on ${e.sample_count} samples`));
  });
  return events.sort((a, b) => String(b.time).localeCompare(String(a.time)));
};
