/**
 * Flower servers of federation jobs whose protocol is "flower" (Module 9).
 *
 * The backend starts one Python process per job (`python -m federation.flower_app`), which runs the
 * job's planned rounds over Flower and writes its progress to
 * federation_artifacts/flower/<job_id>/status.json. This service serves that file to the UIs, keeps
 * the hospitals' own step reports (epoch, "sending", ...), asks the server to close a round early
 * (close_round file) and turns the server's events into hospital notifications.
 */
const fs = require('fs');
const net = require('net');
const path = require('path');
const { spawn } = require('child_process');
const { notifyHospitals } = require('./notificationService');

const REPO_ROOT = path.resolve(__dirname, '../../../');
const STALE_MS = 30000;
const SENT_STATES = ['sending', 'waiting_for_others'];
const ID_RE = /^[A-Za-z0-9_.-]+$/;

const running = new Map();   // job_id -> { child, timer }
const reports = new Map();   // job_id -> { hospital_id -> report }
const seenEvents = new Map(); // job_id -> number of events already turned into notifications

const jobDir = (jobId) => {
  if (!ID_RE.test(jobId)) throw Object.assign(new Error('Invalid job id.'), { status: 400 });
  return path.join(REPO_ROOT, 'federation_artifacts', 'flower', jobId);
};
const statusPath = (jobId) => path.join(jobDir(jobId), 'status.json');

const isRunning = (jobId) => running.has(jobId);

const readStatus = (jobId) => {
  let st;
  try {
    st = JSON.parse(fs.readFileSync(statusPath(jobId), 'utf8'));
  } catch (err) {
    if (err.code === 'ENOENT') return { state: 'not_started', running: isRunning(jobId) };
    throw err;
  }
  const quiet = Date.now() - Date.parse(st.updated_at || 0) > STALE_MS;
  st.running = isRunning(jobId);
  st.stale = !['finished', 'failed', 'stopped'].includes(st.state) && quiet;   // the server writes a heartbeat every 5 s
  return st;
};

const freePort = () => new Promise((resolve, reject) => {
  const srv = net.createServer();
  srv.unref();
  srv.on('error', reject);
  srv.listen(0, () => { const { port } = srv.address(); srv.close(() => resolve(port)); });
});

/** Starts the job's Flower server. `job` is the job as list_jobs returns it (with `plan`). */
const start = async (job) => {
  const jobId = job.federation_job_id;
  if (isRunning(jobId)) throw Object.assign(new Error('This job\'s Flower server is already running.'), { status: 409 });
  if (job.plan?.protocol !== 'flower') throw Object.assign(new Error('This job uses manual rounds, not Flower.'), { status: 400 });
  if (job.plan?.finished) throw Object.assign(new Error(job.plan.reason || 'This job is finished.'), { status: 400 });

  const dir = jobDir(jobId);
  fs.mkdirSync(dir, { recursive: true });
  fs.rmSync(path.join(dir, 'close_round'), { force: true });
  const port = Number(process.env.FLOWER_PORT) || await freePort();
  const backendUrl = process.env.BACKEND_SELF_URL || `http://localhost:${process.env.PORT || 5000}/api/v1`;
  const child = spawn('python', ['-m', 'federation.flower_app', '--job-id', jobId, '--port', String(port),
    '--backend-url', backendUrl, '--status-file', statusPath(jobId)], {
    cwd: REPO_ROOT, env: { ...process.env, PYTHONPATH: '.', PYTHONUNBUFFERED: '1' },
  });
  const log = (data) => process.stdout.write(`[flower ${jobId}] ${data}`);
  child.stdout?.on('data', log);
  child.stderr?.on('data', log);
  const timer = setInterval(() => processEvents(jobId).catch((e) => console.warn('Flower events:', e.message)), 3000);
  running.set(jobId, { child, timer });
  child.on('exit', () => {
    clearInterval(timer);
    running.delete(jobId);
    processEvents(jobId).catch(() => {});
  });
  reports.set(jobId, {});
  return { job_id: jobId, port };
};

const stop = (jobId) => {
  const run = running.get(jobId);
  if (!run) throw Object.assign(new Error('This job\'s Flower server is not running.'), { status: 409 });
  run.child.kill();
  try {
    const st = JSON.parse(fs.readFileSync(statusPath(jobId), 'utf8'));
    st.state = 'stopped';
    st.message = 'Stopped by the admin.';
    st.updated_at = new Date().toISOString();
    fs.writeFileSync(statusPath(jobId), JSON.stringify(st));
  } catch (_) { /* no status yet */ }
};

/** Status for the UIs. Admins get everything; a hospital only its own entry, without the event log. */
const status = (jobId, user) => {
  const st = readStatus(jobId);
  const rep = reports.get(jobId) || {};
  const hospitals = {};
  Object.entries(st.hospitals || {}).forEach(([hid, h]) => {
    hospitals[hid] = rep[hid] && rep[hid].round_id === st.round_id ? { ...h, progress: rep[hid] } : h;
  });
  if (user.role === 'admin') return { ...st, hospitals, sent_updates: sentCount(st, rep) };
  const { events, hospitals: _all, pid, ...pub } = st;
  return { ...pub, hospital: hospitals[user.hospital_id] || null };
};

const sentCount = (st, rep) => Object.values(rep)
  .filter((r) => r.round_id === st.round_id && SENT_STATES.includes(r.state)).length;

/** A hospital service tells which step it is on; the answer says whether its round is closing. */
const report = (jobId, hospitalId, body) => {
  const st = readStatus(jobId);
  const rep = reports.get(jobId) || {};
  rep[hospitalId] = {
    round_id: String(body.round_id || ''), state: String(body.state || ''), message: String(body.message || '').slice(0, 300),
    epoch: Number.isInteger(body.epoch) ? body.epoch : null,
    total_epochs: Number.isInteger(body.total_epochs) ? body.total_epochs : null, at: new Date().toISOString(),
  };
  reports.set(jobId, rep);
  return { round_id: st.round_id || null, closing: Boolean(st.closing), state: st.state };
};

/** Admin: close the current round early once at least the minimum number of hospitals sent their update. */
const aggregateNow = (jobId) => {
  const st = readStatus(jobId);
  if (st.state !== 'training' || st.closing) {
    throw Object.assign(new Error('No round of this job is collecting updates right now.'), { status: 409 });
  }
  const sent = sentCount(st, reports.get(jobId) || {});
  if (sent < (st.minimum_participants || 1)) {
    throw Object.assign(new Error(`Only ${sent} of the required ${st.minimum_participants} hospital update(s) have been sent.`), { status: 409 });
  }
  fs.writeFileSync(path.join(jobDir(jobId), 'close_round'), st.round_id);
  return { round_id: st.round_id, sent_updates: sent };
};

/** Turns new status events into notifications (same texts as the manual protocol's). */
const processEvents = async (jobId) => {
  let st;
  try { st = JSON.parse(fs.readFileSync(statusPath(jobId), 'utf8')); } catch (_) { return; }
  const events = st.events || [];
  const from = seenEvents.get(jobId) || 0;
  seenEvents.set(jobId, events.length);
  for (const e of events.slice(from)) {
    if (e.type === 'round_opened') {
      await notifyHospitals({
        type: 'ROUND_OPEN', title: `Round ${e.round_number} of ${jobId} is open`,
        content: `Federation job ${jobId} opened round ${e.round_number}. Training starts automatically on hospitals `
          + 'that are approved and have the hospital app open.',
        job_id: jobId, round_id: e.round_id,
      });
    } else if (e.type === 'round_closed') {
      await notifyHospitals({
        exclude: e.participants || [], type: 'ROUND_CLOSED', title: `Round of ${jobId} is closed`,
        content: `${e.text}. You will be notified when the next round opens.`, job_id: jobId, round_id: e.round_id,
      });
    }
  }
};

// Stop every Flower server when the backend exits, so no orphan keeps a port or a round open.
process.on('exit', () => running.forEach(({ child }) => child.kill()));

const _reset = () => { reports.clear(); seenEvents.clear(); };

module.exports = { start, stop, status, report, aggregateNow, isRunning, readStatus, _processEvents: processEvents, _reset };
