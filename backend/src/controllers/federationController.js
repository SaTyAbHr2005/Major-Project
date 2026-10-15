const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs/promises');
const Hospital = require('../authentication/models/Hospital');
const { notifyHospitals } = require('../services/notificationService');

const callPythonAdapter = (payload) => {
  return new Promise((resolve, reject) => {
    const pythonPath = 'python'; // Ensure python is in PATH
    const adapterPath = path.resolve(__dirname, '../../../federation/api_adapter.py');
    const child = spawn(pythonPath, [adapterPath], {
      cwd: path.resolve(__dirname, '../../../'),
      env: { ...process.env, PYTHONPATH: '.' }
    });
    
    let stdoutData = '';
    let stderrData = '';
    
    child.stdout.on('data', (data) => { stdoutData += data.toString(); });
    child.stderr.on('data', (data) => { stderrData += data.toString(); });
    
    child.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Python process exited with code ${code}. Error: ${stderrData}`));
        return;
      }
      try {
        const result = JSON.parse(stdoutData);
        if (!result.success) {
          reject(new Error(result.error));
        } else {
          resolve(result);
        }
      } catch (err) {
        reject(new Error(`Failed to parse python output: ${stdoutData}`));
      }
    });

    child.on('error', (err) => {
      reject(err);
    });
    
    child.stdin.write(JSON.stringify(payload));
    child.stdin.end();
  });
};

// The architecture catalogue is fixed in code, and building it imports PyTorch (~7 s),
// so it is fetched once per backend process.
let architecturesCache = null;

exports.getArchitectures = async (req, res) => {
  try {
    architecturesCache = architecturesCache || await callPythonAdapter({ action: 'get_architectures' });
    res.json(architecturesCache);
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

exports.listJobs = async (req, res) => {
  try {
    const result = await callPythonAdapter({ action: 'list_jobs' });
    // A hospital sees only its own join request, never other hospitals' eligibility summaries.
    if (req.user.role === 'hospital_operator') {
      (result.jobs || []).forEach((j) => {
        j.my_participation = (j.participants || []).find((p) => p.hospital_identifier === req.user.hospital_id) || null;
        delete j.participants;
      });
    }
    // Admins see which hospital each participant id is; hospitals only get the ids.
    if (req.user.role === 'admin') {
      const ids = [...new Set((result.jobs || []).flatMap((j) => [
        ...(j.rounds || []).flatMap((r) => r.expected_participants || []),
        ...(j.participants || []).map((p) => p.hospital_identifier),
      ]))];
      const hospitals = ids.length ? await Hospital.find({ hospital_id: { $in: ids } }, 'hospital_id name') : [];
      result.hospital_names = Object.fromEntries(hospitals.map((h) => [h.hospital_id, h.name]));
    }
    res.json(result);
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

exports.getJobDetails = async (req, res) => {
  try {
    const result = await callPythonAdapter({ action: 'get_job_details', job_id: req.params.job_id });
    res.json(result);
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

exports.deleteJob = async (req, res) => {
  try {
    const result = await callPythonAdapter({ action: 'delete_job', job_id: req.params.job_id, admin_id: req.user.user_id });
    res.json(result);
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

exports.createJob = async (req, res) => {
  try {
    const result = await callPythonAdapter({ action: 'create_job', ...req.body });
    res.json(result);
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

const notifyRoundOpen = (rnd) => notifyHospitals({
  type: 'ROUND_OPEN',
  title: `Round ${rnd.round_number} of ${rnd.federation_job_id} is open`,
  content: `Federation job ${rnd.federation_job_id} opened round ${rnd.round_number}. It accepts model updates until ${rnd.deadline}. `
    + 'To take part, start federated training for this job under Federation Status.',
  job_id: rnd.federation_job_id, round_id: rnd.round_id,
});

exports.createRound = async (req, res) => {
  try {
    const result = await callPythonAdapter({ action: 'create_round', ...req.body });
    if (result.round) result.notified_hospitals = await notifyRoundOpen(result.round);
    res.json(result);
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

// Hospitals download the round's canonical base checkpoint (one of its two files) before local training.
exports.downloadBaseModel = async (req, res) => {
  try {
    const result = await callPythonAdapter({ action: 'get_base_model', round_id: req.params.round_id });
    if (!result.files.includes(req.params.file)) {
      return res.status(404).json({ success: false, error: 'Unknown base model file.' });
    }
    res.sendFile(path.join(result.dir, req.params.file));
  } catch (err) {
    res.status(404).json({ success: false, error: err.message });
  }
};

exports.deleteRound = async (req, res) => {
  try {
    const result = await callPythonAdapter({ action: 'delete_round', round_id: req.params.round_id, admin_id: req.user.user_id });
    res.json(result);
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

// Hospital asks to join a job, sending the summary of its local eligibility check (numbers only).
exports.requestParticipation = async (req, res) => {
  try {
    const result = await callPythonAdapter({
      action: 'request_participation',
      job_id: req.params.job_id,
      participant_id: req.user.hospital_id, // the authenticated hospital, never a body field
      eligibility: req.body.eligibility,
    });
    res.json(result);
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
};

// Admin approves or rejects a hospital's join request; the hospital is notified.
exports.decideParticipation = async (req, res) => {
  try {
    const decision = String(req.body.decision || '').toUpperCase();
    const result = await callPythonAdapter({
      action: 'decide_participation',
      job_id: req.params.job_id,
      participant_id: req.params.hospital_id,
      decision,
      reason: req.body.reason || '',
      admin_id: req.user.user_id,
    });
    const approved = decision === 'APPROVED';
    result.notified_hospitals = await notifyHospitals({
      only: [req.params.hospital_id],
      type: approved ? 'PARTICIPATION_APPROVED' : 'PARTICIPATION_REJECTED',
      title: `Request to join ${req.params.job_id} ${approved ? 'approved' : 'not approved'}`,
      content: approved
        ? `Your hospital can now take part in federation job ${req.params.job_id}. Join an open round under Federation Status.`
        : `Your request to join federation job ${req.params.job_id} was not approved.${req.body.reason ? ` Reason: ${req.body.reason}` : ''}`,
      job_id: req.params.job_id,
    });
    res.json(result);
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
};

exports.registerParticipant = async (req, res) => {
  try {
    const result = await callPythonAdapter({ 
      action: 'register_participant', 
      job_id: req.params.job_id,
      round_id: req.params.round_id,
      participant_id: req.user.hospital_id // Use authenticated hospital ID
    });
    res.json(result);
  } catch (err) {
    // Not approved for the job (federation/participation.py) is a permission refusal, not a server error.
    res.status(/cannot join/.test(err.message) ? 403 : 500).json({ success: false, error: err.message });
  }
};

exports.submitUpdate = async (req, res) => {
  try {
    const result = await callPythonAdapter({ 
      action: 'submit_update', 
      job_id: req.params.job_id,
      round_id: req.params.round_id,
      participant_id: req.user.hospital_id,
      handoff_dir: req.body.handoff_dir
    });
    if (!result.success) {
        return res.status(400).json(result);
    }
    res.json(result);
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};


exports.aggregateRound = async (req, res) => {
  try {
    const result = await callPythonAdapter({ action: 'aggregate', ...req.body });
    // Hospitals that never joined this round are told it is closed and not to start training for it.
    const rnd = result.round;
    if (rnd) result.notified_hospitals = await notifyHospitals({
      exclude: rnd.expected_participants || [],
      type: 'ROUND_CLOSED',
      title: `Round ${rnd.round_number} of ${rnd.federation_job_id} is closed`,
      content: `Round ${rnd.round_number} of federation job ${rnd.federation_job_id} stopped accepting updates and its aggregation has started. `
        + 'Do not start federated training for this round. You will be notified when the next round opens.',
      job_id: rnd.federation_job_id, round_id: rnd.round_id,
    });
    // The job's round plan may have opened the next round automatically.
    if (result.next_round) result.notified_next_round = await notifyRoundOpen(result.next_round);
    res.json(result);
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

exports.listGlobalModels = async (req, res) => {
  try {
    const result = await callPythonAdapter({ action: 'list_global_models' });
    res.json(result);
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

exports.evaluateModel = async (req, res) => {
  try {
    const result = await callPythonAdapter({ action: 'evaluate', ...req.body });
    res.json(result);
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

// Progress of a running evaluation, written by federation/evaluation.py. Read straight from the
// file (no Python start-up) so the admin screen can poll it every second or two.
exports.getEvaluationProgress = async (req, res) => {
  const id = req.params.global_model_id;
  if (!/^[A-Za-z0-9_.-]+$/.test(id)) {
    return res.status(400).json({ success: false, error: 'Invalid global model id.' });
  }
  const file = path.resolve(__dirname, '../../../federation_artifacts/evaluation_progress', `${id}.json`);
  try {
    res.json({ success: true, progress: JSON.parse(await fs.readFile(file, 'utf8')) });
  } catch (err) {
    if (err.code === 'ENOENT') return res.json({ success: true, progress: null });
    res.status(500).json({ success: false, error: err.message });
  }
};

exports.promoteModel = async (req, res) => {
  try {
    const result = await callPythonAdapter({ action: 'promote', ...req.body });
    res.json(result);
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

exports.getHospitalStatus = async (req, res) => {
  try {
    // M1-M3 authoritative identity chain:
    //   JWT → User document (via authMiddleware: User.findOne({ user_id: decoded.user_id }))
    //   User.hospital_id  — set at registration from req.body.hospital_id
    //                     — verified against Hospital collection at registration time
    //   user_id (USR_xxxx) is a separate random identifier; it is NEVER the hospital_id.
    const hospital_id = req.user.hospital_id;

    if (!hospital_id) {
      return res.status(400).json({
        success: false,
        error: 'No hospital_id associated with this account. Ensure you are registered as a hospital operator.'
      });
    }

    const result = await callPythonAdapter({ action: 'get_hospital_status', hospital_id });
    res.json(result);
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};
