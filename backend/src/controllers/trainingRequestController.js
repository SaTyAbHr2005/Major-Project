const crypto = require('crypto');
const TrainingRequest = require('../models/TrainingRequest');
const Hospital = require('../authentication/models/Hospital');
const AuditService = require('../authentication/services/auditService');

const createTrainingRequest = async (req, res) => {
  const { disease, task, description, model_architecture, training_config } = req.body || {};

  try {
    if (!disease || typeof disease !== 'string' || disease.trim().length === 0) {
      return res.status(400).json({
        success: false,
        error: { code: 'VALIDATION_ERROR', message: 'Target disease is required' }
      });
    }

    // Privacy Invariant: Requests are strictly metadata instructions.
    // Raw medical images, DICOMs, or patient data are strictly forbidden.
    if (req.body.raw_images || req.body.patient_data || req.body.dicom_files) {
      return res.status(400).json({
        success: false,
        error: {
          code: 'PRIVACY_VIOLATION',
          message: 'Zero-Raw-Data Invariant: Training requests must only contain metadata. Medical imaging data is strictly prohibited.'
        }
      });
    }

    const request_id = `REQ_${crypto.randomBytes(6).toString('hex')}`;

    const newRequest = await TrainingRequest.create({
      request_id,
      researcher_id: req.user.user_id,
      researcher_name: req.user.name,
      disease: disease.trim(),
      task: task ? task.trim() : 'Image Classification',
      description: description ? description.trim() : '',
      status: 'OPEN',
      model_architecture: model_architecture || 'ResNet-18',
      model_version: 'v1',
      participating_hospitals: [],
      training_config: training_config || {
        target_epochs: 5,
        batch_size: 16,
        learning_rate: 0.001
      }
    });

    await AuditService.logEvent({
      user_id: req.user.user_id,
      action: 'TRAINING_REQUEST_CREATED',
      resource_type: 'TrainingRequest',
      resource_id: request_id,
      ip_address: req.ip,
      user_agent: req.get('User-Agent'),
      success: true,
      metadata: { disease: newRequest.disease, task: newRequest.task }
    });

    res.status(201).json({
      success: true,
      data: {
        training_request: newRequest
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: { code: 'SERVER_ERROR', message: error.message }
    });
  }
};

const getAllTrainingRequests = async (req, res) => {
  try {
    const filter = {};
    if (req.query.status) {
      filter.status = req.query.status;
    }
    if (req.query.disease) {
      filter.disease = new RegExp(req.query.disease, 'i');
    }

    const requests = await TrainingRequest.find(filter).sort({ created_at: -1 });

    res.status(200).json({
      success: true,
      data: {
        training_requests: requests
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: { code: 'SERVER_ERROR', message: error.message }
    });
  }
};

const getMyTrainingRequests = async (req, res) => {
  try {
    const requests = await TrainingRequest.find({
      researcher_id: req.user.user_id
    }).sort({ created_at: -1 });

    res.status(200).json({
      success: true,
      data: {
        training_requests: requests
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: { code: 'SERVER_ERROR', message: error.message }
    });
  }
};

const getTrainingRequestById = async (req, res) => {
  const { request_id } = req.params;

  try {
    const request = await TrainingRequest.findOne({ request_id });
    if (!request) {
      return res.status(404).json({
        success: false,
        error: { code: 'NOT_FOUND', message: 'Training request not found' }
      });
    }

    res.status(200).json({
      success: true,
      data: {
        training_request: request
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: { code: 'SERVER_ERROR', message: error.message }
    });
  }
};

const participateInTrainingRequest = async (req, res) => {
  const { request_id } = req.params;
  const hospital_id = req.user.hospital_id;

  if (!hospital_id) {
    return res.status(403).json({
      success: false,
      error: { code: 'HOSPITAL_ACCESS_DENIED', message: 'Only hospital operators with an assigned hospital can participate' }
    });
  }

  try {
    const request = await TrainingRequest.findOne({ request_id });
    if (!request) {
      return res.status(404).json({
        success: false,
        error: { code: 'NOT_FOUND', message: 'Training request not found' }
      });
    }

    if (['COMPLETED', 'CANCELLED'].includes(request.status)) {
      return res.status(400).json({
        success: false,
        error: { code: 'REQUEST_INACTIVE', message: `Cannot participate in a ${request.status} training request` }
      });
    }

    // Retrieve hospital name
    const hospital = await Hospital.findOne({ hospital_id });
    const hospital_name = hospital ? hospital.name : `Hospital Node (${hospital_id})`;

    const existingIndex = request.participating_hospitals.findIndex(
      p => p.hospital_id === hospital_id
    );

    if (existingIndex >= 0) {
      const current = request.participating_hospitals[existingIndex];
      if (['ACCEPTED', 'TRAINING'].includes(current.status)) {
        return res.status(200).json({
          success: true,
          data: {
            message: 'Hospital is already participating in this training request',
            participation: current,
            training_request: request
          }
        });
      }
      // Re-joining after previous withdrawal
      current.status = 'ACCEPTED';
      current.joined_at = new Date();
      current.withdrawn_at = null;
      current.withdrawal_reason = null;
    } else {
      request.participating_hospitals.push({
        hospital_id,
        hospital_name,
        status: 'ACCEPTED',
        joined_at: new Date(),
        training_metrics: {
          current_epoch: null,
          total_epochs: null,
          loss: null,
          accuracy: null,
          status: 'Waiting for training session'
        }
      });
    }

    if (request.status === 'OPEN') {
      request.status = 'ACTIVE';
    }

    await request.save();

    await AuditService.logEvent({
      user_id: req.user.user_id,
      action: 'HOSPITAL_PARTICIPATION_ACCEPTED',
      resource_type: 'TrainingRequest',
      resource_id: request_id,
      hospital_id,
      ip_address: req.ip,
      user_agent: req.get('User-Agent'),
      success: true,
      metadata: { hospital_id, request_id }
    });

    const participation = request.participating_hospitals.find(p => p.hospital_id === hospital_id);

    res.status(200).json({
      success: true,
      data: {
        message: 'Hospital successfully joined training request',
        participation,
        training_request: request
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: { code: 'SERVER_ERROR', message: error.message }
    });
  }
};

const withdrawFromTrainingRequest = async (req, res) => {
  const { request_id } = req.params;
  const { reason } = req.body || {};
  const hospital_id = req.user.hospital_id;

  if (!hospital_id) {
    return res.status(403).json({
      success: false,
      error: { code: 'HOSPITAL_ACCESS_DENIED', message: 'Only hospital operators can withdraw participation' }
    });
  }

  try {
    const request = await TrainingRequest.findOne({ request_id });
    if (!request) {
      return res.status(404).json({
        success: false,
        error: { code: 'NOT_FOUND', message: 'Training request not found' }
      });
    }

    const hospitalEntry = request.participating_hospitals.find(
      p => p.hospital_id === hospital_id
    );

    if (!hospitalEntry) {
      return res.status(400).json({
        success: false,
        error: { code: 'NOT_PARTICIPATING', message: 'This hospital is not enrolled in this training request' }
      });
    }

    if (hospitalEntry.status === 'WITHDRAWN') {
      return res.status(200).json({
        success: true,
        data: {
          message: 'Hospital is already withdrawn from this training request',
          participation: hospitalEntry
        }
      });
    }

    // Historical record rule: Never delete the record. Update status and save reason.
    hospitalEntry.status = 'WITHDRAWN';
    hospitalEntry.withdrawn_at = new Date();
    hospitalEntry.withdrawal_reason = reason ? String(reason).trim() : 'Voluntary hospital withdrawal';

    await request.save();

    await AuditService.logEvent({
      user_id: req.user.user_id,
      action: 'HOSPITAL_PARTICIPATION_WITHDRAWN',
      resource_type: 'TrainingRequest',
      resource_id: request_id,
      hospital_id,
      ip_address: req.ip,
      user_agent: req.get('User-Agent'),
      success: true,
      metadata: { hospital_id, request_id, reason: hospitalEntry.withdrawal_reason }
    });

    res.status(200).json({
      success: true,
      data: {
        message: 'Hospital participation successfully withdrawn. Historical record preserved.',
        participation: hospitalEntry,
        training_request: request
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: { code: 'SERVER_ERROR', message: error.message }
    });
  }
};

// Hospital reports aggregate progress of its LOCAL training run (Member 1 ML
// pipeline). Only whitelisted numeric metrics are stored - never images,
// paths, or patient data.
const reportTrainingProgress = async (req, res) => {
  const { request_id } = req.params;
  const hospital_id = req.user.hospital_id;
  const { status, current_epoch, total_epochs, loss, accuracy } = req.body || {};

  if (!hospital_id) {
    return res.status(403).json({
      success: false,
      error: { code: 'HOSPITAL_ACCESS_DENIED', message: 'Only hospital operators can report training progress' }
    });
  }

  const isCount = (v) => v === null || v === undefined || (Number.isInteger(v) && v >= 0 && v <= 100000);
  const isNum = (v) => v === null || v === undefined || (typeof v === 'number' && Number.isFinite(v));
  if (!['TRAINING', 'COMPLETED'].includes(status) || !isCount(current_epoch) || !isCount(total_epochs)
      || !isNum(loss) || !isNum(accuracy) || (typeof accuracy === 'number' && (accuracy < 0 || accuracy > 1))) {
    return res.status(400).json({
      success: false,
      error: { code: 'VALIDATION_ERROR', message: 'status must be TRAINING or COMPLETED; metrics must be numeric (accuracy in [0, 1])' }
    });
  }

  try {
    const request = await TrainingRequest.findOne({ request_id });
    if (!request) {
      return res.status(404).json({
        success: false,
        error: { code: 'NOT_FOUND', message: 'Training request not found' }
      });
    }

    const entry = request.participating_hospitals.find(p => p.hospital_id === hospital_id);
    if (!entry || entry.status === 'WITHDRAWN') {
      return res.status(400).json({
        success: false,
        error: { code: 'NOT_PARTICIPATING', message: 'This hospital is not an active participant in this training request' }
      });
    }

    const previousStatus = entry.status;
    entry.status = status;
    entry.training_metrics = {
      current_epoch: current_epoch ?? null,
      total_epochs: total_epochs ?? null,
      loss: loss ?? null,
      accuracy: accuracy ?? null,
      status: status === 'COMPLETED' ? 'Local training completed' : 'Local training in progress'
    };
    await request.save();

    if (previousStatus !== status) {
      await AuditService.logEvent({
        user_id: req.user.user_id,
        action: status === 'COMPLETED' ? 'HOSPITAL_LOCAL_TRAINING_COMPLETED' : 'HOSPITAL_LOCAL_TRAINING_STARTED',
        resource_type: 'TrainingRequest',
        resource_id: request_id,
        hospital_id,
        ip_address: req.ip,
        user_agent: req.get('User-Agent'),
        success: true,
        metadata: { hospital_id, request_id }
      });
    }

    res.status(200).json({ success: true, data: { participation: entry } });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: { code: 'SERVER_ERROR', message: error.message }
    });
  }
};

module.exports = {
  reportTrainingProgress,
  createTrainingRequest,
  getAllTrainingRequests,
  getMyTrainingRequests,
  getTrainingRequestById,
  participateInTrainingRequest,
  withdrawFromTrainingRequest
};
