const express = require('express');
const {
  listJobs,
  getJobDetails,
  createJob,
  createRound,
  aggregateRound,
  listGlobalModels,
  evaluateModel,
  getEvaluationProgress,
  promoteModel,
  getHospitalStatus,
  getArchitectures,
  deleteJob,
  deleteRound,
  downloadBaseModel,
  registerParticipant,
  submitUpdate,
  requestParticipation,
  decideParticipation,
  startFlower,
  stopFlower,
  getFlowerStatus,
  reportFlower,
  aggregateFlowerNow
} = require('../controllers/federationController');

const { requireAuth } = require('../authentication/middleware/authMiddleware');
const { requireRole } = require('../authentication/middleware/roleMiddleware');
const { ROLES } = require('../authentication/permissions/roles');

const router = express.Router();

router.use(requireAuth);

// Admin and Hospital Operator shared endpoints
router.get('/jobs', requireRole(ROLES.ADMIN, ROLES.HOSPITAL_OPERATOR), listJobs);
router.get('/rounds/:round_id/base-model/:file', requireRole(ROLES.ADMIN, ROLES.HOSPITAL_OPERATOR), downloadBaseModel);
router.get('/jobs/:job_id/flower/status', requireRole(ROLES.ADMIN, ROLES.HOSPITAL_OPERATOR), getFlowerStatus);

// Admin-only endpoints
router.get('/architectures', requireRole(ROLES.ADMIN), getArchitectures);
router.post('/jobs', requireRole(ROLES.ADMIN), createJob);
router.get('/jobs/:job_id', requireRole(ROLES.ADMIN), getJobDetails);
router.delete('/jobs/:job_id', requireRole(ROLES.ADMIN), deleteJob);
router.post('/rounds', requireRole(ROLES.ADMIN), createRound);
router.delete('/rounds/:round_id', requireRole(ROLES.ADMIN), deleteRound);
router.post('/rounds/aggregate', requireRole(ROLES.ADMIN), aggregateRound);
router.get('/models', requireRole(ROLES.ADMIN), listGlobalModels);
router.post('/models/evaluate', requireRole(ROLES.ADMIN), evaluateModel);
router.get('/models/:global_model_id/evaluation-progress', requireRole(ROLES.ADMIN), getEvaluationProgress);
router.post('/models/promote', requireRole(ROLES.ADMIN), promoteModel);
router.post('/jobs/:job_id/participants/:hospital_id/decision', requireRole(ROLES.ADMIN), decideParticipation);
router.post('/jobs/:job_id/flower/start', requireRole(ROLES.ADMIN), startFlower);
router.post('/jobs/:job_id/flower/stop', requireRole(ROLES.ADMIN), stopFlower);
router.post('/jobs/:job_id/flower/aggregate-now', requireRole(ROLES.ADMIN), aggregateFlowerNow);

// Hospital Operator endpoints
router.get('/hospital-status', requireRole(ROLES.HOSPITAL_OPERATOR), getHospitalStatus);
router.post('/jobs/:job_id/participation-request', requireRole(ROLES.HOSPITAL_OPERATOR), requestParticipation);
router.post('/jobs/:job_id/rounds/:round_id/participate', requireRole(ROLES.HOSPITAL_OPERATOR), registerParticipant);
router.post('/jobs/:job_id/rounds/:round_id/submit', requireRole(ROLES.HOSPITAL_OPERATOR), submitUpdate);
router.post('/jobs/:job_id/flower/report', requireRole(ROLES.HOSPITAL_OPERATOR), reportFlower);

module.exports = router;
