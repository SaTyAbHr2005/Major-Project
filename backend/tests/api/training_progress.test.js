const request = require('supertest');
const mongoose = require('mongoose');
const { MongoMemoryServer } = require('mongodb-memory-server');

const app = require('../../src/app');
const User = require('../../src/authentication/models/User');
const Hospital = require('../../src/authentication/models/Hospital');
const AuditLog = require('../../src/authentication/models/AuditLog');
const TrainingRequest = require('../../src/models/TrainingRequest');
const { ROLES } = require('../../src/authentication/permissions/roles');

let mongoServer;
let hospitalToken;
let otherHospitalToken;
let researcherToken;

const login = async (email, password) =>
  (await request(app).post('/api/v1/auth/login').send({ email, password })).body.data.access_token;

beforeAll(async () => {
  mongoServer = await MongoMemoryServer.create();
  process.env.JWT_SECRET = 'test-secret-training-progress';
  process.env.JWT_EXPIRES_IN = '1h';
  process.env.NODE_ENV = 'test';
  await mongoose.connect(mongoServer.getUri());
}, 180000);

afterAll(async () => {
  await mongoose.disconnect();
  if (mongoServer) await mongoServer.stop();
});

beforeEach(async () => {
  await Promise.all([User.deleteMany({}), Hospital.deleteMany({}), AuditLog.deleteMany({}), TrainingRequest.deleteMany({})]);
  await Hospital.create({ hospital_id: 'HOSP_000001', name: 'Node A', status: 'active', client_ids: ['c1'] });
  await Hospital.create({ hospital_id: 'HOSP_000002', name: 'Node B', status: 'active', client_ids: ['c2'] });
  await User.create({ user_id: 'USR_h1', account_id: 'HOS-001', name: 'Op A', email: 'a@node-a.org',
    password_hash: 'hospital123', role: ROLES.HOSPITAL_OPERATOR, hospital_id: 'HOSP_000001', status: 'active' });
  await User.create({ user_id: 'USR_h2', account_id: 'HOS-002', name: 'Op B', email: 'b@node-b.org',
    password_hash: 'hospital123', role: ROLES.HOSPITAL_OPERATOR, hospital_id: 'HOSP_000002', status: 'active' });
  await User.create({ user_id: 'USR_r1', account_id: 'RES-001', name: 'Dr R', email: 'r@lab.edu',
    password_hash: 'researcher123', role: ROLES.RESEARCHER, status: 'active' });
  hospitalToken = await login('a@node-a.org', 'hospital123');
  otherHospitalToken = await login('b@node-b.org', 'hospital123');
  researcherToken = await login('r@lab.edu', 'researcher123');

  await TrainingRequest.create({
    request_id: 'REQ_progress', researcher_id: 'USR_r1', researcher_name: 'Dr R', disease: 'Malaria', status: 'ACTIVE',
    participating_hospitals: [{ hospital_id: 'HOSP_000001', hospital_name: 'Node A', status: 'ACCEPTED' }]
  });
});

const report = (token, body) => request(app)
  .post('/api/v1/training-requests/REQ_progress/progress')
  .set('Authorization', `Bearer ${token}`)
  .send(body);

describe('Hospital local-training progress reporting', () => {
  it('stores aggregate metrics for the reporting hospital and audits status changes', async () => {
    let res = await report(hospitalToken, { status: 'TRAINING', current_epoch: 3, total_epochs: 10, loss: 0.21, accuracy: 0.93 });
    expect(res.status).toBe(200);
    expect(res.body.data.participation.status).toBe('TRAINING');

    res = await report(hospitalToken, { status: 'COMPLETED', current_epoch: 10, total_epochs: 10, loss: 0.1, accuracy: 0.97 });
    expect(res.status).toBe(200);

    const doc = await TrainingRequest.findOne({ request_id: 'REQ_progress' });
    expect(doc.participating_hospitals[0].status).toBe('COMPLETED');
    expect(doc.participating_hospitals[0].training_metrics).toMatchObject({ current_epoch: 10, total_epochs: 10, loss: 0.1, accuracy: 0.97 });
    const actions = (await AuditLog.find({ resource_id: 'REQ_progress' })).map(a => a.action).sort();
    expect(actions).toEqual(['HOSPITAL_LOCAL_TRAINING_COMPLETED', 'HOSPITAL_LOCAL_TRAINING_STARTED']);
  });

  it('ignores anything but the whitelisted numeric fields', async () => {
    const res = await report(hospitalToken, { status: 'TRAINING', current_epoch: 1, image_path: 'C:/scans/p1.png', patient_id: 'P1' });
    expect(res.status).toBe(200);
    const raw = JSON.stringify(await TrainingRequest.findOne({ request_id: 'REQ_progress' }).lean());
    expect(raw).not.toContain('scans');
    expect(raw).not.toContain('patient_id');
  });

  it('rejects invalid metrics and statuses', async () => {
    expect((await report(hospitalToken, { status: 'DONE' })).status).toBe(400);
    expect((await report(hospitalToken, { status: 'TRAINING', accuracy: 1.5 })).status).toBe(400);
    expect((await report(hospitalToken, { status: 'TRAINING', current_epoch: -1 })).status).toBe(400);
    expect((await report(hospitalToken, { status: 'TRAINING', loss: 'NaN' })).status).toBe(400);
  });

  it('a hospital cannot report for a request it does not participate in', async () => {
    expect((await report(otherHospitalToken, { status: 'TRAINING', current_epoch: 1 })).status).toBe(400);
  });

  it('researchers and unauthenticated callers cannot report progress', async () => {
    expect((await report(researcherToken, { status: 'TRAINING' })).status).toBe(403);
    const res = await request(app).post('/api/v1/training-requests/REQ_progress/progress').send({ status: 'TRAINING' });
    expect(res.status).toBe(401);
  });

  it('returns 404 for an unknown request', async () => {
    const res = await request(app).post('/api/v1/training-requests/REQ_nope/progress')
      .set('Authorization', `Bearer ${hospitalToken}`).send({ status: 'TRAINING' });
    expect(res.status).toBe(404);
  });
});
