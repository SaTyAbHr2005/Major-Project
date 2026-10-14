const crypto = require('crypto');
const Hospital = require('../authentication/models/Hospital');
const Notification = require('../models/Notification');

/**
 * Stores one notification per active hospital (except `exclude`) and pushes it live
 * to the hospital's Socket.io room. Returns the hospital ids that were notified.
 * Failures are logged, never thrown: a notification must not undo a federation action.
 */
const notifyHospitals = async ({ exclude = [], only = null, type, title, content, job_id = null, round_id = null }) => {
  try {
    // `only` limits the notification to the listed hospitals (e.g. the one whose join request was decided).
    const filter = only ? { $in: only.filter((id) => !exclude.includes(id)) } : { $nin: exclude };
    const hospitals = await Hospital.find({ status: 'active', hospital_id: filter }, 'hospital_id');
    if (!hospitals.length) return [];
    const docs = await Notification.insertMany(hospitals.map(({ hospital_id }) => ({
      notification_id: `NTF_${crypto.randomBytes(8).toString('hex')}`,
      hospital_id, type, title, content, job_id, round_id,
    })));
    try {
      const io = require('../socket').getIo();
      docs.forEach((n) => io.to(`hospital_${n.hospital_id}`).emit('notification', n.toObject()));
    } catch (socketErr) {
      console.warn('Notification push failed:', socketErr.message);
    }
    return docs.map((n) => n.hospital_id);
  } catch (err) {
    console.warn('Storing notifications failed:', err.message);
    return [];
  }
};

module.exports = { notifyHospitals };
