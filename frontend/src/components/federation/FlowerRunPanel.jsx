import React, { useEffect, useState } from 'react';
import { Radio, Square, Zap } from 'lucide-react';
import { getFlowerStatus } from '../../services/federationService';
import { useFederationStore } from '../../stores/federationStore';

const SERVER_STATE = {
  not_started: 'Not started', starting: 'Starting', waiting_for_hospitals: 'Waiting for hospitals', training: 'Hospitals training',
  aggregating: 'Aggregating', calibrating: 'Calibrating BatchNorm', finished: 'Finished', failed: 'Failed', stopped: 'Stopped',
};
const HOSPITAL_STATE = {
  connected: 'Connected', training: 'Training', sent: 'Update accepted', rejected: 'Update rejected', calibrating: 'Calibrating',
  done: 'Round done', failed: 'Failed', missed: 'Missed the round', update_sent: 'Update sent',
};
// Flower hands the server a round's updates only when the round ends; until then the hospital's own report says it sent one.
const shownState = (h) => (h.state === 'training' && ['sending', 'waiting_for_others'].includes(h.progress?.state) ? 'update_sent' : h.state);
const LIVE = ['starting', 'waiting_for_hospitals', 'training', 'aggregating', 'calibrating'];
const tone = (s) => (['failed', 'rejected'].includes(s) ? '#ef4444' : ['missed', 'stopped'].includes(s) ? '#eab308'
  : ['done', 'sent', 'update_sent', 'finished'].includes(s) ? '#10b981' : '#93c5fd');

/** Admin: live view of a Flower job's server (round, hospitals, events) with Aggregate now and Stop. */
export const FlowerRunPanel = ({ job, hospitalNames = {}, onMessage }) => {
  const { stopFlower, aggregateFlowerNow } = useFederationStore();
  const [st, setSt] = useState(null);
  const [busy, setBusy] = useState(null);

  useEffect(() => {
    let alive = true;
    const load = () => getFlowerStatus(job.federation_job_id).then((r) => alive && setSt(r.status)).catch(() => {});
    load();
    const t = setInterval(load, 2000);
    return () => { alive = false; clearInterval(t); };
  }, [job.federation_job_id]);

  if (!st || st.state === 'not_started') return null;
  const live = LIVE.includes(st.state) && st.running && !st.stale;
  const hospitals = Object.entries(st.hospitals || {});
  const stillTraining = hospitals.filter(([, h]) => shownState(h) === 'training').map(([id]) => hospitalNames[id] || id);
  const canClose = st.state === 'training' && !st.closing && (st.sent_updates || 0) >= (st.minimum_participants || 1);

  const act = async (name, fn, confirmText) => {
    if (confirmText && !window.confirm(confirmText)) return;
    setBusy(name);
    try { await fn(job.federation_job_id); } catch (e) { onMessage?.({ tone: 'error', text: e.message }); } finally { setBusy(null); }
  };

  return (
    <div style={{ marginBottom: '12px', padding: '12px', background: 'var(--bg-card)', borderRadius: '6px', border: '1px solid rgba(59,130,246,0.35)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', color: '#fff' }}>
          <Radio size={14} color={live ? '#3b82f6' : '#94a3b8'} style={live ? { animation: 'pulseRing 1.6s infinite', borderRadius: '50%' } : undefined} />
          <strong>Flower server: {SERVER_STATE[st.state] || st.state}</strong>
          {st.stale && <span style={{ color: '#eab308', fontSize: '11px' }}>(no heartbeat — process stopped?)</span>}
        </div>
        <div style={{ display: 'flex', gap: '6px' }}>
          <button className="btn btn-primary" style={{ fontSize: '11px', padding: '4px 10px' }} disabled={!canClose || busy}
            title={canClose ? undefined : `Needs at least ${st.minimum_participants || 1} hospital update(s) sent`}
            onClick={() => act('close', aggregateFlowerNow, `Aggregate round ${st.fl_round} now?`
              + (stillTraining.length ? `\n\nStill training (their update for this round will be skipped): ${stillTraining.join(', ')}` : ''))}>
            <Zap size={12} /> {busy === 'close' ? 'Closing…' : 'Aggregate now'}
          </button>
          {st.running && (
            <button className="btn btn-danger" style={{ fontSize: '11px', padding: '4px 10px' }} disabled={busy}
              onClick={() => act('stop', stopFlower, 'Stop this job\'s Flower server? The current round stays open; you can start again later.')}>
              <Square size={11} /> Stop
            </button>
          )}
        </div>
      </div>

      <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', marginTop: '8px', fontSize: '11px', color: 'var(--text-secondary)' }}>
        {st.fl_round ? <span>Round {st.fl_round}{st.total_rounds ? ` of ${st.total_rounds}` : ''}</span> : null}
        {st.state === 'training' && <span>Updates sent: {st.sent_updates || 0} (minimum {st.minimum_participants})</span>}
        {st.address && <span>Server: {st.address}</span>}
        {st.closing && <span style={{ color: '#eab308' }}>Closing the round…</span>}
      </div>
      {st.message && <div style={{ marginTop: '6px', fontSize: '11px', color: st.state === 'failed' ? '#ef4444' : 'var(--text-muted)' }}>{st.message}</div>}

      {hospitals.length > 0 && (
        <table style={{ width: '100%', marginTop: '10px', fontSize: '11px', borderCollapse: 'collapse' }}>
          <tbody>
            {hospitals.map(([id, h]) => (
              <tr key={id} style={{ borderTop: '1px solid var(--border-subtle)' }}>
                <td style={{ padding: '5px 4px', color: '#fff' }}>{hospitalNames[id] ? `${hospitalNames[id]} (${id})` : id}</td>
                <td style={{ padding: '5px 4px', color: tone(shownState(h)) }}>{HOSPITAL_STATE[shownState(h)] || h.state}</td>
                <td style={{ padding: '5px 4px', color: 'var(--text-muted)' }}>
                  {shownState(h) === 'update_sent' ? 'Waiting for the other hospitals'
                    : h.state === 'training' && h.progress?.epoch ? `epoch ${h.progress.epoch} of ${h.progress.total_epochs}` : h.message}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {(st.events || []).length > 0 && (
        <details style={{ marginTop: '8px' }}>
          <summary style={{ fontSize: '11px', color: 'var(--text-muted)', cursor: 'pointer' }}>What happened ({st.events.length})</summary>
          <ul style={{ listStyle: 'none', padding: 0, margin: '6px 0 0', fontSize: '11px', color: 'var(--text-secondary)' }}>
            {[...st.events].reverse().slice(0, 10).map((e, i) => (
              <li key={i} style={{ padding: '2px 0' }}>
                <span style={{ color: 'var(--text-muted)' }}>{new Date(e.at).toLocaleTimeString()}</span> {e.text}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
};

export default FlowerRunPanel;
