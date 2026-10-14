import React, { useCallback, useState } from 'react';
import Modal from '../common/Modal';
import { Notice, fmtDate } from '../common/Notice';
import { useFederationStore } from '../../stores/federationStore';
import { useUiStore } from '../../stores/uiStore';
import { CheckCircle, Loader2, XCircle } from 'lucide-react';

const participantState = (round, id) => (
  round.accepted_participants.includes(id) ? ['Update accepted', 'var(--status-healthy)']
    : round.rejected_participants.includes(id) || (round.quarantined_participants || []).includes(id) ? ['Update rejected', 'var(--status-danger)']
      : round.received_participants.includes(id) ? ['Update received, not validated', 'var(--status-warning)']
        : ['Training, no update yet', 'var(--status-warning)']
);

/**
 * Admin aggregation of a Module 9 round: confirm (with the participating hospitals),
 * then a progress view while the server closes the round and runs FedAvg, then the result.
 *   const [aggregateDialog, startAggregate] = useAggregateRound(onDone);
 *   startAggregate(job, round);  ...render {aggregateDialog}
 */
export function useAggregateRound(onDone) {
  const aggregateRound = useFederationStore((s) => s.aggregateRound);
  const hospitalNames = useFederationStore((s) => s.hospitalNames);
  const setActiveScreen = useUiStore((s) => s.setActiveScreen);
  const [state, setState] = useState(null);   // { step: confirm | running | done | error, job, round, result?, error? }

  const start = useCallback((job, round) => setState({ step: 'confirm', job, round }), []);
  const close = () => { if (state?.step !== 'running') setState(null); };

  const proceed = async () => {
    const { job, round } = state;
    setState({ ...state, step: 'running' });
    try {
      const result = await aggregateRound(job.federation_job_id, round.round_id, true);
      setState({ step: 'done', job, round, result });
      onDone?.(result);
    } catch (err) {
      setState({ step: 'error', job, round, error: err.message });
    }
  };

  let body = null;
  if (state) {
    const { step, job, round } = state;
    const accepted = round.accepted_participants.length;
    const enough = accepted >= round.minimum_participants;
    const early = new Date(round.deadline) > new Date();

    if (step === 'confirm') body = (
      <>
        {early && (
          <Notice tone="info" style={{ marginBottom: '12px' }}>
            The deadline ({fmtDate(round.deadline)}) has not passed yet. Aggregating now closes the round early:
            updates that arrive afterwards are rejected.
          </Notice>
        )}
        <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600, marginBottom: '6px' }}>
          HOSPITALS PARTICIPATING IN THIS ROUND ({round.expected_participants.length})
        </div>
        <div className="table-container" style={{ marginBottom: '12px' }}>
          <table className="fl-table">
            <thead><tr><th>Hospital</th><th>Hospital ID</th><th>Status</th></tr></thead>
            <tbody>
              {round.expected_participants.map((id) => {
                const [label, color] = participantState(round, id);
                return (
                  <tr key={id}>
                    <td>{hospitalNames[id] || '-'}</td>
                    <td className="font-mono" style={{ fontSize: '11px' }}>{id}</td>
                    <td style={{ color, fontSize: '11px' }}>{label}</td>
                  </tr>
                );
              })}
              {round.expected_participants.length === 0 && <tr><td colSpan={3} style={{ color: 'var(--text-muted)' }}>No hospital has joined this round.</td></tr>}
            </tbody>
          </table>
        </div>
        <div style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '14px' }}>
          <strong>{accepted}</strong> accepted update{accepted === 1 ? '' : 's'} (minimum {round.minimum_participants}); only accepted updates are averaged.
          Hospitals still training are left out of this round. Every other active hospital is notified that the round is closed
          and that it should not start training for it.
        </div>
        {!enough && <Notice tone="error" style={{ marginBottom: '14px' }}>Not enough accepted updates to aggregate yet.</Notice>}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
          <button className="btn btn-secondary" onClick={close}>Cancel</button>
          <button className="btn btn-primary" disabled={!enough} onClick={proceed} autoFocus>
            {early ? 'Close round early and aggregate' : 'Aggregate round'}
          </button>
        </div>
      </>
    );

    if (step === 'running') body = (
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '18px 4px', fontSize: '12px', color: 'var(--text-secondary)' }}>
        <Loader2 size={22} className="spin" color="var(--brand-blue)" />
        <div>
          <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '4px' }}>Aggregation in progress</div>
          Round closed to new updates. Averaging {accepted} accepted update{accepted === 1 ? '' : 's'} (FedAvg) into a new global model...
        </div>
      </div>
    );

    if (step === 'done') {
      const gm = state.result.global_model || {};
      const notified = state.result.notified_hospitals || [];
      const { plan, next_round: next, warnings = [] } = state.result;
      body = (
        <>
          <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start', marginBottom: '14px' }}>
            <CheckCircle size={22} color="var(--status-healthy)" />
            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
              <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Aggregation finished</div>
              Global model <span className="font-mono text-primary">{gm.global_model_id}</span> (version {gm.version}) was created
              from {(gm.source_update_ids || []).length} update{(gm.source_update_ids || []).length === 1 ? '' : 's'}.
              <div>Round status: <span className="font-mono">{state.result.round?.status}</span></div>
              <div>{notified.length} non-participating hospital{notified.length === 1 ? ' was' : 's were'} notified{notified.length ? `: ${notified.join(', ')}` : '.'}</div>
            </div>
          </div>
          {plan?.finished && (
            <Notice tone="success" style={{ marginBottom: '12px' }}>The job is complete. {plan.reason}</Notice>
          )}
          {next && (
            <Notice tone="info" style={{ marginBottom: '12px' }}>
              Round {next.round_number} opened automatically from this global model
              {plan?.total_rounds ? ` (round ${next.round_number} of ${plan.total_rounds})` : ''}. It accepts updates until {fmtDate(next.deadline)};
              {' '}{(state.result.notified_next_round || []).length} hospital(s) were notified.
            </Notice>
          )}
          {warnings.map((w) => <Notice key={w} tone="error" style={{ marginBottom: '12px' }}>{w}</Notice>)}
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
            <button className="btn btn-secondary" onClick={close}>Close</button>
            <button className="btn btn-primary" onClick={() => { setState(null); setActiveScreen('models'); }}>View global models</button>
          </div>
        </>
      );
    }

    if (step === 'error') body = (
      <>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start', marginBottom: '14px', fontSize: '12px', color: 'var(--text-secondary)' }}>
          <XCircle size={22} color="var(--status-danger)" />
          <div><div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Aggregation failed</div>{state.error}</div>
        </div>
        <div style={{ display: 'flex', justifyContent: 'flex-end' }}><button className="btn btn-secondary" onClick={close}>Close</button></div>
      </>
    );
  }

  const dialog = (
    <Modal isOpen={!!state} onClose={close} maxWidth="600px"
      title={state ? `Aggregate round ${state.round.round_number} of ${state.job.federation_job_id}` : ''}>
      {body}
    </Modal>
  );
  return [dialog, start];
}
