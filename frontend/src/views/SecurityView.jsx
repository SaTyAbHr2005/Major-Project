import React, { useEffect, useState } from 'react';
import { useFederationStore } from '../stores/federationStore';
import { Notice, Empty, fmtDate, statusBadge } from '../components/common/Notice';
import { ShieldCheck, RefreshCw } from 'lucide-react';

const ACCEPTED = ['ACCEPTED', 'USED_IN_AGGREGATION'];

/**
 * Module 9 update validation: every hospital update with the server's verdict (checksum,
 * provenance, architecture/class match, duplicate and NaN/Inf checks) and the rejection reason.
 */
export const SecurityView = () => {
  const { jobDetails, loadJobDetails } = useFederationStore();
  const [filter, setFilter] = useState('all');
  useEffect(() => { loadJobDetails(); }, [loadJobDetails]);

  const updates = jobDetails.data
    .flatMap((job) => (job.rounds || []).flatMap((round) => (round.updates || []).map((u) => ({ ...u, job, round }))))
    .sort((a, b) => String(b.received_at).localeCompare(String(a.received_at)));
  const rejected = updates.filter((u) => ['REJECTED', 'QUARANTINED', 'FAILED', 'EXPIRED'].includes(u.status));
  const shown = filter === 'problems' ? rejected : updates;

  const Stat = ({ label, value, color }) => (
    <div className="card">
      <div style={{ fontSize: '11px', color: 'var(--text-secondary)', fontWeight: 600 }}>{label}</div>
      <div className="font-mono" style={{ fontSize: '22px', fontWeight: 700, color }}>{value}</div>
    </div>
  );

  return (
    <div style={{ padding: '16px 20px', height: '100%', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '14px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 style={{ fontSize: '18px', fontWeight: '700', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <ShieldCheck size={18} color="var(--status-healthy)" /> Update Validation
          </h1>
          <p style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Every model update a hospital submits is checked by the federation server before it can be aggregated.
          </p>
        </div>
        <button className="btn btn-secondary" onClick={loadJobDetails} disabled={jobDetails.status === 'loading'}>
          <RefreshCw size={13} className={jobDetails.status === 'loading' ? 'spin' : ''} /> <span>Refresh</span>
        </button>
      </div>

      {jobDetails.error && <Notice tone="error">{jobDetails.error}</Notice>}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
        <Stat label="UPDATES RECEIVED" value={updates.length} />
        <Stat label="ACCEPTED" value={updates.filter((u) => ACCEPTED.includes(u.status)).length} color="var(--status-healthy)" />
        <Stat label="REJECTED / QUARANTINED" value={updates.filter((u) => ['REJECTED', 'QUARANTINED'].includes(u.status)).length} color="var(--status-danger)" />
        <Stat label="EXPIRED / FAILED" value={updates.filter((u) => ['EXPIRED', 'FAILED'].includes(u.status)).length} color="var(--status-warning)" />
      </div>

      <div className="card">
        <div className="card-header">
          <span className="card-title">Submitted updates</span>
          <div style={{ display: 'flex', gap: '6px' }}>
            {[['all', 'All'], ['problems', `Not accepted (${rejected.length})`]].map(([k, label]) => (
              <button key={k} className={`btn ${filter === k ? 'btn-primary' : 'btn-ghost'}`} style={{ fontSize: '11px', padding: '3px 8px' }} onClick={() => setFilter(k)}>{label}</button>
            ))}
          </div>
        </div>
        {shown.length === 0 ? (
          <Empty>{jobDetails.status === 'loading' ? 'Loading...' : filter === 'problems' ? 'No update has been rejected.' : 'No hospital has submitted an update yet.'}</Empty>
        ) : (
          <div className="table-container">
            <table className="fl-table">
              <thead>
                <tr><th>Received</th><th>Hospital</th><th>Job / round</th><th>Architecture</th><th>Samples</th><th>Checksum</th><th>Status</th><th>Reason</th></tr>
              </thead>
              <tbody>
                {shown.map((u) => (
                  <tr key={u.update_id}>
                    <td style={{ color: 'var(--text-muted)' }}>{fmtDate(u.received_at)}</td>
                    <td className="font-mono">{u.participant_id}</td>
                    <td>{u.job.task} <span style={{ color: 'var(--text-muted)' }}>· round {u.round.round_number}</span></td>
                    <td className="font-mono">{u.architecture}</td>
                    <td className="font-mono">{u.num_train_samples}</td>
                    <td className="font-mono" title={u.artifact_checksum}>{u.artifact_checksum ? `${u.artifact_checksum.slice(0, 10)}...` : '-'}</td>
                    <td><span className={statusBadge(u.status)}>{u.status}{u.security_status && u.security_status !== 'OK' ? ` (${u.security_status})` : ''}</span></td>
                    <td style={{ fontSize: '11px', color: 'var(--text-secondary)', maxWidth: '320px' }}>{u.security_reason || u.rejection_reason || ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-header"><span className="card-title">Security & Privacy Implementation Status</span></div>
        <ul style={{ fontSize: '11px', color: 'var(--text-secondary)', lineHeight: 1.6, paddingLeft: '18px', margin: 0 }}>
          <li><strong>Byzantine / malicious update detection (Module 12):</strong> Active — per-update delta norm bound, cosine direction verification (rounds ≥ 3), sample-count consistency check, and coordinate-wise median / trimmed mean aggregation.</li>
          <li><strong>Differential privacy (Module 10):</strong> Active — DP-SGD with Rényi privacy accounting, gradient clipping, and per-job cumulative budget enforcement.</li>
          <li><strong>Secure communication (Module 11):</strong> Pending — local development setup uses plain HTTP; production requires TLS/HTTPS termination.</li>
        </ul>
      </div>
    </div>
  );
};
