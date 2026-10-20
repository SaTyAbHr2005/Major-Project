import React, { useEffect } from 'react';
import { useUiStore } from '../../stores/uiStore';
import { useMockStore } from '../../stores/mockStore';
import { useAuthStore, selectRole } from '../../stores/authStore';
import { useFederationStore, currentRound } from '../../stores/federationStore';
import { useRequestsStore } from '../../stores/requestsStore';
import { statusBadge } from './Notice';
import { Shield, BookOpen, Bell, Sliders } from 'lucide-react';

export const TopNav = () => {
  const setActiveScreen = useUiStore((s) => s.setActiveScreen);
  const isAdmin = useAuthStore(selectRole) === 'admin';
  const jobs = useFederationStore((s) => s.jobs.data);
  const loadJobs = useFederationStore((s) => s.loadJobs);
  const dpJobs = (jobs || []).filter((j) => j.dp && !['ARCHIVED', 'COMPLETED'].includes(j.status) && !j.plan?.finished);
  const myRequests = useRequestsStore((s) => s.myRequests.data);
  const loadMyRequests = useRequestsStore((s) => s.loadMyRequests);
  const notificationsCount = useMockStore((s) => s.notificationsCount);

  useEffect(() => { (isAdmin ? loadJobs : loadMyRequests)(); }, [isAdmin, loadJobs, loadMyRequests]);
  const round = isAdmin ? currentRound(jobs) : null;
  const openRequests = myRequests.filter((r) => ['OPEN', 'ACTIVE'].includes(r.status)).length;

  return (
    <header style={{
      height: '50px',
      minHeight: '50px',
      background: 'var(--bg-header)',
      borderBottom: '1px solid var(--border-subtle)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 16px',
      zIndex: 10
    }}>
      {/* Left: current federation round (admin) or open training requests (researcher) */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px' }}>
        {isAdmin ? (
          round ? (
            <>
              <span className={statusBadge(round.status)} title={round.round_id}>
                {round.job.task} · round {round.round_number} · {round.status}
              </span>
              <span className="badge badge-neutral" title="Accepted updates / expected hospitals">
                {round.accepted_participants.length}/{round.expected_participants.length} updates accepted
              </span>
            </>
          ) : <span className="badge badge-neutral">No federation round</span>
        ) : (
          <span className="badge badge-neutral">
            {openRequests} open training request{openRequests === 1 ? '' : 's'}
          </span>
        )}
      </div>

      {/* Right Controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        {/* Differential privacy is chosen per job; the badge reflects the jobs still running. */}
        {dpJobs.length > 0 ? (
          <span className="badge badge-healthy" title={`DP-SGD is on for: ${dpJobs.map((j) => j.federation_job_id).join(', ')}. Updates from these jobs are noise-protected; other jobs' updates are not.`}>
            <Shield size={11} /> Differential privacy: on for {dpJobs.length} job{dpJobs.length === 1 ? '' : 's'}
          </span>
        ) : (
          <span className="badge badge-neutral" title="Differential privacy is set per job when the job is created. No running job uses it, so current updates are sent without DP noise.">
            <Shield size={11} /> Differential privacy: off
          </span>
        )}

        {/* Action Icons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          {!isAdmin && <button
            onClick={() => setActiveScreen('notifications')}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
              padding: '6px',
              borderRadius: '4px',
              position: 'relative'
            }}
            title="Notifications (mock data: Module 18 not implemented)"
          >
            <Bell size={15} />
            {notificationsCount > 0 && (
              <span style={{
                position: 'absolute',
                top: '4px',
                right: '4px',
                width: '6px',
                height: '6px',
                borderRadius: '50%',
                background: 'var(--status-danger)'
              }} />
            )}
          </button>}

          {isAdmin && <button
            onClick={() => setActiveScreen('audit')}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
              padding: '6px',
              borderRadius: '4px'
            }}
            title="Audit log"
          >
            <BookOpen size={15} />
          </button>}

          <button 
            onClick={() => setActiveScreen('settings')}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
              padding: '6px',
              borderRadius: '4px'
            }}
            title="Platform Settings & Preferences"
          >
            <Sliders size={15} />
          </button>
        </div>
      </div>
    </header>
  );
};
