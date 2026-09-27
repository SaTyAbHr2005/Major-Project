import React from 'react';
import { useApp } from '../../context/AppContext';
import { useML } from '../../context/MLContext';
import { 
  ShieldCheck, 
  Cpu, 
  Bell, 
  Server
} from 'lucide-react';

export const HospitalTopNav = () => {
  const {
    activeScreen,
    hospitalId,
    notifications,
    setActiveScreen
  } = useApp();
  const { hardware, trainingJob } = useML();

  const isTraining = trainingJob?.status === 'running';
  const gpu = hardware?.gpu;
  const gb = (mb) => `${(mb / 1024).toFixed(1)} GB`;
  const unreadCount = (notifications || []).filter(n => !n.read).length;

  const tabLabels = {
    dashboard: 'Hospital Workstation Dashboard',
    datasets: 'Module 4: Dataset Ingestion & Inspection',
    preprocessing: 'Module 5: Automated Preprocessing Engine',
    start_training: 'Module 8: Resource-Aware Training Recommendation & Configuration',
    training_monitor: 'Module 7: Local Training Execution Monitor',
    training_results: 'Module 7 / 9: Training Result & Federation Handoff',
    models: 'Module 6: Approved Local Model Registry',
    inference: 'Module 16: Zero-Leakage Local Inference',
    communication: 'Module 18: Consortium Communications',
    settings: 'Hospital Node Workstation Settings'
  };

  return (
    <header style={{
      height: '56px',
      backgroundColor: 'var(--bg-header)',
      borderBottom: '1px solid var(--border-subtle)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 20px',
      gap: '16px',
      flexShrink: 0
    }}>
      {/* Left: View Title */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>
          {tabLabels[activeScreen] || 'Hospital Workstation'}
        </div>
        <span className="badge badge-teal font-mono" style={{ fontSize: '11px' }}>
          {hospitalId}
        </span>
      </div>

      {/* Right: Hardware Telemetry Bar & Privacy Lock */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        {/* Hardware Status Strip */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          padding: '4px 12px',
          background: '#0a101d',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-md)',
          fontSize: '11px'
        }}>
          {/* CUDA GPU */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Server size={13} color="var(--accent-teal)" />
            <span style={{ color: 'var(--text-secondary)' }}>GPU:</span>
            <span className="font-mono text-cyan" style={{ fontWeight: 600 }}>
              {hardware ? (gpu?.cuda_available ? gpu.gpu_name.replace('NVIDIA GeForce ', '') : 'CPU only') : '-'}
            </span>
            {gpu?.cuda_available && (
              <span className="badge badge-healthy" style={{ padding: '0 4px', fontSize: '9px' }}>
                CUDA
              </span>
            )}
          </div>

          <span style={{ color: 'var(--border-strong)' }}>|</span>

          {/* VRAM Gauge */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Cpu size={13} color="#818cf8" />
            <span style={{ color: 'var(--text-secondary)' }}>VRAM:</span>
            <span className="font-mono" style={{ color: isTraining ? 'var(--status-warning)' : 'var(--text-primary)', fontWeight: 600 }}>
              {gpu?.cuda_available ? `${gb(gpu.free_vram_mb)} free / ${gb(gpu.total_vram_mb)}` : '-'}
            </span>
          </div>

          <span style={{ color: 'var(--border-strong)' }}>|</span>

          {/* Node State */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span className={`pulse-dot ${isTraining ? 'warning' : 'healthy'}`} />
            <span style={{ fontWeight: 600, color: isTraining ? 'var(--status-warning)' : 'var(--status-healthy)' }}>
              {isTraining ? 'TRAINING ACTIVE' : 'NODE IDLE'}
            </span>
          </div>
        </div>

        {/* Zero Raw Egress Lock Indicator */}
        <div 
          title="Privacy Guard Active: Raw patient scans never leave this machine. Only encrypted gradients/weights are shared."
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '4px 10px',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            border: '1px solid rgba(16, 185, 129, 0.3)',
            borderRadius: 'var(--radius-md)',
            fontSize: '11px',
            color: 'var(--status-healthy)',
            fontWeight: 600
          }}
        >
          <ShieldCheck size={14} />
          <span>Zero-Raw-Data Lock</span>
        </div>

        {/* Notifications Icon */}
        <button
          onClick={() => setActiveScreen('communication')}
          title="Consortium Notifications & Messages"
          style={{
            position: 'relative',
            background: '#131b2e',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
            padding: '6px 8px',
            cursor: 'pointer',
            color: 'var(--text-secondary)',
            display: 'flex',
            alignItems: 'center'
          }}
        >
          <Bell size={15} />
          {unreadCount > 0 && (
            <span style={{
              position: 'absolute',
              top: '-4px',
              right: '-4px',
              width: '15px',
              height: '15px',
              borderRadius: '50%',
              backgroundColor: 'var(--status-danger)',
              color: '#ffffff',
              fontSize: '9px',
              fontWeight: 700,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              {unreadCount}
            </span>
          )}
        </button>
      </div>
    </header>
  );
};
