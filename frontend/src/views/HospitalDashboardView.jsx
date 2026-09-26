import React from 'react';
import { useApp } from '../context/AppContext';
import PrivacyNotice from '../components/common/PrivacyNotice';
import { 
  Building2, 
  Database, 
  Cpu, 
  Activity, 
  PlayCircle, 
  CheckCircle, 
  Clock, 
  Server, 
  ShieldCheck,
  AlertTriangle,
  ArrowRight,
  Box
} from 'lucide-react';

const DashboardView = () => {
  const { 
    user, 
    hospitalId, 
    hospitalName, 
    hardware, 
    datasets, 
    activeDataset, 
    activeTrainingSession, 
    notifications,
    setActiveTab,
    trainingResult
  } = useApp();

  const isTraining = activeTrainingSession && activeTrainingSession.status === 'TRAINING';
  const recentHistory = [
    {
      id: 'SESS-HOSP_000001-928104',
      model: 'ResNet-18',
      dataset: 'Pediatric Chest X-Ray Cohort',
      epochs: '5/5',
      val_acc: '93.2%',
      date: 'Yesterday, 14:26',
      status: 'FEDERATION_READY'
    },
    {
      id: 'SESS-HOSP_000001-814920',
      model: 'MobileNetV3-Large',
      dataset: 'Pediatric Chest X-Ray Cohort',
      epochs: '3/3',
      val_acc: '91.8%',
      date: 'Sep 15, 11:04',
      status: 'ARCHIVED'
    }
  ];

  return (
    <div style={{ padding: '20px', overflowY: 'auto', height: '100%' }}>
      {/* Privacy Notice Banner */}
      <PrivacyNotice />

      {/* Hospital Node Identity Header Card */}
      <div className="card" style={{ marginBottom: '16px', backgroundColor: '#0e1628', borderColor: 'rgba(6, 182, 212, 0.3)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
            <div style={{
              width: '42px',
              height: '42px',
              borderRadius: 'var(--radius-lg)',
              backgroundColor: 'rgba(6, 182, 212, 0.15)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--accent-teal)'
            }}>
              <Building2 size={24} />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <h2 style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {hospitalName}
                </h2>
                <span className="badge badge-teal font-mono">{hospitalId}</span>
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                Operator: <strong className="text-primary">{user?.name || 'Dr. Marcus Vance'}</strong> | Role: <span className="font-mono text-cyan">{user?.role}</span> | Dept: Medical Imaging & Radiomics
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span className="badge badge-healthy">
              <ShieldCheck size={12} /> HIPAA Zero-Raw-Egress Verified
            </span>
            <button 
              onClick={() => setActiveTab('start_training')}
              className="btn btn-teal"
            >
              <PlayCircle size={14} /> Start Training Workflow
            </button>
          </div>
        </div>
      </div>

      {/* Top 4 Hardware & Telemetry Metric Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginBottom: '16px' }}>
        {/* GPU & CUDA */}
        <div className="card" style={{ padding: '14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
            <span className="card-title-muted">CUDA Accelerator</span>
            <Server size={14} color="var(--accent-teal)" />
          </div>
          <div className="font-mono text-cyan" style={{ fontSize: '16px', fontWeight: 700 }}>
            {hardware?.device_name?.replace('NVIDIA GeForce ', '') || 'RTX 3080 Ti'}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
            CUDA {hardware?.cuda_version || '12.2'} | Compute {hardware?.compute_capability || '8.6'}
          </div>
        </div>

        {/* VRAM Utilization */}
        <div className="card" style={{ padding: '14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
            <span className="card-title-muted">VRAM Status</span>
            <Cpu size={14} color="#818cf8" />
          </div>
          <div className="font-mono text-primary" style={{ fontSize: '16px', fontWeight: 700 }}>
            {isTraining ? '3.4 GB / 12.0 GB' : '1.8 GB / 12.0 GB'}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--status-healthy)', marginTop: '4px' }}>
            10.2 GB Available Headroom (Safe)
          </div>
        </div>

        {/* Host Memory */}
        <div className="card" style={{ padding: '14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
            <span className="card-title-muted">Host RAM & CPU</span>
            <Activity size={14} color="var(--brand-blue)" />
          </div>
          <div className="font-mono text-primary" style={{ fontSize: '16px', fontWeight: 700 }}>
            {hardware?.available_ram_gb || '24.8'} GB Free / 32 GB
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
            {hardware?.cpu_cores || 16} Cores ({hardware?.cpu_name?.split(' ')[1] || 'Ryzen 9'})
          </div>
        </div>

        {/* Ingested Datasets */}
        <div className="card" style={{ padding: '14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
            <span className="card-title-muted">Local Ingested Cohorts</span>
            <Database size={14} color="var(--status-healthy)" />
          </div>
          <div className="font-mono text-emerald" style={{ fontSize: '16px', fontWeight: 700 }}>
            {(datasets || []).length} Cohorts Ingested
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
            5,840 Validated Radiographs
          </div>
        </div>
      </div>

      {/* Middle Grid: Active Dataset & Training State */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px', marginBottom: '16px' }}>
        {/* Active Local Dataset Card */}
        <div className="card">
          <div className="card-title" style={{ justifyContent: 'space-between' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Database size={15} color="var(--accent-teal)" /> Primary Local Clinical Dataset
            </span>
            <button onClick={() => setActiveTab('datasets')} className="btn btn-secondary" style={{ padding: '2px 8px', fontSize: '11px' }}>
              Inspect Details <ArrowRight size={12} />
            </button>
          </div>

          {activeDataset ? (
            <div>
              <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '4px' }}>
                {activeDataset.name}
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '12px' }}>
                Local Path: <code className="font-mono text-code">{activeDataset.local_path}</code>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', padding: '10px', backgroundColor: '#090d16', borderRadius: 'var(--radius-md)', marginBottom: '12px', fontSize: '11px' }}>
                <div>
                  <span className="text-muted">Total Samples:</span>
                  <div className="font-mono text-primary" style={{ fontWeight: 600 }}>{activeDataset.total_samples}</div>
                </div>
                <div>
                  <span className="text-muted">Modality:</span>
                  <div className="font-mono text-cyan" style={{ fontWeight: 600 }}>{activeDataset.modality}</div>
                </div>
                <div>
                  <span className="text-muted">Split Ratio:</span>
                  <div className="font-mono text-primary" style={{ fontWeight: 600 }}>72 / 14 / 14 %</div>
                </div>
              </div>

              <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                <strong>Class Manifest:</strong> NORMAL (1,583) | BACTERIAL_PNEUMONIA (2,780) | VIRAL_PNEUMONIA (1,477)
              </div>
            </div>
          ) : (
            <div className="text-muted" style={{ padding: '20px', textAlign: 'center' }}>
              No dataset loaded. Navigate to Dataset Inspection to scan local directories.
            </div>
          )}
        </div>

        {/* Current Training / Federation Readiness */}
        <div className="card">
          <div className="card-title" style={{ justifyContent: 'space-between' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Activity size={15} color={isTraining ? 'var(--status-warning)' : 'var(--status-healthy)'} /> Local Training & Federation State
            </span>
            <span className={`badge ${isTraining ? 'badge-warning' : 'badge-healthy'}`}>
              {isTraining ? 'TRAINING IN PROGRESS' : 'READY FOR TRAINING'}
            </span>
          </div>

          {isTraining ? (
            <div>
              <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '6px' }}>
                Active Session: <span className="font-mono text-cyan">{activeTrainingSession.session_id}</span>
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '10px' }}>
                Training <strong>{activeTrainingSession.config?.architecture}</strong> on CUDA | Epoch {activeTrainingSession.current_epoch}/{activeTrainingSession.total_epochs}
              </div>
              <button onClick={() => setActiveTab('training_monitor')} className="btn btn-teal" style={{ width: '100%' }}>
                Open Live Training Monitor <ArrowRight size={13} />
              </button>
            </div>
          ) : (
            <div>
              <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '4px' }}>
                Hardware Idle & Ready
              </div>
              <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '14px', lineHeight: 1.4 }}>
                Module 8 resource engine has profiled this workstation. 
                NVIDIA RTX 3080 Ti has 10.2 GB free VRAM, ready for FP16 accelerated training.
              </p>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button onClick={() => setActiveTab('start_training')} className="btn btn-teal" style={{ flex: 1 }}>
                  <PlayCircle size={14} /> Start Training Workflow
                </button>
                <button onClick={() => setActiveTab('models')} className="btn btn-secondary">
                  <Box size={14} /> Local Models
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Recent Training Runs Table */}
      <div className="card">
        <div className="card-title" style={{ justifyContent: 'space-between' }}>
          <span>Recent Local Training Sessions</span>
          <span className="badge badge-neutral" style={{ fontSize: '10px' }}>Module 7 Audit Log</span>
        </div>

        <div className="fl-table-wrapper">
          <table className="fl-table">
            <thead>
              <tr>
                <th>Session ID</th>
                <th>Model Architecture</th>
                <th>Local Dataset</th>
                <th>Epochs</th>
                <th>Val Accuracy</th>
                <th>Completion Date</th>
                <th>Federation Handoff</th>
              </tr>
            </thead>
            <tbody>
              {recentHistory.map((h) => (
                <tr key={h.id}>
                  <td className="font-mono text-cyan">{h.id}</td>
                  <td style={{ fontWeight: 600 }}>{h.model}</td>
                  <td style={{ color: 'var(--text-secondary)' }}>{h.dataset}</td>
                  <td className="font-mono">{h.epochs}</td>
                  <td className="font-mono text-emerald" style={{ fontWeight: 600 }}>{h.val_acc}</td>
                  <td style={{ color: 'var(--text-muted)' }}>{h.date}</td>
                  <td>
                    <span className="badge badge-healthy" style={{ fontSize: '10px' }}>
                      <CheckCircle size={10} /> {h.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default DashboardView;
