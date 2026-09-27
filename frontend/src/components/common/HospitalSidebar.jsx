import React from 'react';
import { useApp } from '../../context/AppContext';
import { 
  LayoutDashboard, 
  Database, 
  Cpu, 
  PlayCircle, 
  Activity, 
  Award, 
  Box, 
  Search, 
  MessageSquare, 
  Settings, 
  ShieldCheck, 
  LogOut,
  Building2,
  Home
} from 'lucide-react';

export const HospitalSidebar = () => {
  const { activeScreen, setActiveScreen, hospitalId, hospitalName, user, logout } = useApp();

  const navItems = [
    { id: 'dashboard', label: 'Hospital Dashboard', icon: LayoutDashboard, module: 'M3' },
    { id: 'datasets', label: 'Dataset Inspection', icon: Database, module: 'M4' },
    { id: 'preprocessing', label: 'Preprocessing Engine', icon: Cpu, module: 'M5' },
    { id: 'start_training', label: 'Start Local Training', icon: PlayCircle, module: 'M8' },
    { id: 'training_monitor', label: 'Training Monitor', icon: Activity, module: 'M7' },
    { id: 'training_results', label: 'Training Results', icon: Award, module: 'M7/M9' },
    { id: 'models', label: 'Local Models', icon: Box, module: 'M6' },
    { id: 'inference', label: 'Local Inference', icon: Search, module: 'M16' },
    { id: 'communication', label: 'Communications', icon: MessageSquare, module: 'M18' },
    { id: 'settings', label: 'Node Settings', icon: Settings, module: 'M3' }
  ];

  return (
    <aside style={{
      width: '260px',
      height: '100%',
      backgroundColor: 'var(--bg-sidebar)',
      borderRight: '1px solid var(--border-subtle)',
      display: 'flex',
      flexDirection: 'column',
      flexShrink: 0
    }}>
      {/* Hospital Node Brand Header */}
      <div style={{
        padding: '18px 16px',
        borderBottom: '1px solid var(--border-subtle)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '34px',
            height: '34px',
            borderRadius: 'var(--radius-md)',
            background: 'linear-gradient(135deg, #0891b2 0%, #2563eb 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#ffffff',
            fontWeight: 700,
            boxShadow: '0 2px 8px rgba(8, 145, 178, 0.4)'
          }}>
            <Building2 size={20} />
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: '14px', letterSpacing: '-0.02em', color: 'var(--text-primary)' }}>
              MedFL<span style={{ color: 'var(--accent-teal)' }}>.Hospital</span>
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              Clinical Node Workstation
            </div>
          </div>
        </div>

        <button
          onClick={() => setActiveScreen('home')}
          title="Return to Public Homepage"
          style={{
            background: 'var(--bg-nested)',
            border: '1px solid var(--border-subtle)',
            borderRadius: '6px',
            padding: '5px 8px',
            color: 'var(--text-secondary)',
            cursor: 'pointer'
          }}
        >
          <Home size={14} />
        </button>
      </div>

      {/* Hospital Identity Badge */}
      <div style={{
        padding: '12px 16px',
        borderBottom: '1px solid var(--border-subtle)',
        backgroundColor: '#0c1322'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '4px' }}>
          <span style={{ fontSize: '10px', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600 }}>
            Active Clinical Enclave
          </span>
          <span className="badge badge-teal font-mono" style={{ fontSize: '10px' }}>
            {hospitalId}
          </span>
        </div>
        <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {hospitalName}
        </div>
      </div>

      {/* Navigation Links */}
      <nav style={{ flex: 1, padding: '12px 8px', overflowY: 'auto' }}>
        <div style={{ fontSize: '10px', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600, padding: '0 8px 6px 8px' }}>
          Workstation Navigation
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeScreen === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveScreen(item.id)}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '8px 10px',
                marginBottom: '2px',
                borderRadius: 'var(--radius-md)',
                background: isActive ? 'rgba(8, 145, 178, 0.16)' : 'transparent',
                border: isActive ? '1px solid rgba(6, 182, 212, 0.35)' : '1px solid transparent',
                color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                cursor: 'pointer',
                textAlign: 'left',
                transition: 'all 0.15s ease'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <Icon size={16} color={isActive ? 'var(--accent-teal)' : 'var(--text-muted)'} />
                <span style={{ fontSize: '12px', fontWeight: isActive ? 600 : 400 }}>
                  {item.label}
                </span>
              </div>
              <span style={{
                fontSize: '9px',
                fontWeight: 600,
                color: isActive ? 'var(--accent-teal)' : 'var(--text-muted)',
                backgroundColor: isActive ? 'rgba(6, 182, 212, 0.2)' : 'rgba(255, 255, 255, 0.05)',
                padding: '1px 5px',
                borderRadius: '4px'
              }}>
                {item.module}
              </span>
            </button>
          );
        })}
      </nav>

      {/* Privacy Guarantee Footer Card */}
      <div style={{
        padding: '12px',
        margin: '8px',
        backgroundColor: '#0a101d',
        border: '1px solid rgba(16, 185, 129, 0.25)',
        borderRadius: 'var(--radius-md)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
          <ShieldCheck size={14} color="var(--status-healthy)" />
          <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--status-healthy)' }}>
            Zero Raw Egress Locked
          </span>
        </div>
        <p style={{ fontSize: '10px', color: 'var(--text-muted)', lineHeight: 1.3 }}>
          Local DICOM scans stay within hospital boundary. Only differential-privacy gradient updates exit.
        </p>
      </div>

      {/* User Info & Logout */}
      <div style={{
        padding: '12px 14px',
        borderTop: '1px solid var(--border-subtle)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        backgroundColor: '#0b101b'
      }}>
        <div style={{ overflow: 'hidden' }}>
          <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
            {user?.name || 'Hospital Operator'}
          </div>
          <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
            Role: {user?.role || 'hospital_operator'}
          </div>
        </div>
        <button
          onClick={logout}
          title="Sign Out"
          style={{
            background: 'transparent',
            border: 'none',
            color: 'var(--text-muted)',
            cursor: 'pointer',
            padding: '6px',
            borderRadius: 'var(--radius-sm)'
          }}
          onMouseEnter={(e) => e.currentTarget.style.color = 'var(--status-danger)'}
          onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-muted)'}
        >
          <LogOut size={16} />
        </button>
      </div>
    </aside>
  );
};
