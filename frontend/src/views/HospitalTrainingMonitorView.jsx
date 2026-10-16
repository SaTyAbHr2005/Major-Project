import React from 'react';
import { useUiStore } from '../stores/uiStore';
import { useMLStore } from '../stores/mlStore';
import { useFederationStore } from '../stores/federationStore';
import { FlowerSteps } from '../components/federation/FlowerSteps';
import TrainingMonitor from '../components/training/TrainingMonitor';
import PrivacyNotice from '../components/common/PrivacyNotice';
import { Activity, Award } from 'lucide-react';

const TrainingMonitorView = () => {
  const setActiveTab = useUiStore((s) => s.setActiveScreen);
  const { trainingJob, stopTraining, setLastRunId } = useMLStore();
  // Flower jobs: which federated step this machine is on (training below is the round's ordinary training job).
  const flowerConns = Object.values(useFederationStore((s) => s.flower)).filter((c) => c.state !== 'disconnected');

  const handleViewResults = () => {
    if (trainingJob?.result?.model_id) setLastRunId(trainingJob.result.model_id);
    setActiveTab('training_results');
  };

  return (
    <div style={{ padding: '20px', overflowY: 'auto', height: '100%' }}>
      <PrivacyNotice />

      {flowerConns.map((c) => (
        <div key={c.job_id} style={{ marginBottom: '16px' }}><FlowerSteps status={c} /></div>
      ))}

      <div className="card" style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Activity size={18} color="var(--accent-teal)" />
              <h2 style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)' }}>Module 7: Local Training Execution Monitor</h2>
            </div>
            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
              Live output of the local PyTorch training process: per-epoch loss and accuracy, and the process log.
            </p>
          </div>
          {trainingJob?.status === 'succeeded' && (
            <button onClick={handleViewResults} className="btn btn-teal"><Award size={14} /> View Final TrainingResult</button>
          )}
        </div>
      </div>

      <TrainingMonitor job={trainingJob} onHalt={stopTraining} onViewResults={handleViewResults} />
    </div>
  );
};

export default TrainingMonitorView;
