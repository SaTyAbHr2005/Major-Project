import React from 'react';
import { useApp } from '../context/AppContext';
import { useML } from '../context/MLContext';
import TrainingMonitor from '../components/training/TrainingMonitor';
import PrivacyNotice from '../components/common/PrivacyNotice';
import { Activity, Award } from 'lucide-react';

const TrainingMonitorView = () => {
  const { setActiveTab } = useApp();
  const { trainingJob, stopTraining, setLastRunId } = useML();

  const handleViewResults = () => {
    if (trainingJob?.result?.model_id) setLastRunId(trainingJob.result.model_id);
    setActiveTab('training_results');
  };

  return (
    <div style={{ padding: '20px', overflowY: 'auto', height: '100%' }}>
      <PrivacyNotice />

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
