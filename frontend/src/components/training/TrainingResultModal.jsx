import React from 'react';
import Modal from '../common/Modal';
import { ReportText } from '../charts/MLCharts';

/** Full audit view of one run: the raw JSON files Module 7/8 wrote (no raw data inside). */
const TrainingResultModal = ({ isOpen, onClose, run }) => {
  if (!run) return null;
  return (
    <Modal isOpen={isOpen} onClose={onClose} title={`Audit Inspector - ${run.model_id}`} maxWidth="760px">
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <div className="card-title-muted">{run.model_id}_training_result.json (Module 7)</div>
        <ReportText text={JSON.stringify(run.result, null, 2)} />
        {run.resource_statistics && (
          <>
            <div className="card-title-muted">{run.model_id}_resource_statistics.json (Module 8)</div>
            <ReportText text={JSON.stringify(run.resource_statistics, null, 2)} />
          </>
        )}
        {run.curve_metrics && (
          <>
            <div className="card-title-muted">curve_metrics.json (ROC-AUC / average precision)</div>
            <ReportText text={JSON.stringify(run.curve_metrics, null, 2)} />
          </>
        )}
        <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
          * Evaluation metrics are experimental results on this local dataset split, not clinical validation.
        </div>
        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <button onClick={onClose} className="btn btn-teal">Close</button>
        </div>
      </div>
    </Modal>
  );
};

export default TrainingResultModal;
