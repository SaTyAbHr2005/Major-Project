import React, { useEffect, useState } from 'react';
import { useApp } from '../context/AppContext';
import { useML } from '../context/MLContext';
import apiService from '../services/apiService';
import RecommendationCard from '../components/training/RecommendationCard';
import ManualTrainingPanel from '../components/training/ManualTrainingPanel';
import TrainingConfirmModal from '../components/training/TrainingConfirmModal';
import PrivacyNotice from '../components/common/PrivacyNotice';
import PipelineStatus from '../components/training/PipelineStatus';
import { PlayCircle, RefreshCw, Link2 } from 'lucide-react';

const gb = (mb) => (mb == null ? '-' : `${(mb / 1024).toFixed(1)} GB`);

const StartTrainingView = () => {
  const { setActiveTab, hospitalId } = useApp();
  const {
    activeDataset, hardware, hardwareError, refreshHardware, pipeline, rerunRecommendations,
    selection, setSelection, startTraining, trainingJob, linkedRequestId, setLinkedRequestId
  } = useML();

  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const [launchError, setLaunchError] = useState(null);
  const [requests, setRequests] = useState([]);
  const [requestsError, setRequestsError] = useState(null);
  const [joining, setJoining] = useState(false);

  const recSet = activeDataset?.recommendations;
  const recs = recSet?.recommendations || [];
  const alternatives = recSet?.alternatives || [];
  const selected = selection?.config || null;
  const pick = (rec, alt) => setSelection(alt
    ? { key: `alt:${rec.model}`, architecture: rec.model, config: rec }
    : { key: rec.recommendation_type, choice: rec.recommendation_type, config: rec });
  const trainingRunning = trainingJob?.status === 'running';

  useEffect(() => {
    apiService.getAllTrainingRequests()
      .then((list) => setRequests(list.filter((r) => ['OPEN', 'ACTIVE'].includes(r.status))))
      .catch((e) => setRequestsError(e.message));
  }, []);

  const linked = requests.find((r) => r.request_id === linkedRequestId);
  const participation = linked?.participating_hospitals?.find((p) => p.hospital_id === hospitalId);
  const isParticipant = participation && participation.status !== 'WITHDRAWN';

  const handleJoin = async () => {
    setJoining(true);
    try {
      const { training_request } = await apiService.participateInTrainingRequest(linkedRequestId);
      setRequests((list) => list.map((r) => (r.request_id === training_request.request_id ? training_request : r)));
    } catch (e) {
      setRequestsError(e.message);
    } finally {
      setJoining(false);
    }
  };

  const handleConfirmLaunch = async () => {
    setShowConfirmModal(false);
    setLaunchError(null);
    try {
      await startTraining();
      setActiveTab('training_monitor');
    } catch (err) {
      setLaunchError(err.message || 'Error launching local training.');
    }
  };

  return (
    <div style={{ padding: '20px', overflowY: 'auto', height: '100%' }}>
      <PrivacyNotice />

      {/* Header Card */}
      <div className="card" style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <PlayCircle size={18} color="var(--accent-teal)" />
              <h2 style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)' }}>
                Start Training: Module 8 Resource-Aware Orchestration
              </h2>
            </div>
            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
              Module 8 measured this workstation (CPU, RAM, GPU/VRAM, storage) and the prepared dataset, then generated up to three
              safe configurations. Pick one - it becomes the Module 7 TrainingConfig that runs locally.
            </p>
          </div>

          <button onClick={() => { refreshHardware(); rerunRecommendations(); }} disabled={pipeline.running || !activeDataset?.preprocessing}
            className="btn btn-secondary" style={{ padding: '6px 12px' }}>
            <RefreshCw size={13} className={pipeline.running ? 'spin' : ''} />
            {pipeline.running ? 'Re-evaluating...' : 'Re-profile Hardware (M8)'}
          </button>
        </div>
      </div>

      <PipelineStatus />

      {/* Hardware & Dataset Context Strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', marginBottom: '16px' }}>
        <div className="card" style={{ padding: '12px' }}>
          <div className="card-title-muted">Hardware Profile (measured)</div>
          <div className="font-mono text-cyan" style={{ fontSize: '13px', fontWeight: 600, marginTop: '2px' }}>
            {hardware ? (hardware.gpu?.cuda_available ? hardware.gpu.gpu_name : 'CPU only (no CUDA device)') : hardwareError || 'Detecting...'}
          </div>
          {hardware && (
            <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px' }}>
              VRAM free {gb(hardware.gpu?.free_vram_mb)} / {gb(hardware.gpu?.total_vram_mb)} | RAM free {gb(hardware.ram?.available_mb)} | {hardware.cpu?.logical_cores} CPU threads
            </div>
          )}
        </div>

        <div className="card" style={{ padding: '12px' }}>
          <div className="card-title-muted">Active Dataset</div>
          <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)', marginTop: '2px' }}>
            {activeDataset?.name || 'None - locate a dataset first'}
          </div>
          <div style={{ fontSize: '10px', color: 'var(--status-healthy)', marginTop: '2px' }}>
            {activeDataset?.preprocessing
              ? `train ${activeDataset.preprocessing.splits.train} / val ${activeDataset.preprocessing.splits.validation ?? 0} / test ${activeDataset.preprocessing.splits.test ?? 0}`
              : 'Not preprocessed yet'}
          </div>
        </div>

        <div className="card" style={{ padding: '12px' }}>
          <div className="card-title-muted">EfficientNet-B0 Status (Module 8)</div>
          <div className="font-mono text-purple" style={{ fontSize: '13px', fontWeight: 600, marginTop: '2px' }}>
            {recSet?.all_model_assessments?.efficientnet_b0?.status_label || '-'}
          </div>
          <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px' }}>
            {recSet?.all_model_assessments?.efficientnet_b0?.reason}
          </div>
        </div>
      </div>

      {/* Three Training Recommendations */}
      <div style={{ marginBottom: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
          <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)' }}>
            Module 8 Resource-Aware Training Configurations (Choose One)
          </div>
          <span className="badge badge-neutral" style={{ fontSize: '10px' }}>Generated for this machine</span>
        </div>

        {recs.length ? (
          <div style={{ display: 'grid', gridTemplateColumns: `repeat(${recs.length}, 1fr)`, gap: '16px' }}>
            {recs.map((rec) => {
              const alt = alternatives.find((a) => a.recommendation_type === rec.recommendation_type);
              return (
                <div key={rec.recommendation_type} style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <RecommendationCard recommendation={rec} isSelected={selection?.key === rec.recommendation_type}
                    onSelect={(r) => pick(r, false)} />
                  {alt && (
                    <RecommendationCard recommendation={alt} isAlternative isSelected={selection?.key === `alt:${alt.model}`}
                      onSelect={(r) => pick(r, true)} />
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          <div className="card" style={{ textAlign: 'center', padding: '30px' }}>
            <div className="text-muted">
              {pipeline.running ? 'Evaluating hardware & dataset...' : 'No recommendations yet - locate a dataset in Dataset Inspection.'}
            </div>
          </div>
        )}
        {(recSet?.notes || []).map((n, i) => (
          <div key={i} style={{ fontSize: '11px', color: 'var(--status-warning)', marginTop: '8px' }}>{n}</div>
        ))}
      </div>

      {recSet?.all_model_assessments && (
        <ManualTrainingPanel assessments={recSet.all_model_assessments} isSelected={selection?.key === 'manual'}
          onSelect={setSelection} />
      )}

      {/* All model assessments */}
      {recSet?.all_model_assessments && (
        <div className="card" style={{ marginBottom: '20px' }}>
          <div className="card-title"><span>All Architectures Evaluated on This Machine</span></div>
          <div className="fl-table-wrapper">
            <table className="fl-table">
              <thead>
                <tr><th>Model</th><th>Tier</th><th>Params</th><th>Status</th><th>Batch</th><th>Epochs</th><th>Est. Time</th><th>Memory</th></tr>
              </thead>
              <tbody>
                {Object.values(recSet.all_model_assessments).map((a) => (
                  <tr key={a.architecture} title={a.reason}>
                    <td style={{ fontWeight: 600 }}>{a.display_name}</td>
                    <td className="font-mono">{a.resource_tier}</td>
                    <td className="font-mono">{(a.param_count / 1e6).toFixed(2)}M</td>
                    <td><span className={`badge ${a.safe ? 'badge-healthy' : 'badge-danger'}`} style={{ fontSize: '10px' }}>{a.status_label}</span></td>
                    <td className="font-mono">{a.batch_size ?? '-'}</td>
                    <td className="font-mono">{a.epochs ?? '-'}</td>
                    <td className="font-mono">{a.time_estimate?.estimated_training_time_display || '-'}</td>
                    <td className="font-mono">
                      {a.memory_estimate ? `${gb(a.memory_estimate.estimated_total_mb)} ${a.memory_estimate.measured ? '(measured)' : '(estimated)'}` : '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Consortium request link (backend) */}
      <div className="card" style={{ marginBottom: '20px' }}>
        <div className="card-title"><Link2 size={14} color="var(--brand-blue)" /><span>Consortium Training Request (optional)</span></div>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
          <select value={linkedRequestId} onChange={(e) => setLinkedRequestId(e.target.value)}
            style={{ flex: 1, minWidth: '260px', padding: '8px 12px', backgroundColor: '#090d16', border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-md)', color: 'var(--text-primary)', fontSize: '12px' }}>
            <option value="">Local run only (no progress reported)</option>
            {requests.map((r) => (
              <option key={r.request_id} value={r.request_id}>{r.disease} - {r.task} ({r.request_id}, {r.status})</option>
            ))}
          </select>
          {linkedRequestId && !isParticipant && (
            <button className="btn btn-secondary" onClick={handleJoin} disabled={joining}>{joining ? 'Joining...' : 'Participate'}</button>
          )}
          {isParticipant && <span className="badge badge-healthy">PARTICIPATING ({participation.status})</span>}
        </div>
        <div style={{ fontSize: '11px', color: requestsError ? 'var(--status-danger)' : 'var(--text-muted)', marginTop: '6px' }}>
          {requestsError || 'When linked, only aggregate progress (epoch, loss, accuracy) is sent to the backend for the researcher to monitor.'}
        </div>
      </div>

      {/* Launch Bar - only after the operator explicitly picks a configuration */}
      {recs.length > 0 && !selected && (
        <div className="card" style={{ textAlign: 'center', padding: '16px', fontSize: '12px', color: 'var(--text-secondary)' }}>
          Select one of the configurations above (a recommendation, its alternative, or a manual configuration) to continue. Nothing is selected or trained automatically.
        </div>
      )}
      {selected && (
        <div className="card" style={{ backgroundColor: '#0c1626', border: '1px solid rgba(6, 182, 212, 0.4)', padding: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
            <div>
              <div style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--accent-teal)', fontWeight: 600 }}>Ready to Generate Module 7 TrainingConfig</div>
              <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)', marginTop: '2px' }}>
                Selected: {selected.label} ({selected.display_name})
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                Device: <strong className="text-primary">{selected.device}</strong> | Precision: <strong className="text-cyan">{selected.precision}</strong> | Batch Size: {selected.batch_size} | Epochs: {selected.epochs} | Est. Time: <strong className="text-emerald">{selected.estimated_training_time_display}</strong>
              </div>
            </div>
            <button onClick={() => setShowConfirmModal(true)} disabled={trainingRunning || (linkedRequestId && !isParticipant)}
              className="btn btn-teal" style={{ padding: '10px 20px', fontSize: '13px', fontWeight: 600 }}>
              <PlayCircle size={16} /> {trainingRunning ? 'Training In Progress' : 'Proceed to Confirmation & Launch'}
            </button>
          </div>
          {launchError && (
            <div style={{ marginTop: '10px', padding: '8px 12px', backgroundColor: 'var(--status-danger-bg)', border: '1px solid var(--status-danger-border)',
              borderRadius: 'var(--radius-md)', color: 'var(--status-danger)', fontSize: '11px' }}>{launchError}</div>
          )}
        </div>
      )}

      <TrainingConfirmModal isOpen={showConfirmModal} onClose={() => setShowConfirmModal(false)} onConfirm={handleConfirmLaunch}
        recommendation={selected} datasetName={activeDataset?.name} requestId={linkedRequestId} />
    </div>
  );
};

export default StartTrainingView;
