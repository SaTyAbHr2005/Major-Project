import React, { useEffect, useState } from 'react';
import inferenceService from '../services/inferenceService';
import modelService from '../services/modelService';
import PrivacyNotice from '../components/common/PrivacyNotice';
import { Search, CheckCircle, ShieldCheck, Activity, Upload } from 'lucide-react';

const selectStyle = {
  width: '100%', padding: '8px 12px', backgroundColor: '#090d16', border: '1px solid var(--border-subtle)',
  borderRadius: 'var(--radius-md)', color: 'var(--text-primary)', fontSize: '12px', outline: 'none', marginBottom: '10px'
};

const InferenceView = () => {
  const [models, setModels] = useState([]);
  const [selectedKey, setSelectedKey] = useState('');
  const [device, setDevice] = useState('auto');
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isInferring, setIsInferring] = useState(false);
  const [inferenceResult, setInferenceResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    modelService.getLocalModels()
      .then((list) => {
        setModels(list);
        if (list.length) setSelectedKey(`${list[0].model_id}@${list[0].version}`);
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);

  const selected = models.find((m) => `${m.model_id}@${m.version}` === selectedKey);

  const handleFile = (e) => {
    const f = e.target.files?.[0] || null;
    setFile(f);
    setInferenceResult(null);
    setPreview(f ? URL.createObjectURL(f) : null); // browser-local preview only
  };

  const handleRunInference = async () => {
    setError(null);
    setIsInferring(true);
    try {
      setInferenceResult(await inferenceService.runLocalInference(selected?.model_id, selected?.version, file, device));
    } catch (err) {
      setError(err.message || 'Local inference execution failed.');
    } finally {
      setIsInferring(false);
    }
  };

  const out = inferenceResult?.output;
  const probs = Object.entries(out?.class_probabilities || {}).sort((a, b) => b[1] - a[1]);

  return (
    <div style={{ padding: '20px', overflowY: 'auto', height: '100%' }}>
      <PrivacyNotice />

      <div className="card" style={{ marginBottom: '16px' }}>
        <div className="card-title">
          <Search size={18} color="var(--accent-teal)" />
          <span>Module 16: Local Inference</span>
        </div>
        <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
          Run a locally trained model on a new image. The image goes only to the local ML service on this workstation,
          is processed with the model's own preprocessing spec, and its temporary copy is deleted immediately after the prediction.
          <strong> Outputs are model predictions for research use, not clinical diagnoses.</strong>
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px', marginBottom: '16px' }}>
        <div className="card">
          <div className="card-title-muted" style={{ marginBottom: '8px' }}>1. Select Local Model Version (Module 6)</div>
          <select value={selectedKey} onChange={(e) => setSelectedKey(e.target.value)} style={selectStyle}>
            {models.length === 0 && <option value="">No trained models yet</option>}
            {models.map((m) => (
              <option key={`${m.model_id}@${m.version}`} value={`${m.model_id}@${m.version}`}>
                {m.model_id} (v{m.version}) - {m.architecture}
              </option>
            ))}
          </select>
          <select value={device} onChange={(e) => setDevice(e.target.value)} style={selectStyle}>
            <option value="auto">Device: auto (CUDA if available)</option>
            <option value="cuda">Device: CUDA</option>
            <option value="cpu">Device: CPU</option>
          </select>
          {selected && (
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              Classes: <strong>{Object.keys(selected.class_mapping || {}).join(', ')}</strong>
            </div>
          )}
        </div>

        <div className="card">
          <div className="card-title-muted" style={{ marginBottom: '8px' }}>2. Select Local Image</div>
          <label className="btn btn-secondary" style={{ display: 'inline-flex', marginBottom: '10px', cursor: 'pointer' }}>
            <Upload size={14} /> {file ? file.name : 'Choose image...'}
            <input type="file" accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff,.webp" onChange={handleFile} style={{ display: 'none' }} />
          </label>
          {preview && <img src={preview} alt="Selected local image" style={{ maxHeight: '140px', maxWidth: '100%', display: 'block', borderRadius: 'var(--radius-sm)', marginBottom: '8px' }} />}
          <div style={{ fontSize: '11px', color: 'var(--status-healthy)', display: 'flex', alignItems: 'center', gap: '5px' }}>
            <ShieldCheck size={13} />
            <span>Sent only to 127.0.0.1 (this workstation); never to the backend or federated server.</span>
          </div>
        </div>
      </div>

      <div className="card" style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px' }}>
          <div>
            <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>Execute Local Inference</div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>python -m hospital_client.inference predict --json</div>
          </div>
          <button onClick={handleRunInference} disabled={isInferring || !selected || !file} className="btn btn-teal" style={{ padding: '8px 20px', fontSize: '12px' }}>
            <Activity size={14} />
            {isInferring ? 'Running Model...' : 'Run Local Inference'}
          </button>
        </div>
        {error && (
          <div style={{ marginTop: '10px', padding: '8px 12px', backgroundColor: 'var(--status-danger-bg)', border: '1px solid var(--status-danger-border)',
            borderRadius: 'var(--radius-md)', color: 'var(--status-danger)', fontSize: '11px' }}>{error}</div>
        )}
      </div>

      {inferenceResult && (
        <div className="card" style={{ borderColor: inferenceResult.status === 'COMPLETED' ? 'rgba(6, 182, 212, 0.4)' : 'var(--status-danger-border)' }}>
          <div className="card-title" style={{ justifyContent: 'space-between' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <CheckCircle size={16} color="var(--accent-teal)" />
              <span>Module 16 Inference Result ({inferenceResult.status})</span>
            </span>
            <span className="badge badge-teal font-mono">
              {(inferenceResult.inference_duration_seconds * 1000).toFixed(1)} ms on {inferenceResult.device}
            </span>
          </div>

          {out ? (
            <>
              <div style={{ padding: '14px', backgroundColor: '#0a1628', border: '1px solid rgba(6, 182, 212, 0.3)', borderRadius: 'var(--radius-md)',
                marginBottom: '16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div>
                  <div style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600 }}>Predicted Class (model output)</div>
                  <div style={{ fontSize: '18px', fontWeight: 700, color: 'var(--text-primary)', marginTop: '2px' }}>{out.predicted_label}</div>
                  <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                    Model: <code className="font-mono text-cyan">{inferenceResult.model_id} v{inferenceResult.model_version} ({inferenceResult.architecture})</code>
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600 }}>Model Confidence</div>
                  <div className="font-mono text-emerald" style={{ fontSize: '24px', fontWeight: 700 }}>{(out.confidence * 100).toFixed(1)}%</div>
                  <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{out.confidence_kind}</div>
                </div>
              </div>

              <div style={{ marginBottom: '16px' }}>
                <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '8px' }}>Class Probability Distribution</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {probs.map(([label, p]) => (
                    <div key={label}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: '3px' }}>
                        <span style={{ fontWeight: 600 }}>{label}</span>
                        <span className="font-mono text-primary">{(p * 100).toFixed(2)}%</span>
                      </div>
                      <div style={{ height: '6px', backgroundColor: '#090d16', borderRadius: '3px', overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: `${p * 100}%`, backgroundColor: label === out.predicted_label ? 'var(--accent-teal)' : 'var(--brand-blue)' }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div className="text-danger" style={{ fontSize: '12px', marginBottom: '12px' }}>
              {inferenceResult.error_category}: {(inferenceResult.errors || []).join('; ')}
            </div>
          )}

          {(inferenceResult.warnings || []).map((w, i) => (
            <div key={i} style={{ fontSize: '11px', color: 'var(--status-warning)', marginBottom: '6px' }}>{w}</div>
          ))}

          <div style={{ padding: '10px 12px', backgroundColor: '#0a101d', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)',
            fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.4 }}>
            <strong>Notice:</strong> {inferenceResult.disclaimer}
          </div>
        </div>
      )}
    </div>
  );
};

export default InferenceView;
