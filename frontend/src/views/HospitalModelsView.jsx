import React, { useState, useEffect } from 'react';
import { useApp } from '../context/AppContext';
import modelService from '../services/modelService';
import PrivacyNotice from '../components/common/PrivacyNotice';
import { Box, CheckCircle, ShieldCheck, Info, Search } from 'lucide-react';

const keyOf = (m) => `${m.model_id}@v${m.version}`;

const ModelsView = () => {
  const { setActiveTab } = useApp();
  const [models, setModels] = useState([]);
  const [selectedKey, setSelectedKey] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    modelService.getLocalModels()
      .then((list) => {
        setModels(list);
        if (list.length > 0) setSelectedKey(keyOf(list[0]));
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const selectedModel = models.find((m) => keyOf(m) === selectedKey);

  return (
    <div style={{ padding: '20px', overflowY: 'auto', height: '100%' }}>
      <PrivacyNotice />

      <div className="card" style={{ marginBottom: '16px' }}>
        <div className="card-title">
          <Box size={18} color="var(--accent-teal)" />
          <span>Module 6: Local Model Registry</span>
        </div>
        <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
          Every checkpoint saved by local training is a versioned artifact in the Module 6 store (weights as safetensors plus
          metadata.json with a SHA-256 digest that is verified on load). Consortium approval of versions belongs to Module 15 (pending).
        </p>
        {error && <div className="text-danger" style={{ fontSize: '11px', marginTop: '6px' }}>{error}</div>}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(12, 1fr)', gap: '16px' }}>
        <div style={{ gridColumn: 'span 5', display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <div style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600 }}>
            Local Model Versions ({models.length})
          </div>
          {loading && <div className="text-muted" style={{ fontSize: '12px' }}>Loading...</div>}
          {!loading && models.length === 0 && !error && (
            <div className="card text-muted" style={{ fontSize: '12px' }}>No trained models yet. Train one from Start Local Training.</div>
          )}

          {models.map((m) => {
            const isSelected = keyOf(m) === selectedKey;
            return (
              <div key={keyOf(m)} onClick={() => setSelectedKey(keyOf(m))} className="card"
                style={{ cursor: 'pointer', border: isSelected ? '1px solid var(--accent-teal)' : '1px solid var(--border-subtle)',
                  backgroundColor: isSelected ? 'rgba(8, 145, 178, 0.1)' : 'var(--bg-card)', padding: '12px', transition: 'all 0.15s ease' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>{m.model_id}</span>
                  <span className="badge badge-healthy font-mono" style={{ fontSize: '9px' }}><CheckCircle size={10} /> {m.status?.toUpperCase()}</span>
                </div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '6px' }}>
                  Architecture: <strong className="text-primary">{m.architecture}</strong> | Version: <span className="font-mono text-cyan">v{m.version}</span>
                </div>
                <div style={{ display: 'flex', gap: '8px', fontSize: '10px', color: 'var(--text-secondary)' }}>
                  <span>Params: {m.total_parameters ? `${(m.total_parameters / 1e6).toFixed(2)}M` : '-'}</span>
                  <span>•</span>
                  <span>Size: {(m.artifact_size_bytes / 1048576).toFixed(1)} MB</span>
                  <span>•</span>
                  <span>Dataset: {m.dataset_name || '-'}</span>
                </div>
              </div>
            );
          })}
        </div>

        <div style={{ gridColumn: 'span 7' }}>
          {selectedModel ? (
            <div className="card" style={{ height: '100%' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
                <div>
                  <h3 style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)' }}>{selectedModel.model_id}</h3>
                  <div style={{ fontSize: '11px', color: 'var(--accent-teal)', marginTop: '2px' }}>
                    {selectedModel.architecture} - PyTorch {selectedModel.framework_versions?.torch}
                  </div>
                </div>
                <span className="badge badge-teal font-mono">v{selectedModel.version}</span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px', padding: '12px', backgroundColor: '#090d16',
                borderRadius: 'var(--radius-md)', marginBottom: '14px', fontSize: '11px' }}>
                {[
                  ['Input', `${selectedModel.input_spec?.color_mode || ''} ${JSON.stringify(selectedModel.input_spec?.input_size || '')}`],
                  ['Output Classes', `${selectedModel.num_classes} (${selectedModel.task_type || 'classification'})`],
                  ['Parameters', selectedModel.total_parameters?.toLocaleString() ?? '-'],
                  ['Created', new Date(selectedModel.created_at).toLocaleString()],
                  ['Epochs at save', selectedModel.epochs ?? '-'],
                  ['Optimizer / LR / seed', `${selectedModel.optimizer ?? '-'} / ${selectedModel.learning_rate ?? '-'} / ${selectedModel.random_seed ?? '-'}`],
                ].map(([k, v]) => (
                  <div key={k}><span className="text-muted">{k}:</span><div className="font-mono text-primary" style={{ fontWeight: 600 }}>{v}</div></div>
                ))}
              </div>

              <div style={{ marginBottom: '14px' }}>
                <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '6px' }}>Class Index Mapping</div>
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  {Object.entries(selectedModel.class_mapping || {}).map(([label, idx]) => (
                    <div key={label} style={{ padding: '6px 10px', backgroundColor: '#0c1322', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', fontSize: '11px' }}>
                      <span className="font-mono text-muted" style={{ marginRight: '6px' }}>Index {idx}:</span><strong className="text-primary">{label}</strong>
                    </div>
                  ))}
                </div>
              </div>

              <div style={{ padding: '12px', backgroundColor: '#070b14', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', fontSize: '11px', marginBottom: '14px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--status-healthy)', fontWeight: 600, marginBottom: '6px' }}>
                  <ShieldCheck size={14} /> Artifact SHA-256 (verified by Module 6 on every load)
                </div>
                <div className="font-mono text-code" style={{ fontSize: '10px', wordBreak: 'break-all' }}>{selectedModel.artifact_sha256}</div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px', padding: '8px 12px',
                backgroundColor: 'rgba(16, 185, 129, 0.08)', borderRadius: 'var(--radius-md)', fontSize: '11px', color: 'var(--status-healthy)' }}>
                <span><CheckCircle size={14} style={{ verticalAlign: 'middle' }} /> Usable for Module 16 local inference. Module 15 approval: pending.</span>
                <button className="btn btn-teal" style={{ padding: '3px 10px', fontSize: '11px' }} onClick={() => setActiveTab('inference')}>
                  <Search size={12} /> Run inference
                </button>
              </div>
            </div>
          ) : (
            <div className="card" style={{ textAlign: 'center', padding: '40px' }}>
              <Info size={24} color="var(--text-muted)" style={{ margin: '0 auto 8px' }} />
              <div className="text-muted">Select a model version to inspect it.</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ModelsView;
