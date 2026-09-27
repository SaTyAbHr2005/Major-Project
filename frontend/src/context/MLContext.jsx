import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';
import ml from '../services/mlClient';
import datasetService from '../services/datasetService';
import preprocessingService from '../services/preprocessingService';
import resourceService from '../services/resourceService';
import trainingService from '../services/trainingService';
import apiService from '../services/apiService';
import { useApp } from './AppContext';

/**
 * Hospital-side ML state, backed by the local ML service. The operator only
 * locates a dataset; inspection (M4), preprocessing (M5) and resource-aware
 * recommendations (M8) then run automatically on this machine. Training
 * (M8 -> M7) starts when the operator picks one of the three recommendations.
 */
const MLContext = createContext(null);
const ACTIVE_DATASET_KEY = 'medfl_active_dataset_id';

const store = {
  get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k, v) => { try { v ? localStorage.setItem(k, v) : localStorage.removeItem(k); } catch { /* storage unavailable */ } },
};

export const MLProvider = ({ children }) => {
  const { isLoggedIn, userRole } = useApp();
  const enabled = isLoggedIn && userRole === 'hospital_operator';

  const [serviceStatus, setServiceStatus] = useState('unknown'); // unknown | online | offline
  const [hardware, setHardware] = useState(null);
  const [hardwareError, setHardwareError] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [activeDataset, setActiveDataset] = useState(null);
  const [pipeline, setPipeline] = useState({ running: false, step: null, job: null, error: null });
  // { key, choice } for a recommendation, { key, architecture[, batch_size, epochs] } for an alternative/manual.
  const [selection, setSelection] = useState(null); // never pre-selected
  const [trainingJob, setTrainingJob] = useState(null);
  const [trainingError, setTrainingError] = useState(null);
  const [lastRunId, setLastRunId] = useState(null);
  const [linkedRequestId, setLinkedRequestId] = useState('');
  const reported = useRef({ epochs: -1, status: null });

  const refreshDatasets = useCallback(async () => {
    try { setDatasets(await datasetService.listDatasets()); } catch { /* service offline: shown via serviceStatus */ }
  }, []);

  const refreshHardware = useCallback(async () => {
    setHardwareError(null);
    try { setHardware(await resourceService.evaluateHardware()); } catch (e) { setHardwareError(e.message); }
  }, []);

  const selectDataset = useCallback(async (datasetId) => {
    const summary = datasetId ? await datasetService.getDataset(datasetId) : null;
    setActiveDataset(summary);
    setSelection(null);
    store.set(ACTIVE_DATASET_KEY, summary?.dataset_id);
    return summary;
  }, []);

  // Connect to the local service after a hospital operator signs in.
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    ml.health()
      .then(async () => {
        if (cancelled) return;
        setServiceStatus('online');
        refreshDatasets();
        refreshHardware();
        const saved = store.get(ACTIVE_DATASET_KEY);
        // A failed restore (e.g. expired session) must not forget the selection.
        if (saved) selectDataset(saved).catch(() => {});
      })
      .catch(() => !cancelled && setServiceStatus('offline'));
    return () => { cancelled = true; };
  }, [enabled, refreshDatasets, refreshHardware, selectDataset]);

  const onJob = (step) => (job) => setPipeline((p) => ({ ...p, step, job }));

  /** Runs the remaining steps for a dataset, starting at `from`. */
  const continuePipeline = useCallback(async (datasetId, from, preprocessConfig) => {
    const steps = ['preprocess', 'recommend'];
    let summary = null;
    for (const step of steps.slice(steps.indexOf(from))) {
      setPipeline((p) => ({ ...p, step, job: null }));
      if (step === 'preprocess') {
        summary = await preprocessingService.runPreprocessing(
          datasetId, preprocessConfig || preprocessingService.getDefaultConfig(), onJob(step));
        setActiveDataset(summary);
      } else {
        const recs = await resourceService.getTrainingRecommendations(datasetId, onJob(step));
        setActiveDataset((d) => (d && d.dataset_id === datasetId ? { ...d, recommendations: recs } : d));
        setSelection(null); // the operator must pick one explicitly
      }
    }
    return summary;
  }, []);

  const runGuarded = useCallback(async (fn) => {
    setPipeline({ running: true, step: null, job: null, error: null });
    try {
      await fn();
      setPipeline({ running: false, step: 'done', job: null, error: null });
    } catch (e) {
      setPipeline((p) => ({ ...p, running: false, error: e.message }));
    } finally {
      refreshDatasets();
    }
  }, [refreshDatasets]);

  /** Locate -> M4 inspect -> M5 preprocess -> M8 recommend, all local. */
  const runPipeline = useCallback((path) => runGuarded(async () => {
    setPipeline((p) => ({ ...p, step: 'inspect' }));
    const summary = await datasetService.inspectDatasetPath(path, onJob('inspect'));
    setActiveDataset(summary);
    store.set(ACTIVE_DATASET_KEY, summary.dataset_id);
    await continuePipeline(summary.dataset_id, 'preprocess');
  }), [runGuarded, continuePipeline]);

  /** Re-run M5 with non-default options (then M8 recommend again). */
  const rerunPreprocessing = useCallback((config) => runGuarded(
    () => continuePipeline(activeDataset.dataset_id, 'preprocess', config)
  ), [runGuarded, continuePipeline, activeDataset]);

  const rerunRecommendations = useCallback(() => runGuarded(
    () => continuePipeline(activeDataset.dataset_id, 'recommend')
  ), [runGuarded, continuePipeline, activeDataset]);

  const startTraining = useCallback(async () => {
    if (!selection) throw new Error('Select a training configuration first.');
    setTrainingError(null);
    reported.current = { epochs: -1, status: null };
    const job = await trainingService.startTraining(activeDataset.dataset_id, selection, linkedRequestId || null);
    setTrainingJob(job);
    return job;
  }, [activeDataset, selection, linkedRequestId]);

  const stopTraining = useCallback(async () => {
    if (trainingJob) setTrainingJob(await trainingService.stopTraining(trainingJob.job_id));
  }, [trainingJob]);

  // Poll the training job (continues while the operator navigates).
  useEffect(() => {
    if (!trainingJob || trainingJob.status !== 'running') return;
    const t = setTimeout(async () => {
      try {
        const next = await trainingService.getJob(trainingJob.job_id);
        setTrainingJob(next);
        if (next.status === 'succeeded') {
          setLastRunId(next.result.model_id);
          refreshHardware();
        }
        if (next.status === 'failed') setTrainingError(next.error);
      } catch (e) {
        setTrainingError(e.message);
      }
    }, 2000);
    return () => clearTimeout(t);
  }, [trainingJob, refreshHardware]);

  // Report aggregate progress (never data) to the consortium request, if linked.
  useEffect(() => {
    const requestId = trainingJob?.meta?.request_id;
    if (!requestId) return;
    const last = trainingJob.epochs[trainingJob.epochs.length - 1];
    const done = trainingJob.status === 'succeeded';
    if (!done && (!last || reported.current.epochs === trainingJob.epochs.length)) return;
    if (done && reported.current.status === 'COMPLETED') return;
    const result = trainingJob.result?.result;
    const body = done
      ? {
          status: 'COMPLETED', current_epoch: result.epochs_run, total_epochs: result.epochs_run,
          loss: result.validation_metrics?.loss ?? null, accuracy: result.validation_metrics?.accuracy ?? null,
        }
      : {
          status: 'TRAINING', current_epoch: last.epoch, total_epochs: last.total_epochs,
          loss: last.val_loss ?? last.train_loss, accuracy: last.val_acc ?? last.train_acc,
        };
    reported.current = { epochs: trainingJob.epochs.length, status: body.status };
    apiService.reportTrainingProgress(requestId, body).catch((e) => console.warn('Progress report failed:', e.message));
  }, [trainingJob]);

  return (
    <MLContext.Provider value={{
      serviceStatus, hardware, hardwareError, refreshHardware,
      datasets, refreshDatasets, activeDataset, selectDataset,
      pipeline, runPipeline, rerunPreprocessing, rerunRecommendations,
      selection, setSelection, linkedRequestId, setLinkedRequestId,
      trainingJob, trainingError, startTraining, stopTraining, lastRunId, setLastRunId,
    }}>
      {children}
    </MLContext.Provider>
  );
};

export const useML = () => useContext(MLContext);
