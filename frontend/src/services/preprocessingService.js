/**
 * Module 5: Automated Preprocessing Engine (real engine via the local ML service).
 * Source files are never modified; outputs go to the local ML work directory.
 */
import ml from './mlClient';

class PreprocessingService {
  getDefaultConfig() {
    // The CLI defaults: lazy manifests (no dataset copy), stop on an invalid existing split.
    return { mode: 'lazy', invalid_split_policy: 'error', group_id_map_path: '', target_column: '' };
  }

  /** Runs `python -m hospital_client.preprocessing preprocess ...`; resolves to the dataset summary. */
  async runPreprocessing(datasetId, config, onJob) {
    if (!datasetId) throw new Error('Inspect a dataset first.');
    const body = { mode: config.mode, invalid_split_policy: config.invalid_split_policy };
    if (config.group_id_map_path) body.group_id_map_path = config.group_id_map_path;
    if (config.target_column) body.target_column = config.target_column;
    const job = await ml.post(`/datasets/${datasetId}/preprocess`, body);
    return ml.waitForJob(job, onJob);
  }
}

export const preprocessingService = new PreprocessingService();
export default preprocessingService;
