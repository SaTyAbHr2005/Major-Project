/**
 * Module 4: Dataset Ingestion and Inspection (real engine via the local ML service).
 * The dataset is only read on this machine; the UI receives the DatasetProfile summary.
 */
import ml from './mlClient';

class DatasetService {
  /** Opens a native folder picker on this machine; resolves to a path or null. */
  async browseFolder() {
    return (await ml.post('/browse-folder')).path;
  }

  async listDatasets() {
    return (await ml.get('/datasets')).datasets;
  }

  async getDataset(datasetId) {
    return ml.get(`/datasets/${datasetId}`);
  }

  /** Runs `python -m hospital_client.dataset inspect <path>`; resolves to the dataset summary. */
  async inspectDatasetPath(folderPath, onJob) {
    if (!folderPath || !folderPath.trim()) throw new Error('Please locate a dataset folder (or a CSV/XLS/XLSX file).');
    const job = await ml.post('/datasets/inspect', { path: folderPath.trim() });
    return ml.waitForJob(job, onJob);
  }
}

export const datasetService = new DatasetService();
export default datasetService;
