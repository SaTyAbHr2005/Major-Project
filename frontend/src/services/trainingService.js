/**
 * Module 7 local training, launched through Module 8
 * (`resource_training train --choice <c> --test --plots`) by the local ML service.
 */
import ml from './mlClient';

class TrainingService {
  /**
   * Starts a background training job; returns the job snapshot.
   * selection: { choice } for one of Module 8's three recommendations, or
   * { architecture, batch_size?, epochs? } for a same-tier alternative / manual configuration.
   */
  async startTraining(datasetId, selection, requestId = null) {
    const body = selection.choice
      ? { choice: selection.choice }
      : { architecture: selection.architecture, batch_size: selection.batch_size, epochs: selection.epochs };
    return ml.post(`/datasets/${datasetId}/train`, { ...body, request_id: requestId });
  }

  async getJob(jobId) {
    return ml.get(`/jobs/${jobId}`);
  }

  async stopTraining(jobId) {
    return ml.post(`/jobs/${jobId}/cancel`);
  }

  async getTrainingHistory() {
    return (await ml.get('/training-runs')).runs;
  }

  /** Full result: TrainingResult JSON, resource statistics, curve metrics and plot names. */
  async getRun(modelId) {
    return ml.get(`/training-runs/${encodeURIComponent(modelId)}`);
  }

  async getPlotUrl(modelId, name) {
    return ml.plotUrl(modelId, name);
  }
}

export const trainingService = new TrainingService();
export default trainingService;
