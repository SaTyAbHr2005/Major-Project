/**
 * Module 8: Resource-Aware Training (real engine via the local ML service).
 * Hardware is detected on this machine; the three recommendations and every
 * model assessment come straight from `resource_training recommend`.
 */
import ml from './mlClient';

class ResourceService {
  /** `python -m hospital_client.resource_training detect` */
  async evaluateHardware() {
    return ml.get('/resources');
  }

  /** `python -m hospital_client.resource_training recommend --dataset <m5 output>` */
  async getTrainingRecommendations(datasetId, onJob) {
    const job = await ml.post(`/datasets/${datasetId}/recommend`);
    return ml.waitForJob(job, onJob);
  }
}

export const resourceService = new ResourceService();
export default resourceService;
