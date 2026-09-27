/**
 * Module 16: Local Inference. The selected image is sent only to the local ML
 * service on 127.0.0.1, written to a temporary file, run through
 * `python -m hospital_client.inference predict --json`, and deleted.
 * Outputs are model predictions, not clinical diagnoses.
 */
import ml from './mlClient';

class InferenceService {
  async runLocalInference(modelId, version, file, device = 'auto') {
    if (!modelId || !version) throw new Error('Select a model and an explicit version.');
    if (!file) throw new Error('Select a local image.');
    const ext = (file.name.match(/\.[^.]+$/) || [''])[0].toLowerCase();
    const q = new URLSearchParams({ model_id: modelId, version: String(version), device, ext });
    return ml.upload(`/predict?${q}`, file);
  }
}

export const inferenceService = new InferenceService();
export default inferenceService;
