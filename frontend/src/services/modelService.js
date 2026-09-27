/**
 * Module 6: Model Management - lists the versioned artifacts in the local
 * Module 6 store (metadata.json per version; weights never leave the disk).
 */
import ml from './mlClient';

class ModelService {
  async getLocalModels() {
    return (await ml.get('/models')).models;
  }
}

export const modelService = new ModelService();
export default modelService;
