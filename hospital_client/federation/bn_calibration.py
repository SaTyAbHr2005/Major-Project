"""BatchNorm calibration after federated averaging (shared by the Flower bench and the hospital Flower client)."""


def calibrate_batchnorm(model, loader, max_batches: int) -> int:
    """Re-measure every BatchNorm layer's running mean/variance for the model's current weights.

    Averaging models also averages their BatchNorm statistics, but each client's statistics describe its own
    weights, not the averaged ones; on the malaria data the averaged model then predicted one class for every
    image (48% test accuracy, against 93% with statistics measured for the averaged weights). Forward passes
    only: weights do not change. Returns the number of batches used.
    """
    import torch
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return 0
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative average over all calibration batches
    device = next(model.parameters()).device
    model.train()
    used = 0
    with torch.no_grad():
        for images, _ in loader:
            if used >= max_batches:
                break
            model(images.to(device))
            used += 1
    return used
