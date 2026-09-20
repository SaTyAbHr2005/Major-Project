# Module 16: Local Inference

## 1. Purpose

Module 16 runs an approved model on a **new local medical image, entirely on the hospital
machine**, and returns a structured `InferenceResult`. The raw image is read locally, preprocessed
locally, passed to a local model, and discarded from memory - it is never uploaded, copied,
written, or serialized, and the inference package has no network capability at all.

> **Modules 9-15 are not required for standalone local inference testing. Module 16 provides
> integration interfaces for their future implementation.**

Status: **IMPLEMENTED and TESTED** (182 Module 16 tests, 99% production coverage - see §14).
This is a research/engineering component: outputs are *model predictions*, not clinical
diagnoses (§9).

## 2. Architecture

```
hospital_client/inference/
├── errors.py          # typed errors (messages never contain the image path/filename)
├── provisioning.py    # LocalModelArtifact, ApprovedModelProvider (interface), LocalStoreModelProvider
├── model_spec.py      # adapter over Module 6 ModelMetadata -> validated InferenceModelSpec
├── image_input.py     # read-only validated image loading (reuses Module 4)
├── preprocessing.py   # InferencePreprocessor (reuses Module 7's training transform)
├── device.py          # explicit cpu/cuda (Module 6) or opt-in "auto" (Module 8)
├── result.py          # InferenceResult, ClassificationOutput, disclaimer, format_summary()
├── engine.py          # LocalInferenceEngine - the one generic engine for every architecture
├── service.py         # LocalInferenceService - UI-independent facade for Module 3
├── cli.py, __main__.py
└── tests/             # 182 tests
```

```
   ApprovedModelProvider ──► LocalModelArtifact (Module 6 on-disk layout, local)
                                   │
                                   ▼
        Module 6  ModelStore.load_checkpoint()   ← SHA-256 integrity + strict architecture/key/shape
                                   │                 validation, safetensors only (no pickle)
                                   ▼
        InferenceModelSpec  (validated adapter over metadata; class mapping, input spec, status, task)
                                   │
   local image ─► validate (Module 4 checks) ─► preprocess (Module 7 transform) ─► tensor
                                   │
                                   ▼
        model.eval() + torch.inference_mode() ─► logits ─► softmax ─► stored class_mapping ─► InferenceResult
```

Nothing is duplicated from earlier modules: **no second model registry, architecture
implementation, checkpoint format, or preprocessing implementation.**

## 3. Local-only inference flow

1. `LocalInferenceEngine(source, config)` provisions and validates the model once (§6, §5).
2. `engine.predict(image_path, reference=None)`:
   - validates the image (§8), preprocesses it (§7), runs one forward pass under
     `torch.inference_mode()`, applies softmax, maps the winning index through the model's stored
     `class_mapping`, and returns an `InferenceResult`.
3. Image-level problems (missing/corrupt/unsupported image, incompatible tensor) return a
   `FAILED` result with an `error_category` - no exception. Model-level problems (bad artifact,
   incompatible metadata, unavailable device, invalid configuration) raise typed errors at load time.

The engine reads the image and the model directory and writes **nothing** (verified by a
filesystem-snapshot test). There are no temp files, caches, or copies.

## 4. Supported models

All eight Module 6 architectures run through the same engine (`vit_b16`, `efficientnet_b4`,
`efficientnet_b0`, `resnet50`, `resnet18`, `mobilenet_v2`, `mobilenet_v3_small`, `mobilevit_xxs`).
The architecture only decides which Module 6 factory `build_model()` calls - there is no
per-architecture inference code. `tests/test_all_architectures.py` runs a real Module 6 artifact
for each of the eight through the engine.

## 5. Approved model concept

Module 16 will not run an arbitrary model. `build_model_spec()` validates the metadata Module 6
records, and rejects on any problem:

| Field | Source | If missing/inconsistent |
|---|---|---|
| model_id, version, architecture | Module 6 metadata (must agree with stored config) | rejected |
| artifact hash / integrity | Module 6 SHA-256 check on every load | rejected |
| input_size, color_mode | `input_spec` (must agree with stored config) | rejected (never guessed) |
| num_classes | metadata (must agree with config and weights) | rejected |
| class_mapping | metadata; must be a complete unique `0..n-1` label<->index map | rejected (never assumed) |
| artifact format | `safetensors` | fixed by Module 6 |
| local status | `trained`/`available` by default (`draft`/`deprecated` rejected) | rejected |
| task | **not recorded by Modules 6/7** | reported as *inferred* `image_classification`, with the reason |
| preprocessing_reference | optional in Module 6, normally unset | reported as `unavailable` + warning (or rejected with `require_preprocessing_reference=True`) |

Module 6's `status` is a **local lifecycle label, not platform approval** (Module 15 owns that).
Every result therefore carries `approval.platform_approved = False` for locally provisioned models.

## 6. Model provisioning boundary

`ApprovedModelProvider` (alias `ApprovedModelSource`) is the interface future modules implement;
`LocalModelArtifact` is what it hands over (a reference to an artifact in Module 6's own layout,
plus an opaque `provenance` dict). The only implementation today is `LocalStoreModelProvider`
(a local Module 6 store, with an **explicit** version - "latest" is never assumed). It can also be
built from a Module 7 `TrainingResult` via `from_training_result(result, output_dir)`.

Nothing here downloads, authenticates, or approves anything. There is no fake network fetch and no
invented backend API, and Module 16 makes no claim that model provisioning is secure (Module 11).

## 7. Preprocessing compatibility

Inference must see what training saw. The reused code is `hospital_client.training.dataset.
build_transform` - Module 7's actual training/evaluation transform: **convert to the model's
color mode -> torchvision `Resize(input_size)` -> `ToTensor()`; no normalization, no augmentation.**

- **Verified equal, not assumed:** tests compare Module 16's tensor with the tensor Module 7's real
  `ManifestImageDataset` produces, and (for Module 5 materialized data) with the output of Module 5's
  real `BaseTransformer` fed through Module 7.
- **Module 5 contract:** if a Module 5 `ImageConfig` is supplied, Module 6's own
  `check_compatibility()` is applied (input size, color mode). A mismatch is a hard error.
- **Normalization is never invented or ignored:** Module 5's `ImageConfig.normalization` is not
  implemented by Module 5 or Module 7, so a non-null value is rejected.
- **Recorded preprocessing spec (Module 7 -> Module 16):** Module 7 now stores a versioned
  `preprocessing_spec` (pipeline, color mode, input size, resize method, interpolation, normalization
  `null`, value range, channel order, dtype) and `task_type` in the model's Module 6 metadata at training
  time. Module 16 checks that spec against the pipeline it actually runs and **refuses the model on any
  mismatch** (different input size/color mode, any normalization, non-bilinear interpolation, unknown spec
  version) instead of feeding it differently-prepared input. Models saved before this change have no spec:
  they still load, with an explicit warning that the pipeline is not verified against a stored spec.
- **Materialized-mode fidelity (automatic):** Module 5's materialization resizes with LANCZOS and saves with
  Pillow's default encoder (lossy for JPEG/WEBP), whereas lazy-mode training resizes with bilinear. Module 5
  now records `image_preprocessing` (output mode, color mode, target size, resize method) in every manifest;
  Module 7 copies it into the model's `preprocessing_spec["upstream"]`. With the default
  `emulate_module5_materialization=None`, Module 16 follows that record: `materialized` -> it reproduces the
  color conversion, LANCZOS resize and JPEG/WEBP re-encode in memory (no caller-supplied `ImageConfig` needed);
  `lazy` or nothing recorded -> no emulation. `True`/`False` still override explicitly (`True` without a
  recorded upstream needs an `ImageConfig`). Verified `torch.equal` against the real Module 5 -> Module 7
  pipeline, including a real materialized training run. The result reports `module5_emulation_source`.
- **Unsafe conversions fail:** high-bit-depth/float images (`I`, `I;16`, `F`) are rejected because
  conversion would silently clip them; other conversions (e.g. RGBA/palette/gray -> model mode) are
  recorded in the result's `preprocessing.conversions`.

The result records the pipeline, input size, color mode, normalization, interpolation, source
format/mode/size, conversions, tensor shape, `preprocessing_spec_version`/`preprocessing_spec_verified`, and
`preprocessing_reference` (or `unavailable`).

## 8. Input validation and security

The image file is read **once** into memory; every check and the decode use those bytes, so the file cannot
change between validation and use. `.dcm`/`.dicom` files are rejected with an explicit message.

Image: readable, non-zero-byte (Module 4's `check_file_readability`), extension in Module 4's
`SUPPORTED_EXTENSIONS` (JPG/JPEG, PNG, BMP, TIF/TIFF, WEBP), decodable (`verify()` + full decode, so
truncated files fail), real content format allowed (a GIF renamed `.png` is rejected), safe source
mode, byte and pixel limits (decompression-bomb warnings are errors), directory/NUL-byte paths
rejected. The image is never modified; error messages never contain its path or filename.

Model: loaded only through Module 6, which validates the SHA-256, architecture, key set, and tensor
shapes with `strict=True` and reads **safetensors only** - no arbitrary Python object is deserialized.
Model ids/versions are validated by Module 6 (path traversal rejected); a missing store directory is
an error and is never created. Mismatched architecture/checkpoint/class-mapping/preprocessing are
each rejected with a specific error.

## 9. Inference result, confidence, and wording

`InferenceResult`: `status`, `model_id`, `model_version`, `architecture`, `task` (+ `task_source`),
`output`, `device` (+ `device_note`), `inference_duration_seconds`, `total_duration_seconds`,
`peak_memory_mb`, `preprocessing`, `model_info` (hash, integrity, approval), `reference`, `warnings`,
`errors`, `error_category`, `extension_metadata`, `environment` (torch/CUDA versions, device name, precision,
preprocessing spec version), `timestamp`. `reference` must be a short opaque token (`[A-Za-z0-9._:-]`, 1-128
chars); paths, whitespace and control characters raise `InferenceConfigError`. Model-store paths are scrubbed
from provisioning errors.

The output is task-specific: `output` is a `TaskOutput` subclass (`ClassificationOutput` today:
`predicted_index`, `predicted_label`, `confidence`, `class_probabilities`, `output_transform`,
`confidence_kind` = "uncalibrated softmax probability").
A future task adds its own subclass and a handler in `engine._TASK_HANDLERS`; `InferenceResult` does
not change.

**Confidence semantics.** Module 7 trains with `CrossEntropyLoss`, so raw outputs are logits and are
**never assumed to be probabilities**: softmax is applied and validated as a distribution.
`confidence` is the softmax probability of the predicted class - a model output, uncalibrated, and
**not clinical certainty**. Class labels come only from the stored `class_mapping` (never "0 =
benign" by assumption); a reversed mapping is tested against direct logits.

**Wording.** Results/summaries use "predicted class", "model output", "model confidence" and always
include: *"This result is a model prediction for the configured medical imaging task and is not a
clinical diagnosis."* A test scans user-facing text for unsupported claims (definitive diagnosis,
"disease-free", "replaces a doctor", "clinically accurate", ...).

## 10. Device handling

- `device="cpu"` / `"cuda"`: validated with Module 6's `resolve_device`. **Requesting CUDA when it is
  unavailable raises `DeviceUnavailableError`** - there is no silent fallback.
- `device="auto"` (opt-in): delegates to Module 8's existing `select_device` (network probe disabled)
  and records Module 8's reason in `device_note`, so a CPU choice is never hidden. No second resource
  evaluator exists in Module 16. Module 8 is imported lazily. If Module 8 picks CUDA, Module 16 additionally
  checks free VRAM against a conservative heuristic (3x the weight file; not a measurement). When that check
  fails, or placing the model hits CUDA OOM, it falls back to CPU **explicitly**: the reason is appended to
  `device_note` and `result.device` is `cpu`. Explicit `device="cuda"` never falls back.
- Inference is lightweight: `eval()`, `torch.inference_mode()`, parameters `requires_grad=False`, no
  optimizer, no DataLoader, fp32. On CUDA the result records peak allocated memory; on CPU no peak is
  claimed. The forward pass runs with deterministic cuDNN/algorithm settings (`InferenceConfig.deterministic`,
  default on; process-wide flags are restored afterwards) for same-machine repeatability. One prediction runs
  at a time per engine (an internal lock), so the engine can be shared across threads. A CUDA OOM during a prediction becomes a `FAILED` result (`out_of_memory`); other
  `RuntimeError`s propagate. `engine.close()` / `service.unload()` release the model (and CUDA cache).

## 11. Privacy behavior

- The inference package imports no network or central-service library (AST test over its sources:
  `socket`, `http`, `urllib`, `requests`, `httpx`, `flwr`, `pymongo`, `subprocess`, ...).
- With `socket`/`urllib`/`http.client` primitives booby-trapped, engine, service, `auto` device
  selection, and CLI all still work (offline inference).
- The image and model directories are unchanged after inference (bytes and mtimes), and nothing new
  is created. Results contain no pixels, base64, path, or filename; logs contain only model id,
  version, device, duration, status, and error category.
- `reference` is an opaque caller-supplied string echoed back; **the caller must not put PHI in it.**

## 12. Service and CLI

`LocalInferenceService` (for the Hospital Desktop, Module 3, or any caller) owns the engine
lifecycle: `load_model()` (a failed reload keeps the previous model), `predict()` (serialized by a
lock), `model_info()`, `unload()`. It contains no UI code.

```
python -m hospital_client.inference predict --models-root <store> --model-id <id> --version <N> \
    --image <local image> [--device cpu|cuda|auto] [--reference REF] [--json]
python -m hospital_client.inference validate-model --models-root <store> --model-id <id> --version <N>
```

For a Module 7 run, `--models-root` is `<training output dir>/models`. The CLI never prints the image
path and exits non-zero on failure.

## 13. Integration status

**IMPLEMENTED NOW**

| Module | Integration |
|---|---|
| 4 | `SUPPORTED_EXTENSIONS`, `check_file_readability` |
| 5 | `ImageConfig` compatibility via Module 6's `check_compatibility`; exact emulation of the materialized-output transform (verified against the real `BaseTransformer`), driven automatically by the new `image_preprocessing` manifest metadata. **One additive change:** Module 5 writes that metadata into each manifest |
| 6 | `ModelStore.load_checkpoint` (integrity + strict validation), `ModelMetadata`, `resolve_device`. **One additive change:** two optional `ModelMetadata` fields, `task_type` and `preprocessing_spec` (old metadata files still load) |
| 7 | Reuses `build_transform`. **One additive change:** `Trainer` now records `task_type` and `preprocessing_spec` (`training.dataset.build_preprocessing_spec`) in each saved checkpoint's metadata; training behaviour is unchanged. Provisions Module 7's Module 6 artifacts from a `TrainingResult`; real train -> infer test; loaded weights verified identical to the `FederationHandoff` parameters |
| 8 | `select_device` / `build_resource_profile` for opt-in `auto` device selection |

**FUTURE INTEGRATION INTERFACE** (nothing below is implemented or claimed)

| Module | Where it plugs in |
|---|---|
| 9 Federated Learning | A global model must first become a Module 6 artifact; it then arrives via a provider. A `FederationHandoff` is **not** an approved model source. |
| 10 Differential Privacy | `InferenceConfig.extension_metadata` -> `InferenceResult.extension_metadata` (passed through untouched); no DP logic in inference. |
| 11 Secure Communication | An `ApprovedModelProvider` that fetches over an authenticated channel and materializes the artifact locally. |
| 14 Database/Artifact Storage | Provider backend for artifacts; consumer of `InferenceResult.to_dict()` (persistence is deliberately outside Module 16). |
| 15 Model Versioning | Decides which version is approved; supplies `platform_approved`/`provenance` through `LocalModelArtifact`. Module 16 already requires an explicit version and preserves it in results. |

Modules 12 and 13 have no Module 16 integration point today.

## 14. Testing (actually executed)

```
python -m pytest hospital_client/inference/tests/ -q         -> 212 passed, 1 skipped; 99% coverage incl. test files
python -m pytest hospital_client/ -q                            -> 720 passed, 1 skipped, 0 failed (Modules 4-8: 508 + Module 16: 212)
```

Covers: model loading (valid, missing, malformed metadata, tampered weights, architecture/checkpoint/
class-count mismatches, path traversal, unusable status); the adapter (class-mapping and required-metadata
rejection, explicit `unavailable` fields); image validation (all formats, corrupt/zero-byte/unsupported/
disguised/oversized/high-bit-depth images, no modification, no path in errors); preprocessing (equality
with real Module 7 and Module 5 outputs); inference (probability distribution, class-mapping order,
determinism, no gradients via a forward hook, weights unchanged, version preserved, all 8 architectures);
device handling (CUDA-unavailable error, `auto` via Module 8, real CUDA parity and OOM handling);
privacy (no network imports, offline operation, filesystem unchanged, no path/pixels/base64 in
results or logs); service, CLI, wording; the limitation fixes (recorded task/spec accepted, automatic materialized-mode emulation from recorded upstream, deterministic flags restored, concurrent predictions, mismatching spec rejected, JPEG/WEBP re-encode equality with real
Module 5, single-read image loading, DICOM message, reference/path sanitization, auto-device CPU fallback);
and real **Module 4 -> 5 -> 7 -> 16** integration. CUDA tests
skip cleanly on machines without CUDA.

## 15. Limitations

1. **Scope: image classification.** Module 16 supports the project's task of classifying medical images
   into disease categories. Localisation tasks (object detection, segmentation) are intentionally out of
   scope; adding one later would also need box/mask labels and Module 6/7 support.
2. **Models saved before the metadata changes** carry no `preprocessing_spec`/`task_type`, or a spec without
   `upstream`. They still run, with explicit warnings, and are not verified against a stored spec; if they
   were trained on Module 5 materialized images the caller must pass the `ImageConfig` with
   `emulate_module5_materialization=True`. Module 5's `normalization` option is unimplemented, so any non-null
   value is rejected. The bilinear (lazy) vs LANCZOS (materialized) resize difference between Module 5's
   two output modes is pre-existing in Modules 5/7 and unchanged.
3. **Integrity is not authenticity.** Module 6's SHA-256 is stored beside the weights, so it detects
   corruption/partial modification but not a replacement of both files. There is no signature or trusted
   origin, and local status is not platform approval (results state `platform_approved: false`).
   Needs Modules 11 (authenticated delivery), 14 (artifact storage) and 15 (approval).
4. **16-bit/DICOM medical images are not supported.** DICOM and high-bit-depth TIFFs are rejected instead of
   being windowed/scaled, because no model-consistent conversion exists in the training pipeline. Multi-frame
   files use the first frame. One image per call; no batch API.
5. **Confidence is uncalibrated softmax.** It can be over- or under-confident, there is no
   out-of-distribution or uncertainty detection, and the model returns a prediction for *any* image,
   including a wrong modality or non-medical image. Calibration/OOD belong to Module 20.
6. **Not validated on real medical data or trained models.** Tests use synthetic images and randomly
   initialized or 1-epoch models: they establish software correctness only - no accuracy, clinical validity,
   or performance benchmark is claimed. Module 20's scope.
7. **Reproducibility and resource estimates have limits.** Repeated runs are identical only on the same
   machine; CPU/CUDA results agree only within tolerance (tested at 1e-2) and are not promised to match across
   hardware or PyTorch/CUDA versions. PyTorch runs with `warn_only`, so an operation with no deterministic
   kernel warns rather than fails. Peak memory is reported for CUDA only; inference is fp32 only. The `auto`
   device's VRAM check is a heuristic (3x the weight file vs free VRAM at load time), so a prediction can still
   hit CUDA OOM later (reported as a `FAILED` result).
8. **Dependent modules do not exist yet.** Modules 9-15 are not implemented, so there is no federated-model
   provisioning, DP metadata, secure delivery, approval workflow, version governance, persistence, audit, or
   telemetry (Module 16 stores nothing and only exposes the interfaces in §13). Module 3 is not in this
   repository, so desktop integration is untested end to end.
