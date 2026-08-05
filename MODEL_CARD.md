# Model card — Pothole YOLO26n segmentation

## Summary

`models/pothole_yolo26n_seg.onnx` is a single-class pothole instance
segmentation model exported from Ultralytics YOLO26n-seg for CPU inference.
It is intended for research, reproducibility, and portfolio demonstration —
not safety-critical road assessment.

| Property | Value |
|---|---|
| Task | Instance segmentation (`pothole`) |
| Input | `1 × 3 × 512 × 512`, RGB after Ultralytics preprocessing |
| Outputs | detections `1 × 37 × 5376`; mask prototypes `1 × 32 × 128 × 128` |
| ONNX | opset 18, static shape, raw one-to-many head |
| Runtime | ONNX Runtime `CPUExecutionProvider` |
| File size | 11,006,894 bytes |
| SHA-256 | `3ab52bdc4b41cc59b4b845b090bcddc2d927870ce272fdcff2dbb473b3a598c5` |
| License | AGPL-3.0; see `THIRD_PARTY_NOTICES.md` |

## Training data

- Pothole-600 training split.
- PothRGBD training subset after geometry and label-quality filtering.

The final model was trained on 1,008 images / 1,092 instances. Dataset files
are not included in this repository.

## Evaluation

The canonical in-domain accuracy receipt is
`artifacts/verify-final/a1/benchmark.json`:

| Metric | Result |
|---|---:|
| Box mAP@0.5 | 86.2% |
| Box mAP@0.5:0.95 | 53.4% |
| Mask mAP@0.5 | 84.6% |
| Mask mAP@0.5:0.95 | 50.8% |

The Pothole-600 testing split was consulted more than once while comparing
exports and checkpoints. It is therefore an evaluation split, not a pristine
unseen test set.

On the independent Mendeley Pothole Videos test split, box mAP@0.5 drops to
49.9% and mask mAP@0.5 to 45.4%. This domain shift is the most important model
limitation.

## Intended use

- Reproducible CPU inference experiments.
- Pothole segmentation research.
- Input to the repository's stereo depth/area proxy pipeline.
- Demonstrating model evaluation and failure analysis.

## Out-of-scope use

- Safety decisions or autonomous vehicle control.
- Calibrated severity classification without field validation.
- Commercial or closed-source deployment without resolving all model and
  dataset licenses.
- Assuming reliable performance under rain, low light, water-filled potholes,
  unusual cameras, or road surfaces not represented by the training data.

## Known limitations

- Large in-domain to cross-domain accuracy drop.
- A single `pothole` class; severity is computed later from heuristic geometry.
- The current stereo benchmark represents only three physical potholes.
- Accuracy was measured with the corresponding PyTorch checkpoint; deployment
  latency and the tracked ONNX file are recorded separately.
- The raw-head ONNX file relies on Ultralytics postprocessing conventions.

## Reproducibility

Validate the tracked file and run a CPU smoke inference with:

```bash
python tools/validate_portfolio.py
python -m demo.model_smoke
```

See `BENCHMARK.md` and `REPORT.md` for protocols, failure cases, and claim
boundaries.
