# HIVE MIND — YOLO11n Training and Results

## 1. Experiment Objective

A YOLO11n object detector was trained to recognize three classes in BeeAlarmed images: `bee`, `varroa`, and `wasp`.

BeeAlarmed was selected as the primary dataset because its visual context is closer to the intended hive-entrance scenario. `VarroaDataset` and `EV2` were not included in this experiment and remain candidates for future comparative experiments.

## 2. Training Environment

- Platform: Google Colab.
- Reported GPU: NVIDIA Tesla T4 with 14.9 GB memory.
- Framework: Ultralytics 8.4.177.
- Initial checkpoint: `yolo11n.pt`.
- Image size: 640 pixels.
- Random seed: 98.

## 3. Training Configuration

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="/content/drive/MyDrive/beealarmed/yolo_v2/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    weight_decay=0.0005,
    degrees=5.0,
    fliplr=0.5,
    close_mosaic=10,
    seed=42,
    project="/content/drive/MyDrive/beealarmed/runs",
    name="yolo11n_v2"
)
```

`epochs=100` is the maximum number of epochs requested. Early stopping may end training sooner when the `patience=20` criterion is met.

## 4. Dataset

| Split | Images |
|---|---:|
| Train | 2,073 |
| Validation | 143 |
| **Total** | **2,216** |

Evaluation was run against the validation split specified in `yolo_v2/data.yaml`. Because the validation set contains 143 images, metrics should be interpreted with the relatively small sample size in mind.

## 5. Validation Results

The following metrics were reported by Ultralytics:

| Class | Images containing class | Instances | Precision (P) | Recall (R) | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|---:|---:|
| All (`all`) | 143 | 183 | 0.984 | 0.988 | 0.985 | 0.752 |
| `bee` | 44 | 44 | 0.995 | 1.000 | 0.995 | 0.841 |
| `varroa` | 39 | 40 | 0.999 | 0.975 | 0.975 | 0.636 |
| `wasp` | 99 | 99 | 0.957 | 0.990 | 0.985 | 0.780 |

### Metric definitions

- **Precision (P):** proportion of model detections that are correct under the evaluation criteria.
- **Recall (R):** proportion of annotated instances that the model detects.
- **mAP@50:** mean Average Precision at an IoU threshold of 0.50.
- **mAP@50–95:** mean Average Precision averaged over IoU thresholds from 0.50 to 0.95, which is stricter about bounding-box localization.

The overall `mAP@50 = 0.985` is high on this validation split. It is not a guarantee of performance on new hives: the validation set is relatively small, and potential duplicates or rotated variants across splits should be ruled out. Independent images are needed for a stronger generalization assessment.

The `varroa` class has a lower `mAP@50–95` (0.636) than `bee` and `wasp`. This indicates that precise localization of Varroa boxes deserves further review. The metric alone does not identify the cause; predictions and annotations should be inspected before drawing conclusions.

## 6. Generated Artifacts

Best checkpoint:

```text
/content/drive/MyDrive/beealarmed/runs/yolo11n_v2/weights/best.pt
```

Dataset configuration:

```text
/content/drive/MyDrive/beealarmed/yolo_v2/data.yaml
```

Evaluation output directory:

```text
/content/drive/MyDrive/beealarmed/evaluation/yolo11n_v2_val
```

Ultralytics output directories may contain confusion matrices, precision/recall curves, prediction samples, and other evaluation artifacts. These paths belong to the original Google Drive environment and must be adapted when reproducing the experiment elsewhere.

## 7. Known Limitations

1. **Validation size:** 143 images and 183 instances.
2. **Model-assisted annotations:** some annotations originated from an earlier model and were confirmed during review. Residual labeling errors may remain.
3. **Potential related images:** BeeAlarmed includes rotated images. Related variants should be checked for leakage between train and validation.
4. **Evaluation domain:** these metrics describe the BeeAlarmed validation split, not performance across all cameras, apiaries, or lighting conditions.
5. **No independent field test reported:** validation metrics do not replace evaluation on newly captured field images.
6. **No multisensor fusion:** this experiment evaluates computer vision only; it does not include validated integration with acoustic, weight, temperature, or humidity sensors.

## 8. Recommended Next Experiments

- Inspect false positives and false negatives for each class.
- Audit Varroa annotations and small or partially visible objects.
- Detect duplicates and rotated variants and create group-aware partitions.
- Compare the current model against a run that adds only `VarroaDataset` to training while keeping BeeAlarmed validation fixed.
- Evaluate EV2 separately before deciding whether it adds value.
- Preserve a separate test set that is not used for model selection or hyperparameter tuning.

## 9. Conclusion

YOLO11n v2 achieved high metrics on the BeeAlarmed validation split, with overall `P=0.984`, `R=0.988`, `mAP@50=0.985`, and `mAP@50–95=0.752`. These results support continuing with the current model as a baseline, while prioritizing validation-leakage checks, visual error analysis, and independent testing before making claims about real-world deployment.
