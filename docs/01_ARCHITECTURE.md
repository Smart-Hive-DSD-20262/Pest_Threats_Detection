# HIVE MIND — Pest and Threat Detection Architecture

## 1. Project Overview

HIVE MIND is an academic project focused on intelligent beehive monitoring. This stage develops a computer-vision prototype to detect bees and potential threats in images of a beehive entrance.

The current detector targets three classes:

- `bee`: a bee.
- `varroa`: a *Varroa destructor* mite.
- `wasp`: a wasp or related intruder within the project's current scope. Individual wasp and hornet species are not distinguished.

The detector is an experimental visual component, not a deployed monitoring system or a definitive diagnosis of colony health.

## 2. Logical Architecture

```text
Images of the beehive entrance
              |
              v
     Dataset preparation
       and CVAT labeling
              |
              v
      YOLO-format dataset
       train / val + YAML
              |
              v
   Training in Google Colab
        with available GPU
              |
              v
      Trained YOLO11n model
              |
              v
       Inference on images
              |
              v
 Bounding boxes + class + confidence
```

## 3. Main Components

| Component | Responsibility |
|---|---|
| BeeAlarmed / BeeDataset | Primary image source, selected because its visual context resembles a beehive entrance. |
| CVAT | Annotation management, review, and editing of bounding boxes. |
| YOLO dataset | Stores images and labels in the format required for training and validation. |
| Google Colab | Training and evaluation environment; no local GPU is required. |
| Ultralytics YOLO11n | Compact object detector used as the current baseline. |
| Inference outputs | Predictions containing bounding boxes, classes, and confidence scores. |

## 4. Model and Classes

The experiment starts from the pretrained `yolo11n.pt` checkpoint. YOLO11n predicts a bounding box, class, and confidence score for each detection.

| Class ID | Class name |
|---:|---|
| 0 | `bee` |
| 1 | `varroa` |
| 2 | `wasp` |

Keep this mapping consistent across `data.yaml`, label files, evaluation scripts, and inference code.

## 5. Data and Artifact Layout

The working files were organized in Google Drive under `beealarmed`:

```text
beealarmed/
├── yolo_v1/
│   ├── images/{train,val}/
│   ├── labels/{train,val}/
│   └── predict_v2/{images,labels}/
├── yolo_v2/
│   ├── images/{train,val}/
│   ├── labels/{train,val}/
│   └── data.yaml
├── models/
│   └── beearmaled_yolo11n_v1_best.pt
├── runs/
│   └── yolo11n_v2/weights/best.pt
└── evaluation/
    └── yolo11n_v2_val/
```

This reflects the experiment's working layout, not a requirement to commit all data and checkpoints to GitHub. Large image datasets, model weights, and generated runs should be managed separately where appropriate. Check each dataset's license and redistribution conditions before publishing it.

## 6. Inference Flow

Given a new image, the detector returns a predicted class, bounding box, and confidence score. Confidence scores can be used to filter detections, but should not automatically be interpreted as calibrated probabilities. Thresholds should be selected based on validation and the relative cost of false positives and false negatives.

## 7. Current Scope and Future Work

### Implemented in this stage

- BeeAlarmed images prepared in YOLO format.
- Three classes: `bee`, `varroa`, and `wasp`.
- Manual and model-assisted annotation review using CVAT.
- YOLO11n training and evaluation in Google Colab.
- Experimental image inference.

### Not yet established as complete

- Production or edge-device deployment at a hive.
- Operational integration with acoustic, weight, temperature, or humidity sensors.
- Validated multisensor fusion and automated field alerts.
- Final evaluation at an independent apiary or with a separate camera.

`VarroaDataset` and `EV2` are potential supplementary sources for future experiments. They were not included in the results reported here. Their labels, formats, and effect on BeeAlarmed validation should be checked before integration.

## 8. Technology Stack

- Python
- Ultralytics YOLO11n
- Google Colab
- Google Drive
- CVAT
- YOLO annotation format

## 9. Reproducibility

For reproducible experiments, preserve Python and Ultralytics versions, `data.yaml`, class mapping, training parameters, split definitions, checkpoints where licensing and storage allow, and evaluation outputs. Google Drive paths are specific to the original environment and must be updated when running the project from another account.
