# HIVE MIND — Dataset Preparation and Annotation

## 1. Purpose

This document describes the preparation and annotation workflow for the image dataset used to train HIVE MIND's bee and threat detector.

**BeeAlarmed / BeeDataset** is the primary source because its images more closely resemble the intended visual context: bees around a hive entrance with natural backgrounds.

## 2. Annotation Classes

Three classes were defined for object detection using bounding boxes:

| YOLO class ID | Class | General annotation criterion |
|---:|---|---|
| 0 | `bee` | A visible bee. |
| 1 | `varroa` | A visible and localizable Varroa mite. |
| 2 | `wasp` | A wasp or related intruder included in the project's current scope. Individual species are not distinguished. |

Boxes should enclose the object represented by the class. For Varroa, the box should surround the mite rather than the entire bee. Blurred, partially visible, or occluded objects require consistent annotation and review criteria.

## 3. Tooling and Workflow

**CVAT** was used to manage and review annotations. To reduce manual drawing, the workflow combined manually created boxes with predictions from an earlier model.

### Steps performed

1. **Inspect BeeAlarmed.** Images and available labels were reviewed to identify examples relevant to the target classes.
2. **Create initial manual annotations.** Bounding boxes were drawn in CVAT for training and validation examples. Approximately 500 boxes had been manually annotated during the initial stage; this is a historical progress figure, not the final total number of boxes.
3. **Generate preliminary labels.** The checkpoint `beearmaled_yolo11n_v1_best.pt` was used to predict objects in 1,503 images from the prediction image directory.
4. **Prediction parameters.** The inference run used `conf=0.25`, `iou=0.45`, and `imgsz=640`.
5. **Review in CVAT.** Predictions were imported as annotations and reviewed. In the reported review, the imported annotations were confirmed without edits.
6. **Export and organize.** Images and labels were organized in YOLO format for training.

## 4. Technical Validation of the Preliminary Labels

The 1,503-image prediction batch was checked for basic file and label consistency. The reported checks found:

- 1,503 images;
- 1,503 corresponding `.txt` label files;
- no missing or syntactically invalid label files in the automated check;
- 2,572 bounding boxes in total.

The reported class distribution for that batch was:

| Class | Number of boxes |
|---|---:|
| `bee` | 1,075 |
| `varroa` | 1,050 |
| `wasp` | 447 |
| **Total** | **2,572** |

These counts describe the reviewed prediction batch. They are not the total number of objects across the entire BeeAlarmed dataset, and they do not guarantee that every box is semantically correct. Automated checks verify file structure and syntax; visual inspection is needed to assess annotation quality.

## 5. YOLO Label Format

Each image has a `.txt` file with the same base filename. Each line represents one object:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Coordinates are normalized to the image dimensions and fall between 0 and 1. The values represent the class ID, bounding-box center, width, and height.

The `data.yaml` file defines dataset paths and class names. Class IDs must remain consistent across all splits and during inference.

## 6. `yolo_v2` Dataset Layout

```text
yolo_v2/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

Reported image counts:

| Split | Images |
|---|---:|
| Train | 2,073 |
| Validation | 143 |
| **Total** | **2,216** |

The validation split was preserved from the manually annotated data, while the training split incorporated images with reviewed predictions. BeeAlarmed includes rotated images to increase variety. Before interpreting validation metrics as an independent estimate, check that rotated or near-duplicate versions of the same original image have not been split across training and validation. This leakage audit is a recommended follow-up task.

## 7. Limitations and Quality Controls

- Model-generated labels can reproduce the model's errors. Human review helps, but does not eliminate this risk.
- Valid file counts do not replace visual review of bounding boxes.
- `VarroaDataset` and `EV2` were not included in the reported experiment. Their backgrounds, lighting, and annotation formats may differ from the target scenario.
- Before adding other sources, verify class meanings, coordinate formats, and annotation compatibility.
- Images from validation or test splits should not be used to create labels that are subsequently added to training.

## 8. Recommended Next Steps

1. Manually audit a random sample from each class.
2. Review false positives, false negatives, incomplete boxes, and small objects.
3. Identify exact duplicates and rotated variants before finalizing a group-aware split.
4. Version the dataset and record annotation changes.
5. Document dataset licensing and attribution before redistributing BeeAlarmed or any additional source.

This document describes the workflow and checks reported so far. It does not claim that every annotation has been individually reviewed by a human.
