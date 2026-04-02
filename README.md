# DVARF: Drone Vehicle Accident Recognition

Experimental evaluation of Meta's Segment Anything Model 3 (SAM 3) for zero-shot object detection in aerial accident scene imagery.

![SAM 3 Example](assets/sam3_example.png)

---

## Overview

This project investigates the application of **SAM 3**, Meta's foundation model for image segmentation, to the domain of accident scene analysis from drone footage. While SAM 3 was designed as a general-purpose segmentation model, this work explores its effectiveness when applied to a specialized detection task using only text-based prompts, without any fine-tuning on the target domain.

The core research question is whether modern vision-language models like SAM 3 can perform reliably in emergency response scenarios where traditional object detectors would require extensive labeled training data. The project addresses this through a systematic experimental pipeline with three main components:

**Zero-shot Detection:** SAM 3 is queried using simple natural language prompts for three object classes relevant to accident scenes: crashed vehicles, persons, and undamaged vehicles. The model outputs segmentation masks which are then converted to bounding box detections in YOLO format, enabling direct comparison with traditional detection methods using standard metrics like mean Average Precision (mAP).

**Segmentation Preservation:** Beyond detection metrics, the project preserves SAM 3's native segmentation outputs as binary masks. This dual-output approach maintains the detailed spatial information that could be valuable for downstream analysis tasks such as damage assessment or scene reconstruction.

**Feature Extraction:** The SAM 3 wrapper has been extended to extract semantic features for each detected object. Using ROI pooling on the model's backbone feature maps, a 256-dimensional query embedding is computed per detection and then conditioned on the text prompt via mean-pooled language embeddings. The embedding is then concatenated with the model's confidence score to produce a 257-dimensional feature vector. These semantic features capture both visual characteristics from the backbone and prompt-specific context, providing rich representations for downstream tasks beyond simple bounding boxes and masks.

**Linear Probing:** To investigate whether SAM 3's performance can be improved with minimal supervision, a lightweight linear classifier is trained on top of the model's outputs. This classifier learns to re-score detections based on semantic features extracted from SAM 3's internal representations: a 257-dimensional vector combining the 256-dimensional query embeddings from backbone feature maps (conditioned on the text prompt) with the model's confidence score. The linear probe provides a low-cost adaptation method that does not require retraining the foundation model itself.

---

## Hardware Requirements

SAM 3 can run on CPU-only systems, but inference is slow, taking approximately 5-10 minutes per image on moderate consumer hardware. The minimum system requirements are 16 GB of RAM and 10 GB of free disk space for storing the model weights. 

For practical experimentation, a GPU is strongly recommended. An NVIDIA GPU with at least 8 GB of VRAM (such as an RTX 3060 or better) provides reasonable inference speeds. The system should have CUDA 11.7 or later installed for GPU acceleration. All experiments reported in this project were conducted on Google Colab using a Tesla T4 GPU, which provides sufficient performance for research purposes at no cost.

---

## Installation

Install dependencies as specified in `requirements.txt`:

```bash
# 1. Install PyTorch (see https://pytorch.org for GPU version)
pip install torch torchvision torchaudio

# 2. Install SAM 3
pip install git+https://github.com/facebookresearch/segment-anything-3.git

# 3. Install project dependencies
pip install -r requirements.txt
```

See `requirements.txt` for detailed installation instructions and platform-specific notes.

---

## Google Colab

For quick experimentation without local setup, two ready-to-use Jupyter notebooks are provided in the `notebooks/` directory. These notebooks run entirely on Google Colab's free GPU instances and handle all installation and data preparation steps automatically.

**Zero-Shot Inference Notebook** ([`test_sam3.ipynb`](notebooks/test_sam3.ipynb)): Demonstrates SAM 3's zero-shot object detection capabilities on the test set. This notebook provides a streamlined workflow for running inference and evaluating results without any training or fine-tuning.

**Linear Probing Notebook** ([`test_sam3_linearProbing.ipynb`](notebooks/test_sam3_linearProbing.ipynb)): Contains the complete experimental pipeline including running SAM 3 on the training split to generate predictions and features, building the linear probe dataset, training the classifier, and evaluating the enhanced predictions on the test set.

To use these notebooks, upload them to Google Colab, ensure GPU runtime is enabled (Runtime → Change runtime type → T4 GPU), and execute cells sequentially. The notebooks will automatically clone this repository, install dependencies, and guide you through the authentication process for Hugging Face model access.

---

## Dataset Format

The project uses the YOLO annotation format, which organizes data into separate directories for images and labels, with each split (train, validation, test) stored independently.

Each annotation file is a plain text file with the same base name as its corresponding image. For an image file `123.jpg`, the annotations are stored in `123.txt`. Each line in the annotation file represents one object instance with five space-separated values: `class_id center_x center_y width height`. The class ID is an integer (0, 1, or 2 for this project), while all spatial coordinates are normalized to the range [0, 1] relative to image dimensions.

The three object classes detected in this work are: Class 0 (crashed vehicles), Class 1 (persons at the accident scene), and Class 2 (undamaged vehicles). These classes were chosen to cover the primary elements relevant for accident scene documentation and analysis.

---

## Usage

### Zero-shot Inference

To run SAM 3 inference on a dataset split, execute the `run_sam3_on_split.py` script. This script processes each image by querying SAM 3 once per class using the corresponding text prompt, then applies class-wise Non-Maximum Suppression (NMS) to reduce duplicate detections. The script generates two types of output: YOLO-format bounding box predictions saved as text files in `data/processed/predictions/sam3_yolo/{split}/`, and binary segmentation masks saved as PNG images in `data/processed/segmentations/sam3/{split}/`.

```bash
python scripts/run_sam3_on_split.py
```

Configuration parameters such as the export confidence threshold, NMS IoU threshold, and whether to save segmentations can be passed as command-line arguments:

```bash
# Run on train split with a custom export threshold
python scripts/run_sam3_on_split.py --split train --export_threshold 0.05

# Save segmentation masks
python scripts/run_sam3_on_split.py --split test --save_segmentations
```

### Evaluation

The evaluation script computes standard object detection metrics by comparing predictions against ground-truth annotations. Metrics include per-class and overall Precision, Recall, F1 score, AP@0.50, and AP@0.50:0.95. Results are printed to the console and can be redirected to a file for record-keeping.

```bash
python scripts/eval_sam3_on_split.py --split test --eval_threshold 0.35 > results/sam3_test_metrics.txt
```

### Linear Probing

The linear probing pipeline consists of four sequential steps. First, features are extracted from SAM 3's predictions on the training set, creating a dataset where each detection is represented by a feature vector containing its confidence score and geometric properties. Second, a per-class logistic regression model is trained to predict whether each detection is a true positive or false positive based on these features. Third, the trained weights are applied to re-score all predictions on the test set. Finally, the re-scored predictions are evaluated using the same metrics as the zero-shot experiment.

```bash
# Build feature dataset from training predictions
python scripts/build_linear_probe_dataset.py

# Train logistic regression classifiers
python scripts/train_linear_probe.py

# Apply learned weights to test predictions
python scripts/apply_linear_probe_to_split.py

# Evaluate performance with linear probe
python scripts/eval_sam3_linear_probe_on_split.py
```

### Visualization

Two visualization scripts are provided for qualitative analysis. The first displays segmentation masks overlaid on the original images with class-specific colors, allowing visual inspection of SAM 3's segmentation quality. The second provides a side-by-side comparison of ground-truth bounding boxes and predicted boxes, making it easy to identify false positives, false negatives, and localization errors.

```bash
# View segmentation masks overlaid on images
python scripts/show_sam3_masks_on_image.py

# Compare ground truth annotations vs predictions
python scripts/show_gt_vs_sam3.py
```

---

## Metrics Methodology

### Class-Specific Average IoU

The **mean IoU** metric measures the average Intersection over Union for correctly matched predictions:

1. **Matching Algorithm**: For each ground truth box, we find the predicted box with the highest IoU. A prediction is considered a **True Positive (TP)** if:
   - The predicted class matches the ground truth class
   - The IoU is ≥ 0.5
   - The ground truth has not already been matched to another prediction

2. **Per-Class IoU**: For each of the three classes (crashed car, person, undamaged car), we calculate the mean IoU over all True Positives for that class. If a class has zero True Positives, its its mean IoU is 0.0.

3. **Overall Mean IoU**: The arithmetic mean of the three per-class IoU values. This provides a single metric that reflects both localization quality and class-specific performance.

This metric is calculated on the **test set** and complements the standard AP@0.50 metric by directly measuring localization quality for correct detections.

### Inference Speed

The **speed per frame** metric measures the average time required to process a single image:

**Components Included**:
- Image loading from disk (I/O)
- SAM 3 model inference (forward pass for each class prompt)
- Post-processing (YOLO format conversion, NMS)
- Linear probe application (for the linear probing variant only)

**Measurement Procedure**:
1. Load all test set image paths
2. Synchronize GPU (if available)
3. Start timer
4. Process each image through the complete pipeline
5. Synchronize GPU (if available)
6. End timer and divide by number of images

**Note**: The timing includes I/O because SAM 3's `predict_with_text()` method loads images internally from file paths. The reported speed represents the complete end-to-end pipeline as it would be deployed in practice.

Speed is measured separately for:
- **SAM 3 zero-shot**: Model inference + post-processing
- **SAM 3 + linear probe**: Model inference + linear classifier + post-processing

Both metrics are measured on the **test set** and saved to the metrics files alongside precision, recall, F1, and AP scores.

---

## Experimental Results

Results on the test set. Two evaluation paradigms are used:

- **Best-F1**: Each model is evaluated at its own confidence threshold, selected on the validation set to maximise micro-F1 (SAM 3 baseline: thr = 0.70, SAM 3 + LP: thr = 0.35).
- **Fair comparison**: Both models are evaluated at the same threshold (thr = 0.35).

### Best-F1 Global Comparison

Each model evaluated at its own optimal threshold.

| Metric | SAM 3 (thr = 0.70) | SAM 3 + LP (thr = 0.35) | Δ |
|--------|--------------------|-------------------------|---|
| Precision (micro-P) | 0.4587 | 0.4788 | +0.0201 |
| Recall (micro-R) | 0.5996 | 0.6695 | +0.0699 |
| F1 (micro-F1) | 0.5197 | 0.5583 | +0.0386 |
| mAP@0.50 | 0.4439 | 0.5605 | +0.1166 |
| mAP@0.50:0.95 | 0.1793 | 0.2655 | +0.0862 |

### Fair Comparison at thr = 0.35

Both models evaluated at the same confidence threshold.

| Metric | SAM 3 | SAM 3 + LP | Δ |
|--------|-------|------------|---|
| Precision (micro-P) | 0.3918 | 0.4788 | +0.0869 |
| Recall (micro-R) | 0.6716 | 0.6695 | −0.0021 |
| F1 (micro-F1) | 0.4949 | 0.5583 | +0.0634 |
| mAP@0.50 | 0.4565 | 0.5605 | +0.1040 |
| mAP@0.50:0.95 | 0.1934 | 0.2655 | +0.0722 |

Predictions on 472 GT boxes: SAM 3 → 809 · SAM 3 + LP → 660 (linear probe filters 149 detections).

### Per-Class Metrics at thr = 0.35

| Class | Metric | SAM 3 | SAM 3 + LP | Δ |
|-------|--------|-------|------------|---|
| **Crashed car** | Precision | 0.6486 | 0.7027 | +0.0541 |
| | Recall | 0.3913 | 0.4239 | +0.0326 |
| | F1 | 0.4881 | 0.5288 | +0.0407 |
| | AP@0.50 | 0.3192 | 0.4179 | +0.0987 |
| | AP@0.50:0.95 | 0.1838 | 0.2307 | +0.0469 |
| **Person** | Precision | 0.4386 | 0.5439 | +0.1053 |
| | Recall | 0.7812 | 0.9688 | +0.1876 |
| | F1 | 0.5618 | 0.6966 | +0.1348 |
| | AP@0.50 | 0.6906 | 0.8918 | +0.2012 |
| | AP@0.50:0.95 | 0.2514 | 0.3520 | +0.1006 |
| **Car** | Precision | 0.3432 | 0.4207 | +0.0775 |
| | Recall | 0.8594 | 0.8086 | −0.0508 |
| | F1 | 0.4905 | 0.5535 | +0.0630 |
| | AP@0.50 | 0.3595 | 0.3717 | +0.0122 |
| | AP@0.50:0.95 | 0.1449 | 0.2138 | +0.0689 |

### Detection Counts at thr = 0.35

| Class | GT | Pred (SAM 3) | Pred (LP) | TP (SAM 3) | TP (LP) | FP (SAM 3) | FP (LP) |
|-------|----|--------------|-----------|------------|---------|------------|---------|
| Crashed car | 184 | 111 | 111 | 72 | 78 | 39 | 33 |
| Person | 32 | 57 | 57 | 25 | 31 | 32 | 26 |
| Car | 256 | 641 | 492 | 220 | 207 | 421 | 285 |

---

## Model Comparison

Comprehensive comparison of SAM 3 with other zero-shot and supervised models on the test set. SAM 3 AP values are reported at thr = 0.35 (fair comparison baseline); mAP@0.5 for SAM 3 + LP refers to the best-F1 evaluation (thr = 0.35, its own optimal threshold):

| Model | Type/Phase | mIoU | Speed (ms/frame) | AP@0.5 crashed | AP@0.5 person | AP@0.5 car | mAP@0.5 |
|-------|------------|------|------------------|----------------|---------------|------------|---------|
| **Moondream 2** | ZSOD | 0.44 | ~1000 | N/A | N/A | N/A | N/A |
| **OMDET TURBO** | ZSOD | 0.72 | ~2000 | N/A | N/A | N/A | N/A |
| **YOLOe** | ZSOD | 0.53 | ~20 | N/A | N/A | N/A | N/A |
| **YOLOe base** | Pre fine-tuning | N/A | ~20 | 0.547 | 0.458 | 0.294 | 0.433 |
| **YOLOe specialized** | Post fine-tuning | N/A | ~20 | 0.911 | 0.760 | 0.803 | 0.825 |
| **SAM 3** | Zero-shot | N/A | 6146 | 0.319 | 0.691 | 0.360 | 0.457 |
| **SAM 3 + LP** | Linear Probe | 0.609 | 6147 | 0.418 | 0.892 | 0.372 | 0.561 |

**Notes**:
- **mIoU**: Mean Intersection over Union on correctly matched predictions (TP with IoU ≥ 0.5), averaged over the three classes. For ZSOD models (Moondream 2, OMDET TURBO, YOLOe), values are reported per text prompt as in the original thesis. For SAM 3 + LP, computed at thr = 0.35. SAM 3 baseline mIoU is not available at its optimal threshold (thr = 0.70 yields no TPs).
- **Speed**: Average inference time per image including I/O, model forward pass, and post-processing. Measured on test set. For other models, speed values are approximate and derived from the thesis.
- **AP@0.5 / mAP@0.5**: Average Precision at IoU threshold 0.5. SAM 3 values evaluated at thr = 0.35; SAM 3 + LP at its optimal thr = 0.35 (best-F1).
- **ZSOD**: Zero-Shot Object Detection (no training on target domain).
- **N/A**: Metric not applicable or not available for this model/configuration.

**Key Observations**:
- **Linear probing significantly improves SAM 3**: mAP@0.5 rises from 0.444 to 0.561 (+0.117) and mAP@0.50:0.95 from 0.179 to 0.266 (+0.087), with the largest gains on the person class (AP@0.5: 0.691 → 0.892).
- **Fine-tuned YOLOe** achieves the best detection performance (mAP@0.5 = 0.825) but requires domain-specific training data.
- **SAM 3's inference speed** (≈6s/frame) is slower than YOLOe (20ms) but comparable to other vision-language models like Moondream (1s) and OMDET (2s). The linear probe adds only ~1.6 ms/frame.
- **The linear probe filters aggressively**: at thr = 0.35, it reduces predictions from 809 to 660 (−19%), improving precision substantially while keeping recall nearly unchanged.


---

## License

SAM 3 is licensed under Apache 2.0 by Meta AI.

---

## Acknowledgments

- **Meta AI** for [Segment Anything Model 3](https://github.com/facebookresearch/segment-anything-3)

