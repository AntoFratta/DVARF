# DVARF: Drone Vehicle Accident Recognition

Experimental evaluation of Meta's Segment Anything Model 3 (SAM 3) for zero-shot object detection in aerial accident scene imagery.

![SAM 3 Example](assets/sam3_example.png)

---

## Overview

This project investigates the application of **SAM 3**, Meta's foundation model for image segmentation, to the domain of accident scene analysis from drone footage. While SAM 3 was designed as a general-purpose segmentation model, this work explores its effectiveness when applied to a specialized detection task using only text-based prompts, without any fine-tuning on the target domain.

The core research question is whether modern vision-language models like SAM 3 can perform reliably in emergency response scenarios where traditional object detectors would require extensive labeled training data. The project addresses this through a systematic experimental pipeline with three main components:

**Zero-shot Detection:** SAM 3 is queried using simple natural language prompts for three object classes relevant to accident scenes: crashed vehicles, persons, and undamaged vehicles. The model outputs segmentation masks which are then converted to bounding box detections in YOLO format, enabling direct comparison with traditional detection methods using standard metrics like mean Average Precision (mAP).

**Segmentation Preservation:** Beyond detection metrics, the project preserves SAM 3's native segmentation outputs as binary masks. This dual-output approach maintains the detailed spatial information that could be valuable for downstream analysis tasks such as damage assessment or scene reconstruction.

**Linear Probing:** To investigate whether SAM 3's performance can be improved with minimal supervision, a lightweight linear classifier is trained on top of the model's outputs. This classifier learns to re-score detections based on simple geometric features (bounding box dimensions, aspect ratios, confidence scores) extracted from the predictions on a small training set. The linear probe provides a low-cost adaptation method that does not require retraining the foundation model itself.

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

**Linear Probing Notebook** ([`test_sam3_linearProbing.ipynb`](notebooks/test_sam3_linearProbing.ipynb)): Contains the complete experimental pipeline including training SAM 3 on the training split, building the linear probe dataset, training the classifier, and evaluating the enhanced predictions on the test set.

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

Configuration parameters such as confidence threshold, NMS IoU threshold, and whether to save segmentations can be adjusted by editing variables at the top of the script.

### Evaluation

The evaluation script computes standard object detection metrics by comparing predictions against ground-truth annotations. Metrics include per-class and overall Precision, Recall, F1 score, AP@0.50, and AP@0.50:0.95. Results are printed to the console and can be redirected to a file for record-keeping.

```bash
python scripts/eval_sam3_on_split.py > results/sam3_test_metrics.txt
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

## Experimental Results

Results on test split (confidence threshold = 0.26):

### Zero-shot SAM 3

| Class | Precision | Recall | F1 | AP@0.50 | AP@0.50:0.95 |
|-------|-----------|--------|-----|---------|--------------|
| Crashed car | 0.490 | 0.674 | 0.568 | 0.512 | 0.235 |
| Person | 0.700 | 0.656 | 0.677 | 0.603 | 0.195 |
| Undamaged car | 0.380 | 0.074 | 0.124 | 0.061 | 0.042 |
| **Overall** | **0.493** | **0.348** | **0.408** | **0.392** | **0.157** |

### SAM 3 + Linear Probe

| Class | Precision | Recall | F1 | AP@0.50 | AP@0.50:0.95 |
|-------|-----------|--------|-----|---------|--------------|
| Crashed car | 0.490 | 0.674 | 0.568 | 0.520 | 0.248 |
| Person | 0.700 | 0.656 | 0.677 | 0.597 | 0.166 |
| Undamaged car | 0.439 | 0.070 | 0.121 | 0.065 | 0.034 |
| **Overall** | **0.503** | **0.345** | **0.410** | **0.394** | **0.149** |

---

## Project Structure

```
DVARF/
├── data/                    # Dataset (images + labels)
├── scripts/                 # Executable scripts
├── src/                     # Library code
├── notebooks/               # Jupyter notebooks for Google Colab
├── results/                 # Evaluation metrics
├── requirements.txt         # Dependencies
└── README.md
```

See individual script files for detailed configuration options.

---

## License

This project is licensed under the MIT License. SAM 3 is licensed under Apache 2.0 by Meta AI.

---

## Acknowledgments

- **Meta AI** for [Segment Anything Model 3](https://github.com/facebookresearch/segment-anything-3)

