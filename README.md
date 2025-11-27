# Drone Accident Detection with SAM 3

This repository contains the code for a computer vision project that uses Meta's Segment Anything Model 3 (SAM 3) in a zero-shot setting to analyse drone images of road accidents. The model is used to detect crashed vehicles, people involved in the scene and non-damaged vehicles starting from a dataset in YOLO format, and to convert SAM 3 outputs into YOLO-style detections so that standard metrics can be computed and compared with existing detectors.

The project focuses on using SAM 3 as an open-vocabulary model driven only by short English text prompts (for example "crashed car", "person", "car"), without any additional training on this dataset.

## How to use (current state)

Once the Python environment and SAM 3 are installed, a first zero-shot test can be run from the project root with:

```bash
python scripts/test_sam3_single_image.py
