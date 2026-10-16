# FallNet: Privacy-Preserving Fall Detection Using Thermal Infrared Video

FallNet is a hybrid deep learning framework for detecting human falls from thermal infrared video. It aims to support privacy-preserving monitoring by avoiding dependence on conventional RGB camera footage. The framework combines gain-robust thermal preprocessing, motion-channel extraction, temporal deep learning, and ensemble learning to address short-duration falls and class imbalance.

## Overview

Thermal cameras can operate in low-light conditions while revealing less visual detail than conventional RGB cameras. However, automatic gain control (AGC), brief fall events, and imbalanced datasets make reliable fall detection challenging.

FallNet addresses these challenges through a preprocessing pipeline and a comparative evaluation of three deep learning architectures, along with soft-voting ensembles.

## Key Features

- **Privacy-preserving vision:** Uses thermal infrared video instead of conventional RGB footage.
- **Gain-robust preprocessing:** Applies percentile-based normalization and gain-jump suppression to reduce intensity variations.
- **Motion-channel extraction:** Combines background differences and temporal differences to represent movement.
- **Short-fall-aware windowing:** Processes 16-frame video windows and injects centered positive windows when needed.
- **Imbalance-aware training:** Uses a rotating balanced sampler with positive, hard-negative, and easy-negative samples.
- **Multiple architectures:** Evaluates recurrent, Transformer-based, and 3D convolutional approaches.
- **Ensemble learning:** Combines model predictions using soft voting.
- **Leakage-aware evaluation:** Uses participant-disjoint splits for identified fall videos and folder-disjoint splits for anonymous non-fall videos.

## System Architecture

FallNet consists of four main stages:

1. **Thermal Video Preprocessing**
   - Converts frames to RGB and resizes them to 112 × 112 pixels.
   - Applies per-frame percentile normalization using the 1st and 99th percentiles.
   - Detects and suppresses sudden gain-related intensity changes.
   - Estimates the background using the temporal median.
   - Generates an additional motion channel.

2. **Window Generation**
   - Constructs 16-frame video windows.
   - Labels windows using overlap with annotated fall intervals.
   - Adds centered positive windows to improve representation of short falls.
   - Separates negative windows into hard negatives and easy negatives.

3. **Deep Learning Models**
   - ResNet18 + Bidirectional GRU (Bi-GRU)
   - MobileNetV3-Small + Transformer
   - R3D-18

4. **Prediction and Evaluation**
   - Produces fall probabilities for video windows.
   - Evaluates individual models and two-model and three-model soft-voting ensembles.
   - Uses validation-selected thresholds and reports classification metrics.

## Model Input

Each input window contains 16 frames with four channels per frame:

- Three channels representing the normalized thermal image.
- One additional motion channel.

The resulting tensor has the shape:

`(16, 4, 112, 112)`

The first three input channels use pretrained backbone weights where applicable, while the motion-channel filters are initialized from the mean of the pretrained filters.

## Experimental Results

The following results are reported in the accompanying paper on a test set containing 1,531 windows, with a natural fall prevalence of 16.0%.

| Model | ROC-AUC | AUPRC | F1-score | Recall |
|---|---:|---:|---:|---:|
| ResNet18 + Bi-GRU | 0.9179 | 0.5943 | 0.6356 | 0.7796 |
| MobileNetV3 + Transformer | 0.9105 | 0.6145 | 0.5486 | 0.5184 |
| R3D-18 | 0.8615 | 0.5198 | 0.4548 | 0.3796 |
| Two-model ensemble | **0.9274** | **0.6381** | 0.5628 | 0.5306 |
| Three-model ensemble | 0.9154 | 0.6255 | 0.5915 | 0.5673 |

The two-model ensemble of ResNet18 + Bi-GRU and MobileNetV3 + Transformer achieved the highest ROC-AUC and AUPRC. ResNet18 + Bi-GRU achieved the highest recall among the learned models, while the three-model ensemble achieved the highest ensemble F1-score.

These results illustrate the trade-off between detecting more falls and limiting false-positive detections.

## Dataset

The experiments use the **Multi-Visual Modality Fall Detection (MUVIM) dataset**, with thermal video folders and annotated fall intervals.

After excluding videos containing fewer than 16 frames, the study uses:

- 187 fall-video folders
- 89 non-fall-video folders
- 11,362 generated video windows across training, validation, and test splits

The test set contains 245 fall windows and 1,286 non-fall windows.

Refer to the accompanying paper for the dataset processing, split methodology, and experimental protocol.

## Technologies

- Python
- PyTorch
- Deep learning and computer vision
- Convolutional neural networks (CNNs)
- Bidirectional GRU
- Transformer encoder
- 3D CNN
- Ensemble learning

## Limitations and Future Work

The reported results are based on a specific dataset and evaluation protocol. Generalization to new thermal datasets, additional participants, camera viewpoints, and real-world deployment conditions requires further validation.

Future work includes event-level evaluation, detection-delay and false-alarm-rate analysis, probability calibration, and exploration of lighter models for deployment on resource-constrained hardware.

## Reference

**FallNet: A Hybrid Deep Learning Framework for Privacy-Preserving Fall Detection from Thermal Infrared Video**

Refer to the accompanying research paper for the complete methodology, experimental analysis, and references.

## Project Status

Research prototype. The reported performance reflects the experimental setup described in the paper and should not be interpreted as a clinically validated or production-ready fall detection system.
