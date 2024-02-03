# Master Thesis: Choroidal Segmentation and Hyperpermeability Classification

## Description
This repository contains the code and results for the master thesis titled "Choroidal Segmentation and Hyperpermeability Classification using OCT Scans." The thesis explores the development of a pipeline for accurate choroidal segmentation and automatic classification of choroidal hyperpermeability in Optical Coherence Tomography (OCT) scans.

## Methods
The project is implemented in Python using the PyTorch framework for deep learning. The segmentation models, including UNET, UNET++, DRUNET, and SegResNet, are implemented and trained to accurately segment the choroidal layer from OCT scans. The SegResNet architecture emerged as the most suitable for the task.

The thickness maps are generated using the SegResNet segmentation model. The maps are then classified for the presence of choroidal hyperpermeability using a ResNet-50-based classification model. The models are trained using a dataset composed of OCT scans from various pathologies, including healthy subjects, Central Serous Chorioretinopathy (CSCR) patients, and others.

## Results

| Model        | Dice Coefficient | Pixel Accuracy | Recall | Precision |
|--------------|------------------|----------------|--------|-----------|
| UNET         | 93.81            | 99.04          | 94.12  | 93.77     |
| UNET++       | 93.98            | 99.05          | 94.12  | 94.07     |
| DRUNET       | 94.27            | 99.11          | 94.04  | 94.66     |
| SegResNet    | 94.36            | 99.14          | 94.19  | 94.72     |

### Classification Results

| Metric          | Value (%)  |
|-----------------|------------|
| Accuracy        | 81.25      |
| True Positives  | 62.5       |
| True Negatives  | 100.0      |
| False Positives | 0.0        |
| False Negatives | 18.75      |

## Conclusion
The SegResNet architecture showed superior performance in choroidal segmentation, leading to high-quality thickness maps. The automatic classification model achieved satisfactory overall accuracy but struggled with false negatives in positive cases. Future work could explore 3D approaches for improved accuracy and better handling of the challenging cases.

Feel free to explore the code and results in the corresponding folders.
