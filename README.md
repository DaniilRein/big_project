# fMRI Emotion Analysis Pipeline
A Python-based pipeline for analyzing emotional responses in fMRI data, focusing on
positive, negative, and neutral emotional states across multiple subjects.

## Overview
This project implements a group-level fMRI analysis pipeline to study emotional responses in the brain.
The pipeline processes pre-analyzed fMRI data and performs statistical analysis to
identify brain regions associated with different emotional states.

## Project documentation
See the included 'PythonReport - Daniil.pdf'

### Key Features
- Group-level analysis of 40 subjects
- Comparison of positive, negative, and neutral emotional states
- Statistical mapping with Z-score thresholding
- Automated checkpoint system for intermediate results
- Visualization of brain activation patterns

## Prerequisites
- Python 3.12 or higher
- At least 16GB RAM recommended
- 50GB free disk space for the full dataset and analysis results

## Data Sources
- OpenNeuro Dataset: [ds005700](https://openneuro.org/datasets/ds005700)

### Original processing pipelines by the authors
- GitHub Repository: [NeuroEmo](https://github.com/abgeena/NeuroEmo)

### Data Preprocessing
The input data has been preprocessed using SPM12 with the following pipeline:
- Motion correction
- Slice-timing correction
- Co-registration
- Segmentation
- Normalization to MNI space
- Smoothing

### Analysis Parameters
- 30-second condition averaging
- 2 conditions per emotion group (positive/negative)
- 40 subject group-level analysis
- Z-score threshold: 3.1

## Project Structure
fmri-emotion-analysis/
├── checkpoints/        # Intermediate processing results
├── results/           # Final analysis outputs and figures
├── main.py           # Primary analysis pipeline
├── pyproject.toml    # Project dependencies and configuration
└── README.md         # Project documentation

## Data Checkpointing

This pipeline uses Python's pickle module for data serialization and checkpointing. Intermediate results are saved in the `checkpoints` directory after each processing step to:
- Prevent unnecessary recomputation
- Allow for interruption and resumption of the pipeline
- Enable inspection of intermediate results

Checkpoint files are saved with `.pkl` extension and can be loaded using Python's pickle module.

## Installation

All commands should be run under the project root directory:

```bash
# Install Virtualenv
pip install virtualenv

# Create virtual environment
python -m venv venv

# Activate virtual environment
# For Windows:
.\venv\Scripts\activate
# For macOS/Linux:
source venv/bin/activate

# Update pip
python -m pip install --upgrade pip

# Install project dependencies
pip install -e .

# Install development tools (optional)
pip install -e .[dev]