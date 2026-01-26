# Mandarin Chinese Speech Recognition for Pronunciation Evaluation  
**Date:** 2025/2026 

**Technical Stack:** Python, PyTorch, Polars, Librosa, WandB

## Project Overview
This project focuses on the development and implementation of a Mispronunciation Detection and Diagnosis (MDD) model for Mandarin Chinese. The primary objective was to validate a neural network's capability to evaluate phonetic accuracy in non-spontaneous speech using a limited vocabulary dataset. By achieving performance levels comparable to human expert evaluators in pronunciation assessment, this system provides a scalable solution for instant phonetic feedback in language learning environments.

### Core Research Thesis
The project demonstrates that a multimodal Convolutional Neural Network (CNN) can effectively evaluate Mandarin Chinese pronunciation correctness. The model serves as a diagnostic tool, providing binary feedback on phonetic performance based on a dataset validated by native speakers.

---

## Technical Architecture
The model utilizes a multimodal **ContextFusionCNN** architecture designed to process diverse audio representations and linguistic context simultaneously.

* **2D Spectral Analysis:** Extraction of Mel-Frequency Cepstral Coefficients (MFCC) via a Voice Activity Detection (VAD) pipeline to capture phonetic characteristics.
* **1D Temporal Parameters:** Integration of Zero Crossing Rate (ZCR) and RMS Energy to analyze speech rhythm and phonetic transitions.
* **Linguistic Embeddings:** Vector representation of word IDs to provide the model with context-specific phonetic expectations for the given vocabulary.



---

## Key Engineering Milestones
* **Optimized Data Pipeline:** Developed a custom PyTorch data processing workflow utilizing a `MemoryLoadedDataLoader`, achieving a 10x acceleration in training cycles by reducing I/O bottlenecks.
* **Data Quality Engineering:** Implemented noise reduction strategies by identifying and removing poorly performing recordings and handling inconsistent expert ratings through random label selection.
* **Transformation Pipeline:** Automated audio preprocessing involving padding to fixed temporal lengths and stacking spectral deltas for improved feature density.



---

## Evaluation and Results
The model demonstrates high precision and generalization in pronunciation assessment. In a task where expert grading consistency serves as the baseline, the model achieved the following:

* **Training Accuracy:** 80.00%
* **Validation Accuracy:** 74.00%
* **Inference Demo Accuracy:** 79.17%

---

## Testing and Inference
To verify the model's performance on your own environment, an interactive inference script is provided.

### Running the Demo
You can test the model using the provided example recordings located in the repository. To do this:
1. Navigate to the `src/` directory.
2. Open the `demo.ipynb` notebook.
3. Ensure the `ContextFusionCNN.pth` checkpoint is in the root or specified path.
4. Execute the cells to load the model and process the sample audio files. 

The notebook will output a comparison between the expert's evaluation and the model's prediction for each sample, along with a final accuracy report for the demo set.

---

## Future Development
* **Production Inference:** Optimization of the model for real-time deployment to support live user pronunciation assessment.
* **Ambiguity Resolution:** Further refinement of the training process for recordings with high expert disagreement.
* **Dictionary Expansion:** Scaling the model to support a broader vocabulary and spontaneous speech patterns beyond isolated words.

---

**Keywords:** Machine Learning, Speech Processing, Mandarin Chinese, MDD, Signal Processing, Multimodal CNN.