# Brain MRI Compass — six-slide pitch outline

## 1. Problem
Brain MRI review is valuable but time-consuming, and early-stage builders lack
a transparent way to prototype and audit medical-image AI. A benchmark result
is not the same as a clinical diagnosis.

## 2. Insight
The fastest safe MVP is a reproducible research loop: known public data,
training from scratch, visible metrics, explicit limitations, and a human-first
fallback. Trust is part of the product.

## 3. Solution demo
Upload a labeled 2D MRI image, run the trained CNN, show the top class,
confidence, and validation metrics, then expose the exact command and
checkpoint so another builder can reproduce it.

## 4. Technical approach
PyTorch CNN from random initialization; grayscale resize and augmentation;
class-weighted cross-entropy; CUDA training; balanced accuracy, macro-F1,
confusion matrix, and saved best checkpoint. Dataset: public four-class MRI
benchmark. Next technical step: patient-level splits and 3D multi-modal BraTS
segmentation.

## 5. Value and route to users
Useful for students, researchers, and clinicians evaluating AI concepts before
expensive validation. Distribution starts with an open reproducible repository,
mentor feedback, and a short demo; clinical deployment is explicitly out of
scope until validation and regulatory review.

## 6. Roadmap and risks
Now: baseline classifier and timed training. Next: external test set,
patient-level metadata, calibration, explainability, and 3D segmentation.
Risks: dataset bias, leakage, label noise, privacy, and overclaiming results.
