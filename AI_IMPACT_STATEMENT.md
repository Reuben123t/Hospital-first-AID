# AI Impact Statement

Brain MRI Compass is a research prototype that classifies a supplied 2D brain
MRI image into four benchmark labels: glioma, meningioma, pituitary tumor, or
no tumor. It is intended to help builders and researchers explore image
classification workflows; it does not diagnose patients, replace radiologists,
or recommend treatment.

The model is a small CNN trained from random initialization with PyTorch. The
training data is the public Brain Tumor MRI Dataset curated by Masoud Nickparvar
and mirrored on Zenodo. It combines multiple public sources and is released by
the dataset publisher under the stated dataset terms/attribution requirements.
The downloaded archive is not committed to the repository.

Guardrails include displaying the predicted class and model confidence rather
than a clinical probability, preserving validation metrics, recommending
patient-level rather than slice-level splits, and explicitly warning against
clinical use. Risks include source/scanner bias, label noise, duplicated or
related images, lack of demographic metadata, and poor performance on scans
from another hospital or MRI protocol. Before any clinical use, the system
would require expert-reviewed labels, patient-level external validation,
calibration, subgroup analysis, prospective evaluation, security/privacy review,
and applicable regulatory approval.

Expected benefit: a reproducible educational baseline and a faster way to
demonstrate model training, evaluation, and uncertainty-aware UX. Expected
harm if misused: false reassurance, unnecessary alarm, or delayed care.
