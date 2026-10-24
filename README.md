# Brain MRI tumor research baseline

The live website is now **3D-only**. It accepts NIfTI brain volumes (`.nii` or
`.nii.gz`) and shows volume geometry plus an axial preview. The original 2D
classifier is retained only as historical training code and is not used by the
website.

This is a research starter for classifying brain MRI images into four labels:
`no_tumor`, `glioma`, `meningioma`, and `pituitary`. It trains a CNN from random
initialization (no ImageNet weights) and reports accuracy, balanced accuracy,
macro F1, a confusion matrix, and a classification report.

This is not a medical device and must not be used to diagnose or triage a
patient. Public 2D datasets can contain duplicates, scanner/site bias, and
uncertain labels. For a credible study, split by patient, preserve an untouched
external test set, and obtain ethics/clinical review.

## Accredix

The Windows-ready app provides a local hospital research console with Google
OAuth support, an explicit local demo login, a patient registry, scanned-patient
list, high-priority queue, notes, and a local SQLite audit trail. Use
de-identified demo data only.

It supports 2D brain MRI classification, 3D brain-volume inspection, and a
liver CT model slot. More models are coming after validation. The app uses CUDA
when available and falls back to the PC processor.

Run the local web app with `..\venv\Scripts\python.exe web_app.py`, or launch a
Windows desktop window with `.\run_accreditx.ps1`. Use `build_windows.ps1` to
package an executable.

## Dataset sources

Use the source's terms and attribution requirements before downloading or
redistributing data.

- [BraTS / CBICA](https://www.med.upenn.edu/cbica/brats/): multi-modal MRI with
  expert tumor masks; best fit for segmentation and 3D research. Access terms
  vary by challenge year.
- [TCIA collections](https://www.cancerimagingarchive.net/browse-collections/):
  de-identified cancer imaging, including glioma and glioblastoma collections;
  collection-specific terms and metadata apply.
- [OpenNeuro](https://openneuro.org/): BIDS-formatted neuroimaging repository;
  search by diagnosis and inspect each dataset's license.
- [BRISC 2025 on Kaggle](https://www.kaggle.com/competitions/1056lab-brain-tumor-image-classification/data):
  2D MRI classification with glioma, meningioma, no-tumor, and pituitary
  classes; the competition page lists CC BY 4.0.
- [Figshare brain tumor dataset](https://doi.org/10.6084/m9.figshare.1512427.v8):
  commonly used 2D MRI classification source; verify its current license and
  provenance before use.

## Expected data layout

Put images in class folders. PNG/JPEG/BMP/TIFF are supported.

```text
data/brain_mri/
  glioma/*.jpg
  meningioma/*.jpg
  notumor/*.jpg
  pituitary/*.jpg
```

Keep patient-level grouping if filenames identify a patient. Do not put slices
from the same patient in both train and validation sets.

## Install and train

```powershell
cd "C:\Users\andre\Reuben\LLM Trainer\liver 50GB\brain_cancer_ai"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py --data-dir .\data\brain_mri\Training --epochs 30 --batch-size 32
```

The best checkpoint is written to `artifacts/best_model.pt`.

## Predict one image

```powershell
python predict.py --checkpoint artifacts/best_model.pt --image path\to\scan.jpg
```

The confidence is model confidence, not clinical probability. For MRI volumes
(`.nii`/`.nii.gz`) use a BraTS/TCIA-specific 3D pipeline after defining a
patient-level task and a validated preprocessing protocol.

## Run the live website

From this folder, after training has produced `artifacts_80min/best_model.pt`:

```powershell
..\venv\Scripts\python.exe -m pip install -r ..\requirements-web.txt
..\venv\Scripts\python.exe web_app.py
```

Open `http://127.0.0.1:5000`. The website runs inference locally and does not
upload images to a third-party service. It is a research demo, not a clinical
diagnostic tool.

