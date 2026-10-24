Set-Location $PSScriptRoot
& "C:\Users\andre\Reuben\LLM Trainer\liver 50GB\venv\Scripts\python.exe" -m pip install pyinstaller pywebview
& "C:\Users\andre\Reuben\LLM Trainer\liver 50GB\venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --name "Accredix" --distpath "dist_latest" --workpath "build_latest" --windowed --add-data "accreditx_templates;accreditx_templates" --add-data "accreditx_static;accreditx_static" --add-data "models;models" --add-data "artifacts_80min;artifacts_80min" desktop_app.py
