# CrackVision 

**AI-powered steel surface crack detection and segmentation using YOLO11n-seg.**

CrackVision is a computer vision project that detects visible cracks in steel surface images and produces segmentation masks and bounding boxes. It includes a Streamlit interface for image uploads and local camera inspection.

## Features

- Steel surface crack detection and segmentation
- Pixel-level segmentation masks and bounding boxes
- Image upload and annotated result visualization
- Local camera snapshot and live inspection prototype
- Adjustable confidence threshold and inference resolution
- CPU inference support
- Model evaluation and dataset exploration notebooks

## Project Structure

```text
CrackVision/
├── app.py
├── models/
├── notebooks/
├── src/crackvision/
├── tests/
├── data/                 # Local dataset; not committed
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Requirements

- Python 3.10 or newer
- A webcam for local camera inspection
- Sufficient RAM and storage for PyTorch and model inference

## Setup

Clone the repository and enter its directory:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd CrackVision
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the CPU-only PyTorch build first:

```bash
python -m pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  torch torchvision
```

Install the remaining dependencies:

```bash
python -m pip install --no-cache-dir -r requirements.txt
```

## Model Weights

Place the trained weights in the `models/` directory:

- `crackvision-yolo11n-seg-best.pt`
- `crackvision-yolo11n-seg-960-best.pt`

The weights are trained artifacts and must be supplied separately if they are not included in the repository.

## Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, typically `http://localhost:8501`.

## Run Tests

```bash
pytest -q
```

## Limitations

- Performance depends on hardware and inference settings.
- Local OpenCV camera access is different from browser webcam access on a remotely hosted server.
- Physical crack dimensions require suitable image calibration.
- Predictions are model estimates and should not be treated as certified industrial inspection results without further validation.

## License

Choose and add an appropriate open-source license before distributing the project.
