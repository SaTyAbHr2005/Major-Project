# Secure and Privacy-Preserving Federated Deep Learning Training Platform for Medical Imaging

A platform for training and using deep learning models on medical images while keeping raw hospital
data on the hospital's own machine. It has a researcher/admin desktop, a hospital desktop, a backend
API, and a hospital-side ML pipeline.

> Research/engineering project. Model outputs are predictions, not clinical diagnoses.

---

## 1. What is in this repository

```
Major Project/
├── frontend/          React + Vite app: researcher/admin screens and hospital screens (one app)
├── backend/           Node.js + Express API: login, roles, users, hospitals, training requests (MongoDB)
├── hospital_client/   Python ML pipeline that runs on the hospital machine
│   ├── dataset/            Module 4  - dataset inspection
│   ├── preprocessing/      Module 5  - preprocessing and train/validation/test splits
│   ├── model_management/   Module 6  - model definitions and versioned model artifacts
│   ├── training/           Module 7  - local PyTorch training
│   ├── resource_training/  Module 8  - hardware-aware training recommendations
│   ├── inference/          Module 16 - local inference
│   └── local_api/          local ML service that lets the hospital screens run the pipeline
├── docs/              per-module documentation (module1_readme.md ... module16_readme.md)
├── requirements.txt   Python packages
└── *.md               README and architecture documents (see section 9)
```

How the parts talk to each other:

```
Browser (http://localhost:5173)
   |                                   |
   | login, users, training requests   | dataset path, training choices, one image for inference
   v                                   v
Backend API  :5000  <--- verifies ---  Local ML service  127.0.0.1:8765  (hospital machine only)
   |           sessions                     |
   v                                        v
MongoDB                            python -m hospital_client.<module>  (reads the dataset locally)
```

Raw images and datasets never go to the backend. Only aggregate training progress (epoch, loss,
accuracy) is sent to the backend when a hospital links a run to a researcher's training request.

---

## 2. Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.10 or newer (tested with 3.10) | `python --version` |
| Node.js | **20.19 or newer** (tested with 26) | required by mongoose 9; `node --version` |
| npm | 9 or newer | comes with Node.js |
| MongoDB | MongoDB Atlas (free tier) or a local MongoDB server | used by the backend |
| NVIDIA GPU + driver | optional | without a GPU everything runs on the CPU (slower training) |
| Git | any | to clone the repository |

On Linux, the "Browse..." folder picker also needs tkinter: `sudo apt install python3-tk`.

---

## 3. Setup (one time)

All commands are run from the repository root unless a `cd` is shown. Commands are written for
Windows PowerShell; on macOS/Linux use `cp` instead of `Copy-Item`.

### 3.1 Clone

```powershell
git clone https://github.com/SaTyAbHr2005/Major-Project.git
cd Major-Project
```

### 3.2 Python packages (ML pipeline)

**With an NVIDIA GPU**, install the CUDA build of PyTorch first. A plain `pip install torch` on
Windows installs the CPU-only build and the GPU will not be used:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

(If your driver is older, pick the matching command at https://pytorch.org/get-started/locally/.)

Then install the rest (this also installs CPU PyTorch if you skipped the step above):

```powershell
pip install -r requirements.txt
```

Check it:

```powershell
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
```

### 3.3 Backend

```powershell
cd backend
npm install
Copy-Item .env.example .env
```

Open `backend/.env` and set at least:

| Variable | Value |
|---|---|
| `MONGODB_URI` | your MongoDB connection string (Atlas: *Connect -> Drivers*), or `mongodb://localhost:27017/federated_dl_auth` for a local server |
| `JWT_SECRET` | any long random string |
| `JWT_EXPIRES_IN` | e.g. `24h` |
| `PORT` | `5000` |

For MongoDB Atlas, also allow your IP address in *Network Access*.

Create the demo accounts, hospital and sample training requests (safe to run more than once):

```powershell
npm run seed
cd ..
```

### 3.4 Frontend

```powershell
cd frontend
npm install
cd ..
```

The frontend works without a `.env` file (it defaults to the local URLs below). To change them,
create `frontend/.env`:

```
VITE_API_URL=http://localhost:5000/api/v1
VITE_ML_API_URL=http://127.0.0.1:8765/api/ml
```

`.env` files are gitignored - never commit them.

---

## 4. Run the project

Open **three terminals** and keep them running:

**Terminal 1 - backend API** (http://localhost:5000/api/v1)
```powershell
cd backend
npm start
```

**Terminal 2 - local ML service** (http://127.0.0.1:8765, repository root)
```powershell
python -m hospital_client.local_api
```

**Terminal 3 - frontend** (http://localhost:5173)
```powershell
cd frontend
npm run dev
```

Open **http://localhost:5173** in a browser.

### Demo accounts (created by `npm run seed`)

| Role | Login | Password |
|---|---|---|
| Hospital operator | `operator@stjude-clinical.org` (or `HOS-001`) | `hospital123` |
| Researcher | `e.rostova@med.stanford.edu` | `researcher123` |
| Admin | `admin@consortium.org` | `admin123` |

Change these passwords for anything beyond local development.

---

## 5. Using the hospital workflow

Log in with **Secure Login as Hospital**, then:

1. **Dataset Inspection** - click **Browse...** (or type a path) to locate a dataset: a folder of
   images with one sub-folder per class (optionally already split into train/val/test), or a
   CSV/XLS/XLSX file. Module 4 inspection, Module 5 preprocessing and Module 8 recommendations then
   run automatically on this machine. The dataset is only read, never modified or uploaded.
2. **Preprocessing Engine** - review the split and quality report. Optionally re-run with a
   patient/group-id map (JSON `{relative_path: patient_id}`) for patient-level splits.
3. **Start Local Training** - choose one option yourself (nothing is pre-selected):
   Recommended, High-Capacity / Not Recommended, or Fast, a **same-tier alternative** of any of
   them, or a **Manual Configuration** (model, batch size, epochs) with a live load meter. Confirm
   to start. Optionally link the run to a researcher's training request.
4. **Training Monitor** - live per-epoch loss/accuracy charts and the training log.
5. **Training Results** - test metrics, per-class table, confusion matrix, ROC/PR plots, resource
   statistics.
6. **Local Models** / **Local Inference** - browse trained model versions and run a prediction on
   a new image.

All generated files go to `~/medfl_ml_work` (datasets' profiles and splits, trained models,
results, plots). Change it with the `ML_WORK_DIR` environment variable before starting the ML
service.

---

## 6. Running the ML pipeline from the terminal (optional)

The hospital screens run these same commands; you can also run them directly:

```powershell
python -m hospital_client.dataset inspect <dataset_path> --output outputs/m4
python -m hospital_client.preprocessing preprocess <dataset_path> --profile outputs/m4/dataset_profile.json --output outputs/m5 --mode lazy
python -m hospital_client.resource_training detect
python -m hospital_client.resource_training recommend --dataset outputs/m5 --output outputs/trained
python -m hospital_client.resource_training train --dataset outputs/m5 --output outputs/trained --model-id my_model --choice recommended --test --plots
python -m hospital_client.inference predict --models-root outputs/trained/models --model-id my_model --version 1 --image <image_path> --json
```

The `outputs/` folder is gitignored. Never commit datasets or model files. Each module's README lists
all options.

---

## 7. Tests

```powershell
python -m pytest hospital_client -q          # ML pipeline + local ML service (CUDA tests skip without a GPU)
cd backend; npm test; cd ..                  # backend API (first run downloads a test MongoDB binary, ~500 MB)
cd frontend; npx vite build; cd ..           # frontend compiles
```

---

## 8. Configuration and troubleshooting

Local ML service environment variables (all optional):

| Variable | Default | Purpose |
|---|---|---|
| `ML_WORK_DIR` | `~/medfl_ml_work` | where profiles, splits, models and results are written |
| `ML_API_PORT` | `8765` | ML service port (also set `VITE_ML_API_URL` in `frontend/.env`) |
| `ML_API_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | browser origins allowed to call it |
| `ML_BACKEND_URL` | `http://localhost:5000/api/v1` | backend used to verify logins |

| Problem | Fix |
|---|---|
| "Local ML service is offline" | start `python -m hospital_client.local_api` from the repository root |
| "Your session is not valid" on hospital screens | log out and log in again (the backend was restarted or the token expired) |
| "Cannot verify your session: the backend API is unreachable" | start the backend (`npm start` in `backend/`) |
| GPU shows "No CUDA device" | install the CUDA build of PyTorch (section 3.2) and check `nvidia-smi` |
| Backend cannot connect to MongoDB | check `MONGODB_URI` in `backend/.env` and, for Atlas, *Network Access* |
| `No module named hospital_client` | run the ML service from the repository root |
| Port already in use | stop the other instance, or change the port (see table above) |
| Training is very slow | do not run the test suite or other GPU programs during training |
| `npm warn install-scripts ...` during `npm install` | harmless: `bcrypt` ships prebuilt binaries, and the test MongoDB binary downloads on the first `npm test` |

---

## 9. Documentation

| File | Contents |
|---|---|
| `architecture.md` | overall platform architecture and team structure |
| `ml_pipeline_architecture.md` | ML pipeline architecture and the Hospital Desktop integration |
| `docs/module1_readme.md` - `docs/module3_readme.md` | authentication, researcher desktop, hospital desktop |
| `docs/module4_readme.md` - `docs/module8_readme.md`, `docs/module16_readme.md` | the ML modules (dataset, preprocessing, models, training, resource-aware training, inference) |

## 10. Status

| Implemented and tested | Planned (not implemented yet) |
|---|---|
| Modules 1-3 (auth, researcher desktop, hospital desktop) | Module 9 federated learning (Flower) |
| Modules 4-8, 16 (hospital-side ML pipeline) | Module 10 differential privacy |
| Hospital desktop <-> ML pipeline <-> backend integration | Modules 11-15, 17-21 |

Privacy note: the design keeps raw data local, but federated learning alone is not a privacy or
legal-compliance guarantee.
