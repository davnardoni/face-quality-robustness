# Face Quality & Robustness in Face Recognition

## 1. Environment Setup

From the project's root directory, create and activate a Python virtual environment and install the required dependencies.

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

*Note: The activation command above is intended for Git Bash on Windows.*

## 2. Demo Execution

The demo consists of executing the original experimental pipeline, from dataset preparation and protocol definition to clean baselines, robustness evaluation, and quality-aware recognition.

All commands must be executed from the project's root directory with the virtual environment activated.

### Step 1 — Dataset Preparation and Experimental Protocol

First, run `src/dataset.py` to load and prepare the LFW dataset, followed by `src/protocol.py` to generate the training, gallery, and probe splits used throughout the experiments.

### Step 2 — Clean Baselines

Run the three recognition models independently on clean images.

**Eigenfaces (PCA):**

```bash
python -c "from experiments.baseline import run_eigenfaces_baseline; run_eigenfaces_baseline()"
```

**LBPH:**

```bash
python -c "from experiments.baseline import run_lbph_baseline; run_lbph_baseline()"
```

**FaceNet:**

```bash
python -c "from experiments.baseline import run_deep_baseline; run_deep_baseline()"
```

### Step 3 — Robustness Experiments

Evaluate the impact of different image degradations on the three recognition models.

**Gaussian Blur:**

```bash
python -m experiments.robustness.blur
```

**Gaussian Noise:**

```bash
python -m experiments.robustness.noise
```

**Brightness:**

```bash
python -m experiments.robustness.brightness
```

**Resolution:**

```bash
python -m experiments.robustness.resolution
```

### Step 4 — Quality-Aware Recognition

Evaluate the effect of sharpness-based probe rejection on recognition performance.

```bash
python -m experiments.quality_aware.blur
```

Each experiment can be executed independently once the dataset and experimental split are available. The generated metrics and plots are saved in the corresponding subdirectories of `results/`.
