# PLSR-unmixing

Reproducibility code and Raman spectroscopy datasets for the manuscript **"Spatial Chemical Imaging and Quantitative Prediction of Dentin Demineralization via Raman Spectroscopy"**.

This repository contains the cleaned Python scripts and the raw/processed datasets needed to reproduce the PLSR modelling and MCR-ALS unmixing analyses reported in the manuscript.

## Repository Structure

```text
PLSR-unmixing/
|-- train_internal_edta_ca_p_models_clean.py
|-- mcr_als_dentin_unmixing_clean.py
|-- requirements.txt
|-- data/
|   |-- EDTA group/
|   |   |-- EDTA-raw Raman data-mapping/
|   |   |-- EDTA-raw Raman data-pointscan/
|   |   |-- EDTA-Ca P content.xlsx
|   |   |-- EDTA-processed pointscan features.xlsx
|   |   |-- EDTA-processed pointscan full spectrum.xlsx
|   |   |-- EDTA-sample ID and group.xlsx
|   |   |-- HA_collagen_references.xlsx
|   |   |-- m1_mcr_ready.npy
|   |   |-- m1_valid_mask.npy
|   |   `-- m1_wn.npy
|   `-- caries group/
`-- results/                 # Created automatically when scripts are run
```

## Scripts

- `train_internal_edta_ca_p_models_clean.py`: trains internal-validation PLSR models for Ca and P prediction from Raman point-scan features and spectra.
- `mcr_als_dentin_unmixing_clean.py`: performs MCR-ALS unmixing of preprocessed Raman mapping data and compares unmixed components with HA/collagen reference spectra.

Both scripts use paths relative to the repository root. After cloning the repository, no manual path editing should be needed if the `data/` folder is kept in the same location.

## Tested Environment

The scripts were tested with the following environment:

- Operating system: Windows 11
- Python: 3.12.4
- NumPy: 1.26.4
- pandas: 2.2.2
- SciPy: 1.13.1
- scikit-learn: 1.7.2
- Matplotlib: 3.8.4
- joblib: 1.4.2
- tqdm: 4.66.4
- openpyxl: 3.1.2

Python 3.10 or newer is recommended. The code uses standard scientific Python packages and should also run on macOS/Linux after installing the dependencies.

## Installation

Clone the repository:

```bash
git clone https://github.com/lizhixin010203/PLSR-unmixing.git
cd PLSR-unmixing
```

Create and activate a clean environment. For example, with conda:

```bash
conda create -n plsr-unmixing python=3.12
conda activate plsr-unmixing
pip install -r requirements.txt
```

Alternatively, with `venv`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate the `venv` environment with:

```bash
source .venv/bin/activate
```

## Running the Analyses

Run the PLSR modelling script:

```bash
python train_internal_edta_ca_p_models_clean.py
```

Run the MCR-ALS unmixing script:

```bash
python mcr_als_dentin_unmixing_clean.py
```

The scripts read inputs from `data/` and write generated outputs to `results/`.

## Main Input Files

For PLSR modelling:

- `data/EDTA group/EDTA-processed pointscan features.xlsx`
- `data/EDTA group/EDTA-processed pointscan full spectrum.xlsx`
- `data/EDTA group/EDTA-Ca P content.xlsx`
- `data/EDTA group/EDTA-sample ID and group.xlsx`

For MCR-ALS unmixing:

- `data/EDTA group/m1_mcr_ready.npy`
- `data/EDTA group/m1_wn.npy`
- `data/EDTA group/m1_valid_mask.npy`
- `data/EDTA group/HA_collagen_references.xlsx`

The raw Raman data are also included under `data/EDTA group/` and `data/caries group/` for verification and reuse.

## Outputs

The PLSR script saves model summaries, diagnostic tables, model objects, and figures under:

```text
results/plsr_internal/
```

The MCR-ALS script saves unmixed component spectra, abundance maps, similarity results, and figures under:

```text
results/m1/
```

## Notes on Reproducibility

- The repository includes both raw datasets and processed input files used directly by the scripts.
- Intermediate model diagnostics such as weights, loadings, and scores are exported by the PLSR workflow.
- The `results/` directory is excluded from version control and will be regenerated when the scripts are run.
