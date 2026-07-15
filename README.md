# PLSR-unmixing

Reproducibility code and Raman spectroscopy datasets for the manuscript "Spatial Chemical Imaging and Quantitative Prediction of Dentin Demineralization via Raman Spectroscopy".

## Contents

- `train_internal_edta_ca_p_models_clean.py`: PLSR modelling for Ca and P prediction.
- `mcr_als_dentin_unmixing_clean.py`: MCR-ALS unmixing for Raman mapping data.
- `data/EDTA group/`: EDTA-group raw and processed Raman data used by the scripts.
- `data/caries group/`: caries-group raw and processed Raman data.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run PLSR modelling:

```bash
python train_internal_edta_ca_p_models_clean.py
```

Run MCR-ALS unmixing:

```bash
python mcr_als_dentin_unmixing_clean.py
```

Both scripts use paths relative to this repository and write outputs to `results/`.
