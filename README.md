# PGNN-FEPEs

Physics-guided neural network for full-energy peak efficiency (FEPE) calibration of HPGe detectors. Combines an MLP trained on a single reference material with an analytical self-attenuation correction layer using XCOM mass attenuation coefficients. Includes solid-angle weighted ray-tracing and parallel beam corrections.

Associated paper: *A physics-guided neural network with analytical self-attenuation correction for rapid efficiency calibration of HPGe detectors* — J. G. Guerra, Scientific Reports (submitted).

## Repository structure

```
PGNN-FEPEs/
├── Well/
│   ├── entrenar_ANN_FEPE_PINN_POZO_multimat.py   # Training and evaluation
│   ├── autoatenuacion_corregida_POZO.py           # Ray-tracing self-attenuation (cylindrical geometry)
│   └── Data examples/                             # PENELOPE simulation data (well-type detector)
├── Xtra/
│   ├── entrenar_ANN_FEPE_PINN_XtRa_multimat.py   # Training and evaluation
│   ├── autoatenuacion_corregida_XtRa.py           # Ray-tracing self-attenuation (truncated-cone geometry)
│   └── Data examples/                             # PENELOPE simulation data (XtRa detector)
├── LICENSE
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy, Pandas, Matplotlib

Install dependencies:

```bash
pip install torch numpy pandas matplotlib openpyxl
```

## How to run

Each detector folder (Well or Xtra) is self-contained. The data files must be in the same directory as the Python scripts.

1. Copy the contents of `Data examples/` into the detector folder (alongside the `.py` files):

```bash
cd Well
cp Data\ examples/* .
```

2. Run the training script:

```bash
python entrenar_ANN_FEPE_PINN_POZO_multimat.py
```

The training script automatically:
- Loads the calibration material data (IAEA-RGU-1) and discovers all test material files by filename pattern.
- Trains the MLP on a subset of energies and heights, reserving the rest for blind testing.
- Calls the ray-tracing module (`autoatenuacion_corregida_POZO.py`) to precompute photon path lengths for each sample height. This is done once and cached for all energies and materials.
- Evaluates the full pipeline (ANN × f_a) for both PB and RT corrections on all test materials at unseen energies and heights.
- Generates diagnostic plots (loss curves, scatter plots, heatmaps, residuals, FEPE vs height curves).
- Exports all results to a JSON file.

The same procedure applies to the Xtra folder with the corresponding scripts.

## Data format

The PENELOPE simulation data are provided as Excel files (`.xlsx`), one per material, following the naming convention:

- Well detector: `FEPE_barrido_alturas_POZO_{material}.xlsx`
- XtRa detector: `FEPE_barrido_alturas_XtRa_{material}.xlsx`

Each file contains FEPE values at 16 energies (46.5–1764.5 keV) for all simulated sample heights.

The XCOM mass attenuation coefficient table (`XCOM_mu_rho_table.csv`) is also required and included in the data. Elemental mass attenuation coefficients were obtained from the [NIST XCOM database](https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html).

## Configuration

Key parameters can be modified in the `CONFIG` dictionary at the top of each training script: training/test energy and height splits, MLP architecture, learning rate, batch size, early stopping patience, etc. Material compositions and densities are defined in the `MATERIALES` dictionary.

## Output

- Diagnostic figures saved as PNG in the working directory.
- Full results exported to JSON, including per-material and per-energy MAPE, R², residuals, and point-by-point predictions.
- Trained model saved as a `.pth` file (optional, controlled by `guardar_modelo` flag).

## Detectors

- **Well-type**: Canberra GCW4023 (40% relative efficiency) with cylindrical polypropylene vial (Ø 11.5 mm × 51 mm). Characterised in [Guerra et al. (2018), NIMA 908, 206–214](https://doi.org/10.1016/j.nima.2018.08.048).
- **XtRa planar**: Canberra GX3518 (38% relative efficiency) with truncated-cone polypropylene beaker (Ø 47.9–56 mm × 72.9 mm). Characterised in [Guerra et al. (2018), NIMA 880, 67–74](https://doi.org/10.1016/j.nima.2017.10.076).

## Associated paper

J. G. Guerra, *An artificial neural network with analytical self-attenuation correction for rapid efficiency calibration of HPGe detectors*, **Scientific Reports**, accepted for publication (2026). https://doi.org/10.1038/s41598-026-58894-0

## License

The source code in this repository is released under the MIT License.

The associated article is subject to its own publishing licence and is not covered by the MIT License.

