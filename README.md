# Pantara v0.7.4

**Physics-Aware Neural-network Tracking And Regression Analysis**

Pantara is a symbolic regression system that discovers physical laws from raw sensor data. It combines a neural oracle (trained on synthetic equations) with an analytical matching-pursuit to build predictions as sums of interpretable basis functions.

## How it works

Given a dataset `(X, y)`, Pantara runs up to 5 matching-pursuit steps:

1. **Power-law screen** — fits `y = C · x₀^n₀ · x₁^n₁ · …` analytically via least-squares in log-log space (O(N), exact). Accepts if R² > 0.97.
2. **Oracle ranking** — a SetEncoder neural network classifies the residual into one of 8 function families: `lin`, `sq`, `pair`, `qc`, `trip`, `qinv`, `sin`, `cos`.
3. **Multi-space search** — each family is tested in 6 transformation spaces: `original`, `log_y`, `log_x`, `log_log`, `sq_y`, `inv_x`. The space with the highest correlation score is selected.
4. **Block fit** — a linear regression (OLS) maps the chosen feature column to the residual; the prediction is accumulated.
5. **Early stop** — if the residual falls below 6% of the target variance, or if one block explains > 80% of variance, the loop terminates.

The fitted model is a sum of basis functions whose parameters are fully stored, enabling genuine generalization to held-out data.

## Installation

```bash
pip install git+https://github.com/Yapock22/pantara.git
```

Or from source:

```bash
git clone https://github.com/Yapock22/pantara.git
cd pantara
pip install .
```

**Requirements:** Python ≥ 3.8, NumPy ≥ 1.21, PyTorch ≥ 1.12.  
The system automatically detects and uses Apple Silicon MPS, CUDA, or CPU.

## Quick start

```python
import numpy as np
import pantara
import torch

# Load the pre-trained oracle
model = pantara.OracleClassifier(n_classes=pantara.N_CLASSES)
model.load_state_dict(torch.load(pantara.MODEL_PATH, weights_only=True))
model.to(pantara.DEVICE)
model.eval()

# Generate a synthetic dataset: y = 3 * x0 * x1^2
N = 500
X = np.column_stack([np.random.uniform(1, 8, N), np.random.uniform(1, 6, N)])
y = 3.0 * X[:, 0] * X[:, 1] ** 2

# Run the pipeline
precision_5pct, chosen_blocks = pantara.pipeline(model, X, y)
print(f"Precision ±5% : {precision_5pct * 100:.1f}%")
print(f"Blocks chosen : {chosen_blocks}")
```

## fit / predict on held-out data

`PantaraRegressor` runs the same matching pursuit as `pipeline()` and stores every chosen block (family, space, fitted parameters), so the learned law can be applied to new data without `y`:

```python
from pantara import PantaraRegressor

est = PantaraRegressor(random_state=42)
est.fit(X_train, y_train, feature_names=["m", "r"])
y_pred = est.predict(X_test)          # ValueError if fit() was not called

print(est.expression())               # 3 × m^1.00 × r^-2.00
print(est.chosen_)                    # ['power_law[log_log_lstsq]']
print(est.score(X_test, y_test))      # R² on held-out data
```

Stored blocks are replayed exactly: power laws via their lstsq coefficients, trig blocks via `A·sin/cos(ω·x + φ) + C`, linear blocks via their Adam weights and normalisation constants. If `X_test` falls outside a block's domain (e.g. a value ≤ 0 for a power-law or log block), the affected predictions are `NaN` and a `RuntimeWarning` is emitted.

## Command line

Installing the package provides a `pantara` command (also available as `python -m pantara`):

```bash
# Discover the law and print a report
pantara run capteurs.csv --target force
pantara run capteurs.csv --target force --features displacement,velocity
pantara run capteurs.csv --target force --task regression

# Learn once, predict later
pantara fit historique.csv --target force --save modele.pkl
pantara predict mesures.csv --model modele.pkl --output predictions.csv
```

```
Pantara v7d — Découverte automatique de lois physiques
======================================================
Fichier     : capteurs.csv
Tâche       : régression (détectée automatiquement)
Points      : 500
Features    : displacement, velocity (2)
Cible       : force

Analyse en cours...

Loi détectée    : power_law
Expression      : 3.42 × displacement^1.00 × velocity^-2.00
R²              : 1.0000
Précision ±5%   : 100.0%
Temps           : 0.06s
Confiance       : haute

Blocs           : power_law[log_log_lstsq]
```

- **CSV loading:** separator (`,` `;` tab) and encoding (UTF-8, Windows-1252, Latin-1) are detected automatically; decimal commas are accepted in `;`-separated files. Column names may be given with or without their unit, and case or accents don't matter (`"Force (N)"`, `Force`, `force`).
- **Cleaning:** rows with missing values are dropped; non-numeric columns are ignored unless they are the target. Warnings are printed below 30 points or above 8 features.
- **Task detection:** a non-numeric target, ≤ 10 distinct values, or fewer than 5 % distinct values means classification. Classification is not implemented yet, so the command says so and exits. Use `--task regression` to override.
- **predict** writes the input rows plus a `<target>_pred` column (to stdout by default). If the target column is present, it also reports the held-out R².
- A saved model is a pickle of the fitted `PantaraRegressor`. Only load model files you trust.

## SRBench results

Evaluated on **SRBench** (La Cava et al. 2021) — 119 Feynman + 14 Strogatz datasets from PMLB, code frozen before evaluation (no post-hoc tuning).

| Metric | Value |
|---|---|
| Datasets evaluated | 132 / 133 |
| R² ≥ 0.95 | **76 / 132 (58%)** |
| R² ≥ 0.90 | **86 / 132 (65%)** |
| R² < 0 (complete failure) | 20 / 132 (15%) |
| R² mean | 0.641 |
| R² median | **0.970** |
| Precision ±5% mean | 53.4% |
| Mean time per dataset | 1.3 s |

**Distribution is strongly bimodal**: Pantara either solves the equation almost perfectly (R² ≥ 0.97) or fails completely (R² < 0). This reflects its deterministic structure — when the true law matches one of its 8 families, recovery is near-perfect; otherwise, the pipeline produces a partial fit.

**Strengths:** Pure power laws (`y = C · ∏ xᵢ^nᵢ`) are recovered exactly in O(N) via lstsq — 35+ Feynman datasets at R² = 1.000 with prediction time < 0.2 s.

**Limitations:** Functions requiring exponentials with additive arguments (`e^(ax+b)`), implicit equations, differential relationships (Strogatz), or more than 3-variable products are not covered by the current 8-family dictionary.

## Architecture

```
SetEncoder (point-wise MLP + mean/max pooling)
    ↓  256-dim encoding of (X, residual, y) subsampled to 64 points
Concatenate with 56-D statistical state vector
    ↓
3-layer classifier → 9 classes (8 families + STOP)
```

Trained on 15 000 synthetic episodes (300 points each), 22 law types, class-balanced up to 3500 examples per family.

## Repository structure

```
pantara/
├── pantara/
│   ├── __init__.py     ← public API
│   ├── core.py         ← full pipeline (physiqai_agent_v7d.py) + PantaraRegressor
│   ├── cli.py          ← `pantara` command line
│   └── model.pt        ← pre-trained oracle weights (532 KB)
├── tests/              ← pytest suite (python -m pytest tests)
├── setup.py
├── requirements.txt
└── README.md
```

## Citation

If you use Pantara in your research, please cite:

```bibtex
@software{pantara2026,
  author  = {Yapock22},
  title   = {Pantara: Physics-Aware Neural-network Tracking And Regression Analysis},
  year    = {2026},
  url     = {https://github.com/Yapock22/pantara},
  version = {0.7.4}
}
```

## License

MIT
