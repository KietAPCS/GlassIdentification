# Glass Identification — EDA, Modeling

Comprehensive analysis and multi-model benchmarking on the UCI Glass Identification dataset. The project explores data cleaning, exploratory data analysis (EDA), handling class imbalance, and training a range of classifiers. Final results and saved models are included for reuse.


## Overview

- Goal: Predict the glass type using chemical composition and refractive index.
- Dataset: UCI Glass Identification (214 samples, 10 columns inc. label; no class 4 present).
- Notebook: `GlassIdentification.ipynb` contains the full workflow (EDA ➜ preprocessing ➜ modeling ➜ evaluation).
- Outputs:
	- `model_comparison_results.csv`: consolidated metrics for 8 models.
	- `saved_models/`: best tuned linear models serialized for reuse.


## Dataset

Source: UCI Machine Learning Repository — Glass Identification Data Set.

Features (per notebook):
- Id (identifier), RI (refractive index), Na, Mg, Al, Si, K, Ca, Ba, Fe
- Target: `Type` (classes 1–7; Class 4 is absent in this dataset)

Class distribution (from notebook):
- 1 Building Windows – Float: 70
- 2 Building Windows – Non-Float: 76
- 3 Vehicle Windows – Float: 17
- 4 Vehicle Windows – Non-Float: 0 (absent)
- 5 Containers: 13
- 6 Tableware: 9
- 7 Headlamps: 29

Data file: `data/GlassIdentification.data`

Note on paths: ensure the notebook reads the data from `data/GlassIdentification.data`. If a cell uses a parent `../data/...` path, adjust it to `data/GlassIdentification.data` to match this repository structure.


## Project structure

- `GlassIdentification.ipynb` — complete analysis and modeling workflow
- `model_comparison_results.csv` — summary metrics for all evaluated models
- `data/GlassIdentification.data` — raw dataset (with header names assigned in the notebook)
- `saved_models/`
	- `ElasticNet_best.pkl`
	- `Logistic_L1_best.pkl`
	- `Logistic_L2_best.pkl`


## Workflow summary

1) Data cleaning
- Load dataset and assign column names
- Inspect missing values (none) and duplicates (removed)

2) Exploratory Data Analysis (EDA)
- Descriptive statistics and class distribution plots
- Correlation heatmap, pair plots, and KDE distributions
- Notes on potential multicollinearity; PCA considered for exploration

3) Modeling and evaluation
- Handling class imbalance with SMOTE within pipelines (where applicable)
- Scalers explored (e.g., StandardScaler/MinMax/PowerTransformer) + estimator via `Pipeline`
- Stratified cross-validation (`StratifiedKFold`) and `GridSearchCV` for tuning
- Metrics: Accuracy, F1-Score, Precision, Recall (macro or weighted per notebook code)
- Confusion matrices and classification reports for selected models

4) Models evaluated
- Logistic Regression (L1, L2)
- ElasticNet (linear baseline)
- KNN
- SVM
- Random Forest
- XGBoost
- Neural Network (Keras/TensorFlow)


## Results (from `model_comparison_results.csv`)

Top performers on held-out evaluation:
- Random Forest — Accuracy 0.8605, F1 0.8324
- KNN — Accuracy 0.8519, F1 0.8079
- XGBoost — Accuracy 0.8372, F1 0.8249

Full summary:

| Model | Accuracy | F1-Score | Precision | Recall | Complexity | Interpretability | Training Speed | Overall Rank |
|---|---:|---:|---:|---:|---|---|---|---:|
| KNN | 0.8519 | 0.8079 | 0.8861 | 0.8181 | Low | Medium | Fast | 1 |
| Random Forest | 0.8605 | 0.8324 | 0.8448 | 0.8429 | Medium | Medium | Medium | 2 |
| XGBoost | 0.8372 | 0.8249 | 0.7827 | 0.9040 | High | Low | Slow | 3 |
| Logistic L2 | 0.5882 | 0.6885 | 0.7145 | 0.7554 | Low | High | Fast | 4 |
| Logistic L1 | 0.5882 | 0.6885 | 0.7145 | 0.7554 | Low | High | Fast | 5 |
| ElasticNet | 0.5882 | 0.6638 | 0.6649 | 0.7549 | Low | High | Fast | 6 |
| SVM | 0.7407 | 0.6664 | 0.8133 | 0.6258 | Medium | Low | Medium | 7 |
| Neural Network | 0.6296 | 0.6685 | 0.7063 | 0.6739 | High | Low | Slow | 8 |

Interpretation:
- Tree-based and instance-based methods (Random Forest, KNN) perform best on this dataset.
- Linear baselines (Logistic, ElasticNet) underperform due to non-linear class boundaries and imbalance.
- XGBoost trades training speed for strong recall.


## How to run

Prerequisites
- Python 3.9+ recommended
- Jupyter (Lab or Notebook)

Install dependencies (minimal set):
- pandas, numpy, seaborn, matplotlib
- scikit-learn, imbalanced-learn
- xgboost
- tensorflow (for the NN baseline)
- tqdm, scipy

Optional, using Git Bash on Windows:

```bash
# create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate

# install dependencies
pip install -U pip
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn xgboost tensorflow tqdm scipy jupyter

# launch Jupyter and run the notebook
jupyter lab
# or
jupyter notebook
```

Then open `GlassIdentification.ipynb` and run all cells. If needed, update the data path in the loading cell to `data/GlassIdentification.data`.


## Using the saved models

Serialized scikit-learn models are provided in `saved_models/` for quick reuse of tuned linear baselines. Depending on how they were saved in the notebook, they may be standalone estimators or full pipelines.

Example: load and predict

```python
import joblib
import numpy as np

# choose one of: ElasticNet_best.pkl, Logistic_L1_best.pkl, Logistic_L2_best.pkl
model = joblib.load("saved_models/Logistic_L2_best.pkl")

# X must follow the same feature order and preprocessing used during training
# e.g., [RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]
X_new = np.array([[1.517, 13.10, 3.60, 1.36, 72.80, 0.55, 8.60, 0.00, 0.00]])
pred = model.predict(X_new)
print(pred)
```

Notes
- If the pickle contains only an estimator, apply the same scaler/transformers from the notebook before calling `predict`.
- If the pickle contains a `Pipeline`, you can call `predict` directly.


## Reproducing results

1) Ensure the data path is correct (`data/GlassIdentification.data`).
2) Run the notebook end-to-end. It will generate plots and evaluation artifacts.
3) Re-export the comparison table if you modify the experiments and want to update `model_comparison_results.csv`.


## Next steps (ideas)

- Calibrated probabilities and class-weighted training for imbalanced classes
- Feature engineering and domain-driven transforms
- Model ensembling and stacking
- Systematic PCA/umap exploration for structure and leakage checks
- Robust cross-validation with stratification and group-aware splits (if applicable)


## Acknowledgments

- UCI Machine Learning Repository — Glass Identification Data Set
- scikit-learn, imbalanced-learn, XGBoost, TensorFlow/Keras, pandas, numpy, seaborn, matplotlib

