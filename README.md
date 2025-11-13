# WineQuality_End_to_End_ML_MidTerm_Project
## Problem
Predict the quality (score 0-10) of red wine from physicochemical tests. This is a regression problem that demonstrates EDA, feature engineering, model selection, hyperparameter tuning, model serialization, containerized serving, and basic reproducibility.


## Badges

[![UCI Dataset](https://img.shields.io/badge/dataset-UCI%20Wine%20Quality-blue)](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
[![Python](https://img.shields.io/badge/python-3.10%2B-green)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## Table of contents

* Project overview
* Dataset
* Exploratory Data Analysis (EDA)
* Modeling
* Results & selected model
* How to run (local)
* API — serving predictions
* Docker
* Directory structure

---

## Project overview

**Problem.** Predict the quality of wine (score 0–10) from a set of physicochemical measurements such as acidity, residual sugar, alcohol content and density. This is a supervised regression problem.

**Goal.** Produce a small, well-documented pipeline that demonstrates data preparation, EDA, model training, hyperparameter tuning, model selection, serialization and deployment as a simple web service (FastAPI) packaged with Docker.

**Why this dataset?** The UCI Wine Quality dataset is compact, and clean. It allows exploration of feature relationships, quick model iteration and reproducible deployment within limited compute.

---

## Dataset

**Source:** UCI Machine Learning Repository — Wine Quality dataset.

* Red wine CSV: winequality-red.csv 

Each row contains physicochemical measurements and a quality score (0–10) assigned by experts.

**Columns (features):**

* fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
  free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol
* quality (target)

**Download:**

```
mkdir data
curl -o data/winequality-red.csv "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
```

---

## Exploratory Data Analysis (EDA)

Key checks and visualizations included in notebook.ipynb:

* Data overview: .info(), .describe() and missing value counts.
* Distribution of target (quality) — histogram and bar plot.
* Histograms of all features to inspect skewness and outliers.
* Correlation matrix and heatmap (to spot strongly correlated predictors).
* Scatter plots of top features correlated with quality.


---

## Modeling

We train and compare multiple models to follow good model selection practice:

1. Decision Tree Regressor — interpretable tree baseline.
2. Random Forest Regressor — ensemble model; main focus of tuning. Grid-search over n_estimators and max_depth.

**Validation strategy:**

* Split data into train / validation / test. Test set is held out (20% of data). Validation set is 20% of the remaining training data (i.e., train:val:test = 0.64:0.16:0.20).
* Use RMSE (root mean squared error) on validation for model selection. Report RMSE and R^2 on the final test set.

**Hyperparameter tuning:**

* GridSearchCV used for small grids (keeps compute reasonable)
* For Random Forest we also include a manual grid experiment over n_estimators (10..200) vs max_depth to produce RMSE vs trees plots.

---

## Results & selected model

* Models evaluated: Decision Tree, Random Forest.
* Primary metric: RMSE on validation; final evaluation on held-out test set.

**Final model selection**

* The Random Forest model (best n_estimators/max_depth found on validation) is retrained on train+val and evaluated on the test set. The trained model is saved as model.joblib.

**Artifacts produced:**

* model.joblib — saved model
* Plots: correlation heatmap, RMSE vs n_estimators, max_depth grid plot, feature importance bar chart

---

## How to run (local)

1. Setup

```
python -m venv .venv
source .venv/bin/activate      
pip install -r requirements.txt
```

2. Download dataset

```
mkdir data
curl -o data/winequality-red.csv "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
```

3. Run the notebook

Open notebook.ipynb and run all cells to reproduce EDA, experiments and plots.

4. Train & save final model (script)

```
python train.py --data data/winequality-red.csv --out model.joblib
```

5. Run the API (local)

```
uvicorn predict:app --host 0.0.0.0 --port 8000
# test with curl
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"fixed_acidity":7.4, "volatile_acidity":0.7, "citric_acid":0.0, "residual_sugar":1.9, "chlorides":0.076, "free_sulfur_dioxide":11.0, "total_sulfur_dioxide":34.0, "density":0.9978, "pH":3.51, "sulphates":0.56, "alcohol":9.4 }'
```

---

## API — Serving predictions (predict.py)

* Implemented with FastAPI and Pydantic for request validation.
* Endpoint GET / returns health status.
* Endpoint POST /predict accepts a JSON with the 11 physicochemical features and returns a JSON with predicted_quality (float).

Example request body:

```
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "citric_acid": 0.0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}
```

Response example:

```
{"predicted_quality": 5.0}
```

---

## Docker

Build and run the API in Docker.

```
docker build -t wine-ml:v1 .
docker run -p 8000:8000 wine-ml:v1
```

This image installs dependencies from requirements.txt, copies code, exposes port 8000 and runs uvicorn predict:app.

---

## Directory structure

```
├── data/
│   └── winequality-red.csv
├── notebook.ipynb
├── train.py
├── predict.py
├── model.joblib      # generated by train.py
├── requirements.txt
├── Dockerfile
└── README.md
```
