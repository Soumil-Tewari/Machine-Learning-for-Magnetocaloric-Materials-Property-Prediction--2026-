# Machine Learning for Magnetocaloric Materials Property Prediction

A machine learning regression project for predicting the maximum magnetic entropy change (|ΔSmax|) of magnetocaloric materials using material characteristics and experimental parameters.

## Overview

Magnetocaloric materials exhibit a change in temperature or magnetic entropy when exposed to an external magnetic field. The maximum magnetic entropy change, |ΔSmax|, is an important property when evaluating materials for magnetic refrigeration applications.

This project develops a supervised machine learning workflow to estimate |ΔSmax| from material descriptors, Curie temperature, applied magnetic field, transition order, and refrigeration range.

Five regression algorithms are implemented and evaluated to compare their predictive performance.

## Objectives

- Develop a machine learning pipeline for predicting the maximum magnetic entropy change of magnetocaloric materials.
- Process numerical and categorical material descriptors using a unified preprocessing workflow.
- Compare linear and nonlinear regression algorithms.
- Optimise ensemble model hyperparameters using cross-validation.
- Evaluate predictions using standard regression metrics.
- Investigate feature contributions using permutation importance.
- Export the trained model for subsequent predictions.

## Dataset

The project uses the Excel dataset:

magnetocaloric_materials_2000.xlsx

The dataset is loaded with the second row as its header (header=1).

The following columns are removed during preprocessing:

- ID
- Formula
- Source
- RCP (J kg⁻¹)

The prediction target is:

|ΔSmax| (J kg⁻¹ K⁻¹)

### Input Features

1. Curie Temp TC (K) — Numerical
2. Applied Field (T) — Numerical
3. Material Class — Categorical
4. Transition Order — Categorical
5. Refrigeration Range — Categorical

### Target Variable

The model predicts:

|ΔSmax| (J kg⁻¹ K⁻¹)

The target represents the maximum magnetic entropy change.

## Methodology

### 1. Data Preprocessing

A ColumnTransformer combines separate preprocessing pipelines for numerical and categorical features.

Numerical features:
- Missing values are imputed using the median.
- Features are standardised using StandardScaler.

Categorical features:
- Missing values are imputed using the most frequent category.
- Categories are converted into numerical representations using OneHotEncoder.
- Unseen categories are handled using handle_unknown='ignore'.

The preprocessing operations are incorporated into Scikit-learn pipelines to ensure that transformations are fitted within each training fold during cross-validation.

### 2. Train-Test Split

The dataset is divided into:

- Training set: 80%
- Testing set: 20%

The split uses random_state=42 to make the experiment reproducible.

### 3. Regression Models

Five regression algorithms are implemented:

1. Linear Regression — Establishes a linear regression baseline.
2. Ridge Regression — Applies L2 regularisation.
3. Lasso Regression — Applies L1 regularisation.
4. Random Forest Regressor — Captures nonlinear relationships using an ensemble of decision trees.
5. XGBoost Regressor — Uses gradient-boosted decision trees for regression.

### 4. Hyperparameter Optimisation

Randomised hyperparameter search is performed for Random Forest and XGBoost.

Random Forest search space:

- n_estimators: 50, 100, 200
- max_depth: None, 5, 10
- min_samples_split: 2, 5, 10

Configuration:
- Cross-validation folds: 5
- Search iterations: 10
- Scoring metric: Negative Mean Absolute Error

XGBoost search space:

- n_estimators: 100, 200
- learning_rate: 0.01, 0.1
- max_depth: 3, 5, 10

Configuration:
- Cross-validation folds: 5
- Search iterations: 6
- Scoring metric: Negative Mean Absolute Error

The best-performing estimator from each search is subsequently fitted on the training data.

### 5. Model Evaluation

All five models are evaluated on the held-out test set.

The following metrics are calculated:

Mean Absolute Error (MAE)

MAE = (1/n) × Σ |yi - ŷi|

Measures the average absolute difference between actual and predicted values.

Root Mean Squared Error (RMSE)

RMSE = sqrt((1/n) × Σ (yi - ŷi)²)

Penalises larger prediction errors more heavily.

Coefficient of Determination (R²)

R² = 1 - [Σ (yi - ŷi)² / Σ (yi - ȳ)²]

Measures how much of the target variance is explained by the model.

### 6. Feature Importance

Permutation importance is calculated for the optimised Random Forest model using the held-out test set.

The procedure:

1. Evaluates the model's baseline predictive performance.
2. Randomly permutes individual input features.
3. Measures the resulting change in model performance.
4. Ranks features according to their mean importance across five repetitions.

The three highest-ranked transformed features are reported.

### 7. Model Export and Prediction

The trained Random Forest pipeline is exported using Joblib:

mce_randomforest.pkl

Because the exported object is the complete preprocessing and modelling pipeline, new input data can be passed directly to its predict() method.

An example prediction uses:

- Curie Temp TC (K): 300
- Applied Field (T): 2
- Material Class: Intermetallic
- Transition Order: First Order
- Refrigeration Range: Near room-temp (280–330 K)

The model returns an estimated |ΔSmax| value in J kg⁻¹ K⁻¹.

This is an illustrative inference example, not an experimentally verified material property.

## Technology Stack

- Programming Language: Python
- Data Processing: Pandas, NumPy
- Machine Learning: Scikit-learn
- Gradient Boosting: XGBoost
- Model Persistence: Joblib

## Installation

Clone the repository:

    git clone <repository-url>
    cd <repository-name>

Install the required dependencies:

    pip install pandas numpy scikit-learn xgboost openpyxl joblib

Place magnetocaloric_materials_2000.xlsx in the project directory.

Run the Python script:

    python main.py

The script will:

1. Load and preprocess the dataset.
2. Split the data into training and testing sets.
3. Tune Random Forest and XGBoost hyperparameters.
4. Train and evaluate all five regression models.
5. Calculate Random Forest permutation importance.
6. Export the trained Random Forest pipeline.
7. Generate an example prediction.

## Project Structure

    .
    ├── main.py
    ├── magnetocaloric_materials_2000.xlsx
    ├── mce_randomforest.pkl
    └── README.md

- main.py: Data preprocessing, model training, evaluation, feature importance, and inference.
- magnetocaloric_materials_2000.xlsx: Input dataset.
- mce_randomforest.pkl: Exported Random Forest pipeline.

## Reproducibility

The implementation uses fixed random seeds (random_state=42) for the train-test split, RandomizedSearchCV, and model initialisation where supported.

The reported evaluation metrics depend on the supplied dataset and execution environment.

## Limitations

- Model performance depends on the quality, coverage, and representativeness of the available dataset.
- A random train-test split does not necessarily establish generalisation to entirely unseen material families.
- Permutation importance indicates predictive dependence, not physical causality.
- Predictions are model estimates and require independent experimental or literature-based validation before being treated as material-property measurements.
- No experimental validation or external generalisation assessment is established by this implementation alone.

## Future Improvements

- Evaluate generalisation using material-family-based or group-based train-test splits.
- Compare additional regression algorithms and optimisation strategies.
- Investigate feature interactions and model interpretability.
- Assess prediction uncertainty.
- Validate predictions against independently sourced experimental measurements.

## Disclaimer

This project is intended for computational research and exploratory machine learning. Its predictions should not be interpreted as experimentally confirmed material properties or as a substitute for laboratory measurements.
