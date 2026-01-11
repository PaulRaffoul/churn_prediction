# Customer Churn Prediction

A machine learning project designed to predict customer churn for a telecommunications company using the Telco Customer Churn dataset. The project implements a complete ML pipeline from data ingestion to model training and inference, with a focus on achieving high recall to minimize false negatives in churn prediction.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Pipeline](#data-pipeline)
- [Model Training](#model-training)
- [Prediction Pipeline](#prediction-pipeline)
- [Technical Details](#technical-details)
- [Requirements](#requirements)

## Project Overview

This project predicts whether a telecommunications customer will churn (leave the service) based on various customer attributes. The system is designed with modularity in mind, featuring separate components for data ingestion, transformation, model training, and prediction.

### Key Objectives

- **High Recall**: Optimized to achieve 90% recall to minimize missing actual churners (false negatives)
- **Automated Pipeline**: End-to-end ML pipeline from raw data to predictions
- **Modular Design**: Cleanly separated components for maintainability and scalability
- **Production Ready**: Includes preprocessing, model serialization, and inference pipeline

## Features

- **Data Ingestion**: Automated train-test split with stratification on target variable
- **Data Transformation**: Comprehensive preprocessing including:
  - Feature cleaning and type conversion
  - Numerical feature scaling (MinMax)
  - Categorical encoding (One-Hot, Label, Binary mapping)
  - Missing value handling
- **Model Training**: Multiple classification algorithms with hyperparameter tuning:
  - Decision Tree Classifier
  - Logistic Regression
  - Support Vector Classifier (SVC)
- **Threshold Optimization**: Custom threshold calculation targeting 90% recall
- **Prediction Pipeline**: Complete inference pipeline with preprocessing and threshold application
- **Logging & Exception Handling**: Custom logging and exception tracking throughout

## Project Structure

```
churn_prediction/
├── data/
│   └── Telco_customer_churn.xlsx       # Raw dataset
├── artifacts/
│   ├── data.xlsx                        # Processed raw data
│   ├── train.xlsx                       # Training dataset
│   ├── test.xlsx                        # Test dataset
│   ├── model.pkl                        # Trained model (serialized)
│   ├── preprocessor.pkl                 # Preprocessing pipeline (serialized)
│   └── threshold.json                   # Optimized threshold for high recall
├── src/
│   ├── components/
│   │   ├── data_ingestion.py           # Data loading and splitting
│   │   ├── data_transformation.py      # Feature engineering and preprocessing
│   │   └── model_trainer.py            # Model training and evaluation
│   ├── pipeline/
│   │   └── predict_pipeline.py         # Inference pipeline
│   ├── exception.py                     # Custom exception handling
│   ├── logger.py                        # Logging configuration
│   └── utils.py                         # Utility functions
├── data_processing.ipynb                # Data exploration notebook
├── model_experiment.ipynb               # Model experimentation notebook
├── requirements.txt                     # Project dependencies
├── setup.py                             # Package setup file
└── README.md                            # This file
```

## Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/PaulRaffoul/churn_prediction.git
   cd churn_prediction
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv churn_env
   source churn_env/bin/activate  # On Windows: churn_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - pandas, numpy: Data manipulation
   - scikit-learn: ML algorithms and preprocessing
   - xgboost: Gradient boosting (optional)
   - matplotlib, seaborn: Visualization
   - openpyxl: Excel file handling

## Usage

### Training the Model

To run the complete training pipeline from data ingestion to model training:

```python
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Initialize and run data ingestion
data_ingestion = DataIngestion()
train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

# Transform the data
data_transformation = DataTransformation()
train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
    train_data_path, test_data_path
)

# Train the model
model_trainer = ModelTrainer()
model_trainer.initiate_model_trainer(train_arr, test_arr)
```

Or simply run the data ingestion module directly:
```bash
python src/components/data_ingestion.py
```

### Making Predictions

Use the prediction pipeline for inference:

```python
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline

# Create prediction pipeline
predict_pipeline = PredictPipeline()

# Load your data (must match training feature format)
features = pd.read_excel('your_data.xlsx')

# Get predictions
predictions = predict_pipeline.predict(features)
```

## Data Pipeline

### 1. Data Ingestion (`src/components/data_ingestion.py`)

**Purpose**: Load raw data and create train-test splits

**Process**:
- Reads the Telco customer churn dataset from `data/Telco_customer_churn.xlsx`
- Creates an `artifacts/` directory for processed data
- Performs stratified train-test split (80-20) based on 'Churn Value'
- Saves raw data, training set, and test set as Excel files

**Key Features**:
- Stratification ensures balanced class distribution
- Automatic directory creation
- Comprehensive logging at each step

### 2. Data Transformation (`src/components/data_transformation.py`)

**Purpose**: Clean and preprocess data for model training

**Cleaning Steps** (`cleaner` method):
- Drops irrelevant columns: Country, State, City, CustomerID, CLTV, Zip Code, Latitude, Longitude, Lat Long, Count, Churn Label, Churn Reason, Churn Score
- Handles missing values in 'Total Charges' (replaces spaces with 0)
- Converts 'Total Charges' to float type

**Feature Engineering**:

1. **Numerical Features** (MinMax scaling):
   - Monthly Charges
   - Total Charges
   - Tenure Months

2. **Categorical Features - One-Hot Encoding**:
   - Multiple Lines, Internet Service, Online Security, Online Backup
   - Device Protection, Tech Support, Streaming TV, Streaming Movies
   - Contract, Payment Method

3. **Categorical Features - Binary Mapping (Yes/No → 1/0)**:
   - Phone Service, Paperless Billing, Partner, Dependents, Senior Citizen

4. **Gender Feature** (Female → 1, Male → 0)

**Output**:
- Transformed training and test arrays
- Serialized preprocessor object (`artifacts/preprocessor.pkl`)

## Model Training

### Algorithm Selection (`src/components/model_trainer.py`)

The system trains and evaluates multiple classification models:

1. **Decision Tree Classifier**
   - Hyperparameters: criterion (entropy/gini)

2. **Logistic Regression**
   - Hyperparameters: C (regularization strength)
   - Max iterations: 200

3. **Support Vector Classifier (SVC)**
   - Hyperparameters: C, kernel (linear/rbf/poly), gamma

### Model Selection Process

1. **Grid Search Cross-Validation**: 3-fold CV for hyperparameter tuning
2. **Evaluation Metric**: Recall score (prioritizes capturing churners)
3. **Best Model Selection**: Model with highest test recall score

### Threshold Optimization

The system calculates a custom decision threshold to achieve 90% recall:

```python
threshold = find_threshold(model, y_train, X_train, target_recall=0.9)
```

**Process**:
- Uses 3-fold cross-validation to get probability scores
- Computes precision-recall curve
- Finds threshold closest to target recall (90%)
- Saves threshold to `artifacts/threshold.json`

**Why High Recall?**
- In churn prediction, false negatives (missing actual churners) are costly
- 90% recall ensures we capture most customers at risk of churning
- Allows proactive retention strategies

### Model Artifacts

All trained components are saved in the `artifacts/` directory:
- `model.pkl`: Best trained model
- `preprocessor.pkl`: Fitted preprocessing pipeline
- `threshold.json`: Optimized decision threshold

## Prediction Pipeline

### PredictPipeline Class (`src/pipeline/predict_pipeline.py`)

**Purpose**: End-to-end inference on new data

**Process**:
1. Load serialized model and preprocessor
2. Load optimized threshold
3. Transform input features using preprocessor
4. Get decision function scores from model
5. Apply threshold to generate binary predictions
6. Return predictions (0 = No Churn, 1 = Churn)

**Usage Example**:
```python
from src.pipeline.predict_pipeline import PredictPipeline
import pandas as pd

# Prepare your data
new_customers = pd.DataFrame({
    'Monthly Charges': [29.85],
    'Total Charges': [29.85],
    'Tenure Months': [1],
    # ... include all required features
})

# Make predictions
pipeline = PredictPipeline()
predictions = pipeline.predict(new_customers)
print(predictions)  # [1] or [0]
```

## Technical Details

### Custom Exception Handling (`src/exception.py`)

Provides detailed error messages including:
- Script name where error occurred
- Line number
- Error message

```python
raise CustomException(e, sys)
```

### Logging System (`src/logger.py`)

- **Log Directory**: `logs/`
- **Log Format**: `[timestamp] line_number module - level - message`
- **Log File Naming**: `MM_DD_YYYY_HH_MM_SS.log`
- **Level**: INFO (captures all major operations)

### Utility Functions (`src/utils.py`)

| Function | Purpose |
|----------|---------|
| `save_object()` | Serialize objects using pickle |
| `load_object()` | Deserialize objects from pickle |
| `save_json()` | Save threshold values as JSON |
| `load_json()` | Load threshold from JSON |
| `map_yes_no_to_binary()` | Convert Yes/No to 1/0 |
| `map_gender()` | Convert Female/Male to 1/0 |
| `evaluate_models()` | Train and evaluate multiple models with GridSearchCV |
| `find_threshold()` | Calculate optimal threshold for target recall |

## Requirements

```
pandas          # Data manipulation and analysis
numpy           # Numerical computing
seaborn         # Statistical data visualization
matplotlib      # Plotting and visualization
scikit-learn    # Machine learning algorithms
xgboost         # Gradient boosting (optional)
openpyxl        # Excel file handling
```

Install all requirements:
```bash
pip install -r requirements.txt
```

## Dataset

**Source**: Telco Customer Churn Dataset (`data/Telco_customer_churn.xlsx`)

**Target Variable**: `Churn Value` (0 = No Churn, 1 = Churn)

**Features**: Customer demographics, services subscribed, contract details, billing information, and usage metrics.

## Model Performance

- **Optimization Goal**: 90% Recall
- **Evaluation Metric**: Recall Score (prioritizes catching churners)
- **Validation**: 3-fold cross-validation
- **Best Model**: Selected based on test set recall

## Future Enhancements

Potential improvements for this project:
- Web application for user-friendly predictions
- Additional models (XGBoost, Random Forest, Neural Networks)
- Feature importance analysis
- Cost-sensitive learning integration
- Real-time prediction API
- Model monitoring and retraining pipeline
- A/B testing framework for model comparison

## Author

**Paul Raffoul**
- Email: paul_raffoul24@hotmail.com

## License

This project is part of a machine learning portfolio demonstration.

---

**Note**: This project focuses on the technical implementation of a churn prediction system. In a production environment, additional considerations would include data privacy, model monitoring, API development, and continuous integration/deployment.
