# Store Sales Prediction

## Overview

This project focuses on predicting daily store sales for up to 6 weeks in advance using machine learning and deep learning techniques. The primary goal is to help companies plan ahead by forecasting future sales with high accuracy.

The following steps were implemented in this project:

1. **Data Preprocessing**: Handling categorical data, extracting useful features from date columns, and scaling numeric features.
2. **Machine Learning Model Building**: Using Sklearn pipelines and Random Forest for regression tasks.
3. **Loss Function Selection**: Choosing an appropriate loss function to measure model performance.
4. **Post Prediction Analysis**: Feature importance exploration and confidence interval estimation.
5. **Model Serialization**: Saving the trained models with timestamps for daily predictions.
6. **Deep Learning Model (LSTM)**: Building a Long Short-Term Memory (LSTM) model for time series sales prediction.

## Table of Contents

- [Data Preprocessing](#data-preprocessing)
- [Model Building with Sklearn Pipelines](#model-building-with-sklearn-pipelines)
- [Loss Function Selection](#loss-function-selection)
- [Post Prediction Analysis](#post-prediction-analysis)
- [Model Serialization](#model-serialization)
- [Deep Learning Model with LSTM](#deep-learning-model-with-lstm)
- [How to Run](#how-to-run)
- [Requirements](#requirements)

## Data Preprocessing

In the preprocessing phase, several tasks were carried out:

- Categorical features (like `Store`, `DayOfWeek`, `StateHoliday`) were converted into numerical form using one-hot encoding.
- Missing values were handled by filling them with appropriate values (e.g., `0` or the median).
- Features were engineered from the date columns to include:
  - **Weekday vs. Weekend** indicator.
  - **Beginning, middle, and end of the month** labels.
  - **Days to/after holidays**.
  - Other useful features like `WeekOfYear`, `DayOfMonth`, etc.
- Numeric features were scaled using `StandardScaler`.

## Model Building with Sklearn Pipelines

A **Random Forest Regressor** was used as the initial machine learning model. Pipelines were utilized for preprocessing and model training to keep the code modular and reproducible. The pipeline consisted of:

- **Preprocessing steps**: Handling numeric and categorical features using `StandardScaler` and `OneHotEncoder`.
- **Modeling step**: Training a Random Forest Regressor to predict sales.

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100))
])
```
