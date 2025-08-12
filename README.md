# XWhiff: Expected Whiff Rate Prediction Model

## Overview

XWhiff is a machine learning model that predicts the probability of a swing-and-miss (whiff) outcome for individual pitches. The model analyzes pitch characteristics, movement patterns, and game situations to estimate whiff likelihood, providing valuable insights for pitcher development and strategic decision-making.

## What It Predicts

- **Target Variable**: Binary classification (0 = no whiff, 1 = whiff)
- **Output**: Probability score indicating likelihood of swing-and-miss
- **Definition**: Whiff = 1 when `PitchCall == 'StrikeSwinging'`, 0 for all other outcomes

## Key Features

The model analyzes comprehensive pitch characteristics:

### Pitch Metrics
- **Velocity**: Release speed (RelSpeed)
- **Movement**: Vertical break (InducedVertBreak), Horizontal break (HorzBreak)
- **Spin**: Spin rate and spin axis orientation
- **Location**: Release point (height, side, extension) and plate location
- **Approach**: Vertical and horizontal approach angles

### Game Context
- **Count**: Balls and strikes
- **Outs**: Inning situation
- **Handedness**: Pitcher throws and batter side
- **Pitch Type**: Tagged pitch classification

## Model Architecture

### Algorithm
- **Primary Model**: XGBoost (Gradient Boosting)
- **Optimization**: Optuna hyperparameter tuning

### Training Strategy
- **Temporal Split**: 2022-2023 data for training, 2024 for testing
- **Cross-Validation**: K-fold validation for robust performance estimation
- **Feature Engineering**: Comprehensive pitch characteristic analysis

## Data Pipeline

### 1. Data Collection (`xwhiff_explore.ipynb`)
- Sources: Private Trackman data from 2022-2024 seasons
- Filter: Division 1 college baseball only
- Cleaning: Remove missing values, standardize data types
- Target Creation: Binary whiff classification

### 2. Model Development (`xwhiff_predict.ipynb`)
- Feature Selection: Identify most predictive pitch characteristics
- Model Selection: Test modeling all pitch groups together vs modeling all the pitch groups seperately 
- Model Training: XGBoost with hyperparameter optimization
- Performance Evaluation: Cross-validation and temporal validation
- Model Comparison: Unified vs. pitch-type-specific models

### 3. Inference Analysis (`xwhiff_inference.ipynb`)
- Pitch Tunneling Analysis: Study pitch combination effectiveness
- Strategic Insights: Identify optimal difference in pitch velocity, Induced Vertical Break and Horizontal Break to add whiffs

## Data Sources

- **Private Trackman Database**: NCAA Division 1 baseball
- **Seasons**: 2022, 2023, 2024
- **Data Quality**: Verified and cleaned pitch tracking data
- **Scope**: Division 1 college baseball competition



## Usage

### Prerequisites

```bash
pip install pandas numpy xgboost optuna scikit-learn seaborn matplotlib
```

### Running the Analysis

1. **Exploration**: Execute `xwhiff_explore.ipynb` for data cleaning and analysis
2. **Modeling**: Run `xwhiff_predict.ipynb` to train and optimize the model
3. **Inference**: Use `xwhiff_inference.ipynb` for strategic analysis and insights

```

## Future Enhancements

- **Advanced Tunneling**: Use difference in velocity, IVB, HB with primary pitch (generally Fastball) as predictors to xwhiff to see how the model evaluates this difference with the expected whiff. Currently the goal of the inference section is to find correlations between these difference in velo, IVB, HB which would only find correlations if the pitchers/pitches with the highest expected whiff also happen to have the same/optimal velo/HB/IVB difference from their primary pitch
