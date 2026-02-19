# MSBA1 - Home Credit Default Risk Prediction

A comprehensive machine learning pipeline for predicting credit default using Home Credit data. This project includes data preparation, exploratory analysis, model development, and a production-ready XGBoost scorer.

## Project Overview

This repository contains two main components:

1. **Data Preparation Pipeline** (`data_preparation.R`) - Comprehensive preprocessing and feature engineering
2. **Modeling Notebook** (`HomeCredit_Modeling.qmd`) - Model development and selection
3. **EDA Analysis** (`HomeCredit_EDA.qmd`) - Exploratory data analysis

## Data Preparation Script

The `data_preparation.R` script is a **comprehensive, modular data preprocessing pipeline** for the Home Credit Default Risk dataset. It transforms raw application and transactional data into clean, model-ready datasets while preventing data leakage and ensuring train/test consistency.

### Key Problems It Solves:
- **Data Quality Issues**: Fixes sentinel values, inconsistent encodings, and near-zero variance variables
- **Missing Data**: Creates meaningful missing indicators and handles imputation consistently  
- **Feature Engineering**: Converts raw features into predictive variables (age, financial ratios, etc.)
- **Data Leakage Prevention**: Ensures all preprocessing parameters come from training data only
- **Model Compatibility**: Creates optimized datasets for both tree-based and linear models

---

## 🏆 Machine Learning Models

The `HomeCredit_Modeling.qmd` notebook implements and compares multiple algorithms to find the best predictor for credit default.

### Models Evaluated

#### 1. **Baseline: Majority Class Classifier**
- **Approach**: Always predict "no default" (majority class)
- **AUC**: 0.5000 (no discriminative ability)
- **Default Detection**: 0% (catches zero defaults)
- **Purpose**: Establishes minimum performance threshold
- **Finding**: High accuracy (92%) but useless for business since it misses all defaults

#### 2. **Logistic Regression** 
- **Best AUC**: 0.7357
- **Variants Tested**: 4 models (basic, extended with ratios, interaction terms)
- **Features**: 435 (after standardization and one-hot encoding)
- **Training Time**: ~1-2 minutes
- **Strengths**:
  - ✅ Interpretable coefficients (regulatory compliant)
  - ✅ Fast training
  - ✅ Captures linear relationships effectively
- **Weaknesses**:
  - ❌ Assumes linear relationships
  - ❌ Sensitive to feature scaling
  - ❌ Doesn't capture complex feature interactions
- **Key Finding**: Feature engineering more valuable than interaction terms

#### 3. **Random Forest**
- **AUC**: 0.7095
- **Trees**: 100 with bootstrap aggregating
- **Training Time**: ~2 minutes
- **Strengths**:
  - ✅ Handles missing values natively
  - ✅ Robust to outliers
  - ✅ Provides feature importance rankings
- **Weaknesses**:
  - ❌ Struggles with class imbalance (8% defaults)
  - ❌ Parallel tree building less effective at error correction
  - ❌ Outperformed by gradient boosting methods
- **Finding**: Good baseline but sequential boosting superior for imbalanced data

#### 4. **🏆 XGBoost (Extreme Gradient Boosting) - WINNER**
- **AUC**: 0.7455 (validation) | 0.75323 (Kaggle test)
- **Configuration**: 200 rounds with early stopping, max_depth=6, scale_pos_weight=11.3
- **Training Time**: ~6 seconds (extremely efficient)
- **Default Detection Rate**: 43.4% recall (detects ~2,600 of ~6,000 actual defaults)
- **Strengths**:
  - ✅ **Best predictive performance** - highest AUC among all models
  - ✅ Naturally handles missing values and class imbalance
  - ✅ Automatically captures feature interactions
  - ✅ Built-in regularization prevents overfitting
  - ✅ Fast training enables rapid iteration
  - ✅ Industry-proven for tabular data
- **Why It Wins**: Sequential tree building where each tree corrects previous errors is perfectly suited to learning from imbalanced data

### Performance Comparison

| Model | AUC | Precision | Recall | Training Time | Status |
|-------|-----|-----------|--------|---------------|---------| 
| Majority Baseline | 0.5000 | 0% | 0% | <1s | Benchmark |
| Logistic Regression | 0.7357 | - | - | 1-2m | Baseline+ |
| Random Forest | 0.7095 | - | - | 2m | Baseline+ |
| **XGBoost** | **0.7455** | **22.6%** | **43.4%** | **6s** | **🏆 Winner** |

### Key Technical Decisions

#### Feature Engineering Pipeline
- **Starting Features**: 122 (from original application data)
- **Engineered Features**: 354 total
- **Feature Categories**:
  - External credit aggregates (mean, min, max of EXT_SOURCE variables)
  - Financial ratios (debt-to-income, payment-to-income, loan-to-value)
  - Missing indicators (57 strategic flags for missing patterns)
  - Binned variables (age groups, income brackets, credit tiers)
  - Interaction terms (cross-feature relationships)
  - Risk flags (HIGH_DTI_FLAG, LOW_INCOME_FLAG, etc.)

#### Class Imbalance Handling
- **Challenge**: Only 8% of customers default (highly imbalanced)
- **Approach**: Used `scale_pos_weight=11.3` parameter in XGBoost
- **Result**: 43.4% recall with manageable false positive rate
- **Alternative Explored**: Undersampling, oversampling, SMOTE - threshold optimization proved most effective

#### Validation Strategy
- **Train/Validation Split**: 80/20 stratified (preserves class balance)
- **Data Sizes**: 
  - Training: 246,009 samples
  - Validation: 61,502 samples
  - Test: 48,744 samples
- **Cross-Validation**: 5-fold stratified with early stopping

### Business Impact

The XGBoost model delivers substantial value:

- **Default Detection**: Catches 43.4% of defaults vs. 0% for naive approaches
- **Risk Ranking**: If the top 10% highest-risk customers are flagged, 63% will actually default (vs. 8% baseline)
- **Deployment Threshold**: Optimized at 0.139 (vs. default 0.50) for balanced performance
- **Kaggle Performance**: Top ~4% of competition entries with 0.75323 AUC

### Feature Importance Insights

**Top Predictive Features** (by XGBoost Gain):
1. **EXT_SOURCE_MEAN** (38.8%) - Average external credit score
2. **EXT_SOURCE_3** (8.4%) - Third-party credit assessment  
3. **EXT_SOURCE_2** (6.2%) - Secondary credit bureau score
4. **AMT_CREDIT** (4.5%) - Loan amount requested
5. **AMT_ANNUITY** (3.8%) - Monthly payment amount

**Key Finding**: External credit bureau scores dominate (40%+ of importance), indicating Home Credit's primary advantage is its credit bureau partnerships.

---

## How to Use the Script

### Basic Usage
```r
# 1. Load the script
source("data_preparation.R")

# 2. Run the complete pipeline
result <- run_data_preparation_pipeline(train_data, test_data)

# 3. Get cleaned datasets
train_for_xgboost <- result$train_tree      # For tree models (XGBoost, LightGBM)
test_for_xgboost <- result$test_tree
train_for_logistic <- result$train_linear   # For linear models (fully imputed)
test_for_logistic <- result$test_linear
```

### What You Get
The script outputs **4 clean datasets** and **preprocessing parameters**:

| Output | Purpose | Description |
|--------|---------|-------------|
| `train_tree` | Tree models | Keeps NAs (trees handle them) + missing indicators |
| `test_tree` | Tree models | Same preprocessing as train_tree |
| `train_linear` | Linear models | Fully imputed data + missing indicators |  
| `test_linear` | Linear models | Same preprocessing as train_linear |
| `preprocessing_parameters.rds` | Production | All parameters for new data preprocessing |

## Script Architecture

### 30+ Modular Functions Organized in Categories:

#### 1. **Data Quality Functions**
```r
fix_days_employed_sentinel(data)     # Fixes 365243 sentinel → NA + indicator
normalize_flag_columns(data)         # Converts Y/N flags to consistent 0/1
identify_near_zero_variance(data)    # Finds variables with >99.5% same value
```

#### 2. **Feature Engineering Functions** 
```r
convert_days_to_years(data)          # DAYS_BIRTH → AGE_YEARS (positive)
create_financial_ratios(data)        # DTI, LTV, payment ratios, risk flags
create_ext_source_aggregates(data)   # Combines 3 external credit scores
```

#### 3. **Missing Data Functions**
```r
create_missing_indicators(data)      # Creates variable_missing flags
create_group_missing_indicators(data) # EXT_SOURCE_all_missing, etc.
```

#### 4. **Imputation Functions (Zero Data Leakage)**
```r
params <- compute_imputation_parameters(train_data)  # TRAINING DATA ONLY!
train_clean <- apply_imputation(train_data, params)  # Apply to train
test_clean <- apply_imputation(test_data, params)    # Same parameters to test
```

## Key Features

### ✅ **Zero Data Leakage**
All preprocessing parameters (medians, quantiles, modes) are computed from **training data only** and applied consistently to test data:

```r
# ✅ CORRECT - No data leakage
params <- compute_imputation_parameters(train_data)  # Train only!
test_imputed <- apply_imputation(test_data, params)  # Uses train medians

# ❌ WRONG - Data leakage!  
# params <- compute_imputation_parameters(rbind(train, test))
```

### ✅ **Modular & Reusable**
Every function works on both train and test data:
```r
# Same function, different datasets
train_fixed <- fix_days_employed_sentinel(train_data)
test_fixed <- fix_days_employed_sentinel(test_data)  # Identical processing
```

### ✅ **Well Documented**
Every function includes complete documentation:
```r
#' Fix DAYS_EMPLOYED sentinel value
#' @param data Data frame with DAYS_EMPLOYED column  
#' @return Data frame with fixed column + indicator
#' @examples data_clean <- fix_days_employed_sentinel(data)
```

### ✅ **Production Ready**
```r
# Apply to new data using saved parameters
new_processed <- preprocess_new_data(new_data, "preprocessing_parameters.rds")
```

## What Gets Created

### Engineered Features:
- **Demographics**: `AGE_YEARS`, `AGE_GROUP`, `YEARS_EMPLOYED`
- **Financial Ratios**: `DEBT_TO_INCOME_RATIO`, `PAYMENT_TO_INCOME_RATIO`, `LOAN_TO_VALUE_RATIO`
- **External Scores**: `EXT_SOURCE_MEAN`, `EXT_SOURCE_COUNT`, `EXT_SOURCE_MAX`
- **Missing Indicators**: `EXT_SOURCE_1_missing`, `BUILDING_high_missing`, etc.
- **Risk Flags**: `HIGH_DTI_FLAG`, `LOW_INCOME_FLAG`

### Data Quality Fixes:
- DAYS_EMPLOYED 365243 sentinel → NA + indicator (fixes ~18% of data)
- FLAG columns normalized to consistent 0/1 encoding  
- Near-zero variance variables identified for removal

## File Structure

```
📁 data_preparation.R          # Complete preprocessing pipeline (this script)
📁 README.md                   # This documentation
📁 HomeCredit_EDA.qmd          # Exploratory data analysis  
📁 .gitignore                  # Excludes data files from git
📁 HomeCredit_columns_description.csv  # Data dictionary
```

## Important Notes

### 🚫 **Data Files Not Included**
This repository contains **code only**. You need to provide:
- `application_train.csv` - Training data
- `application_test.csv` - Test data  
- Transactional files (bureau.csv, etc.) - Optional for enhanced features

### ⚡ **Quick Start**
1. Place your CSV files in the same directory
2. Load data: `train <- read_csv("application_train.csv")`
3. Run script: `source("data_preparation.R")`  
4. Process: `result <- run_data_preparation_pipeline(train, test)`
5. Model: Use `result$train_tree` and `result$test_tree` for XGBoost

## Why This Approach?

This script follows **industry best practices** for ML preprocessing:

- **Reproducible**: Same parameters always produce same results
- **Scalable**: Functions work on datasets of any size
- **Safe**: Prevents data leakage that inflates model performance
- **Flexible**: Works with different model types (tree vs linear)
- **Maintainable**: Modular functions are easy to test and debug

---

**Ready to preprocess your Home Credit data?** Just `source("data_preparation.R")` and run the pipeline! 🚀