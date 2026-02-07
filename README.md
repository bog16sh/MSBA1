# MSBA1 - Home Credit Data Preparation Script

## What This Script Does

The `data_preparation.R` script is a **comprehensive, modular data preprocessing pipeline** for the Home Credit Default Risk dataset. It transforms raw application and transactional data into clean, model-ready datasets while preventing data leakage and ensuring train/test consistency.

### Key Problems It Solves:
- **Data Quality Issues**: Fixes sentinel values, inconsistent encodings, and near-zero variance variables
- **Missing Data**: Creates meaningful missing indicators and handles imputation consistently  
- **Feature Engineering**: Converts raw features into predictive variables (age, financial ratios, etc.)
- **Data Leakage Prevention**: Ensures all preprocessing parameters come from training data only
- **Model Compatibility**: Creates optimized datasets for both tree-based and linear models

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