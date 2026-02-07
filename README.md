# MSBA1 - Home Credit Data Preparation Pipeline

## Overview

This is my **Home Credit Default Risk** project for MSBA1. Complete, modular data preparation pipeline for Home Credit default risk prediction. All functions are reusable, well-documented, and prevent data leakage.

## 🎯 Key Features

- ✅ **Modular Functions**: 30+ reusable functions that work on both train and test data
- ✅ **Well Documented**: Every function has `@param`, `@return`, and `@examples` 
- ✅ **Zero Data Leakage**: All parameters computed from training data only
- ✅ **Production Ready**: Saved parameters for consistent deployment
- ✅ **Validated**: Automatic consistency checks prevent errors

## 🚀 Quick Start

```r
# Load the pipeline
source("data_preparation.R")

# Run complete preprocessing
result <- run_data_preparation_pipeline(train_data, test_data, save_outputs = TRUE)

# Get model-ready datasets
train_for_xgboost <- result$train_tree      # Keeps NAs + indicators
test_for_xgboost <- result$test_tree
train_for_logistic <- result$train_linear   # Fully imputed  
test_for_logistic <- result$test_linear

# Verify success
if (result$pipeline_success) {
  cat("✅ Pipeline completed successfully!")
}
```

## 📋 What the Pipeline Does

### 1. Data Quality Fixes
- **DAYS_EMPLOYED**: Replace 365243 sentinel → NA + indicator
- **FLAG columns**: Normalize Y/N to consistent 0/1 encoding
- **Near-zero variance**: Identify variables with >99.5% same value

### 2. Feature Engineering  
- **DAYS → YEARS**: Convert negative days to positive years (`AGE_YEARS`)
- **Financial ratios**: DTI, PTI, LTV, income per person, credit terms
- **EXT_SOURCE**: Aggregate 3 external scores (mean, count, max, min)

### 3. Missing Data Intelligence
- **Individual indicators**: `variable_missing` flags for 1-90% missing
- **Group patterns**: `EXT_SOURCE_all_missing`, `BUILDING_high_missing`
- **Overall metrics**: Total missing count and percentage per customer

### 4. Train-Only Parameter Computation 🔒
- **Medians/modes**: Computed from training data only
- **Binning breaks**: Quantiles from training distribution only  
- **Capping thresholds**: 99th percentiles from training only
- **Zero data leakage**: Test data never influences any parameter

### 5. Model-Specific Outputs
- **Tree models**: Keeps NAs (XGBoost handles them) + missing indicators
- **Linear models**: Full imputation + missing indicators

## 🔒 Data Leakage Prevention

```r
# ✅ CORRECT: Parameters from training only
params <- compute_imputation_parameters(train_data)  # Train only!
train_clean <- apply_imputation(train_data, params)   # Uses train median
test_clean <- apply_imputation(test_data, params)     # Same train median!

# ❌ WRONG: Combined data (data leakage!)
# params <- compute_imputation_parameters(rbind(train, test))
```

## 📁 Output Files

- `train_cleaned_tree.rds` / `test_cleaned_tree.rds` - For XGBoost/LightGBM
- `train_cleaned_linear.rds` / `test_cleaned_linear.rds` - For logistic regression  
- `preprocessing_parameters.rds` - All parameters for production
- `nzv_variables_review.csv` - Variables to consider dropping
- `preprocessing_summary.csv` - Dataset statistics

## 🔧 Individual Function Examples

All functions work identically on train and test data:

```r
# Same cleaning functions for both datasets
train_clean <- fix_days_employed_sentinel(train_data)
test_clean <- fix_days_employed_sentinel(test_data)  # Same function!

train_flags <- normalize_flag_columns(train_clean)
test_flags <- normalize_flag_columns(test_clean)    # Same function!

# Feature engineering - same functions
train_features <- convert_days_to_years(train_flags)
test_features <- convert_days_to_years(test_flags)  # Same function!

# Imputation with train-only parameters
params <- compute_imputation_parameters(train_features)  # TRAIN ONLY!
train_imputed <- apply_imputation(train_features, params)
test_imputed <- apply_imputation(test_features, params)  # Same parameters!
```

## 🚀 Production Deployment

```r
# Apply same preprocessing to new data
new_processed <- preprocess_new_data(
  new_data = new_applications,
  params_file = "preprocessing_parameters.rds", 
  model_type = "tree"  # or "linear"
)

# Make predictions
model <- readRDS("trained_model.rds")
predictions <- predict(model, new_processed)
```

## ✅ Validation Checks

The pipeline automatically validates:
- ✅ No data leakage (all parameters from training only)
- ✅ Column consistency (identical features except TARGET)  
- ✅ Imputation integrity (same medians/modes for train/test)
- ✅ Feature alignment (names, types, order match exactly)

## 📖 Function Documentation

Every function includes complete documentation:

```r
#' Fix DAYS_EMPLOYED sentinel value
#' 
#' Replaces the sentinel value (365243 = ~1000 years) with NA and creates
#' an indicator variable to preserve the information that the value was missing
#' 
#' @param data Data frame with DAYS_EMPLOYED column
#' @param sentinel_value Numeric sentinel value to replace (default: 365243)
#' @return Data frame with fixed DAYS_EMPLOYED and new indicator column
#' @examples
#' data_clean <- fix_days_employed_sentinel(data)
```

## 🎉 Ready to Use!

The pipeline is production-ready with:
- **30+ modular functions** - each reusable and testable
- **Complete documentation** - every function has examples
- **Zero data leakage** - training-only parameter computation  
- **Train/test consistency** - identical preprocessing guaranteed
- **Production deployment** - saved parameters for new data

**Start with**: `source("data_preparation.R")` and you're ready to go!