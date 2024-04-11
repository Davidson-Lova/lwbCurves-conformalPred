# Construction of tensile curves via conformal prediction

A python package for constructing lower bound curves using
polynomial conformal ridge regression (or just ridge regression in general) for univariate inputs and outputs.

It was used to predict a median curve for tensile tests data (see **Construction of tensile curves via conformal prediction** (2024)).

## Data

Data has to formated in numpy arrays. It can handle only univariate inputs and outputs.

## Code description

### Python Version

Python 3.11.1

### How to use it

Examples are provided in the test folder:

+ test\test_custom_dataset.py - how the class "test_sample" can be used;
+ test\test_dataLoader.py - how the class "dataLoader" can be instantiated;
+ test\test_model.py - how the class "conformal_ridge_regression_1S" can be used;
+ test\test_method.py - how the package is used to predict a lower bound curve on a simulated data set.

### Tree

+ conformal_poly_ridge_reg_1S
  + data
    + data_prepare.py - format input and output numpy array into "dataLoader"
    + custom_dataset.py - generate a simulated data sample
  + model
    + conf_pred_poly_ridge_reg_1S.py - predict confidence regions
  + utils
    + utils.py - usefull function

+ test
  + test_custom_dataset.py
  + test_dataLoader.py
  + test_method.py
  + test_model.py

## Contributors

All the co-authors of the article **Construction of tensile curve via conformal prediction** (2024) have contributed to the developpement of this package.
