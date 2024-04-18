import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# Input: 2 arrays one containing features and another targets, model alpha, fit_intercept, solver
# Notice! Defaults are that of the model provided by sklearn
# Output: Ridge model
def create_ridge_model(X_train: np.ndarray, y_train: np.ndarray, 
                       in_alpha=1.0, in_fit_intercept=True, in_solver='auto') -> Ridge:

    linear_model = Ridge(alpha=in_alpha, fit_intercept=in_fit_intercept, solver=in_solver)
    linear_model.fit(X_train, y_train)

    return linear_model

# Input: 2 arrays one containing features and another targets, model max_depth, max_features, n_estimators, min_samples_split, min_samples_leaf
# Notice! Defaults are that of the model provided by sklearn
# Output: Random forest model
def create_rf_model(X_train: np.ndarray, y_train: np.ndarray, 
                    in_max_depth=None, in_max_features=1.0, in_n_estimators=100, in_min_samples_split=2, in_min_samples_leaf=1) -> RandomForestRegressor:

    rf_model = RandomForestRegressor(max_depth=in_max_depth, max_features=in_max_features, n_estimators=in_n_estimators, 
                                     min_samples_leaf=in_min_samples_leaf, min_samples_split=in_min_samples_split)
    rf_model.fit(X_train, y_train)

    return rf_model