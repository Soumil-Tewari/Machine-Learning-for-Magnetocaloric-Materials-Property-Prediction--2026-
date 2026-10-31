import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import joblib

df = pd.read_excel("magnetocaloric_materials_2000.xlsx", header=1)

df.drop(columns=['ID', 'Formula', 'Source','RCP (J kg⁻¹)'], inplace=True)

X = df.drop(columns='|ΔSmax| (J kg⁻¹ K⁻¹)')
y = df['|ΔSmax| (J kg⁻¹ K⁻¹)']


num_cols = ['Curie Temp TC (K)', 'Applied Field (T)']
cat_cols = ['Material Class', 'Transition Order', 'Refrigeration Range']


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr_pipe = Pipeline([('pre', preprocessor), ('model', LinearRegression())])
ridge_pipe = Pipeline([('pre', preprocessor), ('model', Ridge(random_state=42))])
lasso_pipe = Pipeline([('pre', preprocessor), ('model', Lasso(random_state=42, max_iter=5000))])
rf_pipe = Pipeline([('pre', preprocessor), ('model', RandomForestRegressor(random_state=42))])
xgb_pipe = Pipeline([('pre', preprocessor), ('model', XGBRegressor(objective='reg:squarederror', random_state=42, verbosity=0))])


param_dist_rf = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 5, 10],
    'model__min_samples_split': [2, 5, 10]
}
rs_rf = RandomizedSearchCV(rf_pipe, param_dist_rf, cv=5, n_iter=10,
                           scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1)
rs_rf.fit(X_train, y_train)
print("RF best params:", rs_rf.best_params_)


param_dist_xgb = {
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.01, 0.1],
    'model__max_depth': [3, 5, 10]
}
rs_xgb = RandomizedSearchCV(xgb_pipe, param_dist_xgb, cv=5, n_iter=6,
                            scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1)
rs_xgb.fit(X_train, y_train)
print("XGB best params:", rs_xgb.best_params_)


best_rf = rs_rf.best_estimator_
best_xgb = rs_xgb.best_estimator_

best_rf.fit(X_train, y_train)
best_xgb.fit(X_train, y_train)

models = {
    'LinearRegression': lr_pipe.fit(X_train, y_train),
    'Ridge': ridge_pipe.fit(X_train, y_train),
    'Lasso': lasso_pipe.fit(X_train, y_train),
    'RandomForest': best_rf,
    'XGBoost': best_xgb
}


print("\nTest Set Performance:")
for name, model in models.items():
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name:14s}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")


perm_rf = permutation_importance(best_rf, X_test, y_test,
                                 n_repeats=5, random_state=42, n_jobs=1)

feature_names = (num_cols + 
                 list(best_rf.named_steps['pre'].named_transformers_['cat']
                      .named_steps['onehot'].get_feature_names_out(cat_cols)))
importances = perm_rf.importances_mean
top_idx = importances.argsort()[::-1][:3]
print("\nTop 3 Permutation Importances (RF):")
for idx in top_idx:
    print(f"{feature_names[idx]}: {importances[idx]:.4f}")


joblib.dump(best_rf, "mce_randomforest.pkl")


new_sample = pd.DataFrame([{
    'Curie Temp TC (K)': 300,
    'Applied Field (T)': 2,
    
    'Material Class': 'Intermetallic',
    'Transition Order': 'First Order',
    'Refrigeration Range': 'Near room-temp (280–330 K)'
}])
predicted = best_rf.predict(new_sample)[0]
print(f"\nExample Prediction for new sample: |ΔSmax| = {predicted:.3f} J/kg·K")
