# Applied Machine Learning - Earthquake Aftershock Prediction

## Project Overview
This group project (Group 11) implements machine learning models to predict earthquake aftershocks in the South Japan Sea and North Philippine Sea Region.

## Dataset
- **Source**: United States Geological Survey (USGS)
- **Period**: 40 years (1983-2023)
- **Size**: 59,392 seismic events
- **Region**: South Japan Sea and North Philippine Sea

## Key Features
- Magnitude conversion and normalization
- Distance-based aftershock identification
- Sequence tracking
- Main earthquake correlation

## Models Implemented
1. **Linear Regression** (Ridge)
   - Test MSE: 0.483
   - R²: 0.219

2. **Random Forest Regressor**
   - Test MSE: 0.452
   - R²: 0.269

3. **Gradient Boosting**
   - Test MSE: 0.463
   - R²: 0.251

4. **Neural Network (MLP)**
   - Test MSE: 0.527
   - R²: 0.148

## Technologies
- Python
- Scikit-learn
- Keras/TensorFlow
- Pandas, NumPy
- Matplotlib

## Best Performance
Random Forest model achieved the best performance with the lowest MSE and highest R² score.
