# Delhi House Price Prediction

Regression models that estimate house prices across Delhi localities, built on a public Kaggle real-estate dataset (MagicBricks listings).

## Highlights

- **Dataset:** 1,259 listings, 11 features
- **Best model:** Random Forest Regressor, **R² = 0.85** on the held-out test set
- **Baseline:** Decision Tree Regressor (tuned with GridSearchCV), R² = 0.83
- **Stack:** Python, Pandas, NumPy, Matplotlib, scikit-learn

## Dataset

| Feature | Description |
|---|---|
| Area | House area in square feet |
| BHK | Number of bedrooms |
| Bathroom | Number of bathrooms |
| Furnishing | Furnished / Semi-Furnished / Unfurnished |
| Locality | Locality of the property |
| Parking | Number of parking spaces |
| Status | Ready to move or under construction |
| Transaction | New property or re-sale |
| Type | Builder Floor or Apartment |
| Per_Sqft | Price per square foot |
| **Price** | **Target: price in INR** |

## Approach

1. **Cleaning:** Checked missing values and duplicates, and removed outliers with a z-score filter (|z| < 3).
2. **EDA:** Explored price against area, BHK, locality, furnishing, transaction type and property type.
3. **Feature engineering:** Added `Area_Yards` from area in square feet.
4. **Preprocessing:** Label-encoded categorical columns and applied Min-Max scaling to Area, Price, Per_Sqft and Area_Yards.
5. **Modeling:** 80/20 train-test split (`random_state=42`); Decision Tree tuned with 5-fold GridSearchCV; Random Forest with default parameters.

## Results

| Model | Test R² | MAE | RMSE |
|---|---|---|---|
| Decision Tree Regressor | 0.829 | 0.054 | 0.079 |
| **Random Forest Regressor** | **0.850** | **0.045** | **0.074** |

*MAE and RMSE are on the Min-Max scaled price (0 to 1), not in rupees.*

## Key Insights

- Price is driven mainly by area, number of bedrooms and locality.
- Punjabi Bagh, Lajpat Nagar and Vasant Kunj are among the priciest localities.
- Buyers show a strong preference for new builder-floor properties, which suggests demand for customization and independence.

## Limitations and Next Steps

- `Per_Sqft` is derived from price, so it can leak information into the model. Removing it would give a more realistic estimate.
- The Random Forest's training R² (0.96) is well above its test R² (0.85), which suggests overfitting. Tuning it with cross-validation is the next step.
- Scaling is fitted before the train-test split. A scikit-learn Pipeline would keep test data fully separate.

## Repository

```
├── MagicBricks.csv                      # dataset
├── delhi_house_price_prediction.ipynb   # full analysis and models
├── notebook_export.pdf                  # PDF export of the notebook

```
