# Car Price Prediction Model

---

**Name:** **[Jain Prasannakumar]**

**Organization:** **Entri Elevate**

**Date:** **15-10-2026**

---

## 1. Overview of Problem Statement

The price of a car depends on many interacting factors: the make, body style, engine size, power, weight, fuel economy, and more. Buyers, sellers and dealers need a reasonable price estimate to negotiate fairly, but working one out by hand from a long list of specifications is hard. This project builds a regression model that predicts a car's price from its specifications.

## 2. Objective

To develop a machine learning model that predicts the price of a car from its technical and descriptive attributes, and to deploy it through a Flask web interface so that a user can enter specifications and get an estimated price with a likely range.

## 3. Data Description

- **Source:** `CarPrice_Assignment.csv` (Car Price Prediction dataset, imported cars sold in the US market; prices in US dollars)
- **Rows:** 205 cars
- **Columns:** 26 in the raw file. After cleaning, the model uses **24 input features** and one target.
  - **Identity** — `car_ID` (dropped), `CarName` (reduced to `brand`)
  - **Body and drivetrain** — `carbody`, `doornumber`, `drivewheel`, `enginelocation`
  - **Dimensions and weight** — `wheelbase`, `carlength`, `carwidth`, `carheight`, `curbweight`
  - **Engine** — `fueltype`, `aspiration`, `enginetype`, `cylindernumber`, `enginesize`, `fuelsystem`, `boreratio`, `stroke`, `compressionratio`
  - **Performance and economy** — `horsepower`, `peakrpm`, `citympg`, `highwaympg`
  - **Risk rating** — `symboling` (insurance risk, -2 safest to 3 riskiest)
- **Target:** `price` (log-transformed to `price_log` for modeling)

## 4. Data Collection

The dataset was imported directly from the provided CSV file. No scraping or API collection was required, because the data was already assembled and labeled.

## 5. Data Preprocessing - Data Cleaning

- **Missing values:** none found in any column (`df.isnull().sum()` is 0 for all), so no imputation was needed.
- **Duplicates:** none found (`df.duplicated().sum()` is 0).
- **Irrelevant columns:** `car_ID` is only a row number and was dropped, so the model cannot learn from it.
- **Brand extraction and spelling fixes:** the brand was taken as the first word of `CarName` and lower-cased. Misspellings were corrected: `maxda` to `mazda`, `porcshce` to `porsche`, `toyouta` to `toyota`, `vokswagen` and `vw` to `volkswagen`. This gives 22 clean brands. `CarName` was then dropped.
- **Outliers:** checked for every numeric column with the IQR method (1.5 x IQR). Only a few mild outliers were found (for example 8 values in `carwidth` and 3 in `wheelbase`). They were kept, because with only 205 rows each car is valuable information and tree-based models handle extreme values well.
- **Skewed data:** `price` is heavily right-skewed (mean about $13,277, with a few very expensive cars). A `log1p` transformation was applied to the target, and predictions are converted back with `expm1`.

## 6. Exploratory Data Analysis (EDA)

- Summary statistics (`describe`, `info`, `shape`, `dtypes`) were reviewed for all columns.
- Brand frequencies were checked with `value_counts`: Toyota (32), Nissan (18) and Mazda (17) are the most common, while Mercury has only 1 car.
- Outlier limits and counts were calculated for each numeric feature (Section 5).
- Visual EDA (distribution plots, correlation heatmap) is not included yet and is listed under Future Work.

## 7. Feature Engineering

- **Brand:** extracted from `CarName` and cleaned (Section 5).
- **Target engineering:** `price_log = log1p(price)`.
- **Encoding:** categorical columns are one-hot encoded with `OneHotEncoder(handle_unknown="ignore")`, inside the model pipeline.

## 8. Feature Selection

No separate feature selection step was applied. All 24 features are used, because the dataset is small and the final models (tree ensembles) handle less useful features well. Feature importance and a reduced feature set are listed under Future Work.

## 9. Split Data into Training and Testing Sets

80/20 train-test split (`train_test_split`, `random_state=42`): 164 training rows and 41 test rows. Because the test set is small, `model.py` also reports 5-fold cross-validation R² to give a steadier estimate.

## 10. Feature Scaling

`StandardScaler` is applied to the numeric features and `OneHotEncoder` to the categorical features. Both are bundled in a `ColumnTransformer` inside each model's `Pipeline`, so raw input can be passed directly at training and prediction time without any manual preprocessing.

## 11. Build the ML Model

Five regression algorithms were implemented and compared (test split, log-price scale):

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.9183 | 0.1129 | 0.1488 |
| SVR (RBF) | 0.8810 | 0.1371 | 0.1797 |
| Decision Tree Regressor | 0.8948 | 0.1340 | 0.1689 |
| Random Forest Regressor | 0.9480 | 0.0995 | 0.1187 |
| Gradient Boosting Regressor | 0.9361 | 0.1074 | 0.1316 |

> These notebook runs still included `car_ID` as a feature. Re-run them after dropping it for fully comparable numbers. The deployed `model.py` drops `car_ID`.

## 12. Model Evaluation

Regression metrics used: **MAE, MSE, RMSE, R² Score**, computed on the held-out test set against `price_log` for every model. Because the target is on a log scale, `model.py` also reports the MAE in dollars (`expm1` of the predictions) so the error is easy to interpret.

## 13. Hyperparameter Tuning

Random Forest was tuned with `GridSearchCV` (cv=5, scoring='r2', 108 parameter combinations, 540 fits) over `n_estimators`, `max_depth`, `min_samples_split` and `min_samples_leaf`.

- **Best parameters:** `n_estimators=300`, `max_depth=10`, `min_samples_split=2`, `min_samples_leaf=1`
- **Best cross-validated R²:** 0.8989

**Tuned result:**

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Random Forest (tuned) | 0.9516 | 0.0946 | 0.1145 |

Tuning gave a small gain over the default Random Forest (R² 0.9480 to 0.9516). Linear Regression has no hyperparameters to search, and the other models were left at their defaults in the notebook.

## 14. Save the Model

The final model is a scikit-learn `Pipeline` (preprocessing + `GradientBoostingRegressor`, 100 trees) saved with `pickle` as `model_pipeline.pkl`. `model_meta.json` is saved alongside it and holds the dropdown options, the typical (median) values used to pre-fill the form, the training price range, and the evaluation metrics. The Flask app reads both files at startup.

## 15. Test with Unseen Data

The model was evaluated on the 20% held-out test split (Sections 11 and 13) and with 5-fold cross-validation. No separate external dataset has been tested yet. The web app also warns the user when an entered value is outside the range seen in training.

## 16. Interpretation of Results (Conclusion)

The models explain most of the variation in car prices (R² of about 0.94 to 0.95 on the log-price scale). Tree ensembles (Random Forest and Gradient Boosting) performed best, ahead of Linear Regression, Decision Tree and SVR. Tuned Random Forest was the best model in the notebook, and Gradient Boosting is the model currently deployed in the app; switching is a one-line change (see Future Work).

Key limitations: the dataset has only 205 cars and describes old models, so estimates are not current market prices. Some brands have very few examples, and the model cannot predict reliably for specifications outside the training ranges. The estimate is best used as a guide, with the likely range shown beside it, rather than as an exact value.

## 17. Screenshot

screenshots/Screenshot 2026-10-15 122037.png

## 18. Future Work

- Deploy the tuned Random Forest instead of Gradient Boosting (it scored higher in the notebook). In `model.py` replace the model step with `RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=42)`.
- Tune Gradient Boosting, SVR and Decision Tree as well, and compare all models after dropping `car_ID`.
- Add visual EDA (price distribution, correlation heatmap, price by brand and body style).
- Apply feature selection and compare a reduced feature set against the full one.
- Collect more recent and larger data so estimates reflect current prices.
- Deploy the Flask application to a public host (for example Render, Railway or PythonAnywhere).

---

## Project Structure

```
├── model.py                      # Cleans data, trains and saves the final model
├── app.py                        # Flask application serving the page and predictions
├── CarPricePredictionModel.ipynb # Exploration, model comparison, tuning
├── CarPrice_Assignment.csv       # Dataset
├── model_pipeline.pkl            # Trained Pipeline (preprocessing + GradientBoostingRegressor)
├── model_meta.json               # Form options, typical values, metrics (used by app.py)
├── requirements.txt
├── static/
│   ├── style.css
│   ├── script.js
│   └── images/bg.png
└── templates/
    └── index.html                # Prediction form UI
```

## How to Run

```bash
pip install -r requirements.txt
python model.py     # trains the model, creates model_pipeline.pkl and model_meta.json
python app.py
```

Then open `http://127.0.0.1:5000` in your browser. Run `model.py` before `app.py`, and run it again whenever the data or the model changes.
