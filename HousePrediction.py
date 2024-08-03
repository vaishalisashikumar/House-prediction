import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# Load dataset
file_path = 'HousePricePrediction.xlsx'
try:
    df = pd.read_excel(file_path)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Display basic info and check for missing values
print("Dataset Information:")
print(df.info())
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Drop rows with missing target values and fill missing categorical values
df = df.dropna(subset=['SalePrice'])
df = df.fillna({
    'MSZoning': df['MSZoning'].mode()[0],
    'Exterior1st': df['Exterior1st'].mode()[0],
    'BsmtFinSF2': df['BsmtFinSF2'].mean(),
    'TotalBsmtSF': df['TotalBsmtSF'].mean()
})
print("Missing values handled.")

# Define features and target
X = df[['MSZoning', 'LotConfig', 'BldgType', 'Exterior1st', 'LotArea', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFinSF2', 'TotalBsmtSF']]
y = df['SalePrice']
print("Features and target defined.")

# Define preprocessing for categorical and numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), ['LotArea', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFinSF2', 'TotalBsmtSF']),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), ['MSZoning', 'LotConfig', 'BldgType', 'Exterior1st'])
    ])
print("Preprocessor defined.")

# Define and train the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
print("Pipeline created.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("Data split into training and testing sets.")

# Fit the model
pipeline.fit(X_train, y_train)
print("Model trained.")

# Predict and evaluate the model
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.2f}")

# Plot Actual vs Predicted Prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.xlabel('Actual Sale Prices')
plt.ylabel('Predicted Sale Prices')
plt.title('Actual vs Predicted Sale Prices')
plt.grid(True)
plt.show()

# Function to get user input with explanations
def get_user_input():
    print("Please enter the details for the new property:")
    ms_zoning = input("MSZoning (e.g., RL for Residential Low Density, RM for Residential Medium Density): ")
    lot_config = input("LotConfig (e.g., Inside for interior lot, Corner for corner lot): ")
    bldg_type = input("BldgType (e.g., 1Fam for single-family home, 2fmCon for two-family conversion): ")
    exterior_1st = input("Exterior1st (e.g., VinylSd for Vinyl Siding, MetalSd for Metal Siding): ")
    lot_area = float(input("LotArea (numeric, size of the lot in square feet): "))
    overall_cond = int(input("OverallCond (numeric, condition of the property on a scale from 1 to 10): "))
    year_built = int(input("YearBuilt (numeric, year of construction): "))
    year_remod_add = int(input("YearRemodAdd (numeric, year of last remodeling): "))
    bsmt_fin_sf2 = float(input("BsmtFinSF2 (numeric, area of basement finished after 1950 in square feet): "))
    total_bsmt_sf = float(input("TotalBsmtSF (numeric, total basement area in square feet): "))

    user_input = {
        'MSZoning': ms_zoning,
        'LotConfig': lot_config,
        'BldgType': bldg_type,
        'Exterior1st': exterior_1st,
        'LotArea': lot_area,
        'OverallCond': overall_cond,
        'YearBuilt': year_built,
        'YearRemodAdd': year_remod_add,
        'BsmtFinSF2': bsmt_fin_sf2,
        'TotalBsmtSF': total_bsmt_sf
    }

    return pd.DataFrame([user_input])

# Get user input and make prediction
try:
    new_property_df = get_user_input()

    # Preprocess and predict
    new_property_pred = pipeline.predict(new_property_df)
    print(f"\nPredicted Price for the new property: ${new_property_pred[0]:,.2f}")
except Exception as e:
    print(f"An error occurred: {e}")
