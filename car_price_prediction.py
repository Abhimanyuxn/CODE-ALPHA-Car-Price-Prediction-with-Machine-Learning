# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# STEP 1: Load Dataset
# =========================

df = pd.read_csv("car data.csv")

print("First 5 Records:")
print(df.head())

# =========================
# STEP 2: Data Preprocessing
# =========================

# Convert categorical columns into numerical values
df = pd.get_dummies(
    df,
    columns=['Fuel_Type', 'Selling_type', 'Transmission'],
    drop_first=True
)

# Remove Car_Name because it is not useful for prediction
df.drop('Car_Name', axis=1, inplace=True)

# =========================
# STEP 3: Define Features and Target
# =========================

# Selling_Price is the value we want to predict
X = df.drop('Selling_Price', axis=1)

# Target variable
y = df['Selling_Price']

# =========================
# STEP 4: Split Dataset
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# STEP 5: Train Regression Model
# =========================

model = LinearRegression()

model.fit(X_train, y_train)

# =========================
# STEP 6: Make Predictions
# =========================

y_pred = model.predict(X_test)

# =========================
# STEP 7: Evaluate Model
# =========================

print("\nR2 Score:")
print(r2_score(y_test, y_pred))

print("\nMean Absolute Error:")
print(mean_absolute_error(y_test, y_pred))

print("\nMean Squared Error:")
print(mean_squared_error(y_test, y_pred))

# =========================
# STEP 8: Predict New Car Price
# =========================

sample_car = X.iloc[[0]]

predicted_price = model.predict(sample_car)

print("\nPredicted Price:")
print(predicted_price[0])