
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print("Loading Dataset...")
df = pd.read_csv("crop_data.csv")
print("\nFirst 5 Records:")
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nChecking Missing Values...")
print(df.isnull().sum())
df.dropna(inplace=True)
print("\nMissing Values Removed")
le_state = LabelEncoder()
le_district = LabelEncoder()
le_season = LabelEncoder()
le_crop = LabelEncoder()

df["State"] = le_state.fit_transform(df["State"])
df["District"] = le_district.fit_transform(df["District"])
df["Season"] = le_season.fit_transform(df["Season"])
df["Crop"] = le_crop.fit_transform(df["Crop"])

plt.figure(figsize=(8,5))
plt.hist(df["Production"], bins=10)
plt.title("Crop Production Distribution")
plt.xlabel("Production")
plt.ylabel("Frequency")
plt.show()
X = df[["State", "District", "Season", "Crop", "Area"]]
y = df["Production"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel Training Completed")
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("\n========== MODEL PERFORMANCE ==========")
print("R2 Score:", round(r2, 2))
print("Mean Absolute Error:", round(mae, 2))
print("Mean Squared Error:", round(mse, 2))
sample = pd.DataFrame(
    [[1, 2, 1, 3, 500]],
    columns=[
        "State",
        "District",
        "Season",
        "Crop",
        "Area"
    ]
)
prediction = model.predict(sample)
print("\nPredicted Crop Production:",
      round(prediction[0], 2))
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Production")
plt.ylabel("Predicted Production")
plt.title("Actual vs Predicted Production")
plt.show()
print("\nProject Executed Successfully")