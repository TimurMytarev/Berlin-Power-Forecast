"""
Berlin Power Forecast
---------------------
Weather-based hourly electrical load forecast.
Author: Mytarev Timur
Date: 2025
"""

        # Importing libraries #
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

        # Loading and preparing data #
df = pd.read_csv(r'C:\Users\user\Desktop\Berlin_power_forecast\Data\Berlin_load_weather_synthetic.csv')
print(df.head())
df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])

        # Visualization of load dynamics #
plt.figure(figsize=(10, 5))
plt.plot(df['utc_timestamp'], df['load_MW'])
plt.title("Hourly Power Load (Synthetic Berlin Data)")
plt.xlabel("Time")
plt.ylabel("Load (MW)")
plt.grid(True)
plt.show()
sns.pairplot(df[['load_MW', 'temperature_C', 'humidity', 'wind_speed']])
plt.show()
print(df.info())
print(df.describe())
print(df.isna().sum())

        # Correlation and distributions #
sns.histplot(df['load_MW'], bins=50, kde=True)
plt.title('Distribution of Power Load')
plt.show()

corr = df[['load_MW', 'temperature_C', 'humidity', 'wind_speed']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

        # Baseline model (Linear Regression) #
#X = df[['temperature_C', 'humidity', 'wind_speed']]
#y = df['load_MW']

        # Split data into training and testing sets #
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialization and training of a linear model #
#model = LinearRegression()
#model.fit(X_train, y_train)
        # Predict test data #
#y_pred = model.predict(X_test)

        # Evaluate model performance #
#mae = mean_absolute_error(y_test, y_pred)
#r2 = r2_score(y_test, y_pred)
#print("Estimation models:")
#print(f"Mean absolute error (MAE): {mae:.2f} MW")
#print(f"Coefficient of determination (R²): {r2:.3f}")

        # Visualization of predicted vs actual load #
#comparison = pd.DataFrame({
#    'Real_Load': y_test.values,
#    'Predicted_Load': y_pred
# }, index=y_test.index)

#comparison = comparison.sort_index()

#plt.figure(figsize=(12, 6))
#plt.plot(comparison.index, comparison['Real_Load'], label='Real values', color='blue')
#plt.plot(comparison.index, comparison['Predicted_Load'], label='Model forecast', color='orange', linestyle='--')
#plt.title('Comparison of forecast and actual load (Linear Regression)')
#plt.xlabel('Observations (temporary order)')
#plt.ylabel('Load (MW)')
#plt.legend()
#plt.grid(True)
#plt.show()

        # Adding time features #
df['hour'] = df['utc_timestamp'].dt.hour
df['dayofweek'] = df['utc_timestamp'].dt.dayofweek

        # Feature and target definition #
X = df[['temperature_C', 'humidity', 'wind_speed', 'hour', 'dayofweek']]
y = df['load_MW']

        # Data separation #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression

        # Building an ensemble of models #
estimators = [
    ('rf', RandomForestRegressor(n_estimators=150, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42))
]

stack_model = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression()
)
        # Model training #
stack_model.fit(X_train, y_train)

        # Ensemble prediction #
y_pred = stack_model.predict(X_test)

        # Model evaluation #
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n Ensemble model performance:")
print(f"Mean absolute error (MAE): {mae:.2f} MW")
print(f"Coefficient of determination (R²): {r2:.3f}")

        # Visualization of predictions #
comparison = pd.DataFrame({
    'Real_Load': y_test.values,
    'Predicted_Load': y_pred
}, index=y_test.index).sort_index()

plt.figure(figsize=(12,6))
plt.plot(comparison.index, comparison['Real_Load'], label='Real values', color='blue')
plt.plot(comparison.index, comparison['Predicted_Load'], label='Forecast (Ensemble)', color='orange', linestyle='--')
plt.title('Comparison of the ensemble forecast and actual values')
plt.xlabel('Observations (temporary order)')
plt.ylabel('Load (MW)')
plt.legend()
plt.grid(True)
plt.show()
