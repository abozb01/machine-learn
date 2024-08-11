import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def find_longest_consecutive_increases(series):
    """Find the longest consecutive increase sequences in a Pandas Series."""
    longest_increase = []
    current_increase = []

    for i in range(1, len(series)):
        if series[i] > series[i - 1]:
            current_increase.append(series[i - 1])
            if i == len(series) - 1:
                current_increase.append(series[i])
            continue
        else:
            if len(current_increase) > len(longest_increase):
                longest_increase = current_increase
            current_increase = []

    # Handle case if the last sequence is the longest
    if len(current_increase) > len(longest_increase):
        longest_increase = current_increase

    return longest_increase

# Example DataFrame with features and sales
data = {
    'Feature1': np.random.rand(50) * 100,  # Random feature
    'Feature2': np.random.rand(50) * 50,   # Another random feature
    'Sales': np.random.rand(50) * 1000     # Sales target variable
}
df = pd.DataFrame(data)

# Prepare data for training the model
X = df[['Feature1', 'Feature2']]  # Features
y = df['Sales']                   # Target variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Regressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Predict sales based on test features
y_pred = model.predict(X_test)

# Add predictions to the DataFrame
df_test = X_test.copy()
df_test['Predicted_Sales'] = y_pred

# Combine predictions with the original DataFrame for plotting
df_combined = df.copy()
df_combined.loc[df_test.index, 'Predicted_Sales'] = df_test['Predicted_Sales']

# Find the longest consecutive increases for original sales data
longest_increase = find_longest_consecutive_increases(df['Sales'])

# Plotting
plt.figure(figsize=(14, 8))
plt.plot(df['Sales'], label='Original Sales Data', marker='o', linestyle='-', color='blue')
plt.plot(range(len(longest_increase)), longest_increase, marker='o', linestyle='-', color='red', label='Longest Consecutive Increase')

# Plot predicted sales
plt.plot(df_combined['Predicted_Sales'], marker='x', linestyle='--', color='green', label='Predicted Sales')

plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Sales Data Analysis and Prediction')
plt.legend()
plt.grid(True)
plt.show()
