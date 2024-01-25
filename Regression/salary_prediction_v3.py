import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('salary-data.csv')

# Print information about null values in the dataset
print(f'Null values: {df.isnull().sum()}')

# Create a subset of the dataset for training
training_data = df.head(10)

# Select features (X) and target (y)
X = df.loc[:, ['Age', 'Years of Experience', 'Gender', 'Education Level', 'Job Title']]
y = df.loc[:, ['Salary']]

# Keep the original X for Linear Regression
X_org = X

# Create dummy variables for categorical features
ct = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(drop='first'), ['Gender', 'Education Level', 'Job Title'])
], remainder='passthrough')
X = ct.fit_transform(X)

# Split the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the data to be used for Linear regression before data scaling to get correct results
X_train_lr, X_test_lr, y_train_lr, y_test_lr = X_train, X_test, y_train, y_test

# Data scaling
scaler_x = StandardScaler(with_mean=False)
X_train = scaler_x.fit_transform(X_train)  # first time fit_transform then transform
X_test = scaler_x.transform(X_test)
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)

def randomForest(n_estimators):
    # Random Forest Regression model
    model_rf = RandomForestRegressor(n_estimators=n_estimators)
    model_rf.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = scaler_y.inverse_transform(model_rf.predict(X_test).reshape(-1, 1))
    
    model_name = 'Random Forest Regression'
    print(f'\n{model_name}:')

    # Plot the results for Random Forest Regression
    visualizeResults(y_pred, model_name)
        
    # Evaluate the performance of the machine learning model
    evaluateModel(y_pred)
    
    # Predict the salary of the new employee using the Random Forest model
    predicted_salary = scaler_y.inverse_transform(model_rf.predict(new_employee).reshape(-1, 1))

    # # Print the predicted salaries for the new employee
    print(f'Predicted salary for new employee: {predicted_salary[0][0]:.2f}')


def decisionTree(max_depth, min_samples_split):
    # Decision Tree Regression model
    model_name = 'Decision Tree Regression'
    print(f'\n{model_name}:')
    model_dt = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)
    model_dt.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = scaler_y.inverse_transform(model_dt.predict(X_test).reshape(-1, 1))

    # Plot the results for Decision Tree Regression model
    visualizeResults(y_pred, model_name)
    
    # Evaluate the performance of the machine learning model
    evaluateModel(y_pred)

    # Predict the salary of the new employee using the Decision Tree model
    predicted_salary = scaler_y.inverse_transform(model_dt.predict(new_employee).reshape(-1, 1))

    # # Print the predicted salaries for the new employee
    print(f'Predicted salary for new employee: {predicted_salary[0][0]:.2f}')

def linearRegression():
    # Linear Regression model
    model_name = 'Linear Regression'
    print(f'\n{model_name}:')
    model = LinearRegression()
    model.fit(X_train_lr, y_train_lr)
    
    # Predicting the Test set results
    y_pred = model.predict(X_test_lr)
    
    # Plot the results for Linear Regression
    visualizeResults(y_pred, model_name)
    
    # Evaluate the performance of the machine learning model
    evaluateModel(y_pred)
    
    # Predict the salary of the new employee using the Linear Regression model
    predicted_salary = model.predict(new_employee)

    # # Print the predicted salaries for the new employee
    print(f'Predicted salary for new employee: {predicted_salary[0][0]:.2f}')
    

def evaluateModel(y_pred):
    # Regression metrics    
    # Calculate R2
    r2 = r2_score(y_test, y_pred)
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f'r2 {r2}')
    print(f'mae {mae}')
    print(f'rmse {rmse}')
    
def visualizeResults(y_pred, model_name):
    # Convert the y_test DataFrame to a Series to match the data type of y_pred_lr  
    y_test_plt = y_test.squeeze()
    
    # Plot the results
    plt.scatter(y_test_plt, y_pred, color="red")
    plt.plot([min(y_test_plt), max(y_test_plt)], [min(y_test_plt), max(y_test_plt)], linestyle="--", color="green")
    plt.xlabel('Actual Salary')
    plt.ylabel('Predicted Salary')
    plt.suptitle(model_name)
    plt.title('Actual vs Predicted Salary')
    plt.show()

# Define a new employee 
new_employee = pd.DataFrame({
    'Age': [45],
    'Gender': ['Male'],
    'Education Level': ["Master's"],
    'Job Title': ['Director of Marketing'],
    'Years of Experience': [28]
})

# Create dummy variables for the new employee
new_employee = ct.transform(new_employee)

# Scale the new employee's data
new_employee = scaler_x.transform(new_employee)

# Evaluate Random Forest model
randomForest(100)

# Evaluate Decision Tree model
decisionTree(10, 50)

# Evaluate Linear Regression model
linearRegression()
