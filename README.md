# Hyper-Parameter-Tuning-of-ML-Models
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from google.colab import files

# Load the dataset
file_path = files.upload()

filename = list(file_path.keys())[0]

# Load the dataset from the uploaded CSV file
df = pd.read_csv(filename)

# Drop the "Email No." column as it is an identifier
df.drop(columns=['Email No.'], inplace=True)

# Sample 300 rows from the dataset
df = df.sample(n=300, random_state=42)

# Initialize the scaler
scaler = StandardScaler()

# Get the column names excluding 'Prediction'
columns_to_scale = df.columns.difference(['Prediction'])

# Scale the features
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Display the first few rows of the preprocessed dataframe
print("First few rows of the preprocessed dataframe:")
print(df.head())

# Split the dataset into features and target variable
X = df.drop(columns=['Prediction'])
y = df['Prediction']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to display model performance
def display_performance(model_name, y_test, y_pred):
    print(f"{model_name} Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print()

# Initialize the model
gradient_boosting = GradientBoostingClassifier()

# Initial coarse hyperparameter tuning using Random Search
param_dist_coarse = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0]
}
random_search_coarse = RandomizedSearchCV(GradientBoostingClassifier(), param_distributions=param_dist_coarse, n_iter=10, cv=3, random_state=42)
random_search_coarse.fit(X_train, y_train)

print("Best Hyperparameters from Coarse Random Search for Gradient Boosting:")
print(random_search_coarse.best_params_)

# Refine the search space based on initial results
param_grid_refined = {
    'n_estimators': [random_search_coarse.best_params_['n_estimators'] - 50, random_search_coarse.best_params_['n_estimators'], random_search_coarse.best_params_['n_estimators'] + 50],
    'max_depth': [random_search_coarse.best_params_['max_depth'] - 1, random_search_coarse.best_params_['max_depth'], random_search_coarse.best_params_['max_depth'] + 1],
    'learning_rate': [max(0.01, random_search_coarse.best_params_['learning_rate'] - 0.01), random_search_coarse.best_params_['learning_rate'], random_search_coarse.best_params_['learning_rate'] + 0.01],
    'subsample': [max(0.6, random_search_coarse.best_params_['subsample'] - 0.1), random_search_coarse.best_params_['subsample'], min(1.0, random_search_coarse.best_params_['subsample'] + 0.1)]
}
grid_search_refined = GridSearchCV(GradientBoostingClassifier(), param_grid_refined, cv=3)
grid_search_refined.fit(X_train, y_train)

print("Best Hyperparameters from Refined Grid Search for Gradient Boosting:")
print(grid_search_refined.best_params_)

# Evaluate the performance of the Refined Grid Search tuned model
best_grid_model = grid_search_refined.best_estimator_
y_pred_grid = best_grid_model.predict(X_test)
display_performance("Refined Grid Search Tuned Gradient Boosting", y_test, y_pred_grid)

# Evaluate the performance of the Coarse Random Search tuned model
best_random_model = random_search_coarse.best_estimator_
y_pred_random = best_random_model.predict(X_test)
display_performance("Coarse Random Search Tuned Gradient Boosting", y_test, y_pred_random)

