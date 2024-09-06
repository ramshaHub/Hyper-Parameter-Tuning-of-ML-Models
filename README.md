# Hyper-Parameter Tuning of Machine Learning Models

## Overview
This project demonstrates how to perform hyper-parameter tuning for machine learning models using two common search techniques: **RandomizedSearchCV** and **GridSearchCV**. The model used in this example is a **GradientBoostingClassifier** from the `scikit-learn` library, and the dataset is loaded from a CSV file.

## Project Structure
- **Load Dataset**: The dataset is uploaded and loaded from a CSV file using Google Colab's `files.upload()` function.
- **Preprocessing**: Standard scaling is applied to the features, and unnecessary columns such as identifiers (e.g., `Email No.`) are dropped.
- **Model Training**: The `GradientBoostingClassifier` is used for classification, and its hyper-parameters are fine-tuned using Randomized and Grid Search methods.
- **Performance Evaluation**: After tuning, the performance of the model is evaluated using metrics like accuracy, precision, recall, and F1-score.

## Dependencies
This project requires the following Python libraries:
- `pandas`: For data manipulation and analysis
- `scikit-learn`: For machine learning models, scaling, and evaluation metrics
- `google.colab`: To upload files when running in Google Colab

You can install the required packages using the following commands:
```bash
pip install pandas scikit-learn
```

## Dataset
The dataset used in this project is uploaded from a CSV file. The file contains various features that are scaled before training the model. The target variable for classification is stored in the `Prediction` column.

## Code Breakdown

1. **Loading the Dataset**:  
   The dataset is uploaded via Google Colab's file upload functionality. After loading, the `Email No.` column (an identifier) is dropped.

2. **Preprocessing**:  
   The features are scaled using `StandardScaler` to standardize the data. This is important when working with machine learning algorithms that are sensitive to feature scaling.

3. **Splitting the Dataset**:  
   The dataset is split into training and testing sets using an 80/20 ratio, ensuring that the model is trained on 80% of the data and tested on the remaining 20%.

4. **Randomized Search for Coarse Hyper-Parameter Tuning**:  
   In this step, we perform an initial broad hyper-parameter tuning using `RandomizedSearchCV`. This method randomly samples a set of hyper-parameter combinations from the provided parameter grid.

5. **Grid Search for Refined Hyper-Parameter Tuning**:  
   Based on the results from the randomized search, we narrow down the hyper-parameter search space and use `GridSearchCV` for a more refined hyper-parameter search.

6. **Performance Evaluation**:  
   After tuning, the best models from both search methods are evaluated using key performance metrics: accuracy, precision, recall, and F1-score. These metrics help compare the performance of the coarse and refined hyper-parameter tuned models.

## Model Evaluation Metrics
The performance of each tuned model is evaluated using:
- **Accuracy**: Proportion of correctly predicted instances.
- **Precision**: Proportion of positive predictions that are actually correct.
- **Recall**: Proportion of actual positives that are correctly identified.
- **F1 Score**: A weighted harmonic mean of precision and recall.

## Usage

1. **Upload your dataset**: 
   The dataset should be a CSV file that contains features and a target column named `Prediction`.
2. **Run the Code**: 
   Simply run the provided code in a Google Colab environment.
3. **Model Evaluation**: 
   The script will display the best hyper-parameters and performance metrics for both the coarse and refined hyper-parameter tuned models.

## Example Output

```plaintext
Best Hyperparameters from Coarse Random Search for Gradient Boosting:
{'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8}

Best Hyperparameters from Refined Grid Search for Gradient Boosting:
{'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.06, 'subsample': 0.9}

Refined Grid Search Tuned Gradient Boosting Performance:
Accuracy: 0.87
Precision: 0.88
Recall: 0.85
F1 Score: 0.86
```

## Conclusion
This project illustrates how to apply hyper-parameter tuning to improve the performance of a machine learning model. By using both Randomized Search and Grid Search, we can explore a broad range of hyper-parameters and then refine them to find the optimal configuration.
