# Bank Customer Churn Prediction

This project aims to predict customer churn for a bank using machine learning. Customer churn, also known as customer attrition, is the phenomenon of customers discontinuing their relationship with a business. By predicting churn, the bank can take proactive measures to retain valuable customers.

## Project Overview

This project uses a Random Forest Classifier to predict customer churn based on various features such as demographics, account details, and transaction history. The model is trained and evaluated using a dataset of bank customers, and a Streamlit app is built to visualize the results interactively.

## Features

* **Data Preprocessing:** Handles missing values, performs feature engineering, and prepares the data for modeling.
* **Model Building:** Trains a Random Forest Classifier using optimized hyperparameters.
* **Model Evaluation:** Evaluates the model's performance using metrics like accuracy, precision, recall, and F1-score.
* **Visualization:** Provides interactive visualizations of the confusion matrix, feature importance, ROC curve, and SHAP summary plot using Streamlit.

## Getting Started

1. Clone the repository
2. Install the required libraries
3. Run the Streamlit app

## Dataset

The project uses a dataset containing information about bank customers. You can find the dataset in the `data` folder.

## Usage

The Streamlit app allows you to interact with the model and visualize its results. You can select different metrics from the sidebar to display the corresponding charts.

## Contributing

Contributions are welcome! Please feel free to open issues or pull requests.

## License

This project is licensed under the MIT License.
