# Car Price Prediction using Machine Learning

## Overview

This project uses Machine Learning techniques to predict the selling price of a car based on various features such as manufacturing year, present price, kilometers driven, fuel type, transmission type, and ownership history.

The project demonstrates the complete machine learning workflow, including data preprocessing, feature engineering, model training, and performance evaluation.

## Dataset

The dataset contains information about different cars and their selling prices.

### Features

* Year
* Present_Price
* Driven_kms
* Fuel_Type
* Selling_type
* Transmission
* Owner

### Target Variable

* Selling_Price

## Technologies Used

* Python
* Pandas
* Scikit-Learn

## Machine Learning Algorithm

* Linear Regression

## Project Workflow

1. Load the dataset.
2. Perform data preprocessing.
3. Convert categorical variables into numerical values.
4. Split data into training and testing sets.
5. Train a Linear Regression model.
6. Predict car prices.
7. Evaluate model performance using regression metrics.

## Evaluation Metrics

* R² Score
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)

## How to Run

### Install Required Libraries

```bash
pip install pandas scikit-learn
```

### Run the Project

```bash
python car_price_prediction.py
```

## Project Structure

```text
Car-Price-Prediction/
│
├── car data.csv
├── car_price_prediction.py
└── README.md
```

## Applications

Car price prediction models are widely used by:

* Used car marketplaces
* Automobile dealerships
* Insurance companies
* Vehicle valuation platforms

## Conclusion

This project successfully predicts car selling prices using a Linear Regression model. It demonstrates how machine learning can be applied to real-world business problems involving price estimation and decision-making.
