# Car Price Prediction Using Machine Learning

## Project Overview

This project uses Machine Learning to predict the selling price of used cars based on various features such as manufacturing year, present price, fuel type, transmission type, kilometers driven, and ownership details.

A Random Forest Regressor model is trained on historical car data to estimate the selling price of a vehicle.

---

## Dataset

The dataset contains information about used cars with the following attributes:

* Car_Name
* Year
* Selling_Price (Target Variable)
* Present_Price
* Driven_kms
* Fuel_Type
* Selling_type
* Transmission
* Owner

---

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Google Colab

---

## Machine Learning Algorithm

**Random Forest Regressor**

Random Forest Regression is an ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

---

## Project Workflow

### 1. Import Dataset

```python
df = pd.read_csv("car data.csv")
```

### 2. Data Preprocessing

* Remove missing values
* Convert categorical variables into numerical format using One-Hot Encoding

```python
df = df.dropna()
df = pd.get_dummies(df, drop_first=True)
```

### 3. Feature Selection

Input Features:

* Year
* Present_Price
* Driven_kms
* Fuel_Type
* Selling_type
* Transmission
* Owner

Target Variable:

* Selling_Price

```python
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]
```

### 4. Train-Test Split

The dataset is divided into training and testing sets.

```python
train_test_split(X, y, test_size=0.2)
```

### 5. Model Training

```python
model = RandomForestRegressor()
model.fit(X_train, y_train)
```

### 6. Prediction

```python
y_pred = model.predict(X_test)
```

### 7. Model Evaluation

The model performance is evaluated using the R² Score.

```python
print("R2 Score:", r2_score(y_test, y_pred))
```

---

## Results

### Model Accuracy

R² Score:

```text
0.9243
```

This indicates that the model explains approximately 92.43% of the variance in car selling prices, demonstrating strong predictive performance.

---

## Data Visualization

A scatter plot is used to compare:

* Actual Selling Prices
* Predicted Selling Prices

```python
plt.scatter(y_test, y_pred)
```

The closer the points are to a straight line, the better the model predictions.

---

## Output

* Predicts the selling price of used cars.
* Achieved an R² Score of approximately 0.92.
* Visual comparison between actual and predicted values.

---

## Future Enhancements

* Hyperparameter tuning for improved performance.
* Compare different regression algorithms.
* Deploy the model as a web application using Flask or Streamlit.
* Add feature importance analysis.

---

## Conclusion

This project demonstrates how Machine Learning can be applied to estimate used car prices accurately. By using Random Forest Regression and proper data preprocessing techniques, the model achieves high prediction accuracy and can assist buyers and sellers in making informed decisions.

---

## Author

**Rithiga**

Machine Learning Project using Python and Scikit-learn.

