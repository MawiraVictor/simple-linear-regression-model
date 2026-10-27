
# Simple Linear Regression Model

A simple linear regression model models the relationship between a continuous target variable and explanatory features. This project trains a model to predict CO2 emissions based on engine size.

## Setup and Data Loading

First, `numpy`, `pandas`, `scikit-learn`, and `matplotlib` are installed and imported.

Then, the dataset is obtained using the following URL:
```python
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"


The data is read using `pandas`:
```python
df = pd.read_csv(url)
```

Using `df.sample(1000)`, we can verify that the data was loaded by selecting 1000 random values from the CSV dataset. Basic statistical summaries are also generated using `df.describe()`.

## Data Visualization

During visualization, a scatter plot was generated showing the relationship between fuel consumption combined (`FUELCONSUMPTION_COMB`) and emissions (`CO2EMISSIONS`):

<img width="447" height="243" alt="Scatter plot of Fuel Consumption vs Emissions" src="https://github.com/user-attachments/assets/f178f298-4446-4a47-836c-84f58db874ce" />

We then extracted the input feature (`ENGINESIZE`) and the target label (`CO2EMISSIONS`) into separate NumPy arrays.

## Data Splitting

The dataset is split into training and testing sets using `train_test_split` from `scikit-learn`. We allocate 80% of the data for training and 20% for testing to evaluate the model's performance on unseen data.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Model Training

A Linear Regression model is built using `scikit-learn`'s `LinearRegression` class. Because `X_train` is a 1-D array, it is reshaped to a 2-D array `(-1, 1)` before fitting the model.

```python
from sklearn import linear_model

# Create a model object
regressor = linear_model.LinearRegression()

# Train the model on the training data
regressor.fit(X_train.reshape(-1, 1), y_train)
```

After training, we can extract the model's coefficients and intercept:
```text
Coefficients:  38.99297872443403
Intercept:  126.28970217408735
```

## Model Output Visualization

Finally, we visualize the model's predictions against the training data. The red line represents the linear regression fit.

```python
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
```
<img width="548" height="311" alt="image" src="https://github.com/user-attachments/assets/7d2556b2-409a-4958-b856-2fe2a4bc003e" />
