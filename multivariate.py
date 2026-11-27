"""# multivariate"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------------
# 1. Load dataset
# -----------------------------------

df = pd.read_csv("dataset/co2.csv")

# -----------------------------------
# 2. Features and target
# -----------------------------------

features = [
    "population",
    "coal_co2",
    "oil_co2",
    "gas_co2",
    "cement_co2"
]

X = df[features]
y = df["co2"]

# -----------------------------------
# 3. Split by year
# -----------------------------------

train = df[df["year"] <= 2018]
test = df[df["year"] > 2018]

X_train = train[features]
y_train = train["co2"]

X_test = test[features]
y_test = test["co2"]

# -----------------------------------
# 4. Create and train model
# -----------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

# -----------------------------------
# 5. Predict future/test values
# -----------------------------------

y_pred = model.predict(X_test)

# -----------------------------------
# 6. Print coefficients
# -----------------------------------

print("Intercept:", model.intercept_)

print("\nCoefficients:")

for feature, coefficient in zip(features, model.coef_):
    print(f"{feature}: {coefficient}")

# -----------------------------------
# 7. Evaluation
# -----------------------------------

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nR² =", r2)
print("MSE =", mse)

# -----------------------------------
# 8. Print actual vs predicted
# -----------------------------------

results = pd.DataFrame({
    "Year": test["year"],
    "Actual CO2": y_test,
    "Predicted CO2": y_pred
})

print("\nPredictions:")
print(results)

# -----------------------------------
# 9. Plot
# -----------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    test["year"],
    y_test,
    marker="o",
    label="Actual CO2"
)

plt.plot(
    test["year"],
    y_pred,
    marker="o",
    label="Predicted CO2"
)

plt.xlabel("Year")
plt.ylabel("CO2")
plt.title("Actual vs Predicted CO2")
plt.legend()
plt.grid(True)

plt.show()

"""**POLYNOMIAL REG **"""

import csv
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# ============================================================
# SINGLE VARIABLE POLYNOMIAL REGRESSION
#
# Model:
#
# CO2 = b + w1*x + w2*x^2
#
# x = population
# y = CO2 emissions
#
# Training: 1961 - 2018
# Testing : 2019 - 2024
# ============================================================


# ============================================================
# 1. READ CSV
# ============================================================

years = []
population = []
co2 = []

with open(
    "dataset/co2.csv",
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        years.append(int(row["year"]))
        population.append(float(row["population"]))
        co2.append(float(row["co2"]))


# ============================================================
# 2. TRAIN / TEST SPLIT
#
# We manually separate by year first because this is time-series
# data. We do NOT randomly mix future years into training.
# ============================================================

train_years = []
test_years = []

train_x = []
train_y = []

test_x = []
test_y = []


for year, pop, value in zip(
    years,
    population,
    co2
):

    if 1961 <= year <= 2018:

        train_years.append(year)
        train_x.append(pop)
        train_y.append(value)

    elif 2019 <= year <= 2024:

        test_years.append(year)
        test_x.append(pop)
        test_y.append(value)


# ============================================================
# 3. CONVERT DATA INTO FORMAT REQUIRED BY SCIKIT-LEARN
#
# sklearn expects X to be a 2D structure:
#
# [[population],
#  [population],
#  [population],
#  ...]
# ============================================================

X_train = []

for x in train_x:

    X_train.append([x])


X_test = []

for x in test_x:

    X_test.append([x])


# ============================================================
# 4. CREATE POLYNOMIAL FEATURES
#
# degree = 2
#
# population
#
# becomes:
#
# [1, population, population^2]
#
# ============================================================

degree = 2

polynomial = PolynomialFeatures(
    degree=degree
)


X_train_polynomial = polynomial.fit_transform(
    X_train
)


X_test_polynomial = polynomial.transform(
    X_test
)


# ============================================================
# 5. TRAIN LINEAR REGRESSION MODEL
#
# LinearRegression is used AFTER polynomial transformation.
#
# It learns:
#
# CO2 = b + w1*x + w2*x^2
# ============================================================

model = LinearRegression()

model.fit(
    X_train_polynomial,
    train_y
)


# ============================================================
# 6. GET PARAMETERS
# ============================================================

bias = model.intercept_

weight_1 = model.coef_[1]

weight_2 = model.coef_[2]


# ============================================================
# 7. PREDICTION FUNCTION
#
# This function allows us to manually give a population
# and get its predicted CO2.
# ============================================================

def predict(x):

    X = [[x]]

    X_polynomial = polynomial.transform(X)

    return model.predict(
        X_polynomial
    )[0]


# ============================================================
# 8. MEAN
# ============================================================

def mean(data):

    total = 0

    for value in data:

        total += value

    return total / len(data)


# ============================================================
# 9. MSE
# ============================================================

def mse(actual, predicted):

    total_error = 0

    for y, prediction in zip(
        actual,
        predicted
    ):

        error = prediction - y

        total_error += error ** 2

    return total_error / len(actual)


# ============================================================
# 10. R2
# ============================================================

def r2(actual, predicted):

    y_mean = mean(actual)

    sse = 0
    sst = 0

    for y, prediction in zip(
        actual,
        predicted
    ):

        sse += (
            y - prediction
        ) ** 2

        sst += (
            y - y_mean
        ) ** 2

    return 1 - (
        sse / sst
    )


# ============================================================
# 11. PREDICTIONS
# ============================================================

train_predictions = []

for x in train_x:

    train_predictions.append(
        predict(x)
    )


test_predictions = []

for x in test_x:

    test_predictions.append(
        predict(x)
    )


# ============================================================
# 12. METRICS
# ============================================================

training_mse = mse(
    train_y,
    train_predictions
)

testing_mse = mse(
    test_y,
    test_predictions
)

training_r2 = r2(
    train_y,
    train_predictions
)

testing_r2 = r2(
    test_y,
    test_predictions
)


# ============================================================
# 13. PRINT RESULTS
# ============================================================

print()
print("====================================================")
print("      SINGLE VARIABLE POLYNOMIAL REGRESSION")
print("====================================================")

print()

print("Training : 1961 - 2018")
print("Testing  : 2019 - 2024")

print()

print("X = Population")
print("Y = CO2 emissions")

print()

print("Degree =", degree)

print()

print("Bias =", bias)

print("Weight 1 =", weight_1)

print("Weight 2 =", weight_2)

print()

print("Equation:")

print(
    "CO2 =",
    bias,
    "+",
    weight_1,
    "* Population",
    "+",
    weight_2,
    "* Population^2"
)

print()

print("Training MSE =", training_mse)
print("Training R2  =", training_r2)

print()

print("Testing MSE  =", testing_mse)
print("Testing R2   =", testing_r2)

print()
print("====================================================")


# ============================================================
# 14. PLOT
# ============================================================

plt.figure(figsize=(10, 6))


plt.scatter(
    train_x,
    train_y,
    label="Training data"
)


plt.scatter(
    test_x,
    test_y,
    label="Testing data"
)


# ============================================================
# CREATE SMOOTH POLYNOMIAL CURVE
# ============================================================

minimum = min(population)
maximum = max(population)

line_x = []
line_y = []


for i in range(200):

    x = minimum + (
        i * (maximum - minimum) / 199
    )

    line_x.append(x)

    line_y.append(
        predict(x)
    )


plt.plot(
    line_x,
    line_y,
    linewidth=2,
    label="Degree-2 polynomial"
)


plt.xlabel("Population")

plt.ylabel("CO2 emissions (Mt)")

plt.title(
    "India: Population vs CO2 - Polynomial Regression"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 15. ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(10, 6))


plt.plot(
    test_years,
    test_y,
    marker="o",
    label="Actual"
)


plt.plot(
    test_years,
    test_predictions,
    marker="x",
    label="Predicted"
)


plt.xlabel("Year")

plt.ylabel("CO2 emissions (Mt)")

plt.title(
    "Polynomial Regression: Actual vs Predicted"
)

plt.legend()

plt.grid(True)

plt.show()


