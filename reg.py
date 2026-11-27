# linear reg


import csv
import matplotlib.pyplot as plt


# ============================================================
# 1. READ CSV
# ============================================================

years = []
population = []
co2 = []

with open("dataset/co2.csv",
          "r", encoding="utf-8") as file:

    reader = csv.DictReader(file)

    for row in reader:
        years.append(int(row["year"]))
        population.append(float(row["population"]))
        co2.append(float(row["co2"]))


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

train_x = []
train_y = []

test_x = []
test_y = []
test_years = []

for year, pop, value in zip(years, population, co2):

    if 1961 <= year <= 2018:
        train_x.append(pop)
        train_y.append(value)

    elif 2019 <= year <= 2024:
        test_x.append(pop)
        test_y.append(value)
        test_years.append(year)


# ============================================================
# 3. SCALE X
# ============================================================
# Population is very large, so scale it before gradient descent.

def mean(data):
    return sum(data) / len(data)


x_mean = mean(train_x)

x_std = (
    sum((x - x_mean) ** 2 for x in train_x)
    / len(train_x)
) ** 0.5


def scale(x):
    return (x - x_mean) / x_std


train_x = [scale(x) for x in train_x]
test_x = [scale(x) for x in test_x]


# ============================================================
# 4. LOSS
#
# Lasso:
#
# Loss = MSE + lambda * |weight|
# ============================================================

def loss(x, y, w, b, lam):

    total = 0

    for xi, yi in zip(x, y):

        prediction = w * xi + b

        total += (prediction - yi) ** 2

    mse = total / len(y)

    return mse + lam * abs(w)


# ============================================================
# 5. GRADIENT DESCENT
# ============================================================

def train(x, y, learning_rate, epochs, lam):

    w = 0.0
    b = 0.0

    history = []
    steps = []

    n = len(y)

    for epoch in range(1, epochs + 1):

        dw = 0
        db = 0

        # -----------------------------
        # Calculate gradients
        # -----------------------------

        for xi, yi in zip(x, y):

            prediction = w * xi + b

            error = prediction - yi

            dw += error * xi
            db += error

        dw = (2 / n) * dw
        db = (2 / n) * db


        # -----------------------------
        # Lasso gradient
        # -----------------------------

        if w > 0:
            sign = 1
        elif w < 0:
            sign = -1
        else:
            sign = 0

        dw += lam * sign


        # -----------------------------
        # Gradient descent update
        # -----------------------------

        w = w - learning_rate * dw
        b = b - learning_rate * db


        current_loss = loss(
            x, y, w, b, lam
        )

        history.append(current_loss)


        # Save a few steps
        if epoch in [1, 2, 3, 10, 100, 500, epochs]:

            steps.append(
                (epoch, w, b, current_loss)
            )


    return w, b, history, steps


# ============================================================
# 6. PREDICTION
# ============================================================

def predict(x, w, b):

    return [
        w * xi + b
        for xi in x
    ]


# ============================================================
# 7. METRICS
# ============================================================

def mse(actual, predicted):

    return sum(
        (y - p) ** 2
        for y, p in zip(actual, predicted)
    ) / len(actual)


def r2(actual, predicted):

    y_mean = mean(actual)

    sse = sum(
        (y - p) ** 2
        for y, p in zip(actual, predicted)
    )

    sst = sum(
        (y - y_mean) ** 2
        for y in actual
    )

    return 1 - sse / sst


# ============================================================
# 8. EXPERIMENT WITH DIFFERENT STEP SIZES
# ============================================================

learning_rates = [
    0.001,
    0.005,
    0.01,
    0.05
]

epochs = 1000
lam = 0.01


results = []


for lr in learning_rates:

    w, b, history, steps = train(
        train_x,
        train_y,
        lr,
        epochs,
        lam
    )

    train_predictions = predict(
        train_x, w, b
    )

    test_predictions = predict(
        test_x, w, b
    )


    results.append({
        "lr": lr,
        "w": w,
        "b": b,
        "history": history,
        "steps": steps,
        "train_mse": mse(
            train_y,
            train_predictions
        ),
        "train_r2": r2(
            train_y,
            train_predictions
        ),
        "test_mse": mse(
            test_y,
            test_predictions
        ),
        "test_r2": r2(
            test_y,
            test_predictions
        ),
        "test_predictions": test_predictions
    })


# ============================================================
# 9. PRINT METRICS
# ============================================================

print()
print("==========================================================")
print("       LINEAR REGRESSION + LASSO")
print("==========================================================")

print()

print("Training : 1961 - 2018")
print("Testing  : 2019 - 2024")

print()

print("Epochs =", epochs)
print("Lambda =", lam)

print()

print("----------------------------------------------------------")
print("LEARNING RATE COMPARISON")
print("----------------------------------------------------------")

print(
    f"{'Step Size':<12}"
    f"{'Weight':<15}"
    f"{'Bias':<15}"
    f"{'Train MSE':<15}"
    f"{'Train R2':<15}"
    f"{'Test MSE':<15}"
    f"{'Test R2':<15}"
)

for result in results:

    print(
        f"{result['lr']:<12}"
        f"{result['w']:<15.6f}"
        f"{result['b']:<15.6f}"
        f"{result['train_mse']:<15.4f}"
        f"{result['train_r2']:<15.4f}"
        f"{result['test_mse']:<15.4f}"
        f"{result['test_r2']:<15.4f}"
    )


# ============================================================
# 10. SHOW GRADIENT DESCENT STEPS
# ============================================================

for result in results:

    print()
    print(
        "----------------------------------------------------------"
    )

    print(
        "STEP SIZE =", result["lr"]
    )

    print(
        f"{'Epoch':<10}"
        f"{'Weight':<18}"
        f"{'Bias':<18}"
        f"{'Loss':<18}"
    )

    for epoch, w, b, current_loss in result["steps"]:

        print(
            f"{epoch:<10}"
            f"{w:<18.6f}"
            f"{b:<18.6f}"
            f"{current_loss:<18.6f}"
        )


# ============================================================
# 11. LOSS VS EPOCH
# ============================================================

plt.figure(figsize=(10, 6))

for result in results:

    plt.plot(
        result["history"],
        label=f"step size = {result['lr']}"
    )

plt.xlabel("Epoch")
plt.ylabel("Lasso Loss")

plt.title(
    "Lasso Loss vs Epoch for Different Step Sizes"
)

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 12. ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    test_years,
    test_y,
    marker="o",
    label="Actual"
)

# Show predictions for each step size

for result in results:

    plt.plot(
        test_years,
        result["test_predictions"],
        marker="x",
        label=f"step size = {result['lr']}"
    )

plt.xlabel("Year")
plt.ylabel("CO2 emissions")

plt.title(
    "Actual vs Predicted CO2"
)

plt.legend()
plt.grid(True)

plt.show()












import csv
import matplotlib.pyplot as plt


# ============================================================
# 1. READ CSV
# ============================================================

years = []
population = []
co2 = []

with open("dataset/co2.csv", "r", encoding="utf-8") as file:

    reader = csv.DictReader(file)

    for row in reader:

        years.append(int(row["year"]))
        population.append(float(row["population"]))
        co2.append(float(row["co2"]))


# ============================================================
# 2. SPLIT DATA
#
# Training: 1961 - 2018
# Testing : 2019 - 2024
# ============================================================

train_years = []
test_years = []

train_x = []
train_y = []

test_x = []
test_y = []


for year, pop, value in zip(years, population, co2):

    if 1961 <= year <= 2018:

        train_years.append(year)
        train_x.append(pop)
        train_y.append(value)

    elif 2019 <= year <= 2024:

        test_years.append(year)
        test_x.append(pop)
        test_y.append(value)


# ============================================================
# 3. MEAN
# ============================================================

def mean(data):

    total = 0

    for value in data:
        total += value

    return total / len(data)


# ============================================================
# 4. FIND WEIGHT AND BIAS
#
#             SUM((x-x_mean)(y-y_mean))
# weight =    ---------------------------
#                SUM((x-x_mean)^2)
#
# bias = y_mean - weight*x_mean
# ============================================================

x_mean = mean(train_x)
y_mean = mean(train_y)


numerator = 0
denominator = 0


for x, y in zip(train_x, train_y):

    x_deviation = x - x_mean
    y_deviation = y - y_mean

    numerator += x_deviation * y_deviation
    denominator += x_deviation ** 2


weight = numerator / denominator

bias = y_mean - weight * x_mean


# ============================================================
# 5. PREDICTION
#
# y = wx + b
# ============================================================

def predict(x):

    return weight * x + bias


# ============================================================
# 6. MSE
# ============================================================

def mse(actual, predicted):

    total_error = 0

    for y, prediction in zip(actual, predicted):

        error = prediction - y

        total_error += error ** 2

    return total_error / len(actual)


# ============================================================
# 7. R2
#
# R2 = 1 - SSE/SST
# ============================================================

def r2(actual, predicted):

    y_mean = mean(actual)

    sse = 0
    sst = 0

    for y, prediction in zip(actual, predicted):

        sse += (y - prediction) ** 2

        sst += (y - y_mean) ** 2

    return 1 - (sse / sst)


# ============================================================
# 8. TRAINING PREDICTIONS
# ============================================================

train_predictions = []

for x in train_x:

    train_predictions.append(
        predict(x)
    )


# ============================================================
# 9. TESTING PREDICTIONS
# ============================================================

test_predictions = []

for x in test_x:

    test_predictions.append(
        predict(x)
    )


# ============================================================
# 10. METRICS
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
# 11. PRINT RESULTS
# ============================================================

print()
print("==============================================")
print("       SINGLE VARIABLE LINEAR REGRESSION")
print("==============================================")

print()

print("Training : 1961 - 2018")
print("Testing  : 2019 - 2024")

print()

print("X = Population")
print("Y = CO2 emissions")

print()

print("Mean X =", x_mean)
print("Mean Y =", y_mean)

print()

print("Weight =", weight)
print("Bias   =", bias)

print()

print("Regression Equation:")

print(
    "CO2 =",
    weight,
    "* Population +",
    bias
)

print()

print("Training MSE =", training_mse)
print("Training R2  =", training_r2)

print()

print("Testing MSE  =", testing_mse)
print("Testing R2   =", testing_r2)

print()
print("==============================================")


# ============================================================
# 12. PLOT POPULATION VS CO2
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


# Regression line

minimum = min(population)
maximum = max(population)

line_x = []
line_y = []


for i in range(100):

    x = minimum + (
        i * (maximum - minimum) / 99
    )

    line_x.append(x)
    line_y.append(predict(x))


plt.plot(
    line_x,
    line_y,
    linewidth=2,
    label="Best fit line"
)


plt.xlabel("Population")
plt.ylabel("CO2 emissions (Mt)")

plt.title(
    "India: Population vs CO2 Emissions"
)

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 13. ACTUAL VS PREDICTED TEST DATA
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
    "India CO2: Actual vs Predicted"
)

plt.legend()
plt.grid(True)

plt.show()
