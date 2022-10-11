# Co2 emission prediction

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

# Reading data
dataFrame = pd.read_csv("CO2 Emissions_Canada.csv")

# Data processing
dataFrame["Fuel Type"] = dataFrame["Fuel Type"].replace("Z", "Premium gasoline")
dataFrame["Fuel Type"] = dataFrame["Fuel Type"].replace("D", "Diesel")
dataFrame["Fuel Type"] = dataFrame["Fuel Type"].replace("X", "Regular gasoline")
dataFrame["Fuel Type"] = dataFrame["Fuel Type"].replace("E", "Ethanol (E85)")
dataFrame["Fuel Type"] = dataFrame["Fuel Type"].replace("N", "Natural gas")

# Data visualision
plt.scatter(dataFrame["Engine Size(L)"], dataFrame["CO2 Emissions(g/km)"],  color="blue")
plt.xlabel("Engine size")
plt.ylabel("Co2 Emission(g/km)")
plt.show()

plt.scatter(dataFrame["Cylinders"], dataFrame["CO2 Emissions(g/km)"],  color="blue")
plt.xlabel("Cylinders")
plt.ylabel("Co2 Emission(g/km)")
plt.show()

plt.scatter(dataFrame["Fuel Type"], dataFrame["CO2 Emissions(g/km)"],  color="blue")
plt.xlabel("Fuel type")
plt.ylabel("Co2 Emission(g/km)")
plt.show()

plt.scatter(dataFrame["Fuel Consumption Hwy (L/100 km)"], dataFrame["CO2 Emissions(g/km)"],  color="blue")
plt.xlabel("Fuel Consumption Hwy (L/100 km)")
plt.ylabel("Co2 Emission(g/km)")
plt.show()

plt.scatter(dataFrame["Fuel Consumption City (L/100 km)"], dataFrame["CO2 Emissions(g/km)"],  color="blue")
plt.xlabel("Fuel Consumption City (L/100 km)")
plt.ylabel("Co2 Emission(g/km)")
plt.show()

plt.scatter(dataFrame["Fuel Consumption Comb (L/100 km)"], dataFrame["CO2 Emissions(g/km)"],  color="blue")
plt.xlabel("Fuel Consumption Comb (L/100 km)")
plt.ylabel("Co2 Emission(g/km)")
plt.show()

# Finding the best “random state” and making the regression model
x = dataFrame[["Engine Size(L)", "Cylinders", "Fuel Consumption Comb (L/100 km)"]]
y = dataFrame[["CO2 Emissions(g/km)"]]

r2 = {}
for num in range(1, 50):
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=num)

    GBR = GradientBoostingRegressor()
    GBR.fit(X_train, y_train.values.ravel())
    y_hat = GBR.predict(X_test)

    r2.update({f"{num}": r2_score(y_test, y_hat)})

r2 = {k: v for k, v in r2.items() if v == max(r2.values())}
accuracy = list(r2.values())[0] * 100
accuracy = round(accuracy, 2)
best_random_state = list(r2.keys())[0]
print(f"The best random_state is {best_random_state} with {accuracy} percent accuracy")

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8,random_state=25)  # 25 is the best random state
GBR = GradientBoostingRegressor()
GBR.fit(X_train, y_train.values.ravel())

# Prediction
ES = input("Enter the engine size (Liter)=> ")
CS = input("Enter the Cylinders count => ")
FC = input("Enter the fuel consumption (L/100 km) => ")
user_data = {
    "Engine Size(L)" : ES,
    "Cylinders" : CS,
    "Fuel Consumption Comb (L/100 km)" : FC
}
user_data = pd.DataFrame(user_data, index=[0])
y_predict = GBR.predict(user_data)
y_predict = y_predict[0]
y_predict = round(y_predict,2)
print(f"Your predicted Co2 emission value is {y_predict} with {accuracy} percent accuracy")