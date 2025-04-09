import statistics
import math
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.tree import tree
from sklearn.linear_model import LinearRegression
import numpy as np

print("Welcome to the Capstone Project!")
print("This program will calculate the mean, median, mode, and standard deviation of a list of Monetization Failure.")

# Data
Monetization_Failure = [
    0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0
]

# Statistical calculations
X = statistics.mean(Monetization_Failure)
Y = statistics.median(Monetization_Failure)
Z = statistics.mode(Monetization_Failure)
A = statistics.stdev(Monetization_Failure)
B = statistics.variance(Monetization_Failure)

print("The mean of the Monetization Failure is: ", X)
print("The median of the Monetization Failure is: ", Y)
print("The mode of the Monetization Failure is: ", Z)
print("The standard deviation of the Monetization Failure is: ", A)
print("The variance of the Monetization Failure is: ", B)

# Plotting the scores
plt.hist(Monetization_Failure, bins=2, edgecolor='black', alpha=0.7)
plt.title("Frequency of Monetization Failure")
plt.xlabel("Monetization Failure")
plt.ylabel("Frequency")
plt.xticks([0, 1])  # Ensuring the x-axis only shows 0 and 1
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Decision Tree
# Preparing data for the decision tree
X_data = np.array([[score] for score in Monetization_Failure])  # Reshape scores into 2D array
y_data = np.array(Monetization_Failure)  # Labels are the same as scores for simplicity

# Creating and training the decision tree
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_data, y_data)

# Visualizing the decision tree
plt.figure(figsize=(10, 6))
tree.plot_tree(clf, filled=True, feature_names=["Monetization Failure"], class_names=["0", "1"])
plt.title("Decision Tree for Monetization Failure")
plt.show()

# Printing the decision tree rules
tree_rules = export_text(clf, feature_names=["Monetization Failure"])
print("Decision Tree Rules:")
print(tree_rules)

# Linear Regression
# Preparing data for linear regression
X_reg = np.array(range(len(Monetization_Failure))).reshape(-1, 1)  # Independent variable (index)
y_reg = np.array(Monetization_Failure)  # Dependent variable (scores)

# Creating and training the linear regression model
reg = LinearRegression()
reg.fit(X_reg, y_reg)

# Predicting values
y_pred = reg.predict(X_reg)

# Plotting the linear regression
plt.scatter(X_reg, y_reg, color='blue', label='Actual Data')
plt.plot(X_reg, y_pred, color='red', label='Regression Line')
plt.title("Linear Regression for Monetization Failure")
plt.xlabel("Index")
plt.ylabel("Monetization Failure")
plt.legend()
plt.grid()
plt.show()

print("The program has completed successfully.")
print("Thank you for using the Capstone Project!")