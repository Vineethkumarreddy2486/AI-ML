import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
import joblib

#1 load dataset
houseData = fetch_california_housing()
rows = houseData.data
columnNames = houseData.feature_names

  #  define rows and columns in dataframe
df = pd.DataFrame(rows, columns=columnNames)
df['Price'] = houseData.target

df.head()

#2 EDA (Exploratory Data Analysis)
df.info()
df.describe()
df.isnull().sum() 


X = df.drop("Price", axis=1)      # removes price column
X.info()
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data shape :" , X_train.shape)
print("Testing data shape :" , X_test.shape)

print(" train data before scaling :",X_train[:3])
print(" test data before scaling :",X_test[:3])

#2 Apply Scaling ( StandardScalar for numerical features )
scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

print("after scaling train data",X_train_scaled[:5])
print("after scaling test data",X_test_scaled[:5])

#3 train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print("first 5 predictions:",y_pred[:5])


#4 Evaluate Model ( Checking performance of model )
mae = mean_absolute_error(y_test, y_pred)
print("mean absolute error is : ", mae)

mse = mean_squared_error(y_test, y_pred)
print("mean squared error is : ", mse)

r2 = r2_score(y_test, y_pred)
print("r squared score is : ", r2)


#5 Trying with advanced level model

rf_model = RandomForestRegressor(random_state = 42)    #no scaling needed
rf_model.fit(X_train, y_train) 
y_pred_rf = rf_model.predict(X_test) 
print("random forest r2 score", r2_score(y_test, y_pred_rf)) 

#6 feature importance or modeling
feature_importances = rf_model.feature_importances_

important_df = pd.DataFrame({
    "Feature" : X.columns,
    "Importance": feature_importances
}).sort_values(by='Importance', ascending=False)
print(important_df)

#7 vertical plotting between fetures and its importance

plt.figure(figsize=(10,6))
plt.bar(important_df.Feature, important_df.Importance)
plt.title("Feature Importance Graph - Random Forest")
plt.xlabel("Features")
plt.ylabel("Importance Values")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Residual Analysis for random forest
  #  histogram
residuals = y_test - y_pred_rf
sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()



#8 cross validation( Instead of one train_test_split, we validate multiple times)
kf = KFold(n_splits=3, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    rf_model, X, y, cv=kf, scoring="r2"
    )
print("Cross Validation Score :",cv_scores)
print("Average Of Cross Validation Score  :",cv_scores.mean())


# #9 HyperTuning Parameters
param_list = {
    "n_estimators" : randint(100, 500),
    "max_depth": randint(5, 50),
    "min_samples_split": randint(2,10),
    "min_samples_leaf" : randint(1,10)
}

random_search = RandomizedSearchCV(
    estimator = rf_model,
    param_distributions = param_list,
    n_iter=20,
    cv=kf,
    scoring="r2",
    n_jobs = 2,
    random_state=42
)
random_search.fit(X_train, y_train)
print("Best Parameters:", random_search.best_params_)
print("Best randomized search CV Score:", random_search.best_score_)

best_model = random_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print("Final Test R2 Score :", r2_score(y_test, y_pred_best))

comparison = pd.DataFrame({
    "Actual Values" : y_test[:5],
    "Predicted Values" : y_pred_rf[:5]
})
print(comparison)


# saving the model
joblib.dump(rf_model, "california_rf_model.p1")
print(" model saved successfully")



