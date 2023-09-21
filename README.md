# out_and_in
 ** out_and_in ** is a backward feature elimination followed with forwared features selection strategy. 

## Basic Idea
This function will itinerary remove one feature and from features, if the metrics has beed optimized, we will add it to dropped-features.
we will drop dropped-features after that repeat, and calculate a metrics based on remaining features, and add those dropped_fatures 
 back to the remaining features, if the performance has been increased, we will remove that features from dropped features. and finially,
it will return dropped features. 


## Instalation
```commandline
pip intall out_and_in
```

## Example
We will use the auto_mpg as demo for the selection process of out_and_in
auto_mpg include 80 features and 1460 entries. those features include 
numerical features and categorical features

```python
import pandas as pd
from out_and_in import out_and_in
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load dataset
auto_mpg = fetch_openml(data_id=42165)
X = auto_mpg.data
y = auto_mpg.target
print(f"the shape of X: {X.shape()}")
# encoded X
X_encoded = pd.get_dummies(X, drop_first=True)
# fill NA with mean
X_encoded = X_encoded.fillna(X_encoded.mean())


def model_train(X: pd.DataFrame,
                y: pd.DataFrame):
    """
    This function will train a basic model and return a metric for that model , we take random forest as a example here.
    :param X: pandas dataframe, it will include all features and the categorical features have been encoded.
    :param y: pandas datasrame, it will include prediction target
    :return: metric.
    """
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    rmse = sqrt(mean_squared_error(y_test, y_pred))

    return rmse

# select features
dropped_features = out_and_in(X, y, model_train)
print(f"dropped features: {dropped_features}")
```


