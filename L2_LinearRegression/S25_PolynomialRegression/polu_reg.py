import pandas as pd
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Assign the data to predictor and outcome variables
# TODO: Load the data
data = pd.read_csv('data.csv')
data.head(10)
X = data['Var_X'].values.reshape(-1, 1) #不可以用data[['Var_X']]會出錯, 莫名其妙!For X, make sure it is in a 2-d array of 20 rows by 1 column. 
y = data['Var_Y'].values #can be 1-D or 2-D

# Create polynomial features
# TODO: Create a PolynomialFeatures object, then fit and transform the
# predictor feature
poly_feat = PolynomialFeatures(4)
X_poly = poly_feat.fit_transform(X) 

# Make and fit the polynomial regression model
# TODO: Create a LinearRegression object and fit it to the polynomial predictor
# features
poly_model = LinearRegression().fit(X_poly, y)

# Once you've completed all of the steps, select Test Run to see your model
# predictions against the data, or select Submit Answer to check if the degree
# of the polynomial features is the same as ours!
