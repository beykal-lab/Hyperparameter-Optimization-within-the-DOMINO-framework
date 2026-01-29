from numpy import *
import nlopt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import time
import random

import os
import time

sc=StandardScaler()

df = pd.read_csv("../ETOH.csv")

df = df.reset_index(drop = True) 
df[df.columns]= sc.fit_transform(df[df.columns])
X=df.drop(["Purity","Q_R1","Q_R2","QC1","QC2","QC3"], axis=1).astype("float64")
y=df.Purity

ridge=Ridge()

polynomial_converter = PolynomialFeatures(degree=4,include_bias=True)
Xpolyn = polynomial_converter.fit_transform(X)
X_train, X_blind, y_train,y_blind=train_test_split(Xpolyn, y, test_size=0.3, random_state=32)

folds = 5
kfold = KFold(n_splits=folds, random_state=7, shuffle=True)

iter = 1
verbosity = {}

def black_box (x,grad = [],verbose = 1):
    #ridge = Ridge() if your ML model keeps the coefficients in memory re-build it here
    global iter, verbosity
    verbosity['x']=list(x)
    ## Get x --> train the model (lower-level) --> evaluate the validation score
    f = cross_val_score(ridge.set_params(alpha= x[0]), X_train, y_train, scoring='neg_mean_squared_error',cv=kfold)
    f=f.mean()*-1000
    if verbose==1:
        verbosity['x']=[float(round(i,6)) for i in x]
        verbosity['f']=f
        print(f"Iteration {iter}: {verbosity['x']}\t{verbosity['f']:1.8F}")
    return f

interval0 = np.arange(1,1000,0.000001,dtype=float) #Create an array to pick a random initial guess
initial_guess = random.choice(interval0)

# Build the optimization model; call the solver and variable size
opt1=nlopt.opt(nlopt.GN_DIRECT_L,1) 
lower_bound = np.array([0.0001]) 
upper_bound = np.array([1000])
opt1.set_lower_bounds(lower_bound)  # Set the bounds
opt1.set_upper_bounds(upper_bound)
opt1.set_ftol_abs(1e-8) #Set stop criteria
opt1.set_maxeval(50000)
# opt1.set_stopval() # Use this if you have stop value for objective function

opt1.set_min_objective(black_box) 
print(f"Method of optimization : {opt1.get_algorithm_name()}\n")
print("computing the optimization algorithm...\n")
tic=time.process_time()
xopt1 = opt1.optimize([initial_guess])
minf = opt1.last_optimum_value()
toc=time.process_time()

print(f"The computation cost is : {toc-tic: 1.6f} sec\n")
print(f"Starting point : ({initial_guess:1.6f})")
print (f"Minimum value of MSE at {xopt1[0] :1.6f}= {minf:2.6f}\n")
print(f"No. of samples : {opt1.get_numevals()}")


# Model assessment with the found solution
ridge= Ridge()                                                                                      
ridge.set_params(alpha=xopt1[0])
ridge.fit(X_train,y_train)
ypred_blind=ridge.predict(X_blind)
MSE_blind=mean_squared_error(y_blind,ypred_blind)
R2=r2_score(y_blind,ypred_blind)


print("MODEL ACCURACY ON BLIND DATA: \n")
print(f"MSE: {MSE_blind:1.6f}\n")
print(f"R2 Score: {R2:1.6f}\n")
