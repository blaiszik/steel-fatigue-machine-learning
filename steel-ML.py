### Standard library imports
import pandas as pd
import numpy as np

### sklearn imports
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_validation import KFold

### Load the dataset
data_file = "./data/40192_2013_16_MOESM1_ESM.csv"
df_loaded = pd.read_csv(data_file)

### Define scaler
#scaler = RobustScaler()
scaler = StandardScaler()

### Create a dataframe with all input values
X = pd.DataFrame(scaler.fit_transform(df_loaded.drop(['Fatigue'], axis=1)))
X.columns = df_loaded.drop(['Fatigue'], axis=1).columns
### Similarly, extract the y values as a vector
y = df_loaded['Fatigue']
print(X.head())

### Initialize model tracking variables
rmses, r2s, results, predicted, actual = ([] for i in range(0,5))
n_estimators = 200    # Set number of estimators
n_folds = 15          # Set number of folds
shuffle = True        # Shuffle the data before folding?
num_tests = 1

### Set up validation using kFolds
k_fold = KFold(len(X), n_folds, shuffle=shuffle)

for i in range(0,num_tests):
    # Train and test each fold, and track results
    for k, (train, test) in enumerate(k_fold):
        # Initialize model
        model_name, model = ("Gradient Boosting Regression %s"%(n_estimators), 
                          GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=0.075, 
                          max_depth=4, loss='ls'))

        #Fit the model on the kfold training set and predict the kfold test set
        model.fit(X.iloc[train], y.iloc[train])
        pred = model.fit(X.iloc[train], y.iloc[train]).predict(X.iloc[test])

        #Save r^2 and root mean squared error for each fold
        r2s.append(r2_score(y.iloc[test], pred))
        rmses.append(np.sqrt(mean_squared_error(y.iloc[test], pred)))

        #Save predictions vs actual values for later plotting
        predicted.append(pred)
        actual.append(y.iloc[test])
print("***** %s - %s tests *****"%(model_name, num_tests))
print("rMSE:%s"%(np.mean(rmses)))
print("r^2:%s"%(np.mean(r2s)))