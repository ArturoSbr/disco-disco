# Import libraries
import numpy as np
import pandas as pd
from statsmodels.api import OLS
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

# Get metric function
metrics = {
  'mse': mean_squared_error,
  'mae': mean_absolute_error
}

def cv_bandwidths(
    data: pd.DataFrame = None,
    running_variable: str = None,
    dependent_variable: str = None,
    n_bandwidths: int = 10,
    folds: int = 5,
    criteria: str = 'mse',
    random_state: int = None
):

  # Get metric function
  metric = metrics[criteria]

  # Init splitter
  splitter = KFold(
    n_splits=folds,
    random_state=random_state,
    shuffle=True
  )

  # Get cuts to create bandwidths
  cuts = np.linspace(
      start=data[running_variable].min(),
      stop=data[running_variable].max(),
      num=n_bandwidths*2
  )

  # Init list to store results from each bandwidth
  h = []

  # Iterate over bandwidths
  for i in range(n_bandwidths):

    # Select subset within bandwidth
    lb = cuts.item(i)
    ub = cuts.item(-(i+1))
    t = data[data[running_variable].between(lb, ub)].reset_index()

    # Init list to store MSEs
    l = [lb, ub]

    # Iterate over splits
    for tr_idx, tt_idx in splitter.split(t):

      # Train and test from current fold
      tr = t.iloc[tr_idx]
      tt = t.iloc[tt_idx]
      
      # Fit on train
      m = OLS(
        endog=tr[dependent_variable],
        exog=tr[running_variable]
      ).fit()

      # Get metric on test (fold)
      l.append(
        metric(
          y_true=tt['y'],
          y_pred=m.predict(exog=tt['x'])
        )
      )
    
    # Append all {folds} MSEs
    h.append(l)
  
  # Colum names for final DataFrame
  cols = ['lowerBound', 'upperBound'] + [f'mse{j+1}' for j in range(folds)]
  ret = pd.DataFrame(data=h, columns=cols)
  ret['cvScore'] = ret.iloc[:, 2:].mean(axis=1)

  return ret