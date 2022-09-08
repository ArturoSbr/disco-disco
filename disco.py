# Import libraries
import numpy as np
import pandas as pd
from statsmodels.api import OLS
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

# Dict to select metric function from user input
metrics = {
    'mse': mean_squared_error,
    'mae': mean_absolute_error
}

# Function to prepare data
def prep_data(
    data: pd.DataFrame = None,
    dependent_variable: str = None,
    running_variable: str = None,
    cutoff: float = 0,
    treated: str = 'right',
    degree: int = 1
):

    # Reindex data
    ret = data.reset_index(drop=True)

    # Declare constant column
    ret['const'] = 1

    # Recenter running variable at zero
    ret[running_variable] = ret[running_variable] - cutoff

    # Declare treatment column
    if treated == 'right':
        ret['treat'] = ret[running_variable].ge(0).astype(int)
    else:
        ret['treat'] = ret[running_variable].le(0)

    # Declare powers and interactions
    for d in range(degree):
        ret[f'{running_variable}_pow{d+1}'] = ret[running_variable].pow(d+1)
        ret[f'{running_variable}_treat_pow{d+1}'] = (ret[running_variable] * ret['treat']).pow(d+1)
    
    # Columns with _pow in their name
    cols = [dependent_variable, 'const', 'treat'] + [col for col in ret.columns if '_pow' in col]

    # Organize prepped data
    ret = ret[cols]

    # Return DataFrame
    return ret

# Function to return CV scores
def cv_bandwidths(
    data: pd.DataFrame = None,
    dependent_variable: str = None,
    running_variable: str = None,
    degree: int = 1,
    n_bandwidths: int = 10,
    folds: int = 5,
    criteria: str = 'mse',
    random_state: int = None
):

    # Get scorer
    metric = metrics[criteria]

    # Instantiate splitter
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

            # Get metric on test
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

    # Return DataFrame
    return ret