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
    treated: str = 'above',
    degree: int = 1
):
    """Takes in a pandas.DataFrame object and transforms it to make it compatible
    with a sharp regression discontinuity design.


    Parameters:
    data (pandas.DataFrame): A pandas.DataFrame object that contains the dependent
        and running variables.
    dependent_variable (str): The name of the dependent variable as it appears in
        `data`.
    running_variable (str): The name of the running variable as it appears in
        `data`.
    cutoff (float): The cutoff value that determines the assignment to treatment.
        Defaults to zero. This value is used to recenter the running variable around
        zero. Omit this parameter if the running variable in `data` has already been
        centered.
    treated (str): Indicates whether observations above or below the cutoff are
        assigned to treatment.
        Pass 'above' if observations whose running variable is greater or equal to
        the threshold are treated.
        Pass 'below' if observations whose running variable is less than or equal to
        the threshold are treated.
    degree (int): Indicates the degree of the polynomial to be fitted (i.e. linear,
        quadratic, cubic, etc.).
    """
    # Reindex data
    ret = data.reset_index(drop=True)

    # Declare constant
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
    
    # Columns necessary for regressions
    cols = [dependent_variable, 'const', 'treat']
    cols = cols + [col for col in ret.columns if '_pow' in col]

    # Organize prepped data
    ret = ret[cols]

    # Return DataFrame
    return ret

# Function to return CV scores
def cv_bandwidth(
    data: pd.DataFrame = None,
    dependent_variable: str = None,
    running_variable: str = None,
    cutoff: int = 0,
    treated: str = 'right',
    degree: int = 1,
    n_bandwidths: int = 10,
    folds: int = 5,
    criteria: str = 'mse',
    random_state: int = None
):

    # Prep data
    ret = prep_data(
        data=data,
        dependent_variable=dependent_variable,
        running_variable=running_variable,
        cutoff=cutoff,
        treated=treated,
        degree=degree
    )
    
    # Rename running variable
    running_variable += '_pow1'

    # Instantiate KFold splitter
    splitter = KFold(
        n_splits=folds,
        random_state=random_state,
        shuffle=True
    )

    # Get cuts to create bandwidths
    cuts = np.linspace(
        start=ret[running_variable].min(),
        stop=ret[running_variable].max(),
        num=n_bandwidths*2
    )

    # Columns to train with
    X = ['const'] + [col for col in ret.columns if f'{running_variable}_pow' in col]
    X = X + [col for col in ret.columns if f'{running_variable}_treat_pow' in col]
    
    # Get scorer
    metric = metrics[criteria]

    # Init list to store results from each bandwidth
    h = []

    # Iterate over bandwidths
    for i in range(n_bandwidths):

        # Select subset within bandwidth
        lb = cuts.item(i)
        ub = cuts.item(-(i+1))
        t = ret[ret[running_variable].between(lb, ub)].reset_index(drop=True)

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
                exog=tr[X]
            ).fit()

            # Get metric on test
            l.append(
                metric(
                    y_true=tt[dependent_variable],
                    y_pred=m.predict(exog=tt[X])
                )
            )
    
        # Append all {folds} MSEs
        h.append(l)
  
    # Colum names for final DataFrame
    cols = ['lowerBound', 'upperBound'] + [f'{criteria}{j+1}' for j in range(folds)]
    ret = pd.DataFrame(data=h, columns=cols)
    ret['cvScore'] = ret.iloc[:, 2:].mean(axis=1)

    # Return DataFrame
    return ret