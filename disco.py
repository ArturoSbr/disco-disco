# Import libraries
import numpy as np
import pandas as pd
from statsmodels.api import OLS
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import KFold

# Dict to select metric function from user input
metrics = {
    'mae': mean_absolute_error,
    'mape': mean_absolute_percentage_error,
    'mse': mean_squared_error,
    'msle': mean_squared_log_error
}

# Function to prepare data
def prep_data(
    data: pd.DataFrame = None,
    dependent_variable: str = None,
    running_variable: str = None,
    cutoff: float = None,
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
    treated (str): Indicates whether observations 'above' or 'below' the cutoff are
        assigned to treatment.
        Pass 'above' if observations whose running variable is greater or equal to
        the threshold are treated.
        Pass 'below' if observations whose running variable is less than or equal to
        the threshold are treated.
    degree (int): Indicates the degree of the polynomial to be fitted (i.e. linear,
        quadratic, cubic, etc.).
    """
    # Copy data (keep same index)
    ret = data.copy()

    # Declare constant
    ret['const'] = 1

    # Recenter running variable at zero
    if cutoff is not None:
        ret[running_variable] = ret[running_variable] - cutoff

    # Declare treatment column
    if treated == 'above':
        ret['treat'] = ret[running_variable].ge(0).astype(int)
    else:
        ret['treat'] = ret[running_variable].le(0).astype(int)

    # Declare columns for powers and interactions
    for d in range(degree):
        ret[f'{running_variable}_pow{d+1}'] = ret[running_variable].pow(d+1)
        ret[f'{running_variable}_treat_pow{d+1}'] = (ret[running_variable] * ret['treat']).pow(d+1)
    
    # Columns necessary for regressions
    cols = [dependent_variable, 'const', 'treat']
    cols = cols + [col for col in ret.columns if f'{running_variable}_pow' in col]
    cols = cols + [col for col in ret.columns if f'{running_variable}_treat_pow' in col]

    # Organize prepped data
    ret = ret[cols]

    # Return prrocessed DataFrame
    return ret

# Function to return CV scores
def cv_bandwidth(
    data: pd.DataFrame = None,
    dependent_variable: str = None,
    running_variable: str = None,
    cutoff: int = 0,
    treated: str = 'above',
    degree: int = 1,
    n_bandwidths: int = 10,
    folds: int = 5,
    criteria: str = 'mse',
    random_state: int = None
):
    """Takes in a pandas.DataFrame object, transforms it to make it compatible
    with a sharp regression discontinuity design and returns the cross-validated
    errors of the model for multiple different bandwidths.

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
    treated (str): Indicates whether observations 'above' or 'below' the cutoff are
        assigned to treatment.
        Pass 'above' if observations whose running variable is greater or equal to
        the threshold are treated.
        Pass 'below' if observations whose running variable is less than or equal to
        the threshold are treated.
    degree (int): Indicates the degree of the polynomial to be fitted (i.e. linear,
        quadratic, cubic, etc.).
    n_bandwidths (int): The number of different bandwidths to be tested. Must be an
        even integer. The bandwidths are calculated by taking the minimum and
        maximum values of the recentered running variable and then making a linear
        partition of length `n_bandwidths` in this space.
    folds (int): The number of folds used to approximate the prediction error through
        cross-validation. For each bandwidth, the model will be trained `folds` times,
        and in each turn, the prediction error will be calculated on the portion of
        the data left out for testing.
    criteria (str): The metric used to validate the models with. Accepted values are
        mean absolute error ('mae'), mean absolute percentage error ('mape'), mean
        squared error ('mse') and mean squared log error (`msle`).
    random_state (int): The seed used to instantiate the pseudo-random splitter for
        cross-validation.
    """

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
    X = ['const', 'treat']
    X += [col for col in ret.columns if f'{running_variable}_pow' in col]
    X += [col for col in ret.columns if f'{running_variable}_treat_pow' in col]
    
    # Get scorer
    metric = metrics[criteria]

    # Init list to store results from each bandwidth
    h = []

    # Iterate over bandwidths
    for i in range(n_bandwidths):

        # Select subset within bandwidth
        lb = cuts.item(i)
        ub = cuts.item(-(i+1))
        mask = ret[running_variable].between(lb, ub)
        t = ret[mask].reset_index(drop=True)

        # Init list to store MSEs
        l = [lb, ub, mask.sum()]

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
    
        # Append all MSEs
        h.append(l)
  
    # Colum names for final DataFrame
    cols = ['lowerBound', 'upperBound', 'nObs']
    cols += [f'{criteria}{j+1}' for j in range(folds)]

    # CV results to DataFrame
    ret = pd.DataFrame(data=h, columns=cols)
    ret['cvScore'] = ret.loc[:, f'{criteria}1':].mean(axis=1)

    # Return DataFrame
    return ret