"""Utils used in the preprocessing stage."""

import pandas as pd


def one_hot_encoding(data, categorical_features):
    """Encode a list of categorical features using a one-hot scheme.

    Parameters
    ----------
    data : Pandas DataFrame
        DataFrame containing the categorical features.
    categorical_features : list
        Names of the categorial features to encode

    Returns
    -------
    data : DataFrame
        Returns a pandas DataFrame with one-hot scheme binary columns.
    """
    for feature in categorical_features:
        data_dummy = pd.get_dummies(data[feature], prefix=feature)
        data.drop([feature], axis=1, inplace=True)
        data = pd.concat((data, data_dummy), axis=1)

    return data


def get_weekday(date):
    """Compute the weekday of the given date."""
    return date.weekday()


def process_secs_elapsed(user_session):
    """Wrapper for process_user_secs_elapsed."""
    return process_user_secs_elapsed(*user_session)


def process_user_secs_elapsed(user, sessions):
    """Compute statistical values of the elapsed seconds of a given user.

    Parameters
    ----------
    user : str
        User ID.
    user_secs : array
        Seconds elapsed by each action.

    Returns
    -------
    user_processed_secs : Series
        Returns a pandas Series with the statistical values.
    """
    user_secs = sessions.loc[sessions['user_id'] == user, 'secs_elapsed']
    user_processed_secs = pd.Series()
    user_processed_secs['id'] = user

    user_processed_secs['secs_elapsed_sum'] = user_secs.sum()
    user_processed_secs['secs_elapsed_mean'] = user_secs.mean()
    user_processed_secs['secs_elapsed_min'] = user_secs.min()
    user_processed_secs['secs_elapsed_max'] = user_secs.max()
    user_processed_secs['secs_elapsed_quantile_1'] = user_secs.quantile(0.25)
    user_processed_secs['secs_elapsed_quantile_3'] = user_secs.quantile(0.75)
    user_processed_secs['secs_elapsed_median'] = user_secs.median()
    user_processed_secs['secs_elapsed_std'] = user_secs.std()
    user_processed_secs['secs_elapsed_var'] = user_secs.var()
    user_processed_secs['secs_elapsed_skew'] = user_secs.skew()

    return user_processed_secs


def process_sessions(user_session):
    """Wrapper for process_user_session."""
    return process_user_session(*user_session)


def process_user_session(user, sessions):
    """Count the elapsed seconds per action.

    Parameters
    ----------
    user : str
        User ID.
    user_session : Pandas DataFrame
        Session of the user.

    Returns
    -------
    user_session_data : Series
        Returns a pandas Series with the elapsed second per each action.
    """
    # Get the user session
    user_session = sessions.loc[sessions['user_id'] == user]
    user_session_data = pd.Series()

    # Length of the session
    user_session_data['session_lenght'] = len(user_session)
    user_session_data['id'] = user

    suffix = '_secs_elapsed'

    for column in ['action', 'action_type', 'action_detail', 'device_type']:
        column_data = user_session.groupby(column).secs_elapsed.sum()
        column_data.rename(lambda x: x + suffix, inplace=True)
        user_session_data = user_session_data.append(column_data)

    # Get the most used device
    user_session_data['most_used_device'] = user_session['device_type'].max()

    return user_session_data.groupby(level=0).sum()
