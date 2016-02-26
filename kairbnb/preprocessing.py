"""Utils used in the preprocessing stage."""

import pandas as pd
import numpy as np
from datetime import datetime
import holidays
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import PolynomialFeatures


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

    Warning
    -------
    Is the same as pd.get_dummies(data, columns=categorical_features)
    """
    for feature in categorical_features:
        data_dummy = pd.get_dummies(data[feature], prefix=feature)
        data.drop([feature], axis=1, inplace=True)
        data = pd.concat((data, data_dummy), axis=1)

    return data


class XGBFeatureSelection(XGBClassifier):
    """A custom XGBClassifier with feature importances computation.

    This class implements XGBClassifier and also computes feature importances
    based on the fscores. Implementing feature_importances_ property allow us
    to use `SelectFromModel` with XGBClassifier.
    """

    def __init__(self, n_features, *args, **kwargs):
        """Init method adding n_features."""
        super(XGBFeatureSelection, self).__init__(*args, **kwargs)
        self._n_features = n_features

    @property
    def n_features(self):
        """Number of classes to predict."""
        return self._n_features

    @n_features.setter
    def n_features(self, value):
        self._n_features = value

    @property
    def feature_importances_(self):
        """Return the feature importances.

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        booster = self.booster()
        fscores = booster.get_fscore()

        importances = np.zeros(self.n_features)

        for k, v in fscores.iteritems():
            importances[int(k[1:])] = v

        return importances


def _sanitize_holiday_name(name):
    """Remove weird character from the input string."""
    new_name = [c for c in name if c.isalpha() or c.isdigit() or c == ' ']
    new_name = "".join(new_name).lower().replace(" ", "_")
    return new_name


def distance_to_holidays(date):
    """Append the distance of several holidays in days to the users DataFrame.

    Parameters
    ----------
    user : Pandas Series
        User containing the dates.

    Returns
    -------
    user : Series
        Returns the original pandas Series with the new features.
    """
    distances = pd.Series()
    # Get US holidays for this year
    holidays_dates = holidays.US(years=[date.year], observed=False)

    for holiday_date, name in holidays_dates.items():
        # Compute difference in days
        holiday_date = datetime.combine(holiday_date, datetime.min.time())
        days = (holiday_date - date).days

        # Clean holiday name
        name = _sanitize_holiday_name(name)

        distances['days_to_' + name] = days

    return distances


def process_user_actions(sessions, user):
    """Count the elapsed seconds per action.

    Parameters
    ----------
    sessions : Pandas DataFrame
        Sessions of the users.
    user : int or str
        User ID.

    Returns
    -------
    user_session_data : Series
        Returns a pandas Series with the count of each action.
    """
    # Get the user session
    user_session = sessions.loc[sessions['user_id'] == user]
    user_session_data = pd.Series()

    # Length of the session
    user_session_data['session_lenght'] = len(user_session)
    user_session_data['id'] = user

    # Take the count of each value per column
    columns = ['action', 'action_type', 'action_detail']  # device_type
    for column in columns:
        column_data = user_session[column].value_counts()
        column_data.index = column_data.index + '_count'
        user_session_data = user_session_data.append(column_data)

    # Get the most used device
    session = user_session_data.groupby(user_session_data.index).sum()

    session['most_used_device'] = user_session['device_type'].mode()
    session['most_used_device'] = np.sum(session['most_used_device'])

    if session['most_used_device'] == 0:
        session['most_used_device'] = np.nan

    # For Python 2 it's only needed to do:
    # session['most_used_device'] = user_session['device_type'].max()

    # Grouby ID and add values
    return session


def process_user_secs_elapsed(sessions, user):
    """Compute some statistical values of the elapsed seconds of a given user.

    Parameters
    ----------
    sessions : Pandas DataFrame
        Sessions of the users.
    user : int or str
        User ID.

    Returns
    -------
    user_processed_secs : Series
        Returns a pandas Series with the statistical values.
    """
    # Locate user in sessions file
    user_secs = sessions.loc[sessions['user_id'] == user, 'secs_elapsed']
    user_processed_secs = pd.Series()
    user_processed_secs['id'] = user

    user_processed_secs['secs_elapsed_sum'] = user_secs.sum()
    user_processed_secs['secs_elapsed_mean'] = user_secs.mean()
    user_processed_secs['secs_elapsed_min'] = user_secs.min()
    user_processed_secs['secs_elapsed_max'] = user_secs.max()
    user_processed_secs['secs_elapsed_quantile_1'] = user_secs.quantile(0.1)
    user_processed_secs['secs_elapsed_quantile_2'] = user_secs.quantile(0.25)
    user_processed_secs['secs_elapsed_quantile_3'] = user_secs.quantile(0.75)
    user_processed_secs['secs_elapsed_quantile_3'] = user_secs.quantile(0.9)
    user_processed_secs['secs_elapsed_median'] = user_secs.median()
    user_processed_secs['secs_elapsed_std'] = user_secs.std()
    user_processed_secs['secs_elapsed_var'] = user_secs.var()
    user_processed_secs['secs_elapsed_skew'] = user_secs.skew()

    # Number of elapsed seconds greater than 1 day
    user_processed_secs['day_pauses'] = user_secs[user_secs > 86400].count()

    # Clicks with less than one hour of differences
    user_processed_secs['short_sessions'] = user_secs[user_secs < 3600].count()

    # Long breaks
    user_processed_secs['long_pauses'] = user_secs[user_secs > 300000].count()

    return user_processed_secs


def _empty_columns(x):
    return np.all(x == 0)


def interaction_features(data, degree):
    """Generate polynomial features given a dataset and a degree."""
    poly = PolynomialFeatures(degree, interaction_only=True)

    interaction_data = poly.fit_transform(data)

    interaction = pd.DataFrame(interaction_data).drop(0, axis=1)

    base_columns = np.shape(data)[1] + 1
    interaction = interaction.ix[:, base_columns:]

    # Drop empty columns
    drop_columns = interaction.columns[interaction.apply(_empty_columns)]
    df = interaction.drop(drop_columns, axis=1)

    return df
