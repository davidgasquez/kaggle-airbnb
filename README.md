New User Bookings
=================

This repository contains the code developed for the [Airbnb's Kaggle
competition][competition]. It's written in **Python**, mostly in the form
of **Jupyter Notebooks**, and in regular Python to make small libraries.
Feel free to contribute to the code or open an issue if you see something wrong.

[competition]: https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings


Description
-----------

New users on *Airbnb* can book a place to stay in 34,000+ cities across 190+
countries. By accurately predicting where a new user will book their first
travel experience, *Airbnb* can share more personalized content with their
community, decrease the average time to first booking, and better forecast
demand.

In this competition, the objective is to predict in which country a new user
will make his or her first booking. There are **12** possible outcomes of the
destination country and the datasets consists in a list of users with their
demographics, web session records, and some summary statistics.

Data
----

Due to the [*Competition Rules*][rules], the dataset's can not be shared. If
you want to take a look to the data, head over the [competition][competition]
page and download it.

You need to download `train_users_2.csv`, `test_users.csv` and `sessions.csv`
files and unzip them into the 'data/raw' folder.

**Note**: Since the train users file is the one re-uploaded by the competition
administrators, rename `train_users_2.csv` as `train_users.csv`.

[rules]: https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/rules

Guidelines
----------

If you want to generate a submission you need to follow the next steps assuming
you have already downloaded and placed the *competition data* in the correct
folder:

1. Run `preprocessing.py` to generate the files `processed_train_users.csv` and
`processed_test_users.csv`. Those files contains the processed and encoded users
and sessions data.

2. Run `gradient_boosting.py` to generate the submission file.

Main Ideas
----------

1. The provided datasets have lot of NaNs and some other weirds values, so, a
good preprocessing will be primary to obtain a good solution.

2. Those kind of classification task works nicely with tree based methods, it
would be interesting to take a look at Random Forest and Gradient Boosting.

Requirements
------------
To replicate the findings and execute the code in this repository you will need
basically the next Python packages:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [Jupyter](http://jupyter.org/)
- [SciKit-Learn](http://scikit-learn.org/stable/)
- [Matplotlib](http://matplotlib.org/)

Resources
---------

- [Unbalanced Dataset](https://github.com/fmfn/UnbalancedDataset) - Library with
implementation to handle unbalanced datasets.
- [XGBoost Documentation](https://xgboost.readthedocs.org) - A library that is
designed, and optimized for boosted (tree) algorithms.

License
-------

Copyright (c) 2015 David Gasquez
Licensed under the MIT license.
