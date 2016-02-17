# Airbnb Kaggle Competition: *New User Bookings*

[![Build Status](https://travis-ci.org/davidgasquez/kaggle-airbnb.svg?branch=master)](https://travis-ci.org/davidgasquez/kaggle-airbnb) [![Code Issues](https://www.quantifiedcode.com/api/v1/project/c75f3e6167d940fd89484b651b062109/badge.svg)](https://www.quantifiedcode.com/app/project/c75f3e6167d940fd89484b651b062109)

This repository contains the code developed for the [Airbnb's Kaggle
competition][competition]. It's written in **Python**, some in the form
of **Jupyter Notebooks**, and other in pure Python 3.

The code produces predictions with scores around 0.88090% in the public
leader-board, enough to be in the best 5% participants(0.001% behind the best)
and 0.88509% in the private leader-board(0.0018% behind the winner)

The entire run should not take more than 4 hours(thanks to the parallel
preprocessing) in a modern/recent computer, though you may run into memory
issues with less than 8GB RAM.

Feel free to contribute to the code or open an issue if you see something wrong.

[competition]: https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings


## Description

New users on *Airbnb* can book a place to stay in 34,000+ cities across 190+
countries. By accurately predicting where a new user will book their first
travel experience, *Airbnb* can share more personalized content with their
community, decrease the average time to first booking, and better forecast
demand.

In this competition, the goal is to predict in which country a new user
will make his or her first booking. There are **12** possible outcomes of the
destination country and the datasets consist of a list of users with their
demographics, web session records, and some summary statistics.

## Data

Due to the [*Competition Rules*][rules], the data sets can not be shared. If
you want to take a look at the data, head over the [competition][competition]
page and download it.

You need to download `train_users_2.csv`, `test_users.csv` and `sessions.csv`
files and unzip them into the 'data' folder.

**Note**: Since the train users file is the one re-uploaded by the competition
administrators, rename `train_users_2.csv` as `train_users.csv`.

[rules]: https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/rules

## Main Ideas

1. The provided datasets have lot of NaNs and some other *random* values, so, a
good preprocessing is the primary key to get a good solution:
    - Replace *-unknown-* values with NaNs
    - Clean age values
    - Extract day, weekday, month, year from `date_account_created`
    and `timestamp_first_active`
    - Add number of missing values per user
    - General user session information:
        - Number of different values in `action`, `action_type`,
        `action_detail` and `device_type`

2. That kind of classification task works nicely with tree-based methods, I
used `xgboost` library and the Gradient Boosting Classifier that provides along
`scikit-learn` to make the probabilities predictions.

## Requirements

To replicate the findings and execute the code in this repository you will need
basically the next Python packages:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [Jupyter](http://jupyter.org/)
- [SciKit-Learn](http://scikit-learn.org/stable/)
- [Matplotlib](http://matplotlib.org/)
- [Unbalanced Dataset](https://github.com/fmfn/UnbalancedDataset)

## Resources

- [XGBoost Documentation](https://xgboost.readthedocs.org) - A library designed
and optimized for boosted (tree) algorithms.
- [Pattern Classification](https://github.com/rasbt/pattern_classification) -
Tutorials, examples, collections, and everything else that falls into the
categories: pattern classification, machine learning, and data mining.

## License

Copyright © 2015 David Gasquez
Licensed under the MIT license.
