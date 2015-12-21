
# coding: utf-8

# # Airbnb New User Bookings - Gradient Boosting with H2O

# Load the required libraries

# In[ ]:

import h2o
import pandas as pd
import numpy as np


# Initialize H2O cluster:

# In[ ]:

h2o.init()
h2o.remove_all() 


# ## Data Loading

# In[ ]:

train_users = h2o.import_file("../datasets/raw/train_users.csv")


# Remove train users's ID's since we don't need them to train our classifier

# ## Preprocessing

# In[ ]:

train_users = train_users.drop('id')


# In[ ]:

train_users, validation_users = train_users.split_frame(ratios=[0.8])


# In[ ]:

X =[
    u'date_account_created',
    u'timestamp_first_active',
    u'gender',
    u'age',
    u'signup_method',
    u'signup_flow',
    u'language',
    u'affiliate_channel',
    u'affiliate_provider',
    u'first_affiliate_tracked',
    u'signup_app',
    u'first_device_type',
    u'first_browser',
#     u'country_destination'
]

y = u'country_destination'


# ### Deep Learning Model

# In[ ]:

from h2o.estimators.gbm import H2OGradientBoostingEstimator

gbe = H2OGradientBoostingEstimator(
    ntrees=50,
    max_depth=8, 
    distribution='multinomial',
    stopping_metric="logloss",
    learn_rate=0.25,
    sample_rate=0.6,
    balance_classes=True, 
    nfolds=0
)


# In[ ]:

gbe.train(X, y, training_frame=train_users, validation_frame=validation_users)


# In[ ]:

print "R2:", gbe.r2()


# ### Predictions

# In[ ]:

test_users = h2o.import_file("../datasets/raw/test_users.csv")


# In[ ]:

predictions = gbe.predict(test_users.drop('id'))


# ### Make CSV for submission

# In[ ]:

# Load the predictions into a DataFrame
country_predictions = predictions.as_data_frame()
country_predictions = pd.DataFrame(country_predictions).transpose()

# Use first row as column names
country_predictions.columns = country_predictions.iloc[0]
country_predictions = country_predictions.reindex(country_predictions.index.drop(0))

# Drop labeled prediction
country_predictions.drop('predict', axis=1, inplace=True)


# In[ ]:

# Load the test users into a DataFrame
test_users = pd.DataFrame(test_users.as_data_frame()).transpose()

# Use first row as column names
test_users.columns = test_users.iloc[0]
test_users = test_users.reindex(test_users.index.drop(0))


# In[ ]:

submission = []
number_of_users = country_predictions.shape[0]

# Iterate over each user
for user in range(number_of_users):
    # Get the 5 most provable destination countries
    user_prediction = country_predictions.iloc[user].astype(float)
    
    # Sort in descending order
    user_prediction = user_prediction.sort_values(ascending=False)
    
    # Append the 5 with higher provability
    for country in range(5):
        user_id = test_users.iloc[user]['id']
        destination = list(user_prediction.index[0:5])[country]
        submission.append([user_id, destination])


# In[ ]:

sub_file = pd.DataFrame(submission, columns = ['id', 'country'])


# Write CSV file:

# In[ ]:

sub_file.to_csv('../datasets/submissions/h2o-gb.csv', index=False)


# In[ ]:

h2o.shutdown()

