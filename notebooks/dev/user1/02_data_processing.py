#!/usr/bin/env python
# coding: utf-8

# ## The following is done in this file
#
# 1.

# In[1]:


# standard third party imports
import os
import os.path as op
import shutil
import tarfile
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

pd.options.mode.use_inf_as_na = True



# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


import warnings

from numba import NumbaDeprecationWarning

warnings.filterwarnings('ignore', message="The default value of regex will change from True to False in a future version.",
                        category=FutureWarning)

warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)


# In[4]:


import ta_lib.eda.api as eda

# standard code-template imports
from ta_lib.core.api import (
        create_context,
        display_as_tabs,
        get_dataframe,
        get_feature_names_from_column_transformer,
        get_package_path,
        initialize_environment,
        list_datasets,
        load_dataset,
        merge_info,
        save_dataset,
        string_cleaning,
)

# In[5]:


initialize_environment(debug=False, hide_warnings=True)


# ## Utility functions

# # 1. Initialization

# In[11]:


config_path = op.join('conf', 'config.yml')
context = create_context(config_path)
housing = load_dataset(context, 'cleaned/house-value-prediction')


# # 2. Data cleaning and consolidation

# In[7]:


def income_cat_proportions(data):
        """
        Downloads and extracts the housing dataset.

        Args:
            data (Dataframe): The data that is to be used in model creation.

        Returns:
            Series: Proportion of income
        """
        # Returns proportion of income
        return (data["income_cat"].value_counts() / len(data))

housing["income_cat"] = pd.cut(housing["median_income"],
                                       bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                       labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=float(context.job_catalog["jobs"][0]["test_size"]),
                               random_state=int(context.job_catalog["jobs"][0]["random_seed"]))
for train_index, test_index in split.split(housing,
                                           housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
train_set, test_set = train_test_split(housing, test_size=0.2,
                                       random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"]   \
    / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] \
    / compare_props["Overall"] - 100

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# In[8]:


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, df):
        rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

        # Avoid division by zero by adding a small epsilon
        households = df.iloc[:, households_ix]
        epsilon = 1e-10  # Small constant to avoid division by zero

        # Calculate features
        df['rooms_per_household'] = df.iloc[:, rooms_ix] / (
            households + epsilon)
        df['population_per_household'] = df.iloc[:, population_ix] / (
            households + epsilon)

        df['bedrooms_per_room'] = df.iloc[:, bedrooms_ix] / (
                df.iloc[:, rooms_ix] + epsilon)
        return df

    def inverse_transform(self, df):
        # Drop derived columns added during the transform step
        df = df.drop(columns=['rooms_per_household', 'population_per_household'], errors='ignore')

        if self.add_bedrooms_per_room:
            df = df.drop(columns=['bedrooms_per_room'], errors='ignore')

        return df

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


imputer = SimpleImputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)

imputer.fit(housing_num)
X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing.index)

attr_adder = CombinedAttributesAdder()
housing_extra_attribs = attr_adder.transform(
    housing_tr)

housing_cat = housing[['ocean_proximity']]

housing_prepared = housing_extra_attribs.join(pd.get_dummies(
    housing_cat, drop_first=False))

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_num = X_test.drop("ocean_proximity", axis=1)
X_test_prepared = imputer.transform(X_test_num)
X_test_prepared = pd.DataFrame(X_test_prepared,
                               columns=X_test_num.columns,
                               index=X_test.index)
col_names = list(X_test_prepared.columns)
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=True)
X_test_prepared_df = attr_adder.transform(
    X_test_prepared)

X_test_cat = X_test[['ocean_proximity']]
X_test = X_test_prepared_df.join(pd.get_dummies
                                          (X_test_cat,
                                           drop_first=False))


# In[12]:


save_dataset(context, housing_prepared, 'train/house-value-prediction/features')
save_dataset(context, housing_labels, 'train/house-value-prediction/target')
save_dataset(context, X_test, 'test/house-value-prediction/features')
save_dataset(context, y_test, 'test/house-value-prediction/target')