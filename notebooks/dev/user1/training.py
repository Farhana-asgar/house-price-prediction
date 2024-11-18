#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pprint import pprint
import os
import os.path as op
import shutil

# standard third party imports
import numpy as np
import pandas as pd
import tarfile

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

pd.options.mode.use_inf_as_na = True



# In[3]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[4]:


import warnings
from numba import NumbaDeprecationWarning

warnings.filterwarnings('ignore', message="The default value of regex will change from True to False in a future version.", 
                        category=FutureWarning)

warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)


# In[5]:


# standard code-template imports
from ta_lib.core.api import (
    create_context, get_dataframe, get_feature_names_from_column_transformer, get_package_path,
    display_as_tabs, string_cleaning, merge_info, initialize_environment,
    list_datasets, load_dataset, save_dataset
)
import ta_lib.eda.api as eda

from taregression.linear_api import LinearRegression
from taregression.elastic_net_api import ElasticNet
from taregression.bayes_api import BayesianRegression


# In[29]:


initialize_environment(debug=False, hide_warnings=True)


# In[63]:


config_path = op.join('conf', 'config.yml')
context = create_context(config_path)
housing_prepared = load_dataset(context, 'train/house-value-prediction/features')
housing_labels = load_dataset(context, 'train/house-value-prediction/target')
x_test = load_dataset(context, 'test/house-value-prediction/features').rename(columns={'ocean_proximity_<1H OCEAN': 'ocean_proximity_1H_OCEAN'}).rename(columns={'ocean_proximity_NEAR OCEAN': 'ocean_proximity_NEAR_OCEAN'}).rename(columns={'ocean_proximity_NEAR BAY': 'ocean_proximity_NEAR_BAY'})
y_test = load_dataset(context, 'test/house-value-prediction/features')
housing_prepared = housing_prepared.rename(columns={'ocean_proximity_<1H OCEAN': 'ocean_proximity_1H_OCEAN'}).rename(columns={'ocean_proximity_NEAR OCEAN': 'ocean_proximity_NEAR_OCEAN'}).rename(columns={'ocean_proximity_NEAR BAY': 'ocean_proximity_NEAR_BAY'})
housing_prepared=pd.concat([housing_prepared,housing_labels],axis=1)
x_train=housing_prepared.drop(["median_house_value"],axis=1)
y_train=housing_labels



# In[56]:





# In[58]:


model_equation = (
    "median_house_value ~  longitude + latitude + "
    + "housing_median_age + total_rooms + "
    + "total_bedrooms + population + "
    + "households + median_income + "
    + "ocean_proximity_1H_OCEAN + ocean_proximity_INLAND + ocean_proximity_ISLAND + ocean_proximity_NEAR_BAY + ocean_proximity_NEAR_OCEAN"
)


# In[59]:


alpha = context.job_catalog["jobs"][1]["stages"][0]["tasks"][0]["params"]["alpha"]
lmbd = context.job_catalog["jobs"][1]["stages"][0]["tasks"][0]["params"]["lmbd"]
max_itr = context.job_catalog["jobs"][1]["stages"][0]["tasks"][0]["params"]["max_itr"]
print(housing_prepared.columns)
elastic_model = ElasticNet(model_equation, backend="sklearn")
elastic_model.fit(housing_prepared, alpha=alpha, lmbd=lmbd, max_iter=max_itr)
elastic_model.get_coefficients()


# In[60]:


alpha = context.job_catalog["jobs"][1]["stages"][1]["tasks"][0]["params"]["alpha"]
lmbd = context.job_catalog["jobs"][1]["stages"][1]["tasks"][0]["params"]["lmbd"]
max_itr = context.job_catalog["jobs"][1]["stages"][1]["tasks"][0]["params"]["max_itr"]

lin_reg_model = LinearRegression(model_equation, backend="sklearn")
lin_reg_model.fit(housing_prepared, alpha=alpha, lmbd=lmbd, max_iter=max_itr)
lin_reg_model.get_coefficients()


# In[61]:


params = context.job_catalog["jobs"][1]["stages"][2]["tasks"][0]["params"]
bay_reg_model = BayesianRegression(model_equation, backend="bambi")
bay_reg_model.fit(housing_prepared,
             model_params=params
            )
bay_reg_model.get_summary()


# In[64]:


from ta_lib.regression.api import RegressionComparison, RegressionReport
pred_train_elas=elastic_model.predict(x_train)
pred_test_elas=elastic_model.predict(x_test)
elastic_model_report = RegressionReport(
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
    yhat_train=pred_train_elas,
    yhat_test=pred_test_elas,
)
elastic_model_report.get_report(include_shap=False, file_path="reports/Elastic Model Report")


pred_train_lin=lin_reg_model.predict(x_train)
pred_test_lin=lin_reg_model.predict(x_test)
lin_reg_report = RegressionReport(
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
    yhat_train=pred_train_lin,
    yhat_test=pred_test_lin,
)
lin_reg_report.get_report(include_shap=False, file_path="reports/Linear Regression Model Report")

pred_train_bay=bay_reg_model.predict(x_train)
pred_test_bay=bay_reg_model.predict(x_test)
bayes_report = RegressionReport(
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
    yhat_train=pred_train_bay,
    yhat_test=pred_test_bay,
)
bayes_report.get_report(include_shap=False, file_path="reports/Bayesian Regression Model")


# In[ ]:




