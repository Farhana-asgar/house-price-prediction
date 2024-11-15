#!/usr/bin/env python
# coding: utf-8

# ### In this file, the data discovery happens.
#
# 1. Read data
# 2. Perform Data health analysis
# 3. Get missing values list
# 4. Store summary report as html (in reports/).
# 5. Dataset saved for future usage

# In[15]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # %%time
#
# # Third-party imports
# import os.path as op
# import pandas as pd
# import great_expectations as ge
#
# # Project imports
# from ta_lib.core.api import display_as_tabs, initialize_environment
#
# # Initialization
# initialize_environment(debug=False, hide_warnings=True)

# In[3]:


from ta_lib.core.api import create_context, list_datasets, load_dataset, save_dataset


# In[4]:


config_path = op.join('conf', 'config.yml')
context = create_context(config_path)


# In[5]:


list_datasets(context)


# In[6]:


# load datasets

house_value_prediction = load_dataset(context, '/raw/house-value-prediction')
print(house_value_prediction)


# In[7]:


# Import the eda API
import ta_lib.eda.api as eda


# In[8]:


display_as_tabs([('house-value-prediction', house_value_prediction.shape)])


# In[9]:


sum1 = eda.get_variable_summary(house_value_prediction)
display_as_tabs([('house_value_prediction', sum1)])


# In[10]:


house_value_prediction.isna().sum()


# ## Health Analysis

# In[11]:


sum1, plot1 = eda.get_data_health_summary(house_value_prediction, return_plot=True)

display_as_tabs([('house_value_prediction', plot1)])


# In[12]:


sum1, plot1 = eda.get_missing_values_summary(house_value_prediction, return_plot=True)
display_as_tabs([('house_value_prediction', plot1)])


# ## Health Analysis report

# In[13]:


from ta_lib.reports.api import summary_report

summary_report(house_value_prediction, 'reports/house_value_prediction.html')


# In[14]:


save_dataset(context, house_value_prediction, 'cleaned/house-value-prediction')