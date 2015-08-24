# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 11:28:20 2015

@author: elf
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

def read_data():
    # read the demographics data
    county_df = pd.read_csv('demo.csv', converters={'geographyId': lambda x: str(x)})
    # read the m-lab data
    mlab_df = pd.read_csv('NewEngland/combined_values.csv', converters={'geoid': lambda x: str(x)})
    county_df.rename(columns={'geographyId': 'geoid'}, inplace=True)
    df = pd.merge(mlab_df, county_df, on=['geoid'])
    list_keep = ['geoid', 'MedianRTT', 'download_avg', 'download_count', 'download_median', 'upload_avg', 'upload_count', 'upload_median', 'medianIncome', 'population', 'households']
    result = df[list_keep]
    return result

data = read_data()
'''
fig, axs = plt.subplots(3, 1, sharey=True)
data.plot(kind='scatter', x='MedianRTT', y='population', ax=axs[0], figsize=(16, 8))
data.plot(kind='scatter', x='download_median', y='population', ax=axs[1])
data.plot(kind='scatter', x='upload_median', y='population', ax=axs[2])
plt.savefig('Relationship_feature_population.png', dpi=125)
plt.close()

fig, axs = plt.subplots(3, 1, sharey=True)
data.plot(kind='scatter', x='MedianRTT', y='medianIncome', ax=axs[0], figsize=(16, 8))
data.plot(kind='scatter', x='download_median', y='medianIncome', ax=axs[1])
data.plot(kind='scatter', x='upload_median', y='medianIncome', ax=axs[2])
plt.savefig('Relationship_feature_medianIncome.png', dpi=125)
plt.close()
'''
fig, axs = plt.subplots(1, 2, sharey=True)
data.plot(kind='scatter', x='population', y='MedianRTT', ax=axs[0], figsize=(16, 8))
data.plot(kind='scatter', x='medianIncome', y='MedianRTT', ax=axs[1])
plt.savefig('Relationship_feature_MedianRTT.png', dpi=125)
plt.close()

fig, axs = plt.subplots(1, 2, sharey=True)
data.plot(kind='scatter', x='population', y='download_median', ax=axs[0], figsize=(16, 8))
data.plot(kind='scatter', x='medianIncome', y='download_median', ax=axs[1])
plt.savefig('Relationship_feature_download.png', dpi=125)
plt.close()

fig, axs = plt.subplots(1, 2, sharey=True)
data.plot(kind='scatter', x='population', y='upload_median', ax=axs[0], figsize=(16, 8))
data.plot(kind='scatter', x='medianIncome', y='upload_median', ax=axs[1])
plt.savefig('Relationship_feature_upload.png', dpi=125)
plt.close()


def simple_linear_reg(x_str, y_str, form):
    print 'features\n',form
    # create a fitted model in one line
    lm = smf.ols(formula=form, data=data).fit()
    # print the coefficients
    print 'the coefficients\n',lm.params
    
    # create a DataFrame with the minimum and maximum values of the feature
    X_new = pd.DataFrame({x_str: [data[x_str].min(), data[x_str].max()]})
    print 'minimum and maximum values of the feature\n',X_new.head()
    
    # make predictions for those x values and store them
    preds = lm.predict(X_new)
    print 'predictions for minimum and maximum values of the feature\n',preds
    
    # first, plot the observed data
    data.plot(kind='scatter', x=x_str, y=y_str)
    
    # then, plot the least squares line
    plt.plot(X_new, preds, c='red', linewidth=2)
    title = form+'.png'
    plt.savefig(title, dpi=125)
    plt.close()
    
    # print the confidence intervals for the model coefficients
    print 'confidence intervals\n',lm.conf_int()
    
    # print the p-values for the model coefficients
    print 'p-values for model coefficients\n',lm.pvalues
    
    # print the R-squared value for the model
    print 'R-squared value for the model\n',lm.rsquared
    
def multiple_regression(form):
    print 'feature\n',form
    # create a fitted model with all three features
    lm = smf.ols(formula=form, data=data).fit()

    # print the coefficients
    print '\nthe coefficients\n',lm.params
    
    # print a summary of the fitted model
    print '\nthe summary of the model\n',lm.summary()

'''
x_str=['population','medianIncome']
y_str=['MedianRTT']
form=['MedianRTT ~ population','MedianRTT ~ medianIncome']

i = 0
while i < len(x_str):
    simple_linear_reg(x_str[i], y_str, form[i])
    i = i + 1

y_str=['download_median']
form=['download_median ~ population','download_median ~ medianIncome']

i = 0
while i < len(x_str):
    simple_linear_reg(x_str[i], y_str, form[i])
    i = i + 1

y_str=['upload_median']
form=['upload_median ~ population','upload_median ~ medianIncome']

i = 0
while i < len(x_str):
    simple_linear_reg(x_str[i], y_str, form[i])
    i = i + 1
'''

form='MedianRTT ~ population + medianIncome'
multiple_regression(form)
form='download_median ~ population + medianIncome'
multiple_regression(form)
form='upload_median ~ population + medianIncome'
multiple_regression(form)

