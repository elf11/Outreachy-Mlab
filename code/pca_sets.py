# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 06:42:02 2015

@author: elf
"""

import pandas as pd
import pca_analysis as pca_an

def pca_set_up( df, file_name ):
    idx_demo = df['geographyId'].str.contains('^09|^23|^25|^33|^44|^50')
    df = df[idx_demo]
    df.set_index('geographyId', inplace=True)
    df.index.name = None
    # eliminate the columns that are not useful for the PCA analysis (Anything that's not numeric data)
    del df['Unnamed: 0']
    del df['geographyName']
    
    load_names = df.T.index.values
    #print load_names
    target_names = df.index.values
    #print target_names
    tran_ne = df.values
    pca_an.pca_analysis(tran_ne, load_names, target_names, str_name=file_name)

demo_df = pd.read_csv('demo.csv', converters={'geographyId': lambda x: str(x)})
broad_df = pd.read_csv('broad_sum.csv', converters={'geographyId': lambda x: str(x)})

#broad = broad[['geographyId','downloadSpeedLessThan3Mbps','downloadSpeedGreaterThan3Mbps','uploadSpeedLessThan3Mbps','uploadSpeedGreaterThan3Mbps']]

result = pd.merge(demo_df, broad_df, on=['geographyId'])
idx = result['geographyId'].str.contains('^09|^23|^25|^33|^44|^50')
result = result[idx]
result.set_index('geographyId', inplace=True)
result.index.name = None

# eliminate the columns that are not useful for the PCA analysis (Anything that's not numeric data)
del result['Unnamed: 0_x']
del result['Unnamed: 0_y']
del result['geographyName_x']
del result['geographyName_y']

load_names = result.T.index.values
#print load_names
target_names = result.index.values
#print target_names
tran_ne = result.values
pca_an.pca_analysis(tran_ne, load_names, target_names, str_name='Complete')


# do pca analysis for the demographics and broadband data sets separately
pca_set_up(demo_df, 'Demographics')
pca_set_up(broad_df, 'Broadband')
