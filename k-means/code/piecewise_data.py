# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 17:46:47 2015

@author: elf
"""
import urllib2
import requests
import pandas as pd

def gather_data(url, fileName, columns):
    sum_list = []
    try:
        data_json = requests.get(url).json()
        sum_list.append(data_json)
    except urllib2.URLError, e:
        print 'No kittez. Got an error code:', e

    df = pd.DataFrame(sum_list[0]['features'])
    len_df = len(df)
    
    prop = pd.DataFrame(sum_list[0]['features'][0]['properties'], columns)
    prop = prop.drop_duplicates('geoid', take_last=True)
    
    i = 1
    while i < len_df:
        temp = pd.DataFrame(sum_list[0]['features'][i]['properties'], columns)
        temp = temp.drop_duplicates('geoid', take_last=True)
        frames = [prop, temp]
        prop = pd.concat(frames)
        i = i + 1
    
    prop.set_index('geoid', inplace=True)
    
    prop.to_csv(fileName, encoding='utf-8')

# the urls and the column names that will be used to gather data from piecewise
# first we use Medians
url = 'http://localhost:8080/stats/q/by_county?stats=MedianRTT,MedianUpload,MedianDownload&b.spatial_join=key'
columns = ['upload_median', 'download_median', 'MedianRTT','geoid']
fileName = 'median_values.csv'
gather_data(url, fileName, columns)

# then we use Averages
url = 'http://localhost:8080/stats/q/by_county?stats=AverageRTT,AverageUpload,AverageDownload&b.spatial_join=key'
columns = ['upload_avg', 'download_avg', 'rtt_avg','geoid']
fileName = 'average_values.csv'
gather_data(url, fileName, columns)

# upload and download counts
url = 'http://localhost:8080/stats/q/by_county?stats=UploadCount,DownloadCount&b.spatial_join=key'
columns = ['download_count','upload_count','geoid']
fileName = 'count_values.csv'
gather_data(url, fileName, columns)

# all of them combined
url = 'http://localhost:8080/stats/q/by_county?stats=MedianRTT,MedianUpload,MedianDownload,AverageRTT,AverageUpload,AverageDownload,UploadCount,DownloadCount&b.spatial_join=key'
fileName = 'combined_values.csv'
columns= ['upload_median', 'download_median', 'MedianRTT', 'upload_avg', 'download_avg', 'rtt_avg', 'download_count','upload_count','geoid']
gather_data(url, fileName, columns)