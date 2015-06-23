# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:59:54 2015

@author: oana
"""
import urllib2
import requests
import pandas as pd

data_df = pd.read_csv('national_county.txt', converters={'statefp': lambda x: str(x), 'countyfp': lambda x: str(x)})

counties = []
for index, row in data_df.iterrows():
    code = row['statefp'] + row['countyfp']
    counties.append(code)

lencounties = len(counties)

call_count = []
i = 0
# call the api using a size of 10 counties / batch
while i < lencounties - 10:
    batch = []
    for j in range (0,10):
        batch.append(counties[i])
        i = i + 1
    call_count.append(batch)

sum_list = []
demo_list = []    
for batch in call_count:
    url = 'http://www.broadbandmap.gov/broadbandmap/analyze/jun2014/summary/population/county/ids/'
    url_demo = 'http://www.broadbandmap.gov/broadbandmap/demographic/jun2014/county/ids/'
    for county in batch:
        url = url + str(county) + ','
        url_demo = url_demo + str(county) + ','
    url = url + '?format=json'
    url_demo = url_demo + '?format=json'
    try:
        data_json = requests.get(url).json()
        sum_list.append(data_json)
        data_json = requests.get(url_demo).json()
        demo_list.append(data_json)
    except urllib2.URLError, e:
        print 'No kittez. Got an error code:', e


broad_df = pd.DataFrame(sum_list[0]['Results'])
demo_df = pd.DataFrame(demo_list[0]['Results'])
len_broad = len(sum_list)
print "len_broad"
print len_broad
len_demo = len(demo_list)
print "len_demo"
print len_demo
i = 1

while i < len_broad:
    broad_df = broad_df.append(sum_list[i]['Results'], ignore_index=True)
    i = i + 1

broad_df.to_csv('broad_sum.csv', encoding='utf-8')
#print broad_df.head(5)

i = 1
while i < len_demo:
    demo_df = demo_df.append(demo_list[i]['Results'], ignore_index=True)
    i = i + 1

demo_df.to_csv('demo.csv', encoding='utf-8')

