#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 11:30:48 2018

@author: carsonluuu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import datetime

from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import *


def Hawaii_transform(seg):
    res = []
    for item in seg:
        x = item[0]
        y = item[1]
        res.append((x + 5200000, y-1400000))
    return res

def Alaska_transform(seg):
    res = []
    for item in seg:
        x = item[0]
        y = item[1]
        res.append((0.35*x + 1100000, 0.35*y-1300000))
    return res

"""
IMPORTANT
This is EXAMPLE code.
There are a few things missing:
1) You may need to play with the colors in the US map.
2) This code assumes you are running in Jupyter Notebook or on your own system.
   If you are using the VM, you will instead need to play with writing the images
   to PNG files with decent margins and sizes.
3) The US map only has code for the Positive case. I leave the negative case to you.
4) Alaska and Hawaii got dropped off the map, but it's late, and I want you to have this
   code. So, if you can fix Hawaii and Alask, ExTrA CrEdIt. The source contains info
   about adding them back.
"""

#df1 = pd.read_csv("cp.csv", error_bad_lines=False)
#df2 = pd.read_csv("cn.csv", error_bad_lines=False)
#
#a = df1.groupby('title').mean()
#b = df2.groupby('title').mean()
#a = a['prediction'].replace(1, np.nan)
#b = b['prediction'].replace(1, np.nan)
#c = a.sort_values(ascending=False)
#d = b.sort_values(ascending=False)
#
#res1 = c[0:11]
#res2 = d[0:11]
#
#res1.to_csv()
#
#res2.to_csv()

"""
PLOT 1: SENTIMENT OVER TIME (TIME SERIES PLOT)
"""
# Assumes a file called time_data.csv that has columns
# date, Positive, Negative. Use absolute path.

ts = pd.read_csv("time_data.csv")
# Remove erroneous row.
ts = ts[ts['date'] != 20181231]
ts['date'] = ts['date'].astype("str")
plt.figure(figsize=(12,5))
ts.date = pd.to_datetime(ts['date'], format='%Y-%m-%d')
ts.set_index(['date'],inplace=True)

ax = ts.plot(title="President Trump Sentiment on /r/politics Over Time",
        color=['green', 'red'],
       ylim=(0, 1.05))
ax.plot()

"""
PLOT 2: SENTIMENT BY STATE (POSITIVE AND NEGATIVE SEPARATELY)
# This example only shows positive, I will leave negative to you.
"""

# This assumes you have a CSV file called "state_data.csv" with the columns:
# state, Positive, Negative
#
# You should use the FULL PATH to the file, just in case.

state_data = pd.read_csv("state_data.csv")

"""
You also need to download the following files. Put them somewhere convenient:
https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shp
https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.dbf
https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shx
"""

# Lambert Conformal map of lower 48 states.
m = Basemap(llcrnrlon=-121, llcrnrlat=20, urcrnrlon=-62, urcrnrlat=51,
        projection='lcc', lat_1=32, lat_2=45, lon_0=-95)
shp_info = m.readshapefile('st99_d00','states',drawbounds=True)  # No extension specified in path here.
pos_data = dict(zip(state_data.state, state_data.Positive))
neg_data = dict(zip(state_data.state, state_data.Negative))

# choose a color for each state based on sentiment.
pos_colors = {}
statenames = []
pos_cmap = plt.cm.Greens # use 'hot' colormap

vmin = 0; vmax = 0.6 # set range.
for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia', 'Puerto Rico']:
        pos = pos_data[statename]
        if statename == 'Tennessee':
            pos = 0.3
        pos_colors[statename] = pos_cmap(1. - np.sqrt(( pos - vmin )/( vmax - vmin)))[:3]
    statenames.append(statename)
# cycle through state names, color each one.

# POSITIVE MAP

ax = plt.gca() # get current axes instance
for nshape, seg in enumerate(m.states):
    # skip Puerto Rico and DC
    if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
        if statenames[nshape] == 'Alaska':
            seg = Alaska_transform(seg)
        if statenames[nshape] == 'Hawaii':
            seg = Hawaii_transform(seg)
        color = rgb2hex(pos_colors[statenames[nshape]]) 
        poly = Polygon(seg, facecolor=color, edgecolor=color)
        ax.add_patch(poly)
plt.title('Positive Trump Sentiment Across the US')
plt.show()



m = Basemap(llcrnrlon=-121, llcrnrlat=20, urcrnrlon=-62, urcrnrlat=51,
        projection='lcc', lat_1=32, lat_2=45, lon_0=-95)
shp_info = m.readshapefile('st99_d00','states',drawbounds=True)  # No extension specified in path here.
pos_data = dict(zip(state_data.state, state_data.Positive))
neg_data = dict(zip(state_data.state, state_data.Negative))


# choose a color for each state based on sentiment.
neg_colors = {}
statenames = []
neg_cmap = plt.cm.Greens # use 'hot' colormap

vmin = 0; vmax = 1 # set range.
for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia', 'Puerto Rico']:
        neg = neg_data[statename]
        neg_colors[statename] = neg_cmap(1. - np.sqrt(( neg - vmin )/( vmax - vmin)))[:3]
    statenames.append(statename)
# cycle through state names, color each one.

# NEGATIVE MAP
ax = plt.gca() # get current axes instance
for nshape, seg in enumerate(m.states):
    # skip Puerto Rico and DC
    if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
        if statenames[nshape] == 'Alaska':
            seg = Alaska_transform(seg)
        if statenames[nshape] == 'Hawaii':
            seg = Hawaii_transform(seg)
        color = rgb2hex(neg_colors[statenames[nshape]]) 
        poly = Polygon(seg, facecolor=color, edgecolor=color)
        ax.add_patch(poly)
plt.title('Negative Trump Sentiment Across the US')
plt.show()


m = Basemap(llcrnrlon=-121, llcrnrlat=20, urcrnrlon=-62, urcrnrlat=51,
        projection='lcc', lat_1=32, lat_2=45, lon_0=-95)
shp_info = m.readshapefile('st99_d00','states',drawbounds=True)  # No extension specified in path here.
pos_data = dict(zip(state_data.state, state_data.Positive))
neg_data = dict(zip(state_data.state, state_data.Negative))


# choose a color for each state based on sentiment.
diff_colors = {}
statenames = []
diff_cmap = plt.cm.Greens # use 'hot' colormap

vmin = 0; vmax = 1 # set range.
for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia', 'Puerto Rico']:
        pos = pos_data[statename]
        neg = neg_data[statename]
        diff_colors[statename] = neg_cmap(1. - np.sqrt(( neg - pos - vmin )/( vmax - vmin)))[:3]
    statenames.append(statename)
# cycle through state names, color each one.

# diff MAP
ax = plt.gca() # get current axes instance
for nshape, seg in enumerate(m.states):
    # skip Puerto Rico and DC
    if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
        if statenames[nshape] == 'Alaska':
            seg = Alaska_transform(seg)
        if statenames[nshape] == 'Hawaii':
            seg = Hawaii_transform(seg)
        color = rgb2hex(diff_colors[statenames[nshape]]) 
        poly = Polygon(seg, facecolor=color, edgecolor=color)
        ax.add_patch(poly)
plt.title('Difference of Negative and Positive for Trump Sentiment Across the US')
plt.show()

# SOURCE: https://stackoverflow.com/questions/39742305/how-to-use-basemap-python-to-plot-us-with-50-states
# (this misses Alaska and Hawaii. If you can get them to work, EXTRA CREDIT)

"""
PART 4 SHOULD BE DONE IN SPARK
"""

"""
PLOT 5A: SENTIMENT BY STORY SCORE
"""
# What is the purpose of this? It helps us determine if the story score
# should be a feature in the model. Remember that /r/politics is pretty
# biased.

# Assumes a CSV file called submission_score.csv with the following coluns
# submission_score, Positive, Negative

story = pd.read_csv("submission_score.csv")
plt.figure(figsize=(12,5))
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(story['submission_score'], story['Positive'], s=10, c='b', marker="s", label='Positive')
ax1.scatter(story['submission_score'], story['Negative'], s=10, c='r', marker="o", label='Negative')
plt.legend(loc='lower right');

plt.xlabel('President Trump Sentiment by Submission Score')
plt.ylabel("Percent Sentiment")
plt.show()

"""
PLOT 5B: SENTIMENT BY COMMENT SCORE
"""
# What is the purpose of this? It helps us determine if the comment score
# should be a feature in the model. Remember that /r/politics is pretty
# biased.

# Assumes a CSV file called comment_score.csv with the following columns
# comment_score, Positive, Negative

story = pd.read_csv("comment_score.csv")
plt.figure(figsize=(12,5))
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(story['comment_score'], story['Positive'], s=10, c='b', marker="s", label='Positive')
ax1.scatter(story['comment_score'], story['Negative'], s=10, c='r', marker="o", label='Negative')
plt.legend(loc='lower right');

plt.xlabel('President Trump Sentiment by Comment Score')
plt.ylabel("Percent Sentiment")
plt.show()


states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']

m = Basemap(llcrnrlon=-121, llcrnrlat=20, urcrnrlon=-62, urcrnrlat=51,
        projection='lcc', lat_1=32, lat_2=45, lon_0=-95)
shp_info = m.readshapefile('st99_d00','states',drawbounds=True)  # No extension specified in path here.
pos_data = dict(zip(state_data.state, state_data.Positive))
neg_data = dict(zip(state_data.state, state_data.Negative))

# choose a color for each state based on sentiment.
pos_colors = {}
statenames = []
pos_cmap = plt.cm.Greens # use 'hot' colormap

a = 0.8449/0.3173
vmin = 0; vmax = 0.6 # set range.
for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia', 'Puerto Rico']:
        pos = pos_data[statename]
        neg = neg_data[statename]
        if statename == 'Tennessee':
            pos = 0.31
        ratio = a*pos/neg
        print(ratio)
        if ratio >= 1.05:
            pos_colors[statename] = pos_cmap(0.8)[:3]
        elif ratio >= 0.99:
            pos_colors[statename] = pos_cmap(0.6)[:3]
        elif ratio >= 0.96:
            pos_colors[statename] = pos_cmap(0.4)[:3]
        else:
            pos_colors[statename] = pos_cmap(0.2)[:3]  
    statenames.append(statename)
# cycle through state names, color each one.

# EXTRA MAP

ax = plt.gca() # get current axes instance
for nshape, seg in enumerate(m.states):
    # skip Puerto Rico and DC
    if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
        if statenames[nshape] == 'Alaska':
            seg = Alaska_transform(seg)
        if statenames[nshape] == 'Hawaii':
            seg = Hawaii_transform(seg)
        color = rgb2hex(pos_colors[statenames[nshape]]) 
#        print(color)
        poly = Polygon(seg, facecolor=color, edgecolor=color)
        ax.add_patch(poly)
plt.title('Approval Rate Over State')

l4 = Patch(color='#157f3b', label='>1.05')
l3 = Patch(color='#98d594', label='0.96-1')
l2 = Patch(color='#d3eecd', label='<0.96')
l1 = Patch(color='#4bb062', label='1-1.05')
plt.legend(handles=[l4, l1, l3, l2],title='Ratio', loc='lower right')

plt.show()







