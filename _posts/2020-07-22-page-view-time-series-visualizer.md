---
title: "Page View Time Series Visualizer"
date: 2020-07-22
tags: [data wrangling, data science, messy data]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Data Wrangling, Data Science, Messy Data"
mathjax: "true"
---

# Page View Time Series Visualizer

For this project you will visualize time series data using a line chart, bar chart, and box plots. 

You will use Pandas, matplotlib, and seaborn to visualize a dataset containing the number of page views each day on the freeCodeCamp.org forum from 2016-05-09 to 2019-12-03. 

The data visualizations will help you understand the patterns in visits and identify yearly and monthly growth.

The datasets can be found in my github account using the link below or from https://www.freecodecamp.org/learn/


```python
# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Import data (Make sure to parse dates. Consider setting index column to 'date'.)
df = pd.read_csv('fcc-forum-pageviews.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Clean data
df = df[(df['value'] >= df['value'].quantile(0.025)) & (df['value'] <= df['value'].quantile(0.975))]


def draw_line_plot():
    # Draw line plot
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.plot(df, label='lineplots', color='r', linewidth=1.0)
    plt.xlabel('Date')
    plt.ylabel('Page Views')
    plt.title('Daily freeCodeCamp Forum Page Views 5/2016-12/2019')
    




    # Save image and return fig (don't change this part)
    fig.savefig('line_plot.png')
    return fig

def draw_bar_plot():
    # Copy and modify data for monthly bar plot
    df_bar = df.resample('M').mean()
    df_bar = df_bar.reset_index()
    df_bar['date'] = pd.to_datetime(df_bar['date'])
    df_bar['Years'] = df_bar['date'].dt.year
    df_bar['Months'] = df_bar['date'].dt.month_name()
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    mapping = {month: i for i, month in enumerate(months)}
    key = df_bar['Months'].map(mapping)
    df_bar.iloc[key.argsort()]
    
    
    # Draw bar plot
    fig, ax = plt.subplots()
    df_bar.pivot_table(index='Years', columns=key, values='value', fill_value=0).plot(kind='bar', label='Months', stacked=False, figsize=(9,6), ax=ax)
    my_labels =  ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    plt.legend(labels=my_labels, title='Months')
    plt.xlabel('Years')
    plt.ylabel('Average Page Views')
    plt.tight_layout()
    
    


    # Save image and return fig (don't change this part)
    fig.savefig('bar_plot.png')
    return fig

def draw_box_plot():
    # Prepare data for box plots (this part is done!)
    df_box = df.copy()
    df_box.reset_index(inplace=True)
    df_box['year'] = [d.year for d in df_box.date]
    df_box['month'] = [d.strftime('%b') for d in df_box.date]

    # Draw box plots (using Seaborn)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20,8))
    a1 = sns.boxplot(y='value', x='year', data=df_box, ax=ax1)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Page Views')
    ax1.set_title('Year-wise Box Plot (Trend)')
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    mapping = {month: i for i, month in enumerate(months)}
    key = df_box['month'].map(mapping)
    key = pd.DataFrame(key)
    df_box['key'] = key['month']
    a2 = sns.boxplot(x='key', y='value', data=df_box, ax=ax2)
    my_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax2.set_xticklabels(labels=my_labels)
    plt.xlabel('Month')
    plt.ylabel('Page Views')
    plt.title('Month-wise Box Plot (Seasonality)')
    

    # Save image and return fig (don't change this part)
    fig.savefig('box_plot.png')
    return fig

```
