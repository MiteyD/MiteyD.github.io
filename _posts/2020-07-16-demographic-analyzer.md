---
title: "Demographic Analyzer Project"
date: 2020-07-16
tags: [data wrangling, data science, messy data]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Data Wrangling, Data Science, Messy Data"
mathjax: "true"
---

# Demographic Analyzer

This project involves analyzing demographic data using Pandas. 

The dataset contains demographic data that was extracted from the 1994 Census database.


```python
# Importing the Libraries
import pandas as pd
import math

def calculate_demographic_data(print_data=True):
    # Read data from file
    df = pd.read_csv('adult.data.csv')

    # How many of each race are represented in this dataset? This should be a Pandas series with race names as the index labels.
    race_count = df['race'].value_counts()

    # What is the average age of men?
    def round_half_up(n, decimals=1):
        multiplier = 10 ** decimals
        return math.floor(n*multiplier + 0.5) / multiplier
    average_age_men = round_half_up(df.groupby('sex').mean()['age']['Male'])

    # What is the percentage of people who have a Bachelor's degree?
    percentage_bachelors = round_half_up((df['education'].value_counts()['Bachelors'] / sum(df['education'].value_counts())) * 100)

    # What percentage of people with advanced education (`Bachelors`, `Masters`, or `Doctorate`) make more than 50K?
    # What percentage of people without advanced education make more than 50K?
    
    # with and without `Bachelors`, `Masters`, or `Doctorate`
    Bachelors = df['education'].value_counts()['Bachelors']
    Masters = df['education'].value_counts()['Masters']
    Doctorate = df['education'].value_counts()['Doctorate']
    higher_education_tot = Bachelors + Masters + Doctorate
    lower_education_tot = sum(df['education'].value_counts()) - higher_education_tot
   
    # percentage with salary >50K
    grouped = df.groupby(['salary', 'education'])['education'].count().unstack('salary').reset_index()
    higher_education = (round_half_up(grouped.loc[grouped['education']== 'Bachelors'].iloc[:, -1].values) + 
          round_half_up(grouped.loc[grouped['education']== 'Masters'].iloc[:, -1].values) + 
          round_half_up(grouped.loc[grouped['education']== 'Doctorate'].iloc[:, -1].values))
    higher_education_rich = round_half_up((higher_education / higher_education_tot) * 100)
    
    lower_education = (round_half_up(grouped.loc[grouped['education']== '10th'].iloc[:, -1].values) + 
          round_half_up(grouped.loc[grouped['education']== '11th'].iloc[:, -1].values) + 
          round_half_up(grouped.loc[grouped['education']== '12th'].iloc[:, -1].values) +
          round_half_up(grouped.loc[grouped['education']== '1st-4th'].iloc[:, -1].values) + 
          round_half_up(grouped.loc[grouped['education']== '5th-6th'].iloc[:, -1].values) + 
          round_half_up(grouped.loc[grouped['education']== '7th-8th'].iloc[:, -1].values) +
          round_half_up(grouped.loc[grouped['education']== '9th'].iloc[:, -1].values) + 
          round_half_up(grouped.loc[grouped['education']== 'Assoc-acdm'].iloc[:, -1].values) + 
          round_half_up(grouped.loc[grouped['education']== 'Assoc-voc'].iloc[:, -1].values) +
          round_half_up(grouped.loc[grouped['education']== 'HS-grad'].iloc[:, -1].values) + 
          round_half_up(grouped.loc[grouped['education']== 'Prof-school'].iloc[:, -1].values) +
          round_half_up(grouped.loc[grouped['education']== 'Some-college'].iloc[:, -1].values))
    lower_education_rich = round_half_up((lower_education / lower_education_tot) * 100)
   
    # What is the minimum number of hours a person works per week (hours-per-week feature)?
    min_work_hours = df['hours-per-week'].min()

    # What percentage of the people who work the minimum number of hours per week have a salary of >50K?
    grouped1 = df.groupby(['salary', 'hours-per-week'])['hours-per-week'].count().unstack('salary').reset_index()
    less_than_50K = grouped1.iloc[:, 1]
    more_than_50K = grouped1.iloc[:, 2]
    summed = less_than_50K.add(more_than_50K)
    num_min_workers = round_half_up(float(more_than_50K.head(1) / summed.head(1)))

    rich_percentage = round_half_up(float(more_than_50K.head(1) / summed.head(1)) * 100)
    # What country has the highest percentage of people that earn >50K?
    grouped2 = df.groupby(['salary', 'native-country'])['native-country'].count().unstack('salary').reset_index()
    first = float(grouped2.loc[grouped2['native-country']== '?'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== '?'].iloc[:, 1:].values.sum())
    Cambodia = float(grouped2.loc[grouped2['native-country']== 'Cambodia'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Cambodia'].iloc[:, 1:].values.sum())
    Canada = float(grouped2.loc[grouped2['native-country']== 'Canada'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Canada'].iloc[:, 1:].values.sum())
    China = float(grouped2.loc[grouped2['native-country']== 'China'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'China'].iloc[:, 1:].values.sum())
    Columbia = float(grouped2.loc[grouped2['native-country']== 'Columbia'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Columbia'].iloc[:, 1:].values.sum())
    Cuba = float(grouped2.loc[grouped2['native-country']== 'Cuba'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Cuba'].iloc[:, 1:].values.sum())
    Dominican_Republic = float(grouped2.loc[grouped2['native-country']== 'Dominican-Republic'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Dominican-Republic'].iloc[:, 1:].values.sum())
    Ecuador = float(grouped2.loc[grouped2['native-country']== 'Ecuador'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Ecuador'].iloc[:, 1:].values.sum())
    El_Salvador = float(grouped2.loc[grouped2['native-country']== 'El-Salvador'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'El-Salvador'].iloc[:, 1:].values.sum())
    England = float(grouped2.loc[grouped2['native-country']== 'England'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'England'].iloc[:, 1:].values.sum())
    France = float(grouped2.loc[grouped2['native-country']== 'France'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'France'].iloc[:, 1:].values.sum())
    Germany = float(grouped2.loc[grouped2['native-country']== 'Germany'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Germany'].iloc[:, 1:].values.sum())
    Greece = float(grouped2.loc[grouped2['native-country']== 'Greece'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Greece'].iloc[:, 1:].values.sum())
    Guatemala = float(grouped2.loc[grouped2['native-country']== 'Guatemala'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Guatemala'].iloc[:, 1:].values.sum())
    Haiti = float(grouped2.loc[grouped2['native-country']== 'Haiti'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Haiti'].iloc[:, 1:].values.sum())
    Honduras = float(grouped2.loc[grouped2['native-country']== 'Honduras'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Honduras'].iloc[:, 1:].values.sum())
    Hong = float(grouped2.loc[grouped2['native-country']== 'Hong'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Hong'].iloc[:, 1:].values.sum())
    Hungary = float(grouped2.loc[grouped2['native-country']== 'Hungary'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Hungary'].iloc[:, 1:].values.sum())
    India = float(grouped2.loc[grouped2['native-country']== 'India'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'India'].iloc[:, 1:].values.sum())
    Iran = float(grouped2.loc[grouped2['native-country']== 'Iran'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Iran'].iloc[:, 1:].values.sum())
    Ireland = float(grouped2.loc[grouped2['native-country']== 'Ireland'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Ireland'].iloc[:, 1:].values.sum())
    Italy = float(grouped2.loc[grouped2['native-country']== 'Italy'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Italy'].iloc[:, 1:].values.sum())
    Jamaica = float(grouped2.loc[grouped2['native-country']== 'Jamaica'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Jamaica'].iloc[:, 1:].values.sum())
    Japan = float(grouped2.loc[grouped2['native-country']== 'Japan'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Japan'].iloc[:, 1:].values.sum())
    Laos = float(grouped2.loc[grouped2['native-country']== 'Laos'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Laos'].iloc[:, 1:].values.sum())
    Mexico = float(grouped2.loc[grouped2['native-country']== 'Mexico'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Mexico'].iloc[:, 1:].values.sum())
    Nicaragua =float(grouped2.loc[grouped2['native-country']== 'Nicaragua'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Nicaragua'].iloc[:, 1:].values.sum())
    Peru = float(grouped2.loc[grouped2['native-country']== 'Peru'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Peru'].iloc[:, 1:].values.sum())
    Philippines = float(grouped2.loc[grouped2['native-country']== 'Philippines'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Philippines'].iloc[:, 1:].values.sum())
    Poland = float(grouped2.loc[grouped2['native-country']== 'Poland'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Poland'].iloc[:, 1:].values.sum())
    Portugal = float(grouped2.loc[grouped2['native-country']== 'Portugal'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Portugal'].iloc[:, 1:].values.sum())
    Puerto_Rico = float(grouped2.loc[grouped2['native-country']== 'Puerto-Rico'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Puerto-Rico'].iloc[:, 1:].values.sum())
    Scotland = float(grouped2.loc[grouped2['native-country']== 'Scotland'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Scotland'].iloc[:, 1:].values.sum())
    South = float(grouped2.loc[grouped2['native-country']== 'South'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'South'].iloc[:, 1:].values.sum())
    Taiwan = float(grouped2.loc[grouped2['native-country']== 'Taiwan'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Taiwan'].iloc[:, 1:].values.sum())
    Thailand = float(grouped2.loc[grouped2['native-country']== 'Thailand'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Thailand'].iloc[:, 1:].values.sum())
    Trinadad_n_Tobago = float(grouped2.loc[grouped2['native-country']== 'Trinadad&Tobago'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Trinadad&Tobago'].iloc[:, 1:].values.sum())
    United_States = float(grouped2.loc[grouped2['native-country']== 'United-States'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'United-States'].iloc[:, 1:].values.sum())
    Vietnam = float(grouped2.loc[grouped2['native-country']== 'Vietnam'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Vietnam'].iloc[:, 1:].values.sum())
    Yugoslavia = float(grouped2.loc[grouped2['native-country']== 'Yugoslavia'].iloc[:, -1].values / 
                          grouped2.loc[grouped2['native-country']== 'Yugoslavia'].iloc[:, 1:].values.sum())
    
    
    country_data = {'Country': ['?','Cambodia','Canada','China','Columbia','Cuba',
                         'Dominican_Republic','Ecuador','El_Salvador','England',
                         'France','Germany','Greece','Guatemala','Haiti',
                         'Honduras','Hong','Hungary','India','Iran','Ireland','Italy','Jamaica',
                         'Japan','Laos','Mexico','Nicaragua','Peru',
                         'Philippines','Poland','Portugal','Puerto_Rico','Scotland','South',
                         'Taiwan','Thailand','Trinadad_n_Tobago','United_States','Vietnam','Yugoslavia'],
                    'Earnings': [first, Cambodia, Canada, China, Columbia, Cuba,
                          Dominican_Republic, Ecuador, El_Salvador, England,
                          France, Germany, Greece, Guatemala, Haiti,
                          Honduras, Hong, Hungary, India, Iran, Ireland, Italy, Jamaica,
                          Japan, Laos, Mexico, Nicaragua, Peru,
                          Philippines, Poland, Portugal, Puerto_Rico, Scotland, South,
                          Taiwan, Thailand, Trinadad_n_Tobago, United_States, Vietnam, Yugoslavia]}
    
    arranged_country_data = pd.DataFrame(country_data)
    
    highest_earning_country = (arranged_country_data.groupby(['Country', 'Earnings']).max().sort_values("Earnings", ascending = False).
                               head(1).reset_index().iloc[:, 0].values.all())
    highest_earning_country_percentage = round_half_up(float(arranged_country_data.groupby(['Country', 'Earnings']).max().sort_values("Earnings", ascending = False).
                                                             head(1).reset_index().iloc[:, 1].values) * 100)
   

    # Identify the most popular occupation for those who earn >50K in India.
    grouped3 = df.groupby(['salary', 'occupation', 'native-country'])['native-country'].count().unstack('salary').reset_index()
    popular_occupation = grouped3.pivot_table(index=['occupation'], columns='native-country', values='>50K').reset_index()
    top_IN_occupation = popular_occupation.groupby('India', sort=False).max().sort_index(ascending=False).head(1).iloc[:, 0].all()
    

```
