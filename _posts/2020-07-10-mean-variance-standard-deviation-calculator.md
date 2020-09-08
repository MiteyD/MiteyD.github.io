---
title: "Mean-Variance-Standard Deviation Calculator"
date: 2020-07-10
tags: [data wrangling, data science, messy data]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Data Wrangling, Data Science, Messy Data"
mathjax: "true"
---

# Mean-Variance-Standard Deviation Calculator

This project entailed creating a function that uses Numpy to output the mean, variance, and standard deviation of the rows, columns, and elements in a 3 x 3 matrix.


```python

import numpy as np

def calculate(list):
  if len(list) != 9:
    raise ValueError("List must contain nine numbers.")
  x = np.array(list).reshape(3, 3)
  calculations = {
    k: [func(x, axis=ax).tolist()
      for ax in [0, 1, None]] 
      for (k, func) 
      in zip(["mean", "variance", "standard deviation", "max", "min", "sum"], 
             [np.mean, np.var, np.std, np.max, np.min, np.sum])
  } 


  return calculations

try:
    calculate([1,2,3,4,5,6,7,8,9])
except ValueError as error:
    print(error)

```
