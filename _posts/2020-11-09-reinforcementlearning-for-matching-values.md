---
title: "Reinforcement Learning for Matching Values of an Engineering Model"
date: 2020-11-09
tags: [data wrangling, data science, messy data]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Data Wrangling, Data Science, Messy Data"
mathjax: "true"
---

# Reinforcement Learning for Matching Values of an Engineering Model

This was to test if reinforcement learning is a good candidate to calibrate an engineering model. We had a 3D model that has various properties. By varying these properties the model can be properly calibrated. The first setup was a test case using a very simple model, varying only 1 parameter, and an reinforcement learning loop and check if it will work. A successful outcome is the algorithm can identify the correct value(or close) of the parameter that correctly calibrates the engineering model.

## Create a folder and name it 'data'

This is where you will put your dataset for the model.

## Create a folder and name it 'env'

This is where you will create you own custom environment.

Create 'GasProdEnv.py' file.


```python
# This is the code for 'GasProdEnv.py'

''' 
We are going to be utilising OpenAI Gym for the creation of the environment
So our dataset has 8 columns as follows:
a)Conductivity Layers
b) Conductivity
c) Days
d) Injection Rate
e) Injection Pressure 
f) Oil Rate
g) Water Rate
h) Production Pressure

Key words and values that we have to define in the environment 
a) Action Space
b) Observation_Space
c) Reset

for 3D env rendering we may need PyOpenGL
Remember that each layer has a conductivity value but is not unique, for example, conductivity layer 1,6,8,9 share the conductivity 
value 25
By varying the pressure (Independent variable) the conductivity values and production values (first of all I find this very vague)
are affected
Formulae for this correlation

~pressure(what we are searching for) = ~conductivity values(we already have this values) and 
                                        ~ production values(we are also searching for this values)
'''


#import necessary dependencies
import gym
import random
import numpy as np
from numpy import *
import pandas as pd

from gym  import spaces

#Create the environment constants
MIN_production_pressure = 0
Max_production_Pressure = 36000
Production_Pressure_Capacity = 36000
MAX_Injection_pressure = 48276
MIN_Injection_pressure = 0
MAX_Injection_Rate = 100
MIN_Injection_Rate = 0
MAX_Conductivity_Layers = 77
MIN_Conductivity_Layers = 1
MAX_Conductivity_Value = 7200
MIN_Conductivity_Value = 0
MAX_Number_of_Days = 30194
MIN_Number_of_Days = 1
MAX_Water_Rate= 81
MIN_Water_Rate = 1
MAX_Oil_Rate = 100
MIN_Oil_Rate = 1
MAX_STEPS = 20000

INITIAL_Pressure = 0

class GasProdEnv(gym.Env):
    # a custom gas production environment for OpenAI gymn environment
    metadata = {"render.modes": ["human"]}
    
    #create and define the observation and action space
    def __init__(self, df):
        super(GasProdEnv, self).__init__()
        
        
        #the reward for our case should be headed towards a constant pressure
        self.df = df
        
        self.Production_Pressure_Capacity = Production_Pressure_Capacity
        self.MIN_Conductivity_Layers = MIN_Conductivity_Layers
        self.MAX_Conductivity_Layers = MAX_Conductivity_Layers
        self.MIN_production_pressure = MIN_production_pressure
        self.Max_production_Pressure = Max_production_Pressure
        self.MIN_Conductivity_Value = MIN_Conductivity_Value
        self.MAX_Conductivity_Value = MAX_Conductivity_Value
        
        #The pressure range is between 0 and 35548.52734
        self.reward_range = range(-1, 1)
        
        #create the action_space which am assuming am going to be using Discrete values if I can
        self.action_space = spaces.Discrete(self.Production_Pressure_Capacity)

        self.observation_space = spaces.Box(low=self.MIN_Conductivity_Value, high=self.MAX_Conductivity_Value, shape=(10, ), dtype=int32)
        """
        In arrays we have what we call the dtypes and are immutable just like strings and numbers in Python
        We have diffrent types of tensors ex: a scalar or a rank-0 tensor which is a scalar that contains one value
        and has no axes

        In the observation space below we create a vector or a rank-1 tensor and since it is not a numpy array and more of a tensor use the dtype as int32
        self.InitialProductionPressure = self.observation_space.sample()
        .sample returns a tuple of values and the random.randit() returns a scalar value
        """
    
    
    def _next_observation(self):
        if self.CurrentConductivityValue.shape > (5,):
          frame = np.take(self.CurrentConductivityValue, [-5, -4, -3, -2, -1])
        else:
          frame = self.CurrentConductivityValue
        # Append additional data and scale each value to between 0-1 for the observation space which is filled with whatever values that you defined in the reset
        added = np.array([self.Production_Pressure_Capacity,self.MIN_Conductivity_Value,self.MAX_Conductivity_Value,
            self.MIN_production_pressure,self.Max_production_Pressure]) 
        obs = np.append(frame, added)
        
        return obs
    
    
    def _take_action(self, action):
        
        self.rewards = []

        for l in range(0, 77):#l here represents the no. of layers
            for t in range(0, 16):#t here represents the no. of iterations
                self.i = random.randint(0, 76)
                self.k = random.randint(0, 15)
                self.real_conductivity = self.df.loc[:, 'Conductivity'].values[self.i][self.k]
                self.real_days = self.df.loc[:, 'Days'].values[self.i][self.k]
                self.real_injection_rate = self.df.loc[:, 'Injection rate'].values[self.i][self.k]
                self.real_injection_pressure = self.df.loc[:, 'Injection Pressure'].values[self.i][self.k]
                self.real_oil_rate = self.df.loc[:, 'Oil rate'].values[self.i][self.k]
                self.real_water_rate = self.df.loc[:, 'Water rate'].values[self.i][self.k]
                self.real_production_pressure = self.df.loc[:, 'Production Pressure'].values[self.i][self.k]
                if self.conductivity == self.df.loc[:, 'Conductivity'].values[self.i][self.k]:
                    self.reward = 1
                    self.days = self.df.loc[:, 'Days'].values[self.i][self.k]
                    self.injection_rate = self.df.loc[:, 'Injection rate'].values[self.i][self.k]
                    self.injection_pressure = self.df.loc[:, 'Injection Pressure'].values[self.i][self.k]
                    self.oil_rate = self.df.loc[:, 'Oil rate'].values[self.i][self.k]
                    self.water_rate = self.df.loc[:, 'Water rate'].values[self.i][self.k]
                    self.production_pressure = self.df.loc[:, 'Production Pressure'].values[self.i][self.k]
                elif self.conductivity - 50 <= self.df.loc[:, 'Conductivity'].values[self.i][self.k] <= self.conductivity + 50:
                    self.reward = 0.5
                    self.days = self.df.loc[:, 'Days'].values[self.i][self.k]
                    self.injection_rate = self.df.loc[:, 'Injection rate'].values[self.i][self.k]
                    self.injection_pressure = self.df.loc[:, 'Injection Pressure'].values[self.i][self.k]
                    self.oil_rate = self.df.loc[:, 'Oil rate'].values[self.i][self.k]
                    self.water_rate = self.df.loc[:, 'Water rate'].values[self.i][self.k]
                    self.production_pressure = self.df.loc[:, 'Production Pressure'].values[self.i][self.k]
                elif self.conductivity - 100 <= self.df.loc[:, 'Conductivity'].values[self.i][self.k] <= self.conductivity + 100:
                    self.reward = 0.3
                    self.days = self.df.loc[:, 'Days'].values[self.i][self.k]
                    self.injection_rate = self.df.loc[:, 'Injection rate'].values[self.i][self.k]
                    self.injection_pressure = self.df.loc[:, 'Injection Pressure'].values[self.i][self.k]
                    self.oil_rate = self.df.loc[:, 'Oil rate'].values[self.i][self.k]
                    self.water_rate = self.df.loc[:, 'Water rate'].values[self.i][self.k]
                    self.production_pressure = self.df.loc[:, 'Production Pressure'].values[self.i][self.k]               
                elif self.conductivity - 150 <= self.df.loc[:, 'Conductivity'].values[self.i][self.k] <= self.conductivity + 150:
                    self.reward = 0.1
                    self.days = self.df.loc[:, 'Days'].values[self.i][self.k]
                    self.injection_rate = self.df.loc[:, 'Injection rate'].values[self.i][self.k]
                    self.injection_pressure = self.df.loc[:, 'Injection Pressure'].values[self.i][self.k]
                    self.oil_rate = self.df.loc[:, 'Oil rate'].values[self.i][self.k]
                    self.water_rate = self.df.loc[:, 'Water rate'].values[self.i][self.k]
                    self.production_pressure = self.df.loc[:, 'Production Pressure'].values[self.i][self.k]
                elif self.conductivity - 200 <= self.df.loc[:, 'Conductivity'].values[self.i][self.k] <= self.conductivity + 200:
                    self.reward = 0.05
                    self.days = self.df.loc[:, 'Days'].values[self.i][self.k]
                    self.injection_rate = self.df.loc[:, 'Injection rate'].values[self.i][self.k]
                    self.injection_pressure = self.df.loc[:, 'Injection Pressure'].values[self.i][self.k]
                    self.oil_rate = self.df.loc[:, 'Oil rate'].values[self.i][self.k]
                    self.water_rate = self.df.loc[:, 'Water rate'].values[self.i][self.k]
                    self.production_pressure = self.df.loc[:, 'Production Pressure'].values[self.i][self.k]
                elif self.conductivity - 250 <= self.df.loc[:, 'Conductivity'].values[self.i][self.k] <= self.conductivity + 250:
                    self.reward = 0.01
                    self.days = self.df.loc[:, 'Days'].values[self.i][self.k]
                    self.injection_rate = self.df.loc[:, 'Injection rate'].values[self.i][self.k]
                    self.injection_pressure = self.df.loc[:, 'Injection Pressure'].values[self.i][self.k]
                    self.oil_rate = self.df.loc[:, 'Oil rate'].values[self.i][self.k]
                    self.water_rate = self.df.loc[:, 'Water rate'].values[self.i][self.k]
                    self.production_pressure = self.df.loc[:, 'Production Pressure'].values[self.i][self.k]
                else:
                    self.reward = -1
                    self.days = self.df.loc[:, 'Days'].values[self.i][self.k]
                    self.injection_rate = self.df.loc[:, 'Injection rate'].values[self.i][self.k]
                    self.injection_pressure = self.df.loc[:, 'Injection Pressure'].values[self.i][self.k]
                    self.oil_rate = self.df.loc[:, 'Oil rate'].values[self.i][self.k]
                    self.water_rate = self.df.loc[:, 'Water rate'].values[self.i][self.k]
                    self.production_pressure = self.df.loc[:, 'Production Pressure'].values[self.i][self.k]
                self.rewards.append(self.reward)
        
        self.history = (sum(self.rewards)/len(self.rewards))
    
    
    def step(self, action):
        #Execute one time step within the environment as dictated by the _take_action
        self._take_action(action)
        
        self.CurrentConductivityValue += 1
        
        reward = self.history
        done = self.Production_Pressure_Capacity <= 0
        obs = self._next_observation()
        return obs, reward, done, {}
    
    
    def reset(self):
        """
        Reset the Environment
        """
        self.Production_Pressure_Capacity = 0
        self.MIN_Conductivity_Layers = 0
        self.MAX_Conductivity_Layers = 0
        self.MIN_production_pressure = 0
        self.Max_production_Pressure = 0
        
        ## Set the CurrentConductivityValue to a random point within the data frame
        self.CurrentConductivityValue = self.observation_space.sample()
        self.current_step = random.randint(1, len(self.df.loc[:, 'Layer'].values))
        self.conductivity = random.uniform(0, 7200)
        
        return self._next_observation()

    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print('----------------------------------------------------------------------------------------')
        print('!!!!!START!!!!!')
        print('PREDICTED VALUES')
        print(f'Layer: {self.current_step}')
        print(f'Iteration: {self.k}')
        print(f'Conductivity: {self.conductivity}')
        print(f'Days: {self.days}')
        print(f'Injection rate: {self.injection_rate}')
        print(f'Injection Pressure: {self.injection_pressure}')
        print(f'Oil rate: {self.oil_rate}')
        print(f'Water rate: {self.water_rate}')
        print(f'Production Pressure: {self.production_pressure}')
        print(f'Reward: {self.history}')
        print('REAL VALUES')
        print(f'Real Conductivity: {self.real_conductivity}')
        print(f'Real Days: {self.real_days}')
        print(f'Real Injection rate: {self.real_injection_rate}')
        print(f'Real Injection Pressure: {self.real_injection_pressure}')
        print(f'Real Oil rate: {self.real_oil_rate}')
        print(f'Real Water rate: {self.real_water_rate}')
        print(f'Real Production Pressure: {self.real_production_pressure}')
        print('!!!!!END!!!!!')
        print('----------------------------------------------------------------------------------------')


```

## In the parent folder, create 'main.py' file


```python
# This is the code for 'main.py'

""" Installing the packages
! pip install gym
! pip install tensorflow==1.15
! pip install stable-baselines
"""
#import all the necessary dependency libraries
import gym
import datetime as dt
import pandas as pd
import numpy as np

#the stable_baselines is used for the model training in RL
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

#This is used for checking the environment to see whether we will get any errors once the environment is done
from stable_baselines.common.env_checker import check_env

#importing our just created environment and using the class that we have just created 
from env.GasProdEnv import GasProdEnv


#preprocessing the data given
df = pd.read_excel('/content/drive/My Drive/RL_for_matching_values/data/rl_test_data.xlsx')
df = df.drop([0, 1, 2, 3]).reset_index()
df = df.drop(columns=['index', 'Unnamed: 2', 'Unnamed: 9', 'Unnamed: 18', 
                      'Unnamed: 27', 'Unnamed: 36', 'Unnamed: 45','Unnamed: 54', 
                      'Unnamed: 63', 'Unnamed: 72', 'Unnamed: 81', 'Unnamed: 90', 
                      'Unnamed: 99', 'Unnamed: 108', 'Unnamed: 117', 'Unnamed: 126', 
                      'Unnamed: 135'])
#grab the first row for the header
new_header = df.iloc[0]

#take the data less the header row
df = df[1:]
#set the header row as the df header
df.columns = new_header
df.rename(columns=df.iloc[0])
df = df.reset_index()
df = df.drop(columns='index')
df = df[: -1426]
df = df.apply(pd.to_numeric)
df.isnull()
df.isnull().sum()#returns the column names with the null values plus the sum

"""
~~~~~~~~~~~~~~~~~Understanding the dataset given~~~~~~~~~

looking at the data we already know that the conductivity layers column is just 77 but how do we deal with this
From the four guidelines, guideline no. 2 stipulates that each layer in the dataset has a corresponding conductivity value which is 
ideally true and meaning there are as many conductivity values as there are conductivity layers which is 77 but on a lighter
note looking at our data:
a) The no of days still continue to increase past the corresponding 77th layer
b) The Injection rate is a constant past the 77th layer
c) The Injection pressure has a flunctuating trend past the 77th layer
d) The oil Rate has a decreasing trend past the 77th layer
e) The water Rate is a constant value as seen (0)
f)  The production Pressure is a constant value past the 77th layer(3500)
"""
"""
In the case where you want to randomize an agent to run in your custom env here is the code to do so

env = GasProdEnv()
obs = env.reset()
n_steps = 10
for _ in range(n_steps):
    # Random action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
"""
 
#this allows for one to see all the descriptive data even in the case when the dataframe is too large
pd.set_option('max_columns', None)
print(df.head())

#The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: GasProdEnv(df)])
state = GasProdEnv(df).reset()
print(f'Required shape: {GasProdEnv(df).observation_space.shape}, state: {state.shape}')
print(f'low: {GasProdEnv(df).observation_space.low}, checking: {np.all(state >= GasProdEnv(df).observation_space.low)}')
print(f'high: {GasProdEnv(df).observation_space.high}, checking: {np.all(state <= GasProdEnv(df).observation_space.high)}')

check_env(GasProdEnv(df))

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=200)
obs = env.reset()
for i in range(2000):
  action, _states = model.predict(obs)
  obs, rewards, done, info = env.step(action)
  env.render()
```

## Below is the code to run the whole model


```python
! pip install gym
! pip install tensorflow==1.15
! pip install stable-baselines
```


```python
! python /content/drive/My\ Drive/RL_for_matching_values/main.py
```
