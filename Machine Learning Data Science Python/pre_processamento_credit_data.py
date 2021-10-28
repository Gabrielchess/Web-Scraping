# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
base = pd.read_csv('credit_data.csv')
base.describe()
base.loc[base['age'] < 0]