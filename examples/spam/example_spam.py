#!/usr/bin/env python
# -*- coding: utf-8 -*-

# data source: https://archive.ics.uci.edu/ml/datasets/Spambase

from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from fc import FeatureConstructor

clf = RandomForestClassifier(max_depth=3)
fc = FeatureConstructor(clf, 5, 10)
df = pd.read_csv('data.csv')
data = df.as_matrix()
X = np.array(data[:, :-1])
y = np.array(data[:, -1])
fc.fit(X, y)

fc.get_params('most_freq')
fc.get_params()
fc.plot()