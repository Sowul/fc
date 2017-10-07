#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from fc import FeatureConstructor

clf = RandomForestClassifier(max_depth=3)
fc = FeatureConstructor(clf, 5, 3)
iris = load_iris()
fc.fit(iris.data, iris.target)

fc.get_params('most_freq')
fc.get_params()
fc.plot()