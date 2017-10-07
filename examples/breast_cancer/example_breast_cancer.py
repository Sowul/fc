#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

from fc import FeatureConstructor

clf = RandomForestClassifier(max_depth=3)
fc = FeatureConstructor(clf, 5, 3)
breast_cancer = load_breast_cancer()
fc.fit(breast_cancer.data, breast_cancer.target)

fc.get_params('most_freq')
fc.get_params()
fc.plot()