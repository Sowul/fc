#!/usr/bin/env python
# -*- coding: utf-8 -*-

from filecmp import cmp
from os import remove

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from fc import FeatureConstructor

@pytest.fixture(scope='module')
def fc():
    np.random.seed(10)
    clf = RandomForestClassifier(max_depth=3, random_state=2222)
    fc = FeatureConstructor(clf, 5, duration=0.5)
    iris = load_iris()
    fc.fit(iris.data, iris.target)
    return fc

def test_fc_fit(fc):
    assert fc.ga._base_score < fc.ga._best_score.score

def test_fc_save_load(fc):
    fc.save('tests/fc_saved_ind.json')
    loaded_ind = fc.load('tests/fc_saved_ind.json')
    assert np.array_equal(fc.ga._best_score.transformations.tolist(), loaded_ind.transformations) == True
    assert np.array_equal([x.tolist() for x in fc.ga._best_score.columns], loaded_ind.columns) == True
    remove('tests/fc_saved_ind.json')
    fc.save('tests/fc_saved_ind.json', 'best')
    loaded_ind = fc.load('tests/fc_saved_ind.json')
    assert np.array_equal(fc.ga._best_score.transformations.tolist(), loaded_ind.transformations) == True
    assert np.array_equal([x.tolist() for x in fc.ga._best_score.columns], loaded_ind.columns) == True
    remove('tests/fc_saved_ind.json')
    fc.save('tests/fc_saved_ind.json', 'most_freq')
    loaded_ind = fc.load('tests/fc_saved_ind.json')
    assert np.array_equal(fc.ga._most_freq.transformations.tolist(), loaded_ind.transformations) == True
    assert np.array_equal([x.tolist() for x in fc.ga._most_freq.columns], loaded_ind.columns) == True
    remove('tests/fc_saved_ind.json')

def test_fc_transform(fc):
    iris = load_iris()
    new_X = fc.transform(iris.data, fc.ga._best_score)
    assert new_X.shape[0] == iris.data.shape[0]
    assert new_X.shape[1] == iris.data.shape[1] + fc.ga.n_features
