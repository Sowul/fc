#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Bartosz Sowul"

from ga import GeneticAlgorithm

class FeatureConstructor:

    def __init__(self, clf, fold, duration):
        self.clf = clf
        self.fold = fold
        self.duration = duration
        self.ga = GeneticAlgorithm(clf, fold, duration)

    def fit(self, X, y):
        self.ga.fit(X, y)

    def get_params(self):
        self.ga.get_params()

    def save(self, filename):
        self.ga.save(filename)

    def load(self, filename):
        return self.ga.load(filename)

    def transform(self, X, individual):
        return self.ga.transform(X, individual)
