#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Bartosz Sowul"

from ga import GeneticAlgorithm

class FeatureConstructor:
    """Create new features using genetic algorithm."""

    def __init__(self, clf, fold, duration):
        """Init method.

        Args:
            clf : classifier object implementing 'fit'
                Classfier used for scoring new features.

            fold : int, cross-validation generator or an iterable
                Determines the cross-validation splitting strategy,
                see also http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html.

            duration : int
                Determines how many minutes a genetic algorithm runs.

            ga : GeneticAlgorithm
                Object used for creating new sets of features.

        """
        self.clf = clf
        self.fold = fold
        self.duration = duration
        self.ga = GeneticAlgorithm(clf, fold, duration)

    def fit(self, X, y):
        """Fit estimator.

        Args:
            X : array-like
                The data to fit.

            y : array-like
                The target variable.

        """
        self.ga.fit(X, y)

    def get_params(self):
        """Print best set of new features."""
        self.ga.get_params()

    def save(self, filename):
        """Save the best set of features to a file.

        Args:
            filename : string

        """
        self.ga.save(filename)

    def load(self, filename):
        """Load a set of features from a file.

        Args:
            filename : string

        Returns:
            Tuple with a set of features.

        """
        return self.ga.load(filename)

    def transform(self, X, individual):
        """Transform dataset into new one using created features.

        Args:
            X : array-like
                The data to transform.

            individual : tuple
                Tuple with a set of features.

        Returns:
            New dataset, array-like.

        """
        return self.ga.transform(X, individual)
