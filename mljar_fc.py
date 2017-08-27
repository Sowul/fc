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

    def get_params(self, ind='best'):
        """Print best or most frequent set of new features.

        Args:
            ind : string, 'best' or 'most_freq'
                Determines which set of features save to a file.

        """
        self.ga.get_params(ind)

    def save(self, filename, ind='best'):
        """Save the best or most frequent set of features to a file.

        Args:
            filename : string

            ind : string, 'best' or 'most_freq'
                Determines which set of features save to a file.

        """
        if ind == 'best' or ind == 'most_freq':
            self.ga.save(filename)
        else:
            raise ValueError("ind must be 'best' or 'most_freq'.")

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
