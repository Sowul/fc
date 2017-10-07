#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from collections import namedtuple
from copy import deepcopy
from datetime import datetime, timedelta
import io
import json
from operator import attrgetter
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from tqdm import trange, tqdm

class GeneticAlgorithm:
    """Algorithm used for creating new features."""

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

            metric :
                Metric used for scoring new features,
                see also http://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values.

            pop_members : int
                Determines how big the population is.

            elite : int
                Determines how many individuals are guaranteed a place in the next generation.

            migrants : int
                Determines how many new individuals (possible solutions in a search space) are created (migrate) each generation.

            n_operators : int
                Length of a list of operators used for transforming a dataset.

        """
        self.clf = clf
        self.fold = fold
        self.duration = duration
        self.metric = 'neg_log_loss'
        self.pop_members = 100
        self.elite = 4
        self.migrants = 6
        self.__operators = [np.diff,
                            np.gradient,
                            np.nanprod,
                            np.nansum,
                            np.nanmax,
                            np.nanmin,
                            np.ptp,
                            np.nanpercentile,
                            np.nanmedian,
                            np.nanmean,
                            np.nanstd,
                            np.nanvar,
                            np.trapz,
                            np.sin,
                            np.cos,
                            np.around,
                            np.rint,
                            np.fix,
                            np.floor,
                            np.ceil,
                            np.trunc,
                            np.log1p,
                            np.sinc,
                            np.negative,
                            np.sqrt,
                            np.fabs,
                            np.sign,
                            np.add,
                            np.multiply,
                            np.subtract,
                            np.mod]
        self.n_operators = len(self.__operators)
        self._columns = []
        self._individuals = []
        self._Individual = namedtuple('Individual',
                            ['transformations', 'columns', 'score'])
        self._best_individuals = []
        self._BestIndividual = namedtuple('BestIndividual',
                               ['gen_num', 'transformations', 'columns',
                               'score', 'count'])
        self._gen_score = []
        self._Generations = []
        self._Generation = namedtuple('Generation',
                            ['gen_num', 'mean_score', 'best_ind'])

    def _create_individual(self):
        """Return new individual.

        Returns:
            Array of integers.

        """
        return np.reshape(np.random.choice(self.n_operators,
                        self.n_features, replace=True), (-1, self.n_features))

    def _create_population(self):
        """Return new population.

        Returns:
            Array of individuals (integers).

        """
        population = np.empty((0, self.n_features), dtype=np.int8)
        for i, member in enumerate(range(self.pop_members)):
            population = np.append(population, self._create_individual(),
                                   axis=0)
            cols = []
            for feature in population[i]:
                if feature <= 12:
                    cols.append(np.array([-1]))
                elif (feature > 12 and feature <= 26):
                    cols.append(np.random.randint(self.X.shape[1], size=1))
                else:
                    cols.append(np.random.randint(self.X.shape[1], size=2))
            self._columns.append(cols)
        return population

    def _apply_function(self, i, col, feature):
        """Transform a dataset along given axis.

        Args:
            i : int
                Index of an individual in a given population.

            col : int
                Column index to perform operation on.

            feature : int
                Index of an operator used for data transformation.

        Returns:
            Array.

        """
        if (feature <= 1):
            return np.nan_to_num(np.nanmean(np.apply_along_axis(
                                self.__operators[feature], 1, self.X), axis=1))
        elif (feature > 1 and feature <= 12):
            if (feature == 7):
                return np.nan_to_num(np.apply_along_axis(
                                self.__operators[feature], 1, self.X, 70))
            else:
                return np.nan_to_num(np.apply_along_axis(
                                self.__operators[feature], 1, self.X))
        elif (feature > 12 and feature <= 26):
            col1 = self._columns[i][col][0]
            vfunc = np.vectorize(self.__operators[feature])
            return np.nan_to_num(vfunc(self.X[:, col1]))
        else:
            col1 = self._columns[i][col][0]
            col2 = self._columns[i][col][1]
            vfunc = np.vectorize(self.__operators[feature])
            return np.nan_to_num(vfunc(self.X[:, col1], self.X[:, col2]))

    def _transform(self, i, member):
        """Transform a dataset performing mathematical operations.

        Args:
            i : int
                Index of an individual in a given population.

            member : int array
                Array of indices of mathematical operators.

        Returns:
            Transformed dataset, array-like.

        """
        z = np.zeros((self.X.shape[0], self.n_features), dtype=self.X.dtype)
        for col, feature in enumerate(member):
            z[:, col] = self._apply_function(i, col, feature)
        return np.concatenate((z, self.X), axis=1)

    def _get_fitness(self, clf, X, y):
        """Compute the scores based on the testing set for each iteration of cross-validation.

        Args:
            clf : classifier object implementing 'fit'
                Classfier used for scoring new features.

            X : array-like
                The data to fit.

            y : array-like
                The target variable.

        Returns:
            Cross-validation score for a given dataset.

        """
        return cross_val_score(clf, X, y,
                            scoring=self.metric, cv=self.fold, n_jobs=-1).mean()

    def _select_parents(self, q=4):
        """Select parents from a population of individuals using tournament selection.

        Args:
            q : int
                Tournament size.

        Returns:
            Array of integers.

        """
        parents = np.empty((0, q), dtype=np.int8)
        for i in range(self.pop_members-self.migrants-self.elite):
            parent = np.random.choice(
                                self.pop_members, size=(1, q), replace=False)
            parents = np.append(parents, parent, axis=0)
        parents = np.amin(parents, axis=1)
        np.random.shuffle(parents)
        return np.reshape(parents, (len(parents)//2, 2))

    def _crossover(self, parents, population):
        """Produce a child solution from two parents using uniform crossover scheme.

        Args:
            parents : array of integers
                Indices of parents from a given population.

            population : list of tuples
                List of individuals and their fitness.

        Returns:
            Array of integers.

        """
        new_population = np.empty((0, self.n_features), dtype=np.int8)
        children = np.empty((0, self.n_features), dtype=np.int8)
        population = deepcopy(population)
        for couple in parents:
            first_parent = np.reshape(population[couple[0]].transformations,
                                                        (-1, self.n_features))
            second_parent = np.reshape(population[couple[1]].transformations,
                                                        (-1, self.n_features))
            for feature in range(first_parent.shape[1]):
                if 0.5 > np.random.random_sample():
                    first_parent[0][feature], second_parent[0][feature] = (
                    second_parent[0][feature], first_parent[0][feature])
            children = np.append(children, first_parent, axis=0)
            children = np.append(children, second_parent, axis=0)
        new_population = np.append(new_population, children, axis=0)
        return new_population

    def _mutate(self, new_population, std):
        """Alter gene values in an individual.

        Args:
            new_population : array of integers
                Array of individuals.

            std : float
                Standard deviation of the indices of mathematical operators chosen in the previous generation.

        Returns:
            Array of integers.

        """
        std = int(round(std))
        mutation = np.random.randint(-std-1, std+1, size=new_population.shape)
        new_population = (new_population + mutation) % self.n_operators
        return new_population

    def _create_next_generation(self, population):
        """Create next generation.

        Args:
            population : list of tuples
                Previous generation.

        Returns:
            New population, array of integers.

        """
        population = sorted(population, key=attrgetter('score'), reverse=True)
        parents = self._select_parents()
        new_population = self._crossover(parents, population)
        new_popualtion = self._mutate(new_population,
                                        np.std([ind[0] for ind in population]))
        for migrant in range(self.migrants):
            new_population = np.append(new_population,
                                        self._create_individual(), axis=0)
        self._columns = []
        for ind in new_population:
            cols = []
            for feature in ind:
                if feature <= 12:
                    cols.append(np.array([-1]))
                elif (feature > 12 and feature <= 26):
                    cols.append(np.random.randint(self.X.shape[1], size=1))
                else:
                    cols.append(np.random.randint(self.X.shape[1], size=2))
            self._columns.append(cols)
        for i in range(self.elite):
            elitist = np.reshape(population[i].transformations,
                                (-1, self.n_features))
            new_population = np.append(new_population, elitist, axis=0)
            self._columns.append(population[i].columns)
        return new_population

    def fit(self, X, y):
        """Fit estimator.

        Args:
            X : array-like
                The data to fit.

            y : array-like
                The target variable.

        """
        self.X = np.asarray(X)
        self.y = np.asarray(y).reshape(y.shape[0], )
        self.n_features = np.random.random_integers(10)

        self._base_score = cross_val_score(self.clf, self.X, self.y,
                                       scoring=self.metric, cv=self.fold).mean()
        print("Base score: {}\n".format(self._base_score))
        self._best_score = self._base_score

        population = self._create_population()
        gen = 0

        end_time = datetime.now() + timedelta(minutes=self.duration)
        while datetime.now() < end_time:
            for i, member in enumerate(tqdm(population, desc='Individual',
                                                                leave=False)):
                new_X = self._transform(i, member)
                score = self._get_fitness(self.clf, new_X, self.y)
                self._individuals.append(self._Individual(member,
                                            self._columns[i], score))
            self._Generations.append(self._individuals)

            best = sorted(self._individuals, key=lambda tup: tup.score,
                            reverse=True)[0]
            count = 1
            if gen > 0:
                for elem in self._best_individuals:
                    if(list(best.transformations) == list(elem.transformations)):
                        count += 1
            else:
                pass
            self._best_individuals.append(
                self._BestIndividual(gen, best.transformations, best.columns,
                                        best.score, count))
            if (gen == 0):
                self._best_score = self._best_individuals[gen]
            if (best.score > self._best_score.score):
                self._best_score = self._best_individuals[gen]
            else:
                pass
            self._gen_score.append(self._Generation(gen,
            sum([tup[2] for tup in self._individuals])/len(self._individuals),
            self._best_individuals[gen]))
            population = self._create_next_generation(self._individuals)
            self._individuals = []
            gen += 1
        else:
            self._most_freq = sorted(self._best_individuals, key=lambda tup: tup.count,
                            reverse=True)[0]

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
        z = np.zeros((X.shape[0], self.n_features), dtype=X.dtype)
        for col, feature in enumerate(individual.transformations):
            if (feature <= 1):
                z[:, col] = np.nan_to_num(np.nanmean(np.apply_along_axis(
                                    self.__operators[feature], 1, X), axis=1))
            elif (feature > 1 and feature <= 12):
                if (feature == 7):
                    z[:, col] = np.nan_to_num(np.apply_along_axis(
                                    self.__operators[feature], 1, X, 70))
                else:
                    z[:, col] = np.nan_to_num(np.apply_along_axis(
                                    self.__operators[feature], 1, X))
            elif (feature > 12 and feature <= 26):
                col1 = individual.columns[col][0]
                vfunc = np.vectorize(self.__operators[feature])
                z[:, col] = np.nan_to_num(vfunc(X[:, col1]))
            else:
                col1 = individual.columns[col][0]
                col2 = individual.columns[col][1]
                vfunc = np.vectorize(self.__operators[feature])
                z[:, col] = np.nan_to_num(vfunc(X[:, col1], X[:, col2]))
        return np.concatenate((z, X), axis=1)

    def get_params(self, ind='best'):
        """Print best or most frequent set of new features.

        Args:
            ind : string, 'best' or 'most_freq'
                Determines which set of features save to a file.

        """
        if ind == 'best':
            print('Best params:')
            for i, feature in enumerate(self._best_score.transformations):
                if (feature <= 12):
                    print('\tfeature {}: {} every row'.format(i, self.__operators[feature].__name__))
                elif (feature > 12 and feature <= 26):
                    print('\tfeature {}: {} col {}'.format(i, self.__operators[feature].__name__, self._best_score.columns[i][0]))
                else:
                    print('\tfeature {}: {} cols {} and {}'.format(i, self.__operators[feature].__name__,
                            self._best_score.columns[i][0], self._best_score.columns[i][1]))
            else:
                print('neg_log_loss: {}\n'.format(self._best_score.score))
        elif ind == 'most_freq':
            avg = 0
            for elem in self._best_individuals:
                if(list(self._most_freq.transformations) == list(elem.transformations)):
                    avg += elem.score
            print("Most frequent individual ({} times, average score {}): ".format(
                                   self._most_freq.count, avg/self._most_freq.count))
            for i, feature in enumerate(self._most_freq.transformations):
                if (feature <= 12):
                    print('\tfeature {}: {} every row'.format(i, self.__operators[feature].__name__))
                elif (feature > 12 and feature <= 26):
                    print('\tfeature {}: {} col {}'.format(i, self.__operators[feature].__name__, self._most_freq.columns[i][0]))
                else:
                    print('\tfeature {}: {} cols {} and {}'.format(i, self.__operators[feature].__name__,
                            self._most_freq.columns[i][0], self._most_freq.columns[i][1]))
            else:
                print('neg_log_loss: {}\n'.format(self._most_freq.score))
        else:
            pass


    def save(self, filename, ind='best'):
        """Save the best set of features to a file.

        Args:
            filename : string

            ind : string, 'best' or 'most_freq'
                Determines which set of features save to a file.

        """
        def dump_to_dict(ind):
            individual = tuple()
            if ind == 'best':
                individual = self._Individual(self._best_score.transformations.tolist(),
                    [x.tolist() for x in self._best_score.columns], self._best_score.score)
            elif ind == 'most_freq':
                individual = self._Individual(self._most_freq.transformations.tolist(),
                    [x.tolist() for x in self._most_freq.columns], self._most_freq.score)
            else:
                raise ValueError("ind must be 'best' or 'most_freq'.")
            individual = json.dumps(individual._asdict(), indent=4,
                    separators=(',', ': '), ensure_ascii=False)
            return individual

        with io.open(filename, 'w', encoding='utf8') as outfile:
            individual = dump_to_dict(ind)
            from sys import version_info
            if version_info < (3, 0):
                outfile.write(unicode(individual))
            else:
                outfile.write(str(individual))

    def load(self, filename):
        """Load a set of features from a file.

        Args:
            filename : string

        Returns:
            Tuple with a set of features.

        """
        with open(filename) as in_file:
            return self._Individual(**json.load(in_file))

    def plot(self):
        """Plot data from the genetic algorithm."""
        mean_score = [x[1] for x in self._gen_score]
        best_score = [x[2][3] for x in self._gen_score]

        plt.plot(self._best_score.gen_num, self._best_score.score, 'ro')
        plt.plot([0, len(mean_score)], [self._base_score, self._base_score], 'b--', lw=2)
        plt.plot(range(len(mean_score)), mean_score, 'k', label='mean')
        plt.plot(range(len(best_score)), best_score, 'g', label='best')
        plt.xlabel('generation')
        plt.ylabel(self.metric)
        plt.legend(loc='upper left')
        plt.show()