#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from builtins import int
from collections import namedtuple
from copy import deepcopy
from datetime import datetime, timedelta
import io
import json
import multiprocessing
from operator import attrgetter
from timeit import default_timer as timer
import sys

import dill as pickle
import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np
from pathos.multiprocessing import ProcessPool
from six import integer_types, string_types, text_type
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from tqdm import trange, tqdm

class GeneticAlgorithm:
    """Algorithm used for creating new features."""

    def __init__(self, clf, cv=5, duration=None, max_iter=None, base_included=True):
        """Init method.

        Args:
            clf : classifier object implementing 'fit'
                Classfier used for scoring new features.

            cv : int, cross-validation generator or an iterable
                Determines the cross-validation splitting strategy,
                see also http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html.

            duration : float
                Determines how many minutes a genetic algorithm runs.

            max_iter : int
                Determines how many iterations a genetic algorithm runs.

            base_included : bool
                Determines whether or not the base dataset is included during the evaluation of newly created features.
        """
        if all(var is not None for var in (duration, max_iter)):
            raise ValueError('Duration and max_iter variables are both not None. One of them should be None.')
        if all(var is None for var in (duration, max_iter)):
            raise ValueError('Duration and max_iter variables are both None. Only one of them should be None.')
        if duration is not None:
            if isinstance(duration, (integer_types, float)):
                self.duration = duration
                self.max_iter = max_iter
            else:
                raise ValueError('Duration value must be a float or an integer.')
        else:
            if isinstance(max_iter, integer_types):
                self.max_iter = max_iter
                self.duration = duration
            else:
                raise ValueError('Max_iter value must be an integer.')
        self.clf = clf
        self.cv = cv
        self.metric = 'neg_log_loss'
        self.pop_members = 100
        self.elite = 4
        self.migrants = 6
        self.__operators = [np.diff,
                            np.gradient,
                            np.nanprod,#2
                            np.nansum,#3
                            np.nanmax,
                            np.nanmin,
                            np.ptp,
                            np.nanpercentile,
                            np.nanmedian,
                            np.nanmean,
                            np.nanstd,
                            np.nanvar,
                            np.trapz,
                            np.sin,#13
                            np.cos,#14
                            np.around,
                            np.rint,
                            np.fix,
                            np.floor,
                            np.ceil,
                            np.trunc,
                            np.log1p,#21
                            np.sinc,
                            np.negative,
                            np.sqrt,#24
                            np.fabs,#25
                            np.sign,#26
                            np.add,#27
                            np.multiply,#28
                            np.subtract,#29
                            np.mod]#30
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
        self._unique_members = 0
        self._skipped_members = 0
        if isinstance(base_included, bool):
            self._base_included = base_included
        else:
            raise ValueError('base_included value must be a bool.')
        self._created_models = 0

    def __repr__(self):
        return '{}(clf={}, cv={},\n\t\tmetric={}, base_included={},\n\t\t{}={}, pop_members={})'.format(
            self.__class__.__name__, self.clf.__class__.__name__, self.cv, self.metric, self._base_included,
            'duration' if self.duration is not None else 'max_iter',
            self.duration if self.duration is not None else self.max_iter, self.pop_members)

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
            if (feature == 2):#np.nanprod
                arr = self.X
                return np.nan_to_num(ne.evaluate('prod(arr, axis=1)'))
            elif (feature == 3):#np.nansum
                arr = self.X
                return np.nan_to_num(ne.evaluate('sum(arr, axis=1)'))
            elif (feature == 7):
                return np.nan_to_num(np.apply_along_axis(
                                self.__operators[feature], 1, self.X, 70))
            else:
                return np.nan_to_num(np.apply_along_axis(
                                self.__operators[feature], 1, self.X))
        elif (feature > 12 and feature <= 26):
            col1 = self._columns[i][col][0]
            if (feature == 13):#np.sin
                arr = self.X[:, col1]
                return np.nan_to_num(ne.evaluate('sin(arr)'))
            elif (feature == 14):#np.cos
                arr = self.X[:, col1]
                return np.nan_to_num(ne.evaluate('cos(arr)'))
            elif (feature == 15):#np.around
                return np.nan_to_num(self.__operators[feature](self.X[:, col1], decimals=2))
            elif (feature == 21):#np.log1p
                arr = self.X[:, col1]
                return np.nan_to_num(ne.evaluate('log1p(arr)'))
            elif (feature == 24):#np.sqrt
                arr = self.X[:, col1]
                return np.nan_to_num(ne.evaluate('sqrt(arr)'))
            elif (feature == 25):#np.fabs
                arr = self.X[:, col1]
                return np.nan_to_num(ne.evaluate('abs(arr)'))
            elif (feature == 26):#np.sign
                arr = self.X[:, col1]
                return np.nan_to_num(ne.evaluate('-arr'))
            else:
                vfunc = np.vectorize(self.__operators[feature])
                return np.nan_to_num(vfunc(self.X[:, col1]))
        else:
            col1 = self._columns[i][col][0]
            col2 = self._columns[i][col][1]
            if (feature == 27):#np.add
                arr1 = self.X[:, col1]
                arr2 = self.X[:, col2]
                return np.nan_to_num(ne.evaluate('arr1 + arr2'))
            elif (feature == 28):#np.multiply
                arr1 = self.X[:, col1]
                arr2 = self.X[:, col2]
                return np.nan_to_num(ne.evaluate('arr1 * arr2'))
            elif (feature == 29):#np.subtract
                arr1 = self.X[:, col1]
                arr2 = self.X[:, col2]
                return np.nan_to_num(ne.evaluate('arr1 - arr2'))
            elif (feature == 30):#np.mod
                arr1 = self.X[:, col1]
                arr2 = self.X[:, col2]
                return np.nan_to_num(ne.evaluate('arr1 % arr2'))
            else:
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
        z = np.empty((self.X.shape[0], self.n_features), dtype=self.X.dtype)
        for col, feature in enumerate(member):
            z[:, col] = self._apply_function(i, col, feature)
        if self._base_included:
            return np.concatenate((z, self.X), axis=1)
        else:
            return z

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
                            scoring=self.metric, cv=self.cv, n_jobs=1).mean()

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
        mutation = np.random.randint(-int(std)-1, int(std)+1, size=new_population.shape) 
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

    def _score_ind(self, ind):
        new_X = self._transform(ind[0], ind[1])
        score = self._get_fitness(self.clf, new_X, self.y)
        return self._Individual(ind[1], self._columns[ind[0]], score)

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
        self._base_score = cross_val_score(self.clf, self.X, y=self.y,
                                           scoring=self.metric, cv=self.cv).mean()
        print("Base score: {}\n".format(self._base_score))
        self._best_score = self._base_score

        population = self._create_population()
        gen = 0
        total_time = 0

        if self.duration is not None:
            import math
            ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(math.floor(n//10)%10!=1)*(n%10<4)*n%10::4])

            end_time = datetime.now() + timedelta(minutes=self.duration)

            while datetime.now() < end_time:
                p = ProcessPool(nodes=multiprocessing.cpu_count())
                print('Creating {} generation models...'.format(ordinal(gen)))
                start = timer()
                self._individuals = p.map(self._score_ind, list(enumerate(population)))

                total_time = total_time + timer() - start

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
                print('Done.')
            else:
                self._most_freq = sorted(self._best_individuals, key=lambda tup: tup.count,
                                reverse=True)[0]
                print('gen: {}'.format(gen))
                print('avg time per gen: {0:0.1f}'.format(total_time/gen))
        else:
            for i in trange(self.max_iter, desc='Generation', leave=False):
                p = ProcessPool(nodes=multiprocessing.cpu_count())

                start = timer()
                self._individuals = p.map(self._score_ind, list(enumerate(population)))
                total_time = total_time + timer() - start

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
                print('gen: {}'.format(gen))
                print('avg time per gen: {0:0.1f}'.format(total_time/gen))

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
        z = np.empty((X.shape[0], self.n_features), dtype=X.dtype)
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
        if self._base_included:
            return np.concatenate((z, X), axis=1)
        else:
            return z

    def get_params(self, ind='most_freq'):
        """Print best or most frequent set of new features.

        Args:
            ind : string, 'best' or 'most_freq'
                Determines which set of features print.

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


    def save(self, filename, ind='most_freq'):
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

        if isinstance(filename, (string_types, text_type)):
            with io.open(filename, 'w', encoding='utf8') as outfile:
                individual = dump_to_dict(ind)
                from sys import version_info
                if version_info < (3, 0):
                    outfile.write(unicode(individual))
                else:
                    outfile.write(str(individual))
        else:
            raise ValueError('Filename must be a string.')

    def load(self, filename):
        """Load a set of features from a file.

        Args:
            filename : string

        Returns:
            Tuple with a set of features.

        """
        if isinstance(filename, (string_types, text_type)):
            with open(filename) as in_file:
                return self._Individual(**json.load(in_file))
        else:
            raise ValueError('Filename must be a string.')

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