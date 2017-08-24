#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

__author__ = "Bartosz Sowul"

from collections import namedtuple
from copy import deepcopy
from datetime import datetime, timedelta
import io
import json
from operator import attrgetter

import numpy as np
from numpy import diff, gradient
from numpy import nanprod, nansum, nanmax, nanmin, ptp, nanpercentile, nanmedian, nanmean, nanstd, nanvar
from numpy import sin, cos, around, rint, fix, floor, ceil, trunc, trapz, exp, expm1, exp2, log1p, sinc, reciprocal, negative, sqrt, fabs, sign
from numpy import logaddexp, logaddexp2, add, multiply, divide, power, subtract, true_divide, floor_divide, mod
from sklearn.model_selection import cross_val_score
from tqdm import trange, tqdm

class GeneticAlgorithm:

    def __init__(self, clf, fold, duration):
        self.clf = clf
        self.fold = fold
        self.duration = duration
        self.metric = 'neg_log_loss'
        self.pop_members = 100
        self.elite = 4
        self.migrants = 6
        self.__operators = [diff,
                            gradient,
                            nanprod,
                            nansum,
                            nanmax,
                            nanmin,
                            ptp,
                            nanpercentile,
                            nanmedian,
                            nanmean,
                            nanstd,
                            nanvar,
                            trapz,
                            sin,
                            cos,
                            around,
                            rint,
                            fix,
                            floor,
                            ceil,
                            trunc,
                            exp,
                            expm1,
                            exp2,
                            log1p,
                            sinc,
                            reciprocal,
                            negative,
                            sqrt,
                            fabs,
                            sign,
                            logaddexp,
                            logaddexp2,
                            add,
                            multiply,
                            divide,
                            power,
                            subtract,
                            true_divide,
                            floor_divide,
                            mod]
        self.n_operators = len(self.__operators)
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
        self._columns = []

    def _create_individual(self):
        return np.reshape(np.random.choice(len(self.__operators),
                        self.n_features, replace=False), (-1, self.n_features))

    def _create_population(self):
        population = np.empty((0, self.n_features), dtype=np.int8)
        for i, member in enumerate(range(self.pop_members)):
            population = np.append(population, self._create_individual(),
                                   axis=0)
            cols = []
            for feature in population[i]:
                if feature <= 12:
                    cols.append(np.array([-1]))
                elif (feature > 12 and feature <= 30):
                    cols.append(np.random.randint(self.X.shape[1], size=1))
                else:
                    cols.append(np.random.randint(self.X.shape[1], size=2))
            self._columns.append(cols)
        return population

    def _apply_function(self, i, col, feature):
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
        elif (feature > 12 and feature <= 30):
            col1 = self._columns[i][col][0]
            vfunc = np.vectorize(self.__operators[feature])
            return np.nan_to_num(vfunc(self.X[:, col1]))
        else:
            col1 = self._columns[i][col][0]
            col2 = self._columns[i][col][1]
            vfunc = np.vectorize(self.__operators[feature])
            return np.nan_to_num(vfunc(self.X[:, col1], self.X[:, col2]))

    def _transform(self, i, member):
        z = np.zeros((self.X.shape[0], self.n_features), dtype=self.X.dtype)
        for col, feature in enumerate(member):
            z[:, col] = self._apply_function(i, col, feature)
        return np.concatenate((z, self.X), axis=1)

    def _get_fitness(self, clf, X, y):
        return cross_val_score(clf, X, y,
                            scoring=self.metric, cv=self.fold, n_jobs=-1).mean()

    def _select_parents(self, q=4):
        parents = np.empty((0, q), dtype=np.int8)
        for i in range(self.pop_members-self.migrants-self.elite):
            parent = np.random.choice(
                                self.pop_members, size=(1, q), replace=False)
            parents = np.append(parents, parent, axis=0)
        parents = np.amin(parents, axis=1)
        np.random.shuffle(parents)
        return np.reshape(parents, (len(parents)//2, 2))

    def _crossover(self, parents, population):
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
        std = int(round(std))
        mutation = np.random.randint(-std-1, std+1, size=new_population.shape)
        new_population = (new_population + mutation) % len(self.__operators)
        return new_population

    def _create_next_generation(self, population):
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
                elif (feature > 12 and feature <= 30):
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
        self.X = np.asarray(X)
        self.y = np.asarray(y).reshape(y.shape[0], )
        self.n_features = np.random.random_integers(2*self.X.shape[1])

        self._base_score = cross_val_score(self.clf, self.X, self.y,
                                       scoring=self.metric, cv=self.fold).mean()
        print("Base score: ", self._base_score)
        self._best_score = self._base_score

        population = self._create_population()
        gen = 0

        end_time = datetime.now() + timedelta(minutes=self.duration)
        while datetime.now() < end_time:
            for j, member in enumerate(tqdm(population, desc='Individual',
                                                                leave=False)):
                new_X = self._transform(j, member)
                score = self._get_fitness(self.clf, new_X, self.y)
                self._individuals.append(self._Individual(member,
                                            self._columns[j], score))
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
            best = sorted(self._best_individuals, key=lambda tup: tup.count,
                            reverse=True)[0]
            if(best.count > 1):
                avg = 0
                for elem in self._best_individuals:
                    if(list(best.transformations) == list(elem.transformations)):
                        avg += elem.score
                print("Most frequent individual: ", best)
                print("Average score: ", avg/best.count)

    def transform(self, X, individual):
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
            elif (feature > 12 and feature <= 30):
                col1 = individual.columns[col][0]
                vfunc = np.vectorize(self.__operators[feature])
                z[:, col] = np.nan_to_num(vfunc(X[:, col1]))
            else:
                col1 = individual.columns[col][0]
                col2 = individual.columns[col][1]
                vfunc = np.vectorize(self.__operators[feature])
                z[:, col] = np.nan_to_num(vfunc(X[:, col1], X[:, col2]))
        return np.concatenate((z, X), axis=1)

    def get_params(self):
        print('Best params:')
        for i, feature in enumerate(self._best_score.transformations):
            if (feature <= 1 and feature <= 12):
                print('\tfeature {}: {} every row'.format(i, self.__operators[feature].__name__))
            elif (feature > 12 and feature <= 30):
                print('\tfeature {}: {} col {}'.format(i, self.__operators[feature].__name__, self._best_score.columns[i][0]))
            else:
                print('\tfeature {}: {} cols {} and {}'.format(i, self.__operators[feature].__name__,
                        self._best_score.columns[i][0], self._best_score.columns[i][1]))
        else:
            print('Logloss: {}'.format(-self._best_score.score))

    def save(self, filename):
        with io.open(filename, 'w', encoding='utf8') as outfile:
            individual = self._Individual(self._best_score.transformations.tolist(),
                [x.tolist() for x in self._best_score.columns], self._best_score.score)
            ind = json.dumps(individual._asdict(), indent=4,
                                separators=(',', ': '), ensure_ascii=False)
            from sys import version_info
            if version_info < (3, 0):
                outfile.write(unicode(ind))
            else:
                outfile.write(str(ind))

    def load(self, filename):
        with open(filename) as infile:
            return self._Individual(**json.load(infile))
