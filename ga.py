#!/usr/bin/env python

__author__ = "Bartosz Sowul"

from collections import namedtuple
from copy import deepcopy
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
        self.__best_individuals = []
        self.__BestIndividual = namedtuple('BestIndividual',
                               ['gen_num', 'transformations', 'score', 'count'])
        self.__gen_score = []
        self.__generations = []
        self.__Generation = namedtuple('Generation',
                            ['gen_num', 'mean_score', 'best_ind'])
        self.__individuals = []
        self.__Individual = namedtuple('Individual',
                            ['transformations', 'score'])

    def __create_individual(self):
        return np.reshape(np.random.choice(len(self.__operators),
                        self.X.shape[1], replace=False), (-1, self.X.shape[1]))
                        
    def __create_population(self):
        population = np.empty((0, self.X.shape[1]), dtype=np.int8)
        for member in range(self.pop_members):
            population = np.append(population, self.__create_individual(),
                                   axis=0)
        return population

    def __apply_function(self, feature):
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
            col = np.random.randint(self.X.shape[1])
            vfunc = np.vectorize(self.__operators[feature])
            return np.nan_to_num(vfunc(self.X[:, col]))
        else:
            col1 = np.random.randint(self.X.shape[1])
            col2 = np.random.randint(self.X.shape[1])
            vfunc = np.vectorize(self.__operators[feature])
            return np.nan_to_num(vfunc(self.X[:, col1], self.X[:, col2]))

    def __transform(self, member):
        z = np.zeros(self.X.shape, dtype=self.X.dtype)
        for col, feature in enumerate(member):
            z[:, col] = self.__apply_function(feature)
        # with or without old dataset?
        return np.concatenate((z, self.X), axis=1)

    def __get_fitness(self, clf, X, y):
        return cross_val_score(clf, X, y,
                            scoring=self.metric, cv=self.fold, n_jobs=-1).mean()
    
    def __select_parents(self, q=4):
        parents = np.empty((0, q), dtype=np.int8)
        for i in range(self.pop_members-self.migrants-self.elite):
            parent = np.random.choice(
                                self.pop_members, size=(1, q), replace=False)
            parents = np.append(parents, parent, axis=0)
        parents = np.amin(parents, axis=1)
        np.random.shuffle(parents)
        # self-crossover?
        return np.reshape(parents, (len(parents)//2, 2))
        
    def __crossover(self, parents, population):
        new_population = np.empty((0, self.X.shape[1]), dtype=np.int8)
        children = np.empty((0, self.X.shape[1]), dtype=np.int8)
        population = deepcopy(population)
        for couple in parents:
            first_parent = np.reshape(population[couple[0]].transformations,
                                                        (-1, self.X.shape[1]))
            second_parent = np.reshape(population[couple[1]].transformations,
                                                        (-1, self.X.shape[1]))
            for feature in range(first_parent.shape[1]):
                if 0.5 > np.random.random_sample():
                    first_parent[0][feature], second_parent[0][feature] = (
                    second_parent[0][feature], first_parent[0][feature])
            children = np.append(children, first_parent, axis=0)
            children = np.append(children, second_parent, axis=0)        
        new_population = np.append(new_population, children, axis=0)
        return new_population
    
    def __mutate(self, new_population, std):
        std = int(round(std))
        mutation = np.random.randint(-std-1, std+1, size=new_population.shape)
        new_population = (new_population + mutation) % len(self.__operators)
        return new_population

    def __create_next_generation(self, population):
        population = sorted(population, key=attrgetter('score'), reverse=True)
        parents = self.select_parents()
        new_population = self.crossover(parents, population)
        new_popualtion = self.mutate(new_population, np.std(population)[0])      
        for migrant in range(self.migrants):
            new_population = np.append(new_population,
                                        self.__create_individual(), axis=0) 
        for i in range(self.elite):
            elitist = np.reshape(population[i].transformations,
                                (-1, self.X.shape[1]))
            new_population = np.append(new_population, elitist, axis=0)
        return new_population

    def fit(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y).reshape(y.shape[0], )
        
        self._base_score = cross_val_score(self.clf, self.X, self.y,
                                       scoring=self.metric, cv=self.fold).mean()
        print("Base score: ", self._base_score)
        self._best_score = self._base_score
        
        population = self.__create_population()
        
        for i in trange(50, desc='Generation', leave=False):
            for member in tqdm(population, desc='Individual', leave=False):
                new_X = self.__transform(member)
                score = self.__get_fitness(member, self.clf, new_X, self.y)
                self.__individuals.append(self.__Individual(member, score))

            self.__generations.append(self.__individuals)

            best = sorted(self.__individuals, key=lambda tup: tup.score,
                            reverse=True)[0]
            count = 1
            if (i > 0):
                for elem in self.__best_individuals:
                    if(list(best.transformations) == list(elem.transformations)):
                        count += 1
            else:
                pass
            self.__best_individuals.append(
                self.__BestIndividual(i, best.transformations, best.score, count))
            print("Best from gen: {}, count: {}".format(best, count))
            
            if (i == 0):
                self._best_score = self.__best_individuals[i]
            if (best.score > self._best_score.score):
                self._best_score = self.__best_individuals[i]
            else:
                pass
            self.__gen_score.append(self.__Generation(i, 
            sum([tup[1] for tup in self.__individuals])/len(self.__individuals),
            self.__best_individuals[i]))
            population = self.__create_next_generation(self.__individuals)
            self.__individuals = []
        else:
            best = sorted(self.__best_individuals, key=lambda tup: tup.count,
                            reverse=True)[0]
            if(best.count > 1):
                avg = 0
                for elem in self.__best_individuals:
                    if(list(best.transformations) == list(elem.transformations)):
                        avg += elem.score
                print("Most frequent individual: ", best)
                print("Average score: ", avg/best.count)


if __name__ == "__main__":

