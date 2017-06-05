#!/usr/bin/env python

__author__ = "Bartosz Sowul"

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
        return

    def __apply_function(self, feature):

    def __transform(self, member):
        return

    def __get_fitness(self, clf, X, y):
        return
    
    def __select_parents(self, q=4):
        return
        
    def __crossover(self, parents, population):
        return new_population
    
    def __mutate(self, new_population, std):
        return new_population

    def __create_next_generation(self, population):
        return new_population

    def fit(self, X, y):
        self.X = np.asarray(X)
        self.shape = 
        self.y = np.asarray(y).reshape(y.shape[0], )
        


if __name__ == "__main__":

