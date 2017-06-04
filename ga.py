#!/usr/bin/env python

__author__ = "Bartosz Sowul"

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

    def __create_individual(self):
        return

    def __create_population(self):
        return

    def __apply_function(self, feature):

    def __transform(self, member):
        return

    def __get_fitness(self, clf, X, y):
        return

    def __create_next_generation(self, population):
        return new_population

    def mutate(self, new_population, std):
        return new_population

    def crossover(self, parents, population):
        return new_population

    def select_parents(self, q=4):
        return 

    def fit(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y).reshape(y.shape[0], )


if __name__ == "__main__":

