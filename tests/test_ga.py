#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
from copy import deepcopy
from filecmp import cmp
from operator import attrgetter

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from ga import GeneticAlgorithm

@pytest.fixture
def ga():
    np.random.seed(10)
    iris = load_iris()
    clf = RandomForestClassifier(max_depth=3, random_state=2222)
    ga = GeneticAlgorithm(clf, 5, 0.5)
    ga.X = np.asarray(iris.data)
    ga.y = np.asarray(iris.target)
    ga.y = ga.y.reshape(ga.y.shape[0], )
    ga.n_features = np.random.random_integers(2*ga.X.shape[1])
    return ga

@pytest.fixture(scope='module')
def ga_fitted():
    np.random.seed(10)
    clf = RandomForestClassifier(max_depth=3, random_state=2222)
    ga = GeneticAlgorithm(clf, 5, 0.5)
    iris = load_iris()
    ga.fit(iris.data, iris.target)
    return ga

def test_ga_create_individual(ga):
    assert np.array_equal(ga._create_individual(), np.array([[30, 21]])) == True

def test_ga_create_population(ga):
    ga.pop_members = 5
    assert np.array_equal(ga._create_population(),
            np.array([[30, 21], [3, 19], [7, 11], [37, 12], [15, 18]])) == True

def test_ga_apply_function(ga):
    ga.pop_members = 5
    population = ga._create_population()
    z = np.zeros((ga.X.shape[0], ga.n_features), dtype=ga.X.dtype)
    for i, member in enumerate(population):
        for col, feature in enumerate(member):
            z[:, col] = ga._apply_function(i, col, feature)
    assert ga.X.shape[0] == z.shape[0]
    assert z.shape[1] == ga.n_features
    assert np.array_equal(z[:5],
                            [[1.,1.],[1.,1.],[1.,1.],[2.,1.],[1.,1.]]) == True

def test_ga_transform(ga):
    ga.pop_members = 5
    population = ga._create_population()
    for j, member in enumerate(population):
        new_X = ga._transform(j, member)
    assert ga.X.shape[0] == new_X.shape[0]
    assert ga.X.shape[1]+ga.n_features == new_X.shape[1]
    assert np.array_equal(new_X[-4:],
                            [[ 5. ,  5. ,  6.3,  2.5,  5. ,  1.9],
                            [ 5. ,  5. ,  6.5,  3. ,  5.2,  2. ],
                            [ 5. ,  5. ,  6.2,  3.4,  5.4,  2.3],
                            [ 5. ,  5. ,  5.9,  3. ,  5.1,  1.8]]) == True

def test_ga_get_fitness(ga):
    ga.pop_members = 5
    population = ga._create_population()
    scores = []
    for j, member in enumerate(population):
        new_X = ga._transform(j, member)
        score = ga._get_fitness(ga.clf, new_X, ga.y)
        scores.append(score)
    assert all(elem <= 0 for elem in scores) == True

def test_ga_select_parents(ga):
    parents = ga._select_parents()
    assert parents.shape == ((ga.pop_members-ga.migrants-ga.elite) // 2, 2)
    assert np.all(np.logical_and(parents >= 0, parents < ga.pop_members)) == True

def test_ga_crossover(ga):
    population = ga._create_population()
    for j, member in enumerate(population):
        new_X = ga._transform(j, member)
        score = ga._get_fitness(ga.clf, new_X, ga.y)
        ga._individuals.append(ga._Individual(member,
                            ga._columns[j], score))
    population = sorted(ga._individuals, key=attrgetter('score'), reverse=True)
    best_ind = deepcopy(population[0])
    parents = ga._select_parents()
    first_parent = np.reshape(population[parents[2][0]].transformations,
                                                        (-1, ga.n_features))
    second_parent = np.reshape(population[parents[2][1]].transformations,
                                                        (-1, ga.n_features))
    new_population = ga._crossover(parents, population)
    best_ind_after_cross = population[0]
    x = [first_parent[0][0], first_parent[0][1], second_parent[0][0], second_parent[0][1]]
    y = [new_population[4][0], new_population[4][1], new_population[5][0], new_population[5][1]]
    assert len(new_population) == ga.pop_members-ga.elite-ga.migrants
    assert np.array_equal(best_ind.transformations,
                            best_ind_after_cross.transformations) == True
    assert Counter(x) == Counter(y)


def test_ga_mutate(ga):
    population = ga._create_population()
    pop_shape = population.shape
    pop_shape = (pop_shape[0] - ga.elite - ga.migrants, pop_shape[1])
    for j, member in enumerate(population):
        new_X = ga._transform(j, member)
        score = ga._get_fitness(ga.clf, new_X, ga.y)
        ga._individuals.append(ga._Individual(member,
                            ga._columns[j], score))
    population = sorted(ga._individuals, key=attrgetter('score'), reverse=True)
    parents = ga._select_parents()
    new_population = ga._crossover(parents, population)
    mutated_population = ga._mutate(new_population,
                                    np.std([ind[0] for ind in population]))
    assert np.all(np.logical_and(mutated_population >= 0,
                                  mutated_population < ga.n_operators)) == True
    assert pop_shape == new_population.shape == mutated_population.shape

def test_ga_create_next_generation(ga):
    population = ga._create_population()
    for j, member in enumerate(population):
        new_X = ga._transform(j, member)
        score = ga._get_fitness(ga.clf, new_X, ga.y)
        ga._individuals.append(ga._Individual(member,
                            ga._columns[j], score))
    new_population = ga._create_next_generation(ga._individuals)
    assert population.shape == new_population.shape

def test_ga_fit(ga_fitted):
    assert ga_fitted._base_score < ga_fitted._best_score.score

def test_ga_transform(ga_fitted):
    iris = load_iris()
    new_X = ga_fitted.transform(iris.data, ga_fitted._best_score)
    assert new_X.shape[0] == iris.data.shape[0]
    assert new_X.shape[1] == iris.data.shape[1] + ga_fitted.n_features

def test_ga_save_load(ga_fitted):
    ga_fitted.save('tests/ga_saved_ind.json')
    loaded_ind = ga_fitted.load('tests/ga_saved_ind.json')
    assert np.array_equal(ga_fitted._best_score.transformations, loaded_ind.transformations) == True
    assert np.array_equal(ga_fitted._best_score.columns, loaded_ind.columns) == True
