# fc [![Build Status](https://travis-ci.org/Sowul/fc.svg?branch=master)](https://travis-ci.org/Sowul/fc)

What is the best representation of the data to solve a problem? Find out using FeatureConstructor powered by genetic algorithm.<br> All you need is a classifier object implementing 'fit' method and this simple script (works both with Python 2 and 3).<br>
Sometimes it yields great results out of the box (check out sample outputs for 2d_circles and 3d_spheres examples), sometimes you have to do a bit more beforehand (aka pre-processing your data, diabetes and spam examples).

 __Disclaimer:__ I take no responsibility or liability for any losses which may be incurred by any person or persons using the whole or part of the contents of this software, i.e. don't blame me for lost money.<br>

## Prerequisites

```
$ pip install -r requirements.txt
```

## Installation

```
$ cd fc
$ pip install .
```

## Usage
Load data
```python
>>> from sklearn.datasets import load_iris
>>> iris = load_iris()
>>> X = iris.data
>>> y = iris.target
```
Choose your favourite classifier
```python
>>> from sklearn.ensemble import RandomForestClassifier
>>> clf = RandomForestClassifier(max_depth=3)
```
Initialize FeatureConstructor
```python
>>> from fc import FeatureConstructor
>>> fc = FeatureConstructor(clf, fold=5, duration=3)
```
Fit data
```python
>>> fc.fit(X, y)
```
Print best params
```python
>>> fc.get_params()
```
Or most frequent params
```python
>>> fc.get_params('most_freq')
```
Save new features
```python
>>> fc.save('new_features.json', 'best')
```
Load them later
```python
>>> new_features = fc.load('new_features.json')
```
Transform raw dataset into new, hopefully better one
```python
>>> new_X = fc.transform(X, new_features)
```
## Examples

Check out examples directory.

## Tests

```
$ python -m pytest tests
```

## License
[The MIT License](LICENSE.md)
