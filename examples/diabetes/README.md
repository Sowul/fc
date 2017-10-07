# diabetes
[Data source](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)

## Sample output

```
$ python example_diabetes.py
Base score: -0.5100131873

Most frequent individual (1 times, average score -0.491155230831):
	feature 0: nanmax every row
	feature 1: sinc col 7
	feature 2: fix col 3
	feature 3: nanprod every row
	feature 4: add cols 5 and 0
	feature 5: nansum every row
	feature 6: fabs col 2
	feature 7: add cols 1 and 5
neg_log_loss: -0.491155230831

Best params:
	feature 0: nanmin every row
	feature 1: gradient every row
	feature 2: fix col 5
	feature 3: negative col 2
	feature 4: sinc col 2
	feature 5: log1p col 0
	feature 6: fix col 2
	feature 7: sinc col 0
neg_log_loss: -0.48364792757

```
