# breast_cancer

## Sample output

```
$ python example_breast_cancer.py
Base score: -0.151235673812

Most frequent individual (1 times, average score -0.128750840096):
	feature 0: sin col 26
	feature 1: negative col 24
	feature 2: floor col 11
	feature 3: multiply cols 26 and 28
	feature 4: floor col 5
	feature 5: cos col 18
neg_log_loss: -0.128750840096

Best params:
	feature 0: around col 18
	feature 1: nanmedian every row
	feature 2: nanstd every row
	feature 3: multiply cols 1 and 27
	feature 4: fabs col 26
	feature 5: cos col 21
neg_log_loss: -0.120126101022

```
