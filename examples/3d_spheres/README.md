# 3d_spheres

## Sample output

```
$ python example_3d_spheres.py
Base score: -0.4949091459122827

Most frequent individual (1 times, average score -0.06541971906497579):
	feature 0: rint col 0
	feature 1: nanmin every row
	feature 2: floor col 1
	feature 3: ptp every row
	feature 4: nanprod every row
	feature 5: nanmean every row
	feature 6: nanpercentile every row
	feature 7: nanvar every row
neg_log_loss: -0.06541971906497579

Best params:
	feature 0: nanmin every row
	feature 1: nanvar every row
	feature 2: nanpercentile every row
	feature 3: nanmax every row
	feature 4: trunc col 0
	feature 5: floor col 0
	feature 6: nanvar every row
	feature 7: nansum every row
neg_log_loss: -0.029754791532622858

```
