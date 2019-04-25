# Implementation 2

KNN and Decision Trees
David Sahni
Burton Jaursch
Chase McWhirt

## K-Nearest Neighor

Scripts include

* q1.py
* knn.py

To run use:

``` bash
python q1.py knn_train.csv knn_test.csv <int k>
python knn.py
```

### q1.py

This script, given the training and testing set, will make predictions based on training data.
First it normalizes both data sets, shuffles the training data, and removes 10% of the training data to be used for validation.
This calculation is random but seeded so graders will get the same results.
It then calculates the "k" nearest points.
Based on the class of the "k" nearest points, a prediction is made (which class is more prominent in the "k" sample).
By default, the verbose print statement is active.

### knn.py

This script allows you to run q1.py multiple times for odd k's between 1 and 51.
It is recommended that you comment out the verbose print statement and uncomment the lite statement.
Results are available in knnErrors.csv.

## Decision Tree

Scripts include q2_1.py and q2_2.py  
Run using commands specified in the assignment (they will also print usage statements)  
**Note** q2_2.py depends on some functions from q2_1.py so please leave them in the same folder
