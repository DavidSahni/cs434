# Code Explanation

There are two sets of scripts that run for different implementations.
The first set of implementations contains K-means Clustering.
The second set of implementations contains Principal Component Analysis.

## K-means Clustering

To run these set of scripts, run:

``` bash
python kmeans.py {k}
```

Note that "{k}" should be replaced with a natural number.
2 is the recommended number.

## Principal Component Analysis

There are three different scripts that you can run by executing the following:

``` bash
python pca_1.py
```

This will output the 10 largest eigen-values in descending order.

``` bash
python pca_2.py
```

This will output 10 eigen-vector images and the mean image based on all data.

``` bash
python pca_3.py
```

This will use PCA to reduce ten eigen-vectors to 10 dimensions and save that as ten images.
