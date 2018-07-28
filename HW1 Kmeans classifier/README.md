# Kmeans classifier

## How does it work?
Simple kmeans classification that takes an input file with different points (samples) in a n-dim system, 
calculates the cluster centers and assigns each point a probability with which it belongs to one of the cluster centers.

## Input file format
The first row of the input file is reserved for parameters k, n, d:
- k: number of cluster centers
- n: number of data points
- d: number of dimensions

The following k rows are the coordinates of the cluster centers.
The next n rows are the coordinates of the data points.
