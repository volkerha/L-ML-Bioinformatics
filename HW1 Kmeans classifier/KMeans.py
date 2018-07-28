import sys
import math

data_file = "data/input.txt"
output_file = "data/output.txt"


# read parameter values and data points from file
def read_data(file_path):
    file = open(file_path, 'r')
    knd = file.readline().split()
    k = int(knd[0])
    n = int(knd[1])
    d = int(knd[2])

    data = [[float(t) for t in line.split()] for line in file]

    return k, n, d, data


# write data to output file
def write_data(line):
    file = open(output_file, 'w')
    file.write(line)


# check if parameter values are valid
def check_param_bounds():
    if K <= 1 or K >= 6:
        print("Invalid K-Means parameter: k = ", K, ". k must be larger than 1 and smaller than 6")
        return False
    if N <= K or N >= 50:
        print("Invalid K-Means parameter: n = ", N, ". n must be larger than k and smaller than 50")
        return False
    if D <= 1 or D >= 6:
        print("Invalid K-Means parameter: d = ", D, ". n must be larger than 1 and smaller than 6")
        return False

    return True


# distance between data point d and cluster center c
def euclid_dist(d, c):
    dist = 0
    # sum up difference between d and c for each dimension
    for i in range(len(d)):
        dist += pow(d[i]-c[i], 2)

    return math.sqrt(dist)


# this is one iteration of the clustering algorithm k-means
# the data points are returned with an associated cluster center
# and the probabilities that a data point belong to any of the cluster centers
def cluster(data_points):
    dp_num = 0
    for dp, c, p_i in data_points:
        prob = []
        dist_to_clusters = 0
        for cc in cluster_centers:
            dist_to_clusters += pow(euclid_dist(dp, cc), 2)
        for cc in cluster_centers:
            p = (1 - pow(euclid_dist(dp, cc), 2) / dist_to_clusters) / (K - 1)
            prob.append(p)

        m = max(prob)
        max_index = [i + 1 for i, j in enumerate(prob) if j == m]

        data_points[dp_num] = (dp, max_index[0], prob)
        dp_num += 1

    return data_points


# recalculates the cluster centers depending on the last k-means iteration
# all data points and their probabilities to belong to a cluster are used
# to calc the new cluster centers
def calc_cluster_center(data_points):
    cluster_center_new = []
    for k in range(K):
        k_elem = [(dp, p) for (dp, c, p) in data_points]
        cluster_dim = [0] * D
        prob_sum = 0
        for elem, p in k_elem:
            dim_num = 0
            for dim in elem:
                cluster_dim[dim_num] = cluster_dim[dim_num] + dim * p[k]
                dim_num += 1
            prob_sum += p[k]

        cluster_dim = [c / prob_sum for c in cluster_dim]
        cluster_center_new.append(cluster_dim)

    return cluster_center_new


# get parameter values and data points from file
K, N, D, data = read_data(data_file)
if not check_param_bounds():
    sys.exit()

cluster_centers = data[:K]
data_points_init = data[K:]
# create a tupel of coordinates, assigned cluster and probabilities
# for all clusters for every data points
data_points_init = [(dp, 0, 0) for dp in data_points_init]

iterations = 3
if len(sys.argv) > 1:
    iterations = int(sys.argv[1])
output = ""
for i in range(iterations):
    data_points_init = cluster(data_points_init)
    cluster_centers = calc_cluster_center(data_points_init)
    for c_i in cluster_centers:
        for c in c_i:
            output += str(c) + " "
    if i < iterations-1:
	    output += "\n"

for k in range(K+1):
    dp_num = 1
    for dp, c, s in data_points_init:
        if c is k:
            output += str(dp_num) + " "
        dp_num += 1
    output += "\n"

write_data(output)