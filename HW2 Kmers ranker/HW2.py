import math
import operator
import matplotlib.pyplot as plt
import numpy as np
import sys

from sklearn.decomposition import PCA


data_file_1 = "COG1.fasta"
data_file_2 = "COG160.fasta"
data_file_3 = "COG161.fasta"

fasta_files = [data_file_1, data_file_2, data_file_3]
F_x = []
K_F_x = []

k = 5
R = 50

kmers = {}
skewness = {}
rank_product = {}

kmers_rank = {}
skewness_rank = {}

top_kmers = []


# read parameter values and data points from file
def read_data(file_path):
    fasta_data = []
    file = open(file_path, 'r')
    file_content = file.readlines()
    protein_seq = ""
    for line in file_content:
        if line[0] != ">":
            protein_seq += line.rstrip()
        else:
            if len(protein_seq) > 0:
                fasta_data.append(protein_seq)
                protein_seq = ""

    F_x.append(fasta_data)


# fetch total count of kmers from fasta input
def get_kmers_freq():
    for x in F_x:
        kmers_x = {}
        for pseq in x:
            for i in range(len(pseq) - k - 1):
                kmer = pseq[i:i+k]
                if kmer in kmers:
                    kmers[kmer] += 1
                else:
                    kmers[kmer] = 1
                if kmer in kmers_x:
                    kmers_x[kmer] += 1
                else:
                    kmers_x[kmer] = 1

        K_F_x.append(kmers_x)

    kr = sorted(kmers.items(), key=operator.itemgetter(1), reverse=True)
    for i in range(len(kr)):
        kmers_rank[kr[i][0]] = i + 1


def calc_kmers_entropy():
    for kmer in kmers:
        tot = kmers[kmer]
        entropy = 0
        for x in K_F_x:
            if kmer in x:
                prob = x[kmer] / tot
                entropy -= (prob * math.log2(prob))

        # print(kmer, kmers_rank[kmer], entropy)
        skewness[kmer] = entropy

    sr = sorted(skewness.items(), key=operator.itemgetter(1), reverse=True)
    for i in range(len(sr)):
        skewness_rank[sr[i][0]] = i + 1


def calc_rank_product():
    for kmer in kmers:
        rp = math.sqrt(kmers_rank[kmer] * skewness_rank[kmer])
        rank_product[kmer] = rp


def get_sorted_kmers():
    sorted_rp = sorted(rank_product.items(), key=operator.itemgetter(1))
    sorted_kmers = []
    i = 1
    for (kmer, rp) in sorted_rp:
        sorted_kmers.append((i, kmers_rank[kmer], skewness_rank[kmer], rank_product[kmer], kmer))
        i += 1

    return sorted_kmers


def create_feature_matrix():
    pseq_len = 0
    for x in F_x:
        pseq_len += len(x)

    fm = np.zeros((pseq_len, len(top_kmers)))
    i = 0
    for x in F_x:
        for pseq in x:
            j = 0
            for kmer in top_kmers:
                fm[i][j] += pseq.count(kmer)
                j += 1
            i += 1

    return fm


def write_to_file(sorted_kmers):
    file = open("ranked kmers.txt", 'w')
    file.write("Rank" + "\t\t" + "Frequency Rank" + "\t\t" + "Skewness Rank"
               + "\t\t" + "Rank Product" + "\t\t" + "K-mer" + "\n")
    for kmer in sorted_kmers:
        kmer_string = ""
        for el in kmer:
            kmer_string += str(el) + "\t\t\t"
        kmer_string += "\n"
        file.write(kmer_string)


if len(sys.argv) > 2:
    k = int(sys.argv[1])
    R = int(sys.argv[2])
elif len(sys.argv) > 1:
    k = int(sys.argv[1])

for file in fasta_files:
    read_data(file)

get_kmers_freq()
# print(kmers)

# if R > math.pow(20, k):
#     print("R too big, must be smaller 20^k. Select smaller R and bigger k.")
#     sys.exit()

calc_kmers_entropy()
calc_rank_product()
sort_kmers = get_sorted_kmers()

if len(sort_kmers) < R:
    R = len(sort_kmers)
for i in range(R):
    top_kmers.append(sort_kmers[i][len(sort_kmers[-1])-1])

write_to_file(sort_kmers)
# print(sort_kmers)
fm = create_feature_matrix()
# print(feature_matrix)


# >>> PCA >>>
X = fm
y = np.array([0]*len(F_x[0]) + [1]*len(F_x[1]) + len(F_x[2])*[2])
target_names = fasta_files
# Run Principal Component analysis with PC1 & PC2
pca = PCA(n_components=2)
# Transform original data into PC1&PC2 space
X_r = pca.fit(X).transform(X)
# Prepare figure
f = plt.figure()

colors = ['navy', 'turquoise', 'darkorange']
lw = 2 # line width

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
	plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
			label=target_name)

plt.ylabel("PC2")
plt.xlabel("PC1")
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of fasta protein sequences')
plt.show()

f.savefig('fasta_plot.pdf')




