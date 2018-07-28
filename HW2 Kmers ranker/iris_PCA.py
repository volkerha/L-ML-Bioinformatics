import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA

# load iris data
iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names 
print("x values:", X)
print("y values:", y)
print("target names: ", target_names)
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

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
plt.show()

f.savefig('iris_plot.pdf')
