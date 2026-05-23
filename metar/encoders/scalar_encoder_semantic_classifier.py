import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from htm.bindings.encoders import ScalarEncoder, ScalarEncoderParameters

# -----------------------------------
# Create ScalarEncoder
# -----------------------------------
params = ScalarEncoderParameters()

params.minimum = 0
params.maximum = 100

params.size = 120
params.activeBits = 12

params.periodic = False

encoder = ScalarEncoder(params)

# -----------------------------------
# Sample scalar values
# -----------------------------------
values = np.array([5, 10, 15, 35, 40, 45, 75, 80, 85])

# Semantic classes
labels = []

for v in values:
    if v < 25:
        labels.append("Low")
    elif v < 60:
        labels.append("Medium")
    else:
        labels.append("High")

# -----------------------------------
# Encode values into SDRs
# -----------------------------------
vectors = []

for v in values:
    sdr = encoder.encode(v)
    vectors.append(sdr.dense)

vectors = np.array(vectors)

# -----------------------------------
# Reduce SDR dimensions using PCA
# -----------------------------------
pca = PCA(n_components=3)
projection = pca.fit_transform(vectors)

# -----------------------------------
# 3D Visualization
# -----------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot each semantic class
for label in ["Low", "Medium", "High"]:

    indices = [i for i, l in enumerate(labels) if l == label]

    ax.scatter(
        projection[indices, 0],
        projection[indices, 1],
        projection[indices, 2],
        s=120,
        label=label,
    )

# Annotate values
for i, value in enumerate(values):

    x, y, z = projection[i]

    ax.text(x, y, z, str(value), fontsize=10)

# Labels
ax.set_title("3D Semantic Clustering of ScalarEncoder SDRs")

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")

ax.legend()

plt.show()
