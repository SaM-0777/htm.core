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
params.size = 100
params.activeBits = 10
params.periodic = False

encoder = ScalarEncoder(params)

# Values to encode
values = [30, 39, 40, 41, 50]

# -----------------------------------
# Encode scalar values into SDRs
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
# 3D Plot
# -----------------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

for i, value in enumerate(values):
    x, y, z = projection[i]

    ax.scatter(x, y, z, s=120)
    ax.text(x, y, z, str(value), fontsize=11)

# Labels
ax.set_title("3D PCA Projection of ScalarEncoder SDRs")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")

plt.show()
