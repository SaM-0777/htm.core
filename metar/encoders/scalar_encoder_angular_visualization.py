import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from htm.bindings.encoders import ScalarEncoder, ScalarEncoderParameters

# -----------------------------------
# Create ScalarEncoder
# -----------------------------------
params = ScalarEncoderParameters()

params.minimum = 0
params.maximum = 360

# Enable cyclic encoding
params.periodic = True

params.size = 120
params.activeBits = 12

encoder = ScalarEncoder(params)

# -----------------------------------
# Encode angle values
# -----------------------------------
angles = np.arange(0, 361, 10)

vectors = []

for angle in angles:
    sdr = encoder.encode(angle)
    vectors.append(sdr.dense)

vectors = np.array(vectors)

# -----------------------------------
# PCA Projection to 2D
# -----------------------------------
pca = PCA(n_components=2)
projection = pca.fit_transform(vectors)

# -----------------------------------
# 2D Visualization
# -----------------------------------
plt.figure(figsize=(9, 7))

# Plot points
plt.scatter(projection[:, 0], projection[:, 1], s=70)

# Connect neighboring angles
plt.plot(projection[:, 0], projection[:, 1])

# Label selected angles
for i, angle in enumerate(angles):
    if angle % 90 == 0:
        x, y = projection[i]
        plt.text(x, y, f"{angle}°", fontsize=10)

# Labels
plt.title("2D PCA Projection of Cyclic ScalarEncoder")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.grid(True)
plt.show()
