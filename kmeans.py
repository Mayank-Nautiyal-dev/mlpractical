
"""#kmeans

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("dataset/Mall_Customers.csv")

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 2. SELECT FEATURES
# ============================================================

# CustomerID is only an identifier, so don't use it.
# Genre is categorical, so don't use it in basic K-Means.
#
# We cluster customers using:
#   X1 = Annual Income
#   X2 = Spending Score

X = df[['Annual Income (k$)',
        'Spending Score (1-100)']].values.astype(float)


# ============================================================
# 3. K-MEANS FROM SCRATCH
# ============================================================

def kmeans(X, k, max_iter=300, seed=42):

    np.random.seed(seed)

    # Randomly choose k existing customers as initial centroids
    indices = np.random.choice(
        len(X),
        size=k,
        replace=False
    )

    centroids = X[indices].copy()

    for iteration in range(max_iter):

        # ----------------------------------------------------
        # STEP 1: Calculate distance from every point
        #         to every centroid
        # ----------------------------------------------------

        distances = np.zeros((len(X), k))

        for i in range(len(X)):
            for j in range(k):

                distances[i, j] = np.sqrt(
                    np.sum(
                        (X[i] - centroids[j]) ** 2
                    )
                )

        # ----------------------------------------------------
        # STEP 2: Assign each customer to nearest centroid
        # ----------------------------------------------------

        labels = np.argmin(distances, axis=1)

        # ----------------------------------------------------
        # STEP 3: Calculate new centroids
        # ----------------------------------------------------

        new_centroids = np.zeros_like(centroids)

        for j in range(k):

            cluster_points = X[labels == j]

            if len(cluster_points) > 0:
                new_centroids[j] = np.mean(
                    cluster_points,
                    axis=0
                )
            else:
                # Handle empty cluster
                new_centroids[j] = X[
                    np.random.randint(len(X))
                ]

        # ----------------------------------------------------
        # STEP 4: Check convergence
        # ----------------------------------------------------

        if np.allclose(centroids, new_centroids):
            centroids = new_centroids
            break

        centroids = new_centroids

    # --------------------------------------------------------
    # Calculate WCSS
    # --------------------------------------------------------

    wcss = 0

    for i in range(len(X)):

        wcss += np.sum(
            (X[i] - centroids[labels[i]]) ** 2
        )

    return labels, centroids, wcss


# ============================================================
# 4. ELBOW METHOD
# ============================================================

wcss_values = []

for k in range(1, 11):

    labels, centroids, wcss = kmeans(X, k)

    wcss_values.append(wcss)

    print(
        f"K = {k:2d} | WCSS = {wcss:.2f}"
    )


# Visualize elbow
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    wcss_values,
    marker='o'
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method for Choosing K")

plt.grid(True)
plt.show()


# ============================================================
# 5. FINAL K-MEANS
# ============================================================

K = 5

labels, centroids, wcss = kmeans(
    X,
    K,
    seed=42
)

# Add cluster labels to original dataframe
df['Cluster'] = labels


# ============================================================
# 6. PRINT CENTROIDS
# ============================================================

print("\n" + "=" * 60)
print("FINAL K-MEANS RESULTS")
print("=" * 60)

print("\nNumber of clusters:", K)
print("Final WCSS:", round(wcss, 2))

print("\nCentroids:")

for i, centroid in enumerate(centroids):

    print(
        f"Cluster {i}: "
        f"Income = {centroid[0]:.2f} k$, "
        f"Spending Score = {centroid[1]:.2f}"
    )


# ============================================================
# 7. VISUALIZE CLUSTERS
# ============================================================

plt.figure(figsize=(10, 7))

# Plot every cluster
for cluster in range(K):

    cluster_points = X[labels == cluster]

    plt.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        s=70,
        label=f"Cluster {cluster}"
    )


# Plot centroids
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker='X',
    s=300,
    edgecolors='black',
    linewidths=2,
    label='Centroids'
)


# Labels
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-Means Customer Segmentation")

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 8. CLUSTER SUMMARY
# ============================================================

summary = df.groupby('Cluster').agg({

    'Age': 'mean',

    'Annual Income (k$)': 'mean',

    'Spending Score (1-100)': 'mean',

    'CustomerID': 'count'

}).rename(
    columns={
        'Age': 'Average Age',
        'Annual Income (k$)': 'Average Income',
        'Spending Score (1-100)': 'Average Spending Score',
        'CustomerID': 'Number of Customers'
    }
)

print("\n" + "=" * 60)
print("CLUSTER SUMMARY")
print("=" * 60)

print(summary.round(2))


# ============================================================
# 9. SHOW CUSTOMERS WITH THEIR CLUSTER
# ============================================================

print("\n" + "=" * 60)
print("CUSTOMER CLUSTER ASSIGNMENTS")
print("=" * 60)

print(
    df[
        [
            'CustomerID',
            'Genre',
            'Age',
            'Annual Income (k$)',
            'Spending Score (1-100)',
            'Cluster'
        ]
    ].head(20)
)

