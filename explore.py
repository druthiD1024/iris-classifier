import numpy as np
from sklearn.datasets import load_iris

# ─────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────
print("=" * 50)
print("LOADING IRIS DATASET")
print("=" * 50)

iris = load_iris()
X    = iris.data
y    = iris.target

print(f"Features  : {iris.feature_names}")
print(f"Classes   : {iris.target_names}")
print(f"Shape     : {X.shape}")
print(f"Samples   : {len(X)}")

# ─────────────────────────────────────────
# 2. CLASS DISTRIBUTION
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("CLASS DISTRIBUTION")
print("=" * 50)

for i, name in enumerate(iris.target_names):
    count = int(np.sum(y == i))
    print(f"  {name:12s} → {count} samples ({count/len(y)*100:.0f}%)")

print("\n✅ Perfectly balanced — 50 samples per class!")

# ─────────────────────────────────────────
# 3. FEATURE STATISTICS PER CLASS
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("FEATURE STATISTICS PER CLASS")
print("=" * 50)

feature_names = ['Sepal Len', 'Sepal Wid', 'Petal Len', 'Petal Wid']

for i, name in enumerate(iris.target_names):
    print(f"\n  [{name.upper()}]")
    X_class = X[y == i]
    for j, feat in enumerate(feature_names):
        print(f"    {feat:10s} → mean={X_class[:,j].mean():.2f}  "
              f"min={X_class[:,j].min():.2f}  "
              f"max={X_class[:,j].max():.2f}")

# ─────────────────────────────────────────
# 4. KEY OBSERVATIONS
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("KEY OBSERVATIONS")
print("=" * 50)

petal_len = [X[y == i][:, 2].mean() for i in range(3)]
petal_wid = [X[y == i][:, 3].mean() for i in range(3)]

print(f"  Avg Petal Length → Setosa: {petal_len[0]:.2f}cm  "
      f"Versicolor: {petal_len[1]:.2f}cm  Virginica: {petal_len[2]:.2f}cm")
print("  → Petal length is the STRONGEST separator!")

print(f"\n  Avg Petal Width  → Setosa: {petal_wid[0]:.2f}cm  "
      f"Versicolor: {petal_wid[1]:.2f}cm  Virginica: {petal_wid[2]:.2f}cm")
print("  → Petal width is the 2nd strongest separator!")

print("\n  Summary:")
print("  - Setosa is clearly separate (tiny petals)")
print("  - Versicolor & Virginica overlap slightly")
print("  - SVM handles the overlap better than KNN")

# ─────────────────────────────────────────
# 5. SAMPLE ROWS
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("SAMPLE ROWS (first 3 per class)")
print("=" * 50)
print(f"  {'Class':12s} {'SepalL':>8} {'SepalW':>8} {'PetalL':>8} {'PetalW':>8}")
print("  " + "-" * 48)

for i, name in enumerate(iris.target_names):
    for row in X[y == i][:3]:
        print(f"  {name:12s} {row[0]:>8.1f} {row[1]:>8.1f} {row[2]:>8.1f} {row[3]:>8.1f}")

print("\n✅ EDA complete — ready for Step 3: Preprocessing & Scaling")
