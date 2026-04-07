import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ─────────────────────────────────────────
# 1. LOAD & PREPARE
# ─────────────────────────────────────────
print("=" * 50)
print("LOADING & PREPARING DATA")
print("=" * 50)

iris      = load_iris()
X, y      = iris.data, iris.target
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

# ─────────────────────────────────────────
# 2. TRAIN & EVALUATE BOTH MODELS
# ─────────────────────────────────────────
models = [
    ('KNN (k=5)',  KNeighborsClassifier(n_neighbors=5)),
    ('SVM (RBF)',  SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)),
]

results = {}

for name, clf in models:
    print("\n" + "=" * 50)
    print(f"MODEL: {name}")
    print("=" * 50)

    # Train
    clf.fit(X_train_sc, y_train)

    # Predict
    y_pred = clf.predict(X_test_sc)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy : {acc*100:.1f}%")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"              Setosa  Versicolor  Virginica")
    for i, row in enumerate(cm):
        print(f"  {class_names[i]:12s}  {row[0]:>6}  {row[1]:>10}  {row[2]:>9}")

    # Cross validation
    cv_scores = cross_val_score(clf, 
                                scaler.fit_transform(X),  
                                y, cv=5, scoring='accuracy')
    print(f"\n5-Fold CV Accuracy : {[round(s*100,1) for s in cv_scores]}")
    print(f"Mean CV Accuracy   : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    results[name] = {
        'clf':      clf,
        'acc':      acc,
        'cv_mean':  cv_scores.mean()
    }

# ─────────────────────────────────────────
# 3. COMPARE & PICK WINNER
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)
print(f"{'Model':<20} {'Test Acc':>10} {'CV Acc':>10}")
print("-" * 42)
for name, r in results.items():
    print(f"{name:<20} {r['acc']*100:>9.1f}% {r['cv_mean']*100:>9.1f}%")

winner_name = max(results, key=lambda n: results[n]['cv_mean'])
print(f"\n🏆 Winner: {winner_name}")

# ─────────────────────────────────────────
# 4. TEST ON CUSTOM FLOWERS
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("CUSTOM PREDICTIONS")
print("=" * 50)

best_clf = results[winner_name]['clf']

test_flowers = [
    [5.1, 3.5, 1.4, 0.2],   # typical setosa
    [6.4, 3.2, 4.5, 1.5],   # typical versicolor
    [6.3, 3.3, 6.0, 2.5],   # typical virginica
    [5.9, 3.0, 5.1, 1.8],   # tricky — versicolor/virginica overlap
]

labels = ['Typical Setosa    ', 'Typical Versicolor',
          'Typical Virginica ', 'Tricky one!       ']

print(f"  {'Input':40s} {'Predicted':>12} {'Confidence':>12}")
print("  " + "-" * 66)

for flower, label in zip(test_flowers, labels):
    flower_sc = scaler.transform([flower])
    pred      = best_clf.predict(flower_sc)[0]
    prob      = best_clf.predict_proba(flower_sc)[0].max()
    print(f"  {label} {class_names[pred]:>12} {prob*100:>11.1f}%")

print(f"\n✅ Training complete — ready for Step 5: Save Model")
