# 🍷 Wine Classification using Bagging (Ensemble Learning)

> *Combining weak learners into a strong classifier — one bottle at a time.*

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-150458?style=flat-square&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-11557c?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview

This project applies **Bagging (Bootstrap Aggregation)** with Decision Trees to classify wine types from the classic `sklearn` Wine dataset. It demonstrates how ensemble learning reduces overfitting and boosts stability compared to a single Decision Tree.

---

## 📂 Dataset

| Property | Details |
|----------|---------|
| **Source** | `sklearn.datasets.load_wine()` |
| **Type** | Multi-class Classification |
| **Classes** | 3 types of wine |
| **Features** | 13 chemical properties (alcohol, malic acid, ash, etc.) |
| **Samples** | 178 instances |

---

## 🛠️ Tech Stack

- **Python** — Core language
- **Scikit-learn** — ML models & evaluation
- **Pandas** — Data handling
- **Matplotlib** — Visualization

---

## 🧠 How Bagging Works

```
  Training Data
       │
       ▼
┌─────────────────────────────┐
│    Bootstrap Sampling       │  ← Random sampling with replacement
│  D₁    D₂    D₃  ...  Dₙ   │
└──┬──────┬──────┬──────┬────┘
   ▼      ▼      ▼      ▼
  DT₁    DT₂    DT₃  ...DTₙ     ← Train independent Decision Trees
   │      │      │      │
   └──────┴──────┴──────┘
              │
              ▼
       Majority Vote  →  Final Prediction ŷ
```

**Step-by-step:**

1. **Bootstrap Sampling** — Create `n` random subsets of training data (with replacement).
2. **Train Multiple Models** — Fit a separate Decision Tree on each subset.
3. **Predict Independently** — Each model makes its own prediction.
4. **Aggregate (Majority Vote)** — The final output is:

$$\hat{y} = \text{mode}(h_1(x),\ h_2(x),\ \ldots,\ h_n(x))$$

---

## 💡 Why Bagging?

| Problem with Single DT | How Bagging Fixes It |
|------------------------|----------------------|
| High variance (overfits) | Averages out errors across models |
| Sensitive to training data | Each model sees a different subset |
| Unstable predictions | Ensemble smooths out noise |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install scikit-learn pandas matplotlib
```

### Run the Project

```bash
git clone https://github.com/your-username/wine-classification-bagging.git
cd wine-classification-bagging
python wine_bagging.py
```

---

## 📄 Code Walkthrough

### 1. Load Dataset
```python
from sklearn.datasets import load_wine

wine = load_wine()
x = wine.data
y = wine.target
```

### 2. Split into Train & Test
```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
```

### 3. Define Base Model
```python
from sklearn.tree import DecisionTreeClassifier

base = DecisionTreeClassifier()
```

### 4. Apply Bagging
```python
from sklearn.ensemble import BaggingClassifier

bagging = BaggingClassifier(base, n_estimators=50, random_state=72)
```

### 5. Train & Predict
```python
bagging.fit(x_train, y_train)
y_pred = bagging.predict(x_test)
```

### 6. Evaluate
```python
from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

## 📊 Results

| Model | Accuracy |
|-------|----------|
| Single Decision Tree | ~90–92% |
| **Bagging Classifier (n=50)** | **~96–100%** |

> ✅ Bagging significantly improves accuracy and reduces variance over a standalone Decision Tree.

---

## 📁 Project Structure

```
wine-classification-bagging/
│
├── wine_bagging.py        # Main script
├── README.md              # Project documentation
└── requirements.txt       # Dependencies
```

---

## 🔑 Key Concepts

- **Bootstrap Aggregation (Bagging)** — Ensemble method that trains multiple models on different random subsets and combines predictions via majority vote.
- **Variance Reduction** — Particularly effective for high-variance models like Decision Trees.
- **Out-of-Bag (OOB) Estimation** — Can be enabled with `oob_score=True` for internal validation without a separate test set.

---

## 🌱 Future Improvements

- [ ] Compare with Random Forest (which adds feature randomness on top of Bagging)
- [ ] Enable `oob_score=True` for OOB accuracy estimation
- [ ] Tune `n_estimators` and `max_samples` via GridSearchCV
- [ ] Visualize feature importances

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">Made with ❤️ and a love for ensemble learning</p>
