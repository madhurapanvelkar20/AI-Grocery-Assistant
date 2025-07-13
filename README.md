# AI-Grocery-Assistant
Intelligent, health‑aware grocery companion built with Streamlit, Python, and a lightweight cosine‑similarity recommender. Search, compare, plan carts, receive nutrition warnings, discount alerts, and email checkout summaries—everything in one sleek web app.

### 📌 Features

| Category                         | Highlights                                                                 |
| -------------------------------- | -------------------------------------------------------------------------- |
| 🔍 **Search & Health Tags**      | Real‑time product search; filter by vegan, low‑sodium, diabetic‑safe, etc. |
| ⚖️ **Product Comparison**        | Side‑by‑side nutrition & price table.                                      |
| 🚨 **Automatic Health Warnings** | Flags high calories, fat, carbs, or low protein via rule‑based thresholds. |
| 🛒 **Smart Cart**                | Tracks items, quantity, total cost; checkout triggers email summary.       |
| 📉 **Discount Integration**      | Shows best price across stores (optional).                                 |
| 🤖 **ML Recommendations**        | Cosine‑similarity on user‑item sparse matrix (sklearn + scipy).            |


### 🧠 Machine‑Learning Approach

| Component             | Library                                                                 | Details                                                                                                             |
| --------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Recommender**       | `sklearn.metrics.pairwise.cosine_similarity`, `scipy.sparse.csr_matrix` | Builds a sparse user × product matrix from implicit feedback (view / add / buy) and recommends top‑N similar items. |
| **Nutrient Warnings** | Pure Python (rule‑based)                                                | Threshold flags for calories, protein, fat, carbs.                                                                  |


### 🛠️ Tech Stack

* **Streamlit** — reactive web UI
* **Pandas / NumPy** — data wrangling
* **scikit‑learn** — cosine‑similarity recommender
* **SciPy sparse** — memory‑efficient user‑item matrix
* **smtplib / email** — checkout email notification
