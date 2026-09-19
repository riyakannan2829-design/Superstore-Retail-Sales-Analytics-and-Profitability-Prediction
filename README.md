# Superstore Retail Sales Analytics and Profitability Prediction


**Dataset:** Superstore.csv — 9,994 US retail order line items (2011–2014)

---

## Project Overview

A complete 4-tier retail analytics project demonstrating the full data science pipeline:

| Tier | Level | Question |
|------|-------|----------|
| 1 | Descriptive Analytics | What happened? |
| 2 | Diagnostic Analytics | Why did it happen? |
| 3 | Predictive Analytics | What will happen? |
| 4 | Prescriptive Analytics | What should we do? |

**ML Problem:** Binary classification — predict whether an order line item will result in a financial loss  
**Target:** `Loss_Flag = 1` when `Profit < 0`, else `0`

---

## Actual Model Performance (Random Forest, Test Set)

| Metric | Score |
|--------|-------|
| Accuracy | **93.95%** |
| Precision | **83.20%** |
| Recall | **84.76%** |
| F1-Score | **83.97%** |
| ROC-AUC | **98.49%** |

*All metrics are from actual model execution on the held-out test set (1,999 records). No values were manually entered.*

---

## Key Findings

1. **Discount is the #1 predictor** — accounts for 53.7% of the Random Forest's feature importance
2. **Discount ≥ 40% → 100% loss rate** — every transaction above this threshold in the dataset is a loss
3. **Central region** has the highest loss rate (31.9%), driven by 80% discounts on Binders/Appliances
4. **Tables and Bookcases** are the most loss-prone sub-categories by total dollar amount
5. **Technology (Copiers)** is the most profitable sub-category — recommended for growth focus
6. Targeting just the **top 500 high-risk orders** can prevent ~**$125,406** in expected losses

---

## Project Files

```
.
├── Superstore.csv                          # Source dataset (9,994 rows, 21 columns)
├── Rithika_SuperstoreRetailAnalytics.ipynb # Complete Jupyter notebook (all 4 tiers)
├── app.py                                  # Streamlit dashboard (4-tab interactive app)
├── requirements.txt                        # Python dependencies
├── Rithika_SuperstoreProjectReport.docx    # Full project report
├── README.md                               # This file
│
├── build_project.py                        # Build script (runs all analysis, saves outputs)
├── generate_notebook.py                    # Generates the .ipynb file
├── generate_report.py                      # Generates the .docx report
│
├── models/
│   ├── loss_classifier.joblib              # Trained Random Forest model
│   └── feature_columns.joblib             # Feature column list (for dashboard inference)
│
└── outputs/
    ├── metrics.json                        # All computed metrics (used by docs)
    ├── risk_scored.csv                     # Full dataset with loss probability scores
    ├── intervention_priority_list.csv      # Ranked intervention list
    └── figures/
        ├── class_balance.png
        ├── sales_profit_trend.png
        ├── subcategory_profit.png
        ├── discount_vs_profit.png
        ├── loss_heatmap_region_category.png
        ├── confusion_matrix.png
        ├── roc_curve.png
        ├── feature_importance.png
        ├── risk_tier_distribution.png
        └── prescriptive_expected_loss.png
```

---

## Prerequisites

- Python 3.9+
- pip

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Option 1: Run the full analysis pipeline

```bash
python -X utf8 build_project.py
```

This runs all analysis, trains the model, saves all figures, outputs, and `models/`.

### Option 2: Open the Jupyter Notebook

```bash
jupyter notebook Rithika_SuperstoreRetailAnalytics.ipynb
```

Run all cells from top to bottom. The notebook re-runs all analysis end-to-end.

### Option 3: Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

Opens an interactive 4-tab dashboard in your browser at `http://localhost:8501`.

The dashboard includes:
- **Descriptive tab**: KPI cards, category/segment charts, monthly trend
- **Diagnostic tab**: Discount-profit scatter, loss heatmap, Pareto analysis
- **Predictive tab**: Model metrics, ROC curve, feature importance, live order predictor
- **Prescriptive tab**: Intervention priority table, discount guardrails, recommendations

---

## Dataset Notes

| Property | Detail |
|----------|--------|
| Encoding | Latin-1 (CP1252) — not UTF-8 |
| Date format | DD-MM-YYYY (day first — European order) |
| Postal Code | Must be loaded as string — leading zeros stripped in source |
| Discount | Decimal fraction 0.0–0.8 (not a percentage) |
| Profit | Can be negative — 18.7% of rows are losses |

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| pandas | Data loading, cleaning, feature engineering |
| numpy | Numerical operations |
| matplotlib / seaborn | Visualizations |
| scikit-learn | ML model, train/test split, metrics |
| joblib | Model persistence |
| streamlit | Interactive dashboard |
| python-docx | Report generation |
| nbformat | Notebook generation |

---

## Leakage Prevention

`Profit` (the source of the target label) is **never used as a predictor**. The full exclusion list:

- `Profit` — directly constructs `Loss_Flag`
- Any Profit derivative (Profit Margin, Profit Ratio, etc.)
- `Row ID`, `Order ID`, `Customer ID`, `Product ID` — raw identifiers
- `Country` — zero variance (always "United States")
- `Customer Name`, `Product Name` — high-cardinality text
- `City`, `State`, `Postal Code` — high cardinality; Region captures geography
