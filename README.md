# Pakistan E-Commerce Order Cancellation Predictor

A machine learning project that analyzes Pakistan's largest e-commerce dataset (~1 million transactions from 2016–2018) to predict whether an order will be cancelled, achieving **85.87% test accuracy** with a Random Forest classifier.

---

## Problem Statement

Order cancellations are a direct revenue loss for e-commerce businesses. This project builds a binary classifier (`is_canceled`) to flag high-risk orders at the time of placement, enabling businesses to intervene proactively — through targeted support, discount offers, or payment method nudges — before the cancellation happens.

---

## Dataset

| Property | Details |
|---|---|
| **Source** | Pakistan Largest Ecommerce Dataset (CSV) |
| **Raw size** | ~1,048,575 rows, 26 columns |
| **After cleaning** | 434,934 usable orders |
| **Time range** | July 2016 – August 2018 |
| **Categories** | Women's Fashion, Mobiles & Tablets, Beauty & Grooming, Appliances, Men's Fashion, and more |
| **Payment methods** | Cash on Delivery (COD), Bank Alfalah, and others |

---

## Pipeline Overview

### Step 0 — Imports & Setup
Libraries used: `pandas`, `NumPy`, `Matplotlib`, `Plotly`, `Seaborn`, `scikit-learn`

### Step 1 — Data Loading & Cleaning
- Dropped 5 entirely empty trailing columns
- Removed rows that were all-null (except `item_id`)
- Retained only unambiguous terminal order states (`complete` and `canceled`) to avoid label noise from intermediate states like `received`, `paid`, or `order_refunded`

### Step 2 — Preprocessing
- Imputed missing `category_name_1` with mode
- Parsed `created_at` and `Customer Since` as datetime; filled missing tenure dates with median
- Dropped high-cardinality sparse column (`sales_commission_code` — 88K+ nulls)

### Step 3 — Feature Engineering

| Feature | Description |
|---|---|
| `is_canceled` | Binary target (1 = canceled, 0 = complete) |
| `has_discount` | Whether a discount was applied |
| `price_per_unit` | Grand total divided by quantity |
| `is_cod` | Flag for cash-on-delivery payment |
| `has_commission` | Whether a sales commission code was present |
| `category_enc` | Label-encoded product category |
| `payment_enc` | Label-encoded payment method |
| `order_dow` | Day of week the order was placed |
| `order_month` | Month the order was placed |
| `order_year` | Year the order was placed |
| `customer_tenure_days` | How long the customer had been registered at order time |

### Step 4 — Modeling
- Train/test split (80/20)
- **Model:** Random Forest Classifier (`n_estimators=200`, `n_jobs=-1`, `random_state=42`)
- **Test Accuracy: 85.87%**

---

## Results

```
Test Accuracy: 0.8587 (85.87%)
```

---

## Key Insight

Only `complete` and `canceled` statuses were used as labels. This deliberate choice avoids training noise from ambiguous intermediate states and makes the model's predictions directly actionable for business intervention.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

---

## How to Run

1. Clone this repository
2. Upload the dataset CSV to your Google Drive
3. Open the `.ipynb` file in Google Colab
4. Mount your Google Drive and update the dataset path
5. Run all cells in order

---

## Project Structure

```
├── pakistan_ecommerce_analysis.ipynb   # Main notebook
└── README.md                           # Project documentation
```
