# Predicting Airbnb Price Tiers from Photos, Text and Listing Details

Multimodal Data Analysis — Project 2.2 (individual implementation) · Kunning (Ricardo) Liang · Group 19

Classify Melbourne Airbnb listings into **budget / mid-range / premium** (nightly-price tertiles) from three modalities and compare early, late and attention-based intermediate fusion against unimodal baselines.

| Modality | Source | Main encoder | Comparison | Reference |
|---|---|---|---|---|
| Image | main listing photo | EfficientNet-B0 (1,280-d) | ViT-B/16 | CLIP ViT-B/32 |
| Text | title + description (PII and names masked) | BERT-base mean pooling (768-d) | Sentence-BERT all-mpnet-base-v2 | CLIP text |
| Tabular | room/property type, capacity, bedrooms, beds, bathrooms, amenities, reviews, minimum nights, area | 104 standardised features | – | – |

## Main results (test split, n = 2,430)

| Model | Accuracy | Macro-F1 [95% CI] | ROC-AUC | QWK |
|---|---|---|---|---|
| Image only | 0.506 | 0.507 [0.488, 0.526] | 0.695 | 0.361 |
| Text only | 0.555 | 0.556 [0.537, 0.576] | 0.744 | 0.468 |
| Tabular only | 0.697 | 0.701 [0.683, 0.718] | 0.864 | 0.711 |
| Early fusion (XGBoost, PCA-8) | 0.686 | 0.690 [0.672, 0.707] | 0.864 | 0.701 |
| **Late fusion (product rule)** | **0.707** | **0.711 [0.695, 0.728]** | **0.868** | **0.725** |
| Attention fusion | 0.702 | 0.705 [0.688, 0.722] | 0.865 | 0.721 |

Late fusion beats the best unimodal model by +1.09 macro-F1 points (paired bootstrap 95% CI 0.19–2.01). With CLIP encoders, early fusion reaches 0.710.

## Repository layout
```
airbnb_price_tiers.ipynb   end-to-end notebook, executed (all outputs of the reported run)
airbnb_mm.py               module: data & privacy, grouped split, preprocessing, feature extraction, models, evaluation
requirements.txt           dependencies
results/                   tables (CSV) + results.json + name_masking.json from the reported run
figures/                   figures from the reported run
```

## Reproduce
1. Copy this folder to Google Drive as `MyDrive/airbnb-multimodal-price/`.
2. Open `airbnb_price_tiers.ipynb` in Google Colab → *Runtime → Change runtime type → T4 GPU* → *Run all*. The first cell installs `sentence-transformers` and the spaCy model and prints `NOTEBOOK VERSION v3`.
3. First run ≈ 1–1.5 h (photo download ≈ 43 min, feature extraction ≈ 12 min); everything is cached in `cache/`, so re-runs take ≈ 25 min.

Local run (GPU recommended): `pip install -r requirements.txt`, `python -m spacy download en_core_web_sm`, then run the notebook with the environment variable `PROJECT_DIR` set to this folder.

Seeds: 42 for the split and XGBoost; 0, 1, 2 for the attention network (spread reported).

## Data and privacy
Data: Inside Airbnb, Melbourne snapshot 16 September 2026 — <https://insideairbnb.com/get-the-data/> — CC BY 4.0. The notebook downloads `listings.csv.gz` and the photos itself; **no data, photos or text are committed**. Host columns and exact coordinates are removed before analysis; the host id is used only as a salted hash for grouping; e-mails, URLs, phone numbers and names are masked; photos showing people are treated as a missing modality.

## Licence
Code: MIT. Data © Inside Airbnb (CC BY 4.0). Pretrained models: torchvision (BSD-3) and the Hugging Face model cards of `bert-base-uncased`, `sentence-transformers/all-mpnet-base-v2`, `openai/clip-vit-base-patch32`.
