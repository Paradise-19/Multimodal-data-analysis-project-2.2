"""airbnb_mm — multimodal price-tier classification of Inside Airbnb Melbourne listings.

Modalities: main listing photo (image), host-written title + description (text), listing details (tabular).
Task: 3-class classification (budget / mid-range / premium) from nightly price tertiles.

The module is organised in the order of the pipeline:
    1. utilities            set_seed, caching, timing, figure/table helpers
    2. data                 loading, privacy filtering, grouping, grouped split, labels
    3. preprocessing        text cleaning (HTML + PII masking), tabular encoder, image download / quality
    4. feature extraction   CNN / ViT / CLIP image encoders, BERT / Sentence-BERT / CLIP text encoders,
                            person detector (privacy), PCA reduction
    5. models               tuned XGBoost (unimodal + early fusion), late fusion (sum / product rule),
                            attention-based intermediate fusion network (with missing-modality masks)
    6. evaluation           metrics (incl. ordinal + calibration), bootstrap CIs, subgroup / fairness metrics

All course references (Week N) point to the Multimodal Data Analysis lectures and practicals.
"""
from __future__ import annotations

import copy
import hashlib
import html
import io
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, log_loss,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

CLASSES = ["budget", "mid-range", "premium"]
MODALITIES = ["image", "tabular", "text"]
SEED = 42


# =====================================================================================================
# 1. Utilities
# =====================================================================================================
def set_seed(seed: int = SEED):
    """Fix all random number generators used in the project."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def cached_array(path: str, compute_fn, meta_path: str | None = None):
    """Load a .npy array from `path` or compute it with `compute_fn()` (which may return (array, meta))
    and save it. Feature extraction is expensive, so every embedding matrix is computed once and cached."""
    if os.path.exists(path):
        arr = np.load(path, allow_pickle=False)
        meta = json.load(open(meta_path)) if meta_path and os.path.exists(meta_path) else {}
        return arr, meta
    t0 = time.perf_counter()
    out = compute_fn()
    arr, meta = out if isinstance(out, tuple) else (out, {})
    meta = {**meta, "seconds": time.perf_counter() - t0}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, arr)
    if meta_path:
        json.dump(meta, open(meta_path, "w"), indent=1)
    return arr, meta


class Timer:
    """Context manager measuring wall-clock seconds (synchronises CUDA when available)."""

    def __enter__(self):
        self._sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._sync()
        self.seconds = time.perf_counter() - self.t0

    @staticmethod
    def _sync():
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass


# =====================================================================================================
# 2. Data: loading, privacy filtering, grouping, grouped split, labels
# =====================================================================================================
LISTINGS_URL = ("https://data.insideairbnb.com/australia/vic/melbourne/2026-09-16/"
                "data/listings.csv.gz")

# Columns describing the PLACE (kept). Every host_* column, the listing URL and exact coordinates are
# removed (re-identification risk: Szanto & Mehta, 2018). host_id is used only once, as a salted hash, to
# keep a host's listings in the same split, and is then discarded.
KEEP_COLUMNS = ["id", "name", "description", "picture_url", "price", "room_type", "property_type",
                "accommodates", "bedrooms", "beds", "bathrooms_text", "amenities", "minimum_nights",
                "neighbourhood_cleansed", "number_of_reviews", "review_scores_rating"]


def load_listings(cache_dir: str, url: str = LISTINGS_URL) -> pd.DataFrame:
    """Download (once) and read the Inside Airbnb listings file."""
    import requests
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, os.path.basename(url.split("?")[0]))
    if not os.path.exists(path):
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return pd.read_csv(path, compression="gzip", low_memory=False)


def privacy_filter(raw: pd.DataFrame, salt: str = "mda-2026") -> tuple[pd.DataFrame, dict]:
    """Keep only place-level columns; replace host_id by a salted hash used for grouping only."""
    dropped = [c for c in raw.columns if c not in KEEP_COLUMNS]
    df = raw[[c for c in KEEP_COLUMNS if c in raw.columns]].copy()
    if "host_id" in raw.columns:
        df["_host_key"] = raw["host_id"].astype(str).map(
            lambda s: hashlib.sha256((salt + s).encode()).hexdigest()[:16])
    else:
        df["_host_key"] = df["id"].astype(str)
    info = {"n_raw_columns": raw.shape[1], "n_dropped_columns": len(dropped),
            "dropped_host_columns": [c for c in dropped if c.startswith("host_")],
            "dropped_location_columns": [c for c in dropped if c in ("latitude", "longitude")]}
    return df, info


def parse_price(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")


def build_groups(df: pd.DataFrame, min_desc_chars: int = 30) -> np.ndarray:
    """Union-find over two leakage keys: (1) the same host and (2) the same (normalised) description.
    Listings connected by either key form one group, so copies never straddle train / validation / test."""
    parent = np.arange(len(df))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    desc = df["description"].fillna("").str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    for key in [df["_host_key"].values,
                np.where(desc.str.len() >= min_desc_chars, desc.values, None)]:
        first = {}
        for i, k in enumerate(key):
            if k is None:
                continue
            if k in first:
                union(first[k], i)
            else:
                first[k] = i
    roots = np.array([find(i) for i in range(len(df))])
    return pd.factorize(roots)[0]


def grouped_split(strata: np.ndarray, groups: np.ndarray, test=0.15, val=0.15, seed=SEED):
    """70 / 15 / 15 split that is stratified (approximately) and grouped (exactly) via StratifiedGroupKFold.
    20 folds of ~5 %: 3 folds -> test, 3 folds -> validation, 14 folds -> training."""
    n_folds = 20
    fold = np.empty(len(strata), dtype=int)
    for k, (_, idx) in enumerate(StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)
                                 .split(np.zeros(len(strata)), strata, groups)):
        fold[idx] = k
    n_te, n_va = round(test * n_folds), round(val * n_folds)
    split = np.where(fold < n_te, "test", np.where(fold < n_te + n_va, "val", "train"))
    return split


def make_labels(price: pd.Series, split: np.ndarray, cap_q: float = 0.99):
    """Outlier cap and tertile edges are estimated on the TRAINING split only and then applied to all
    splits (the proposal computed them on the full data; this removes a small test-set leak)."""
    train_price = price[split == "train"]
    cap = float(train_price.quantile(cap_q))
    kept = (price <= cap).values
    edges = train_price[train_price <= cap].quantile([1 / 3, 2 / 3]).values.astype(float)
    y = np.digitize(price.values, edges, right=True)          # <= e1 -> 0, (e1, e2] -> 1, > e2 -> 2
    return y, kept, cap, edges


# =====================================================================================================
# 3. Preprocessing
# =====================================================================================================
# ---- Text (Week 2: basic cleaning; Week 7: privacy) -------------------------------------------------
PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "url": re.compile(r"(https?://\S+|www\.\S+)"),
    "phone": re.compile(r"(?:\+?61|0)[\s-]?\d(?:[\s-]?\d){7,9}"),
}
TAG_RE = re.compile(r"<[^>]+>")


def clean_text(name, description) -> str:
    """Title + description, HTML entities/tags removed, PII masked, whitespace normalised.
    Case is kept for BERT-uncased (it lower-cases itself) and for the cased Sentence-BERT tokenizer."""
    text = f"{name if isinstance(name, str) else ''}. {description if isinstance(description, str) else ''}"
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = TAG_RE.sub(" ", text)
    for tag, pat in PII_PATTERNS.items():
        text = pat.sub(f"[{tag.upper()}]", text)
    return re.sub(r"\s+", " ", text).strip(" .")


INTRO_RE = re.compile(
    r"\b(?i:I am|I'm|I’m|my name is|name's|your hosts?(?: are| is)?|hosted by|ask for|contact|call|"
    r"text|message)\s+([A-Z][a-z]{1,15}(?:\s+(?:and|&)\s+[A-Z][a-z]{1,15})?)\b")


NOT_NAMES = {"Us", "Me", "Him", "Her", "Them", "The", "Our", "My", "Your", "Uber", "Airbnb", "Reception", "Management",
             "Melbourne", "Guests", "Guest", "Host", "Hosts", "Team", "Staff", "Please", "Now", "Today"}


def mask_person_names(texts: list[str], max_doc_share: float = 0.002, batch_size: int = 256):
    """Mask host / personal names with [NAME] (proposal, Section 6), using two precise rules:

    1. Self-introduction patterns ("I'm Sarah", "your host John", "contact Emma") - always masked.
    2. spaCy PERSON entities that are RARE in the corpus (in < max_doc_share of listings, i.e. < ~38 of 18.8k)
       and look like names (capitalised alphabetic tokens). A plain NER pass (first run) tagged 56% of texts,
       mostly frequent suburbs, landmarks and brands (St Kilda, Fitzroy, WiFi, Coles), which would erase the
       location and amenity signal; real personal names are listing-specific and therefore rare.

    Returns (masked_texts, share_of_texts_masked, {"masked_top": ..., "kept_top": ...}); spaCy unavailable ->
    only rule 1 is applied."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "tagger", "attribute_ruler"])
        docs = [[(e.start_char, e.end_char, e.text) for e in d.ents if e.label_ == "PERSON"]
                for d in nlp.pipe(texts, batch_size=batch_size)]
    except Exception:
        docs = [[] for _ in texts]
    doc_freq = {}
    for ents in docs:
        for k in {e[2] for e in ents}:
            doc_freq[k] = doc_freq.get(k, 0) + 1
    limit = max_doc_share * len(texts)
    name_like = re.compile(r"^(?:[A-Z][a-z'’-]+)(?:\s+[A-Z][a-z'’-]+){0,2}$")
    masked_count, kept_count, out, n_hit = {}, {}, [], 0
    for t, ents in zip(texts, docs):
        spans = []
        for a, b, k in ents:
            if doc_freq[k] < limit and name_like.match(k):
                spans.append((a, b)); masked_count[k] = masked_count.get(k, 0) + 1
            else:
                kept_count[k] = kept_count.get(k, 0) + 1
        for m in INTRO_RE.finditer(t):
            if m.group(1).split()[0] in NOT_NAMES:
                continue
            spans.append(m.span(1)); masked_count[m.group(1)] = masked_count.get(m.group(1), 0) + 1
        if spans:
            n_hit += 1
            merged = []
            for a, b in sorted(spans):
                if merged and a <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
                else:
                    merged.append((a, b))
            for a, b in reversed(merged):
                t = t[:a] + "[NAME]" + t[b:]
        out.append(t)
    top = lambda d: dict(sorted(d.items(), key=lambda kv: -kv[1])[:25])
    return out, n_hit / max(len(texts), 1), {"masked_top": top(masked_count), "kept_top": top(kept_count)}


def pii_counts(texts: pd.Series) -> dict:
    return {k: float(texts.fillna("").str.contains(p.pattern, regex=True).mean()) for k, p in PII_PATTERNS.items()}


# ---- Tabular ---------------------------------------------------------------------------------------
def parse_amenities(s) -> list[str]:
    if not isinstance(s, str):
        return []
    try:
        return [a.strip().lower() for a in json.loads(s)]
    except json.JSONDecodeError:
        return []


class TabularEncoder:
    """Listing-detail features. Every statistic (medians, scaler, category vocabularies, amenity vocabulary)
    is fitted on the training split only.

    numeric : accommodates, bedrooms, beds, bathrooms, n_amenities, log1p(number_of_reviews),
              review_scores_rating, log1p(min(minimum_nights, 365))
    flags   : missing-value indicators, no_reviews_yet, shared_bath
    one-hot : room_type, property_type (top-k + other), neighbourhood (optional: area ablation)
    multi-hot: the top-k most common amenities (e.g. pool, gym, hot tub)
    """

    NUMERIC = ["accommodates", "bedrooms", "beds", "bathrooms", "n_amenities", "log_reviews",
               "review_scores_rating", "log_min_nights"]

    def __init__(self, include_area: bool = True, top_property: int = 15, top_amenities: int = 40):
        self.include_area, self.top_property, self.top_amenities = include_area, top_property, top_amenities

    @staticmethod
    def base_frame(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        bt = df["bathrooms_text"].fillna("").astype(str).str.lower()
        num = bt.str.extract(r"([\d.]+)")[0].astype(float)
        out["bathrooms"] = np.where(bt.str.contains("half-bath") & num.isna(), 0.5, num)
        out["shared_bath"] = bt.str.contains("shared").astype(float)
        for c in ["accommodates", "bedrooms", "beds", "review_scores_rating"]:
            out[c] = pd.to_numeric(df[c], errors="coerce")
        out["n_amenities"] = df["amenities"].map(lambda s: len(parse_amenities(s)))
        out["log_reviews"] = np.log1p(pd.to_numeric(df["number_of_reviews"], errors="coerce").fillna(0))
        out["log_min_nights"] = np.log1p(pd.to_numeric(df["minimum_nights"], errors="coerce").clip(upper=365))
        out["no_reviews_yet"] = df["review_scores_rating"].isna().astype(float)
        return out

    def fit(self, df: pd.DataFrame):
        base = self.base_frame(df)
        self.medians_ = base[self.NUMERIC].median()
        self.missing_cols_ = [c for c in self.NUMERIC if base[c].isna().any() and c != "review_scores_rating"]
        self.room_types_ = sorted(df["room_type"].dropna().unique())
        self.property_types_ = df["property_type"].value_counts().index[: self.top_property].tolist()
        self.areas_ = sorted(df["neighbourhood_cleansed"].dropna().unique())
        counts = pd.Series([a for s in df["amenities"] for a in parse_amenities(s)]).value_counts()
        self.amenities_ = counts.index[: self.top_amenities].tolist()
        self.scaler_ = StandardScaler().fit(self._raw(df))
        return self

    def _raw(self, df: pd.DataFrame) -> pd.DataFrame:
        base = self.base_frame(df)
        cols = dict(base[self.NUMERIC].fillna(self.medians_))
        for c in self.missing_cols_:
            cols[f"{c}_missing"] = base[c].isna().astype(float)
        cols["no_reviews_yet"] = base["no_reviews_yet"]
        cols["shared_bath"] = base["shared_bath"]
        for r in self.room_types_:
            cols[f"room={r}"] = (df["room_type"] == r).astype(float)
        prop = df["property_type"].where(df["property_type"].isin(self.property_types_), "other")
        for p in self.property_types_ + ["other"]:
            cols[f"property={p}"] = (prop == p).astype(float)
        if self.include_area:
            for a in self.areas_:
                cols[f"area={a}"] = (df["neighbourhood_cleansed"] == a).astype(float)
        am = df["amenities"].map(lambda s: set(parse_amenities(s)))
        for a in self.amenities_:
            cols[f"amenity={a}"] = am.map(lambda s, a=a: float(a in s))
        return pd.DataFrame(cols, index=df.index)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.scaler_.transform(self._raw(df)).astype(np.float32)

    @property
    def feature_names_(self):
        return list(self.scaler_.feature_names_in_)


# ---- Images (download + Week 2 quality metrics) -----------------------------------------------------
def _download_one(args):
    import requests
    from PIL import Image, ImageOps
    url, path, width = args
    if os.path.exists(path):
        return True
    try:
        r = requests.get(url.split("?")[0] + f"?im_w={width}", timeout=20)
        r.raise_for_status()
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(r.content))).convert("RGB")
        im.thumbnail((width, width))
        im.save(path, "JPEG", quality=90)
        return True
    except Exception:
        return False


def download_images(df: pd.DataFrame, img_dir: str, width: int = 480, workers: int = 32) -> np.ndarray:
    """Download the main photo of every listing once (resumable). Returns a boolean 'downloaded' mask."""
    os.makedirs(img_dir, exist_ok=True)
    jobs = [(u, os.path.join(img_dir, f"{i}.jpg"), width) for i, u in zip(df["id"], df["picture_url"])]
    with ThreadPoolExecutor(workers) as pool:
        ok = list(pool.map(_download_one, jobs))
    return np.array(ok)


def image_path(img_dir, listing_id):
    return os.path.join(img_dir, f"{listing_id}.jpg")


def image_quality(paths: list[str], size: int = 224, workers: int = 8) -> pd.DataFrame:
    """Week 2 image-quality indicators: resolution, aspect ratio, brightness, contrast and blur
    (variance of the Laplacian; low = blurry)."""
    from PIL import Image
    from scipy import ndimage

    def one(p):
        try:
            im = Image.open(p)
            w, h = im.size
            g = np.asarray(im.convert("L").resize((size, size)), dtype=np.float32)
            return dict(width=w, height=h, aspect=w / h, brightness=g.mean(), contrast=g.std(),
                        laplacian_var=float(ndimage.laplace(g).var()), readable=True)
        except Exception:
            return dict(width=np.nan, height=np.nan, aspect=np.nan, brightness=np.nan, contrast=np.nan,
                        laplacian_var=np.nan, readable=False)

    with ThreadPoolExecutor(workers) as pool:
        return pd.DataFrame(list(pool.map(one, paths)))


# =====================================================================================================
# 4. Feature extraction (Week 3)
# =====================================================================================================
class _ImageFolderDataset:
    """Minimal torch dataset over listing photos; unreadable files return a black image + flag 0."""

    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        import torch
        try:
            im = Image.open(self.paths[i]).convert("RGB")
            ok = 1
        except Exception:
            im, ok = Image.new("RGB", (224, 224)), 0
        x = self.transform(im)
        return x, torch.tensor(ok)


def _image_loader(paths, transform, batch_size=128, num_workers=2):
    import torch
    from torch.utils.data import DataLoader
    return DataLoader(_ImageFolderDataset(paths, transform), batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


def extract_image_features(paths: list[str], backbone: str = "efficientnet_b0", batch_size: int = 128):
    """Frozen ImageNet backbones as feature extractors (Week 3).
    efficientnet_b0 -> 1280-d global-average-pooled features; vit_b_16 -> 768-d [CLS] embedding."""
    import torch
    import torchvision
    device = get_device()
    if backbone == "efficientnet_b0":
        w = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = torchvision.models.efficientnet_b0(weights=w)
        model.classifier = torch.nn.Identity()
    elif backbone == "vit_b_16":
        w = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        model = torchvision.models.vit_b_16(weights=w)
        model.heads = torch.nn.Identity()
    else:
        raise ValueError(backbone)
    model = model.eval().to(device)
    feats, oks = [], []
    with torch.no_grad(), torch.autocast(device_type=device, enabled=device == "cuda"):
        for x, ok in _image_loader(paths, w.transforms(), batch_size):
            feats.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
            oks.append(ok.numpy())
    return np.concatenate(feats).astype(np.float32), {"backbone": backbone, "readable": int(np.concatenate(oks).sum())}


def _as_tensor(out):
    """transformers versions differ in whether get_*_features returns a tensor or a model output."""
    import torch
    if torch.is_tensor(out):
        return out
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        if getattr(out, attr, None) is not None:
            return getattr(out, attr)
    return out[0]


def extract_clip_image_features(paths: list[str], model_name="openai/clip-vit-base-patch32", batch_size=128):
    """CLIP image embeddings (Radford et al., 2021) - state-of-the-art reference encoder."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    device = get_device()
    model = CLIPModel.from_pretrained(model_name).eval().to(device)
    proc = CLIPProcessor.from_pretrained(model_name)
    feats = []
    with torch.no_grad():
        for s in range(0, len(paths), batch_size):
            ims = []
            for p in paths[s:s + batch_size]:
                try:
                    ims.append(Image.open(p).convert("RGB"))
                except Exception:
                    ims.append(Image.new("RGB", (224, 224)))
            px = proc.image_processor(ims, return_tensors="pt")["pixel_values"].to(device)
            feats.append(_as_tensor(model.get_image_features(pixel_values=px)).float().cpu().numpy())
    return np.concatenate(feats).astype(np.float32), {"model": model_name}


def extract_bert_mean(texts: list[str], model_name="bert-base-uncased", max_length=256, batch_size=64):
    """BERT last-layer mean pooling over real (non-padding) tokens, as in Weeks 4-8."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = get_device()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    feats, n_trunc = [], 0
    with torch.no_grad(), torch.autocast(device_type=device, enabled=device == "cuda"):
        for s in range(0, len(texts), batch_size):
            batch = texts[s:s + batch_size]
            full = tok(batch, add_special_tokens=True, truncation=False)["input_ids"]
            n_trunc += sum(len(ids) > max_length for ids in full)
            enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state.float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            feats.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    return np.concatenate(feats).astype(np.float32), {"model": model_name, "max_length": max_length,
                                                       "truncated_share": n_trunc / len(texts)}


def extract_sbert(texts: list[str], model_name="sentence-transformers/all-mpnet-base-v2", batch_size=64):
    """Sentence-BERT (Reimers & Gurevych, 2019): siamese-trained sentence embeddings."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=get_device())
    emb = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    return emb.astype(np.float32), {"model": model_name, "max_seq_length": model.max_seq_length}


def extract_clip_text(texts: list[str], model_name="openai/clip-vit-base-patch32", batch_size=256):
    import torch
    from transformers import CLIPModel, CLIPTokenizer
    device = get_device()
    model = CLIPModel.from_pretrained(model_name).eval().to(device)
    tok = CLIPTokenizer.from_pretrained(model_name)
    feats = []
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            enc = tok(texts[s:s + batch_size], padding=True, truncation=True, max_length=77,
                      return_tensors="pt").to(device)
            feats.append(_as_tensor(model.get_text_features(**enc)).float().cpu().numpy())
    return np.concatenate(feats).astype(np.float32), {"model": model_name, "max_length": 77}


def detect_people(paths: list[str], score_thr=0.8, min_area=0.05, batch_size=32):
    """Privacy screen (Week 7): flag photos that show a person (COCO label 1) with a box covering at least
    `min_area` of the image, using a pretrained Faster R-CNN (MobileNetV3, 320 px). Flagged photos are
    treated as a MISSING image modality instead of being used."""
    import torch
    import torchvision
    from torchvision.transforms.functional import to_tensor
    from PIL import Image
    device = get_device()
    w = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    det = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=w).eval().to(device)
    flags = np.zeros(len(paths), dtype=bool)
    with torch.no_grad():
        for s in range(0, len(paths), batch_size):
            ims, idx = [], []
            for j, p in enumerate(paths[s:s + batch_size]):
                try:
                    ims.append(to_tensor(Image.open(p).convert("RGB")).to(device))
                    idx.append(s + j)
                except Exception:
                    pass
            if not ims:
                continue
            for k, out in zip(idx, det(ims)):
                h, w_ = ims[idx.index(k)].shape[1:]
                keep = (out["labels"] == 1) & (out["scores"] >= score_thr)
                if keep.any():
                    b = out["boxes"][keep]
                    area = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).max().item() / (h * w_)
                    flags[k] = area >= min_area
    return flags


def fit_pca(X_train: np.ndarray, n_components: int = 128, seed=SEED):
    """PCA fitted on training rows only. Reduces 768/1280-d embeddings so that, in early fusion, the
    high-dimensional blocks do not swamp the ~100 tabular columns (Week 4: single-modality domination)."""
    scaler = StandardScaler().fit(X_train)
    n_components = min(n_components, X_train.shape[1], X_train.shape[0])
    pca = PCA(n_components=n_components, random_state=seed).fit(scaler.transform(X_train))
    return lambda X: pca.transform(scaler.transform(X)).astype(np.float32), float(pca.explained_variance_ratio_.sum())


# =====================================================================================================
# 5. Models
# =====================================================================================================
def xgb_base(device="cpu"):
    return dict(objective="multi:softprob", num_class=3, eval_metric="mlogloss", tree_method="hist",
                subsample=0.8, min_child_weight=1, reg_lambda=1.0, seed=SEED, device=device, verbosity=0)


def tune_xgb_cv(X, y, groups, grid: dict, device="cpu", n_splits=5, max_rounds=2000, early_stopping=50):
    """Grid search with 5-fold group-stratified CV on the TRAINING split (xgb.cv). The number of trees of each
    configuration is chosen by early stopping on the CV log loss (same protocol as Assignment 2)."""
    import xgboost as xgb
    dtrain = xgb.DMatrix(X, label=y, missing=np.nan)
    folds = list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=SEED).split(X, y, groups))
    rows = []
    for combo in product(*grid.values()):
        params = {**xgb_base(device), **dict(zip(grid.keys(), combo))}
        cv = xgb.cv(params, dtrain, num_boost_round=max_rounds, folds=folds,
                    early_stopping_rounds=early_stopping, seed=SEED)
        rows.append({**dict(zip(grid.keys(), combo)), "n_estimators": len(cv),
                     "cv_log_loss": cv["test-mlogloss-mean"].iloc[-1],
                     "cv_log_loss_sd": cv["test-mlogloss-std"].iloc[-1],
                     "cv_train_log_loss": cv["train-mlogloss-mean"].iloc[-1],
                     "curve_test": cv["test-mlogloss-mean"].to_numpy(),
                     "curve_train": cv["train-mlogloss-mean"].to_numpy()})
    return pd.DataFrame(rows).sort_values("cv_log_loss").reset_index(drop=True)


def best_params(search: pd.DataFrame, keys) -> dict:
    b = search.iloc[0]
    p = {k: (int(b[k]) if k in ("max_depth", "n_estimators") else float(b[k])) for k in keys}
    p["n_estimators"] = int(b["n_estimators"])
    return p


def make_xgb(params: dict, device="cpu"):
    import xgboost as xgb
    base = xgb_base(device)
    base.pop("num_class")
    base.pop("seed")
    return xgb.XGBClassifier(**base, random_state=SEED, **params)


class EarlyFusionXGB:
    """Feature-level (early) fusion (Week 4): [image PCA | tabular | text PCA] -> one XGBoost.
    Masked (missing) image rows are passed as NaN, which XGBoost routes along a learned default branch."""

    def __init__(self, params: dict, device="cpu"):
        self.model = make_xgb(params, device)

    @staticmethod
    def concat(blocks: dict, image_present=None):
        img = blocks["image"].copy()
        if image_present is not None:
            img[~image_present] = np.nan
        return np.hstack([img, blocks["tabular"], blocks["text"]]).astype(np.float32)

    def fit(self, blocks, y, image_present=None):
        self.dims_ = [blocks[m].shape[1] for m in ["image", "tabular", "text"]]
        with Timer() as t:
            self.model.fit(self.concat(blocks, image_present), y)
        self.fit_time_ = t.seconds
        return self

    def predict_proba(self, blocks, image_present=None):
        return self.model.predict_proba(self.concat(blocks, image_present))

    def modality_gain_share(self) -> dict:
        """Share of total split gain contributed by each modality block."""
        gain = self.model.get_booster().get_score(importance_type="total_gain")
        edges = np.cumsum([0] + self.dims_)
        tot = {m: 0.0 for m in ["image", "tabular", "text"]}
        for f, g in gain.items():
            j = int(f[1:])
            m = ["image", "tabular", "text"][np.searchsorted(edges, j, side="right") - 1]
            tot[m] += g
        s = sum(tot.values())
        return {m: v / s for m, v in tot.items()}


class LateFusion:
    """Decision-level (late) fusion (Week 4): one model per modality, combined with weights learned on the
    VALIDATION split by minimising log loss (Assignment 2 protocol).
      sum rule     p = sum_m w_m p_m              (w on the simplex)
      product rule p ∝ prod_m p_m ** w_m          (log-linear pool; w >= 0)
    A missing image is represented by the uniform distribution, which is neutral under the product rule."""

    def __init__(self, rule="product"):
        self.rule = rule

    @staticmethod
    def fuse(P: list, w, rule="product"):
        if rule == "sum":
            return sum(wi * Pi for wi, Pi in zip(w, P))
        z = sum(wi * np.log(np.clip(Pi, 1e-7, 1)) for wi, Pi in zip(w, P))
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    @staticmethod
    def optimise(P, y, rule):
        M = len(P)
        loss = lambda w: log_loss(y, np.clip(LateFusion.fuse(P, w, rule), 1e-12, 1), labels=[0, 1, 2])
        if rule == "sum":
            res = minimize(loss, np.full(M, 1 / M), method="SLSQP", bounds=[(0, 1)] * M,
                           constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
            w = np.clip(res.x, 0, 1)
            return w / w.sum(), res.fun
        res = minimize(loss, np.ones(M), method="L-BFGS-B", bounds=[(0, None)] * M)
        return np.clip(res.x, 0, None), res.fun

    def fit(self, P_val: list, y_val):
        self.search_ = {}
        for rule in ("sum", "product"):
            w, l = self.optimise(P_val, y_val, rule)
            self.search_[rule] = {"weights": w, "val_log_loss": float(l)}
        self.rule = min(self.search_, key=lambda r: self.search_[r]["val_log_loss"])
        self.weights_ = self.search_[self.rule]["weights"]
        return self

    def predict_proba(self, P: list):
        return self.fuse(P, self.weights_, self.rule)


def neutral_image_proba(P_img: np.ndarray, image_present: np.ndarray):
    P = P_img.copy()
    P[~image_present] = 1 / 3
    return P


# ---- Attention-based intermediate fusion (Weeks 4-5; Assignment 2 architecture) ----------------------
def build_attention_net(dims: dict, hidden_dim=128, num_heads=4, dropout=0.3, modality_dropout=0.1):
    import torch
    import torch.nn as nn

    class ModalityEncoder(nn.Module):
        """Two-layer MLP projecting one modality into the shared hidden space (two-tower idea, Week 6)."""

        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))

        def forward(self, x):
            return self.net(x)

    class AttentionFusionNet(nn.Module):
        """1. one token per modality (encoders) + learnable modality-type embedding;
        2. transformer block: multi-head self-attention across modality tokens + FFN, residual + LayerNorm
           (Week 5), so each modality is re-written in the context of the others;
        3. additive attention pooling (Week 4) -> per-listing modality weights alpha;
        4. linear classifier.  Missing modalities are excluded through key_padding_mask and from pooling;
        modality dropout (Neverova et al., 2016) randomly hides modality tokens during training."""

        def __init__(self):
            super().__init__()
            self.order = ["image", "tabular", "text"]
            self.encoders = nn.ModuleDict({m: ModalityEncoder(dims[m]) for m in self.order})
            self.type_emb = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)
            self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.ff = nn.Sequential(nn.Linear(hidden_dim, 4 * hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                    nn.Linear(4 * hidden_dim, hidden_dim))
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.drop = nn.Dropout(dropout)
            self.pool = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 3))
            self.modality_dropout = modality_dropout

        def forward(self, xs: dict, present=None):
            B = xs["tabular"].shape[0]
            tok = torch.stack([self.encoders[m](xs[m]) for m in self.order], 1) + self.type_emb
            if present is None:
                present = torch.ones(B, 3, dtype=torch.bool, device=tok.device)
            present = present.clone()
            if self.training and self.modality_dropout > 0:
                # Week 6 balancing strategy: any modality (incl. the dominant tabular one) can be hidden,
                # which forces the network to learn from image and text too; a row never loses all tokens.
                drop = torch.rand(B, 3, device=tok.device) < self.modality_dropout
                keep = present & ~drop
                empty = ~keep.any(1)
                keep[empty] = present[empty]
                present = keep
            h, attn_w = self.attn(tok, tok, tok, key_padding_mask=~present, need_weights=True,
                                  average_attn_weights=True)
            tok = self.norm1(tok + self.drop(h))
            tok = self.norm2(tok + self.drop(self.ff(tok)))
            score = self.pool(tok).squeeze(-1).masked_fill(~present, float("-inf"))
            alpha = torch.softmax(score, 1)
            fused = (alpha.unsqueeze(-1) * tok).sum(1)
            return self.head(fused), alpha, attn_w

    return AttentionFusionNet()


def to_torch(blocks: dict, present: np.ndarray, device):
    import torch
    xs = {m: torch.tensor(np.nan_to_num(blocks[m]), dtype=torch.float32, device=device) for m in MODALITIES}
    return xs, torch.tensor(present, dtype=torch.bool, device=device)


def train_attention(train: tuple, val: tuple, dims: dict, hidden_dim=128, num_heads=4, dropout=0.3,
                    modality_dropout=0.1, lr=1e-3, weight_decay=1e-4, batch_size=256, max_epochs=200,
                    patience=20, seed=SEED, device=None):
    """AdamW + cross-entropy, mini-batches; early stopping on validation loss restores the best weights
    (Weeks 6-7). train / val are tuples (blocks, present, y)."""
    import torch
    import torch.nn as nn
    device = device or get_device()
    set_seed(seed)
    model = build_attention_net(dims, hidden_dim, num_heads, dropout, modality_dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    Xtr, Ptr = to_torch(train[0], train[1], device)
    ytr = torch.tensor(train[2], device=device)
    Xva, Pva = to_torch(val[0], val[1], device)
    yva = torch.tensor(val[2], device=device)
    gen = torch.Generator().manual_seed(seed)
    hist = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc"]}
    best, best_state, best_epoch, wait = np.inf, None, 0, 0
    with Timer() as t:
        for epoch in range(max_epochs):
            model.train()
            order = torch.randperm(len(ytr), generator=gen).to(device)
            tl, tc = 0.0, 0
            for s in range(0, len(ytr), batch_size):
                b = order[s:s + batch_size]
                logits, _, _ = model({m: Xtr[m][b] for m in MODALITIES}, Ptr[b])
                loss = crit(logits, ytr[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                tl += loss.item() * len(b)
                tc += (logits.argmax(1) == ytr[b]).sum().item()
            model.eval()
            with torch.no_grad():
                vl_logits, _, _ = model(Xva, Pva)
                vl = crit(vl_logits, yva).item()
            hist["train_loss"].append(tl / len(ytr))
            hist["train_acc"].append(tc / len(ytr))
            hist["val_loss"].append(vl)
            hist["val_acc"].append((vl_logits.argmax(1) == yva).float().mean().item())
            if vl < best - 1e-4:
                best, best_epoch, wait = vl, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                wait += 1
                if wait >= patience:
                    break
    model.load_state_dict(best_state)
    model.eval()
    model.train_time_, model.best_epoch_, model.history_ = t.seconds, best_epoch, hist
    model.n_params_ = sum(p.numel() for p in model.parameters())
    return model


def attention_predict(model, blocks, present, device=None, batch_size=4096):
    """Returns class probabilities and per-listing modality attention weights alpha (n, 3)."""
    import torch
    device = device or get_device()
    model.eval()
    X, P = to_torch(blocks, present, device)
    probs, alphas = [], []
    with torch.no_grad():
        for s in range(0, len(P), batch_size):
            logits, alpha, _ = model({m: X[m][s:s + batch_size] for m in MODALITIES}, P[s:s + batch_size])
            probs.append(torch.softmax(logits, 1).cpu().numpy())
            alphas.append(alpha.cpu().numpy())
    return np.concatenate(probs), np.concatenate(alphas)


def mc_dropout_predict(model, blocks, present, K=30, device=None):
    """Monte-Carlo dropout (Week 5): K stochastic passes with dropout active (modality dropout off).
    Returns the mean probabilities and the predictive entropy (uncertainty)."""
    import torch
    device = device or get_device()
    X, P = to_torch(blocks, present, device)
    md = model.modality_dropout
    model.modality_dropout = 0.0
    model.train()
    with torch.no_grad():
        samples = torch.stack([torch.softmax(model(X, P)[0], 1) for _ in range(K)]).cpu().numpy()
    model.eval()
    model.modality_dropout = md
    mean = samples.mean(0)
    entropy = -(mean * np.log(np.clip(mean, 1e-12, 1))).sum(1)
    return mean, entropy


# =====================================================================================================
# 6. Evaluation
# =====================================================================================================
def expected_calibration_error(y, P, n_bins=15):
    conf, pred = P.max(1), P.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


def metrics(y, P) -> dict:
    """Classification (accuracy, macro / weighted P-R-F1), ranking (macro OvR ROC-AUC), probabilistic (log
    loss, ECE) and ordinal (quadratic-weighted kappa; share of 'two-tier' errors, i.e. budget <-> premium)."""
    pred = P.argmax(1)
    pw, rw, fw, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "macro_f1": fm, "macro_precision": pm, "macro_recall": rm,
            "weighted_f1": fw, "weighted_precision": pw, "weighted_recall": rw,
            "roc_auc_ovr": roc_auc_score(y, P, multi_class="ovr", labels=[0, 1, 2]),
            "log_loss": log_loss(y, np.clip(P, 1e-12, 1), labels=[0, 1, 2]),
            "ece": expected_calibration_error(y, P),
            "qwk": cohen_kappa_score(y, pred, weights="quadratic"),
            "two_tier_error": float(np.mean(np.abs(pred - y) == 2))}


def per_class(y, P) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(y, P.argmax(1), labels=[0, 1, 2], zero_division=0)
    return pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=CLASSES)


def bootstrap_ci(y, P, fn=lambda y, p: f1_score(y, p, average="macro"), n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    pred = P.argmax(1)
    vals = [fn(y[i], pred[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    return np.percentile(vals, [2.5, 97.5])


def paired_bootstrap_diff(y, P_a, P_b, n_boot=2000, seed=SEED):
    """95 % CI of macro-F1(a) - macro-F1(b) on the same resampled test listings."""
    rng = np.random.default_rng(seed)
    pa, pb = P_a.argmax(1), P_b.argmax(1)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        d.append(f1_score(y[i], pa[i], average="macro") - f1_score(y[i], pb[i], average="macro"))
    d = np.array(d)
    return float(np.mean(d)), np.percentile(d, [2.5, 97.5]), float(np.mean(d <= 0))


def group_metrics(y, P, groups: pd.Series, min_n=30) -> pd.DataFrame:
    """Per-subgroup accuracy, macro-F1, premium recall (equal opportunity, Week 7) and the predicted vs actual
    premium rate."""
    pred = P.argmax(1)
    rows = []
    for g, idx in pd.Series(np.arange(len(y))).groupby(groups.values):
        idx = idx.values
        if len(idx) < min_n:
            continue
        yt, yp = y[idx], pred[idx]
        prem = yt == 2
        rows.append({"group": g, "n": len(idx), "accuracy": accuracy_score(yt, yp),
                     "macro_f1": f1_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0),
                     "premium_recall": (yp[prem] == 2).mean() if prem.any() else np.nan,
                     "actual_premium_rate": prem.mean(), "predicted_premium_rate": (yp == 2).mean()})
    return pd.DataFrame(rows).set_index("group").sort_values("accuracy")


def disparity_summary(gm: pd.DataFrame) -> dict:
    return {"groups": int(len(gm)), "accuracy_min": gm["accuracy"].min(), "accuracy_max": gm["accuracy"].max(),
            "accuracy_gap": gm["accuracy"].max() - gm["accuracy"].min(), "accuracy_sd": gm["accuracy"].std(),
            "premium_recall_gap": gm["premium_recall"].max() - gm["premium_recall"].min()}


def inference_ms_per_1000(predict_fn, n_rows, repeats=5):
    times = []
    for _ in range(repeats):
        with Timer() as t:
            predict_fn()
        times.append(t.seconds)
    return 1000 * np.median(times) / n_rows * 1000
