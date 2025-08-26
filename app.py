# ===============================
# Twitter Anti-India Campaign Detector — Streamlit App (Single File)
# Author: Mukund + ChatGPT
# Run:  
#   pip install -r requirements.txt   (see list below in comments)
#   streamlit run app.py
#
# What it does (end-to-end, in one page):
#  1) Live fetch recent tweets matching your keywords via snscrape (no API key).
#  2) Score each tweet for risk using: keyword hits (+ optional toxicity model).
#  3) Cluster tweets into narratives (Sentence-Transformers; falls back to TF-IDF+KMeans).
#  4) Show dashboard: risky tweets table, cluster stats, trend charts.
#  5) Export evidence: CSV + PDF (summary with top risky tweets).
#
# Notes:
#  - For fastest demo, keep the model toggle OFF (keyword-only). Turn ON for richer scoring.
#  - If model downloads are slow, pre-run once on your machine to warm the cache.
#  - You can extend keyword lexicons in KEYWORDS below.
#
# Suggested requirements.txt (copy to a file and pip install):
# -----------------------------
# streamlit>=1.36.0
# snscrape>=0.7.0.20230622
# pandas>=2.2.0
# numpy>=1.26.0
# scikit-learn>=1.4.0
# sentence-transformers>=3.0.0
# transformers>=4.41.0
# torch>=2.2.0
# fpdf==1.7.2
# clean-text==0.6.0
# -----------------------------
# If wheels are an issue, you can comment out transformers/torch/sentence-transformers
# and run the app in "keyword-only mode".

import re
import io
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import streamlit as st

# Try importing snscrape
try:
    import snscrape.modules.twitter as sntwitter
except Exception as e:
    sntwitter = None

# Optional heavy deps (lazily loaded)
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# ============= Config & Lexicons ============= #
st.set_page_config(page_title="🇮🇳 Anti-India Campaign Detector (Twitter)", layout="wide")

DEFAULT_QUERY = "India OR Bharat OR Kashmir OR Boycott India"
MAX_TWEETS_HARD = 1000  # absolute cap per run

# Minimal bilingual lexicon (extend for your use case)
KEYWORDS_INCITEMENT = [
    # English incitement / harm
    r"kill", r"violence", r"attack", r"burn", r"destroy", r"terror", r"riot", r"genocide",
    # Hindi transliteration (simplified)
    r"maar", r"jalaa", r"tod", r"dhwansh", r"hinsa", r"danga", r"hathiyaar",
]
KEYWORDS_ANTI_INDIA = [
    r"free kashmir", r"boycott india", r"down with india", r"anti[-\s]?india", r"break india",
    r"divide india", r"disintegrate india", r"separate kashmir", r"indian occupation",
    # Hindi translit (examples)
    r"azaad kashmir", r"bharat murdabad", r"india murdabad", r"tukde tukde",
]

# Weighting for risk score
W_KEY_INCITEMENT = 0.6
W_KEY_ANTI = 0.4
W_MODEL = 0.5  # used only if model is enabled; blended later

# ============= Helpers ============= #

def build_query(keywords: str, hours_lookback: int, lang: str | None = None) -> str:
    since = (datetime.utcnow() - timedelta(hours=hours_lookback)).strftime("%Y-%m-%d")
    # snscrape uses since: and until:. We'll fetch recent without until to include now.
    q = f"{keywords} since:{since}"
    if lang:
        q += f" lang:{lang}"
    return q

@st.cache_data(show_spinner=False)
def regex_any(patterns: List[str]) -> re.Pattern:
    return re.compile("|".join(patterns), flags=re.IGNORECASE)

RE_INCITEMENT = regex_any(KEYWORDS_INCITEMENT)
RE_ANTI = regex_any(KEYWORDS_ANTI_INDIA)


def score_keywords(text: str) -> Dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        return {"incitement_hits": 0, "anti_hits": 0, "key_score": 0.0}
    inc_hits = len(re.findall(RE_INCITEMENT, text))
    anti_hits = len(re.findall(RE_ANTI, text))
    # Convert to [0,1] by squashing counts
    inc_score = 1 - np.exp(-inc_hits)
    anti_score = 1 - np.exp(-anti_hits)
    key_score = (W_KEY_INCITEMENT * inc_score + W_KEY_ANTI * anti_score)
    return {"incitement_hits": inc_hits, "anti_hits": anti_hits, "key_score": float(key_score)}


@st.cache_resource(show_spinner=False)
def load_toxicity_model():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-offensive")
        mdl = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-offensive")
        return tok, mdl
    except Exception as e:
        return None, None


def model_offensive_prob(texts: List[str], tok, mdl) -> List[float]:
    if tok is None or mdl is None:
        return [0.0] * len(texts)
    from torch.nn.functional import softmax
    import torch
    probs = []
    for t in texts:
        try:
            inputs = tok(t, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                out = mdl(**inputs)
            sc = softmax(out.logits, dim=-1)
            offensive = sc[0, 1].item()
        except Exception:
            offensive = 0.0
        probs.append(float(offensive))
    return probs


def compute_risk_row(row, use_model=False):
    key_score = row.get("key_score", 0.0)
    if use_model:
        # Blend keyword score with model offensive probability
        mdl_prob = row.get("model_offense", 0.0)
        # Blend using a smooth combining function
        blended = 1 - (1 - key_score) * (1 - W_MODEL * mdl_prob)
        return float(np.clip(blended, 0.0, 1.0))
    return float(np.clip(key_score, 0.0, 1.0))


def fetch_tweets(query: str, limit: int = 300) -> pd.DataFrame:
    if sntwitter is None:
        st.error("snscrape not available. Install with `pip install snscrape`.\nIf you're on Windows, use: pip install git+https://github.com/JustAnotherArchivist/snscrape")
        return pd.DataFrame()
    rows = []
    scraper = sntwitter.TwitterSearchScraper(query)
    for i, tweet in enumerate(scraper.get_items()):
        if i >= min(limit, MAX_TWEETS_HARD):
            break
        # Skip retweets for cleaner signals
        if hasattr(tweet, "retweetedTweet") and tweet.retweetedTweet is not None:
            continue
        rows.append({
            "date": tweet.date,
            "id": tweet.id,
            "url": f"https://x.com/{tweet.user.username}/status/{tweet.id}",
            "user": tweet.user.username,
            "displayname": getattr(tweet.user, "displayname", ""),
            "content": tweet.content,
            "replyCount": tweet.replyCount,
            "retweetCount": tweet.retweetCount,
            "likeCount": tweet.likeCount,
            "lang": tweet.lang,
        })
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def get_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


def cluster_texts(texts: List[str], k: int = 5):
    embedder = get_embedder()
    if embedder is not None:
        X = embedder.encode(texts, show_progress_bar=False)
    else:
        # Fallback: TF-IDF
        tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
        X = tfidf.fit_transform(texts).toarray()
    k = max(2, min(k, len(texts)))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return labels


def top_terms(texts: List[str], top_n: int = 10) -> List[str]:
    # Quick and simple keyword extraction using TF-IDF
    if len(texts) == 0:
        return []
    vec = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    idx = np.asarray(X.sum(axis=0)).ravel().argsort()[::-1]
    feats = np.array(vec.get_feature_names_out())[idx]
    return feats[:top_n].tolist()


def make_pdf(df: pd.DataFrame, clusters: Dict[int, Dict[str, Any]]) -> bytes:
    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Anti-India Campaign Report (Twitter)", ln=True)

    # Summary
    pdf.set_font("Arial", size=11)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    pdf.multi_cell(0, 7, f"Generated: {now}")
    pdf.ln(2)

    total = len(df)
    high = (df["risk_score"] >= 0.7).sum()
    med = ((df["risk_score"] >= 0.4) & (df["risk_score"] < 0.7)).sum()
    low = (df["risk_score"] < 0.4).sum()
    pdf.multi_cell(0, 7, f"Totals — All: {total} | High: {high} | Medium: {med} | Low: {low}")
    pdf.ln(3)

    # Cluster overview
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Clusters (Top Terms)", ln=True)
    pdf.set_font("Arial", size=11)
    for cid, meta in clusters.items():
        terms = ", ".join(meta.get("top_terms", [])[:8])
        count = meta.get("count", 0)
        pdf.multi_cell(0, 7, f"Cluster {cid}  |  {count} tweets  |  Terms: {terms}")
    pdf.add_page()

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Top High-Risk Tweets", ln=True)
    pdf.set_font("Arial", size=11)
    df_top = df.sort_values("risk_score", ascending=False).head(25)
    for _, r in df_top.iterrows():
        text = (r.get("content", "") or "").replace("\n", " ")
        line = f"[{r.get('risk_score',0):.2f}] @{r.get('user','')} — {r.get('url','')}"
        pdf.multi_cell(0, 7, line)
        pdf.multi_cell(0, 7, text)
        pdf.ln(2)

    return pdf.output(dest="S").encode("latin-1", "ignore")


# ============= UI ============= #
st.title("🇮🇳 Anti-India Campaign Detector — Twitter (Live Keywords)")
st.caption("Prototype: fetch → classify → cluster → alert → export. Public data only via snscrape.")

with st.sidebar:
    st.subheader("Search Controls")
    q = st.text_area("Keywords / boolean query", value=DEFAULT_QUERY, height=70, help="Use OR/AND to combine terms. e.g., (India OR Bharat) AND (Kashmir OR Boycott)")
    hours = st.slider("Lookback (hours)", 1, 72, 24)
    lang = st.selectbox("Language filter (optional)", ["", "en", "hi", "ur"], index=0)
    limit = st.slider("Max tweets to fetch", 50, 800, 300, step=50)

    st.divider()
    st.subheader("Scoring Settings")
    use_model = st.toggle("Use toxicity model (slower)", value=False)
    risk_threshold = st.slider("High-risk threshold", 0.1, 0.95, 0.6, 0.05)

    st.divider()
    n_clusters = st.slider("Number of clusters", 2, 12, 5)
    run_btn = st.button("🔎 Scan Now", use_container_width=True)

# Session state storage
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

if run_btn:
    with st.spinner("Fetching tweets…"):
        query = build_query(q, hours_lookback=hours, lang=lang if lang else None)
        df = fetch_tweets(query, limit=limit)
        st.session_state.df = df.copy()

# Work on current df
if st.session_state.df.empty:
    st.info("Enter keywords on the left and click **Scan Now** to fetch recent tweets.")
    st.stop()

# Scoring
with st.spinner("Scoring tweets…"):
    df = st.session_state.df.copy()
    key_parts = df["content"].fillna("").apply(score_keywords).apply(pd.Series)
    df = pd.concat([df, key_parts], axis=1)

    if use_model:
        tok, mdl = load_toxicity_model()
        df["model_offense"] = model_offensive_prob(df["content"].fillna("").tolist(), tok, mdl)
    else:
        df["model_offense"] = 0.0

    df["risk_score"] = df.apply(lambda r: compute_risk_row(r, use_model=use_model), axis=1)

# Clustering
with st.spinner("Clustering narratives…"):
    try:
        labels = cluster_texts(df["content"].fillna("").tolist(), k=n_clusters)
        df["cluster"] = labels
    except Exception as e:
        st.warning(f"Clustering fallback due to error: {e}")
        df["cluster"] = 0

# Save back
st.session_state.df = df.copy()

# ============= Dashboard Panels ============= #
col1, col2, col3, col4 = st.columns(4)
col1.metric("Tweets scanned", f"{len(df):,}")
high = (df["risk_score"] >= risk_threshold).sum()
col2.metric("High-risk tweets", f"{high:,}")
col3.metric("Avg. risk score", f"{df['risk_score'].mean():.2f}")
col4.metric("Clusters", f"{df['cluster'].nunique()}")

st.divider()

# Alerts
if high > 0:
    st.error(f"🚨 {high} tweets flagged over threshold {risk_threshold:.2f}. Review below.")
else:
    st.success("No tweets crossed the risk threshold for this query.")

# Cluster overview
st.subheader("Narrative Clusters")
cl_counts = df["cluster"].value_counts().sort_index()
st.bar_chart(cl_counts)

# Top terms per cluster
cluster_meta: Dict[int, Dict[str, Any]] = {}
with st.expander("Top terms by cluster"):
    for cid in sorted(df["cluster"].unique()):
        texts = df.loc[df["cluster"] == cid, "content"].astype(str).tolist()
        terms = top_terms(texts, top_n=10)
        cluster_meta[int(cid)] = {"count": len(texts), "top_terms": terms}
        st.markdown(f"**Cluster {cid}** — {len(texts)} tweets\n\n`" + ", ".join(terms) + "`")

st.divider()

# High-risk table
st.subheader("High-Risk Tweets")
st.dataframe(
    df.loc[df["risk_score"] >= risk_threshold, ["date", "user", "displayname", "content", "risk_score", "incitement_hits", "anti_hits", "cluster", "url"]]
      .sort_values("risk_score", ascending=False)
      .reset_index(drop=True),
    use_container_width=True,
    height=420,
)

# Full table (toggle)
with st.expander("All scanned tweets"):
    st.dataframe(
        df[["date", "user", "content", "risk_score", "cluster", "url"]]
          .sort_values("date", ascending=False)
          .reset_index(drop=True),
        use_container_width=True,
        height=350,
    )

st.divider()

# Downloads
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV (all tweets)", data=csv_bytes, file_name="tweets_scored.csv", mime="text/csv")

# PDF export (summary + top high-risk)
try:
    clusters_for_pdf = cluster_meta if cluster_meta else {int(c): {"count": int(n), "top_terms": []} for c, n in cl_counts.items()}
    pdf_bytes = make_pdf(df, clusters_for_pdf)
    st.download_button("⬇️ Download PDF Report", data=pdf_bytes, file_name="anti_india_report.pdf", mime="application/pdf")
except Exception as e:
    st.warning(f"PDF export unavailable: {e}")

st.caption("Prototype for research/demo only. Uses public data via snscrape. Focuses on coordination risk via clusters and incitement/anti-India keyword signals. Extend lexicons and add human analyst review for production use.")
