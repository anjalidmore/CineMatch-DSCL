"""
CineMatch — AI-Powered Movie Recommendation System
Netflix-inspired UI with dynamic learning from user feedback.
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import json
import logging
import urllib.parse

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_PATH = os.path.join(os.path.dirname(__file__), "cinematch.log")
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

def log_metrics():
    if os.path.exists("metrics.pkl"):
        m = pickle.load(open("metrics.pkl", "rb"))
        logging.info("METRICS  Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f", m["accuracy"], m["precision"], m["recall"], m["f1_score"])

# ─── User Preferences (persisted to JSON) ─────────────────────────────────────
PREFS_PATH = os.path.join(os.path.dirname(__file__), "user_prefs.json")

def load_prefs():
    if os.path.exists(PREFS_PATH):
        with open(PREFS_PATH, "r") as f:
            return json.load(f)
    return {"liked": [], "disliked": [], "watched": [], "watchlist": []}

def save_prefs():
    with open(PREFS_PATH, "w") as f:
        json.dump(st.session_state.prefs, f, indent=2)

if "prefs" not in st.session_state:
    st.session_state.prefs = load_prefs()
if "active_movie" not in st.session_state:
    st.session_state.active_movie = None
if "browse_sel" not in st.session_state:
    st.session_state.browse_sel = None
if "metrics_logged" not in st.session_state:
    log_metrics()
    st.session_state.metrics_logged = True


def do_like(title):
    p = st.session_state.prefs
    if title in p["liked"]:
        p["liked"].remove(title)
    else:
        p["liked"].append(title)
        if title in p["disliked"]:
            p["disliked"].remove(title)
    save_prefs()

def do_dislike(title):
    p = st.session_state.prefs
    if title in p["disliked"]:
        p["disliked"].remove(title)
    else:
        p["disliked"].append(title)
        if title in p["liked"]:
            p["liked"].remove(title)
    save_prefs()

def do_watched(title):
    p = st.session_state.prefs
    if title in p["watched"]:
        p["watched"].remove(title)
    else:
        p["watched"].append(title)
    save_prefs()

def do_watchlist(title):
    p = st.session_state.prefs
    if title in p["watchlist"]:
        p["watchlist"].remove(title)
    else:
        p["watchlist"].append(title)
    save_prefs()

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

.stApp { background-color: #141414; font-family: 'DM Sans', sans-serif; color: #e5e5e5; }
[data-testid="stAppViewContainer"] { background: transparent; }

.hero { padding: 48px 40px 28px; }
.hero h1 { font-family: 'Space Grotesk', sans-serif; font-size: 2.6rem; font-weight: 700; color: #fff; margin: 0 0 6px; letter-spacing: -1px; }
.hero p { font-size: .95rem; color: #808080; margin: 0 0 24px; }

.section-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem; font-weight: 600; color: #fff; margin: 0 0 16px; }

.mc { background: #1c1c1c; border-radius: 8px; overflow: hidden; transition: transform .2s, box-shadow .2s; margin-bottom: 4px; }
.mc:hover { transform: scale(1.02); box-shadow: 0 6px 24px rgba(0,0,0,.5); }
.mc.dimmed { opacity: 0.4; }
.mc-hdr { background: linear-gradient(135deg,#8b1a1a,#c0392b); padding: 14px 14px 10px; position: relative; }
.mc-hdr.dimmed-hdr { background: linear-gradient(135deg,#333,#444); }
.mc-rank { position: absolute; top: 8px; right: 10px; background: rgba(0,0,0,.35); color: #fff; font-size: .65rem; font-weight: 700; padding: 2px 7px; border-radius: 4px; }
.mc-title { font-family: 'Space Grotesk', sans-serif; font-size: .88rem; font-weight: 600; color: #fff; line-height: 1.3; min-height: 36px; }
.mc-body { padding: 10px 14px 12px; }
.mr { display: flex; align-items: center; justify-content: space-between; margin-bottom: 5px; }
.mr-l { font-size: .65rem; color: #808080; text-transform: uppercase; letter-spacing: .7px; font-weight: 600; }
.mr-v { font-size: .85rem; font-weight: 700; color: #27ae60; font-family: 'Space Grotesk', sans-serif; }
.mb-bg { background: #2a2a2a; border-radius: 3px; height: 4px; width: 100%; margin-bottom: 10px; }
.mb-f { height: 4px; border-radius: 3px; background: #27ae60; }
.gr { display: flex; flex-wrap: wrap; gap: 4px; }
.gt { background: #252525; color: #999; font-size: .58rem; font-weight: 600; padding: 2px 7px; border-radius: 3px; text-transform: uppercase; letter-spacing: .3px; }
.watched-tag { display: inline-block; background: #252525; color: #666; font-size: .58rem; font-weight: 600; padding: 2px 7px; border-radius: 3px; margin-top: 6px; }

.wl-card { background: #1c1c1c; border-radius: 8px; padding: 14px 16px; border-left: 3px solid #c0392b; margin-bottom: 6px; }
.wl-title { font-family: 'Space Grotesk', sans-serif; font-size: .9rem; font-weight: 600; color: #e5e5e5; }
.wl-genres { font-size: .7rem; color: #666; margin-top: 3px; }

.divider { height: 1px; background: #252525; margin: 32px 0; }
.footer-text { text-align: center; color: #333; font-size: .68rem; padding: 32px 0 16px; }

/* ── Flip Cards (All Movies) ──────────── */
.flip-container { perspective: 800px; height: 140px; margin-bottom: 4px; }
.flip-inner { position: relative; width: 100%; height: 100%; transition: transform 0.5s cubic-bezier(.4,0,.2,1); transform-style: preserve-3d; }
.flip-container:hover .flip-inner { transform: rotateY(180deg); }
.flip-front, .flip-back { position: absolute; width: 100%; height: 100%; backface-visibility: hidden; border-radius: 8px; overflow: hidden; }
.flip-front { background: #1c1c1c; display: flex; flex-direction: column; justify-content: center; padding: 16px 14px; }
.flip-front-title { font-family: 'Space Grotesk', sans-serif; font-size: .82rem; font-weight: 600; color: #e5e5e5; line-height: 1.3; margin-bottom: 8px; }
.flip-front-genres { font-size: .6rem; color: #666; }
.flip-back { background: linear-gradient(135deg,#8b1a1a,#c0392b); transform: rotateY(180deg); display: flex; flex-direction: column; justify-content: center; padding: 16px 14px; }
.flip-back-label { font-size: .6rem; color: rgba(255,255,255,.6); text-transform: uppercase; letter-spacing: .8px; font-weight: 600; margin-bottom: 6px; }
.flip-back-title { font-family: 'Space Grotesk', sans-serif; font-size: .82rem; font-weight: 600; color: #fff; line-height: 1.3; margin-bottom: 4px; }
.flip-back-match { font-size: .72rem; color: rgba(255,255,255,.7); }

#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { display: none; }

.stButton > button {
    background: transparent !important; color: #999 !important; border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 4px !important; font-weight: 600 !important; font-size: .52rem !important;
    padding: 2px 4px !important; transition: all .2s ease !important; font-family: 'DM Sans', sans-serif !important;
    white-space: nowrap !important; overflow: hidden !important; text-overflow: ellipsis !important;
    min-height: 0 !important; line-height: 1.2 !important; text-transform: uppercase; letter-spacing: 0.5px;
}
.stButton > button:hover { background: rgba(255,255,255,0.08) !important; border-color: rgba(255,255,255,0.3) !important; color: #fff !important; }

div[data-testid="stTabs"] button { color: #999 !important; font-weight: 500 !important; font-family: 'DM Sans', sans-serif !important; }
div[data-testid="stTabs"] button[aria-selected="true"] { color: #fff !important; border-bottom-color: #c0392b !important; }

/* ── Clickable 3D Cards (Native Streamlit Overlay without Hacks) ──────────── */
/* Only target columns that contain our special browse card */
div[data-testid="column"]:has(.browse-card) {
    position: relative;
    cursor: pointer;
}

/* Make the button wrapper absolute and cover the column */
div[data-testid="column"]:has(.browse-card) > div[data-testid="element-container"]:nth-child(2) {
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    width: 100% !important;
    height: 100% !important;
    z-index: 10 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Force the button to be completely transparent but intercept pointer events */
div[data-testid="column"]:has(.browse-card) div[data-testid="stButton"] button {
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    width: 100% !important;
    height: 100% !important;
    opacity: 0 !important;
    cursor: pointer !important;
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
}

/* Magic: Connect hover from the invisible button to the 3D card flip! */
div[data-testid="column"]:has(.browse-card):has(div[data-testid="stButton"] button:hover) .flip-inner {
    transform: rotateY(180deg);
}
</style>
""", unsafe_allow_html=True)

# ─── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    similarity = pickle.load(open("similarity.pkl", "rb"))
    return movies, similarity

KNOWN_GENRES = ["action","adventure","animation","comedy","crime","documentary","drama","family","fantasy","history","horror","music","mystery","romance","sciencefiction","tvmovie","thriller","war","western"]
GENRE_DISPLAY = {"action":"Action","adventure":"Adventure","animation":"Animation","comedy":"Comedy","crime":"Crime","documentary":"Documentary","drama":"Drama","family":"Family","fantasy":"Fantasy","history":"History","horror":"Horror","music":"Music","mystery":"Mystery","romance":"Romance","sciencefiction":"Sci-Fi","tvmovie":"TV Movie","thriller":"Thriller","war":"War","western":"Western"}

def extract_genres(tags_str):
    if not isinstance(tags_str, str): return []
    words = tags_str.lower().split()
    return [GENRE_DISPLAY.get(g, g.title()) for g in KNOWN_GENRES if g in words]

def get_recommendations(movie, movies_df, sim, prefs, top_n=10, show_watched=False):
    try:
        idx = movies_df[movies_df["title"] == movie].index[0]
    except IndexError:
        return []
    scores = sim[idx].copy().astype(float)
    for liked in prefs.get("liked", []):
        try:
            li = movies_df[movies_df["title"] == liked].index[0]
            scores += sim[li] * 0.15
        except IndexError:
            pass
    for disliked in prefs.get("disliked", []):
        try:
            di = movies_df[movies_df["title"] == disliked].index[0]
            scores -= sim[di] * 0.10
        except IndexError:
            pass
    ranked = sorted(list(enumerate(scores)), reverse=True, key=lambda x: x[1])
    watched_set = set(prefs.get("watched", []))
    results = []
    for i, sc in ranked:
        title = movies_df.iloc[i]["title"]
        if title == movie:
            continue
        is_watched = title in watched_set
        if not show_watched and is_watched:
            continue
        conf = max(0, min(sc * 100, 100))
        results.append({
            "title": title,
            "confidence": round(conf, 1),
            "genres": extract_genres(str(movies_df.iloc[i].get("tags", ""))),
            "watched": is_watched,
        })
        if len(results) >= top_n:
            break
    return results


def render_card(rec, rank, key_prefix):
    """Render a single movie card with action buttons."""
    is_w = rec["watched"]
    card_cls = "mc dimmed" if is_w else "mc"
    hdr_cls = "mc-hdr dimmed-hdr" if is_w else "mc-hdr"
    genres_html = "".join(f'<span class="gt">{g}</span>' for g in rec["genres"][:3])
    conf = rec["confidence"]
    watched_html = '<span class="watched-tag">Watched</span>' if is_w else ''
    title = rec["title"]

    st.markdown(f"""
<div class="{card_cls}">
<div class="{hdr_cls}">
    <div class="mc-rank">{rank}</div>
    <div class="mc-title">{title}</div>
</div>
<div class="mc-body">
    <div class="mr"><span class="mr-l">Match</span><span class="mr-v">{conf:.0f}%</span></div>
    <div class="mb-bg"><div class="mb-f" style="width:{max(conf,3)}%"></div></div>
    <div class="gr">{genres_html}</div>
    {watched_html}
</div>
</div>""", unsafe_allow_html=True)

    prefs = st.session_state.prefs
    is_liked = title in prefs.get("liked", [])
    is_disliked = title in prefs.get("disliked", [])
    is_in_wl = title in prefs.get("watchlist", [])

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.button("Liked" if is_liked else "Like", key=f"{key_prefix}_L{rank}", on_click=do_like, args=(title,))
    with b2:
        st.button("Disliked" if is_disliked else "Dislike", key=f"{key_prefix}_D{rank}", on_click=do_dislike, args=(title,))
    with b3:
        st.button("Seen" if is_w else "Watched", key=f"{key_prefix}_W{rank}", on_click=do_watched, args=(title,))
    with b4:
        st.button("Saved" if is_in_wl else "+ List", key=f"{key_prefix}_A{rank}", on_click=do_watchlist, args=(title,))

# ─── Dialog for Recommendations ───────────────────────────────────────────────
if hasattr(st, "dialog"):
    st_dialog = st.dialog
else:
    st_dialog = st.experimental_dialog

@st_dialog("Movie Recommendations", width="large")
def show_movie_recommendations(movie_title):
    st.markdown(f'<div class="section-title" style="margin-top:0">Because you selected: {movie_title}</div>', unsafe_allow_html=True)
    recs = get_recommendations(movie_title, movies, similarity, st.session_state.prefs, top_n=12, show_watched=True)
    if not recs:
        st.info("No recommendations found.")
        return
    for row_start in range(0, len(recs), 3):
        cols = st.columns(3, gap="small")
        for j, col in enumerate(cols):
            ri = row_start + j
            if ri >= len(recs): break
            with col:
                render_card(recs[ri], ri + 1, f"mod_{movie_title}_{ri}")

movies, similarity = load_data()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_rec, tab_browse, tab_watchlist, tab_about, tab_map = st.tabs(["Recommend", "All Movies", "Watchlist", "How It Works", "Interactive Map"])

# ━━━  TAB 1 — Recommend  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_rec:
    st.markdown('<div class="hero"><h1>CineMatch</h1><p>Tell us a movie you love and we\'ll find 10 you\'ll enjoy just as much.</p></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        selected = st.selectbox("Search for a movie", movies["title"].values, index=0, key="rec_search")
        if st.button("Find Similar Movies", use_container_width=True, key="rec_go"):
            st.session_state.active_movie = selected

    active = st.session_state.active_movie
    if active:
        recs = get_recommendations(active, movies, similarity, st.session_state.prefs, top_n=12, show_watched=True)
        if recs:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="section-title">Because you liked {active}</div>', unsafe_allow_html=True)
            for row_start in range(0, len(recs), 3):
                cols = st.columns(3, gap="small")
                for j, col in enumerate(cols):
                    ri = row_start + j
                    if ri >= len(recs): break
                    with col:
                        render_card(recs[ri], ri + 1, f"rec{row_start}")
                if row_start == 0 and len(recs) > 3:
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        else:
            st.info("No new recommendations. Try toggling 'Show watched' or pick a different movie.")

    st.markdown('<div class="footer-text">CineMatch  /  Content-Based Recommendation Engine  /  Cosine Similarity</div>', unsafe_allow_html=True)


# ━━━  TAB 2 — All Movies  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_browse:
    st.markdown('<div class="hero" style="padding-bottom:14px"><h1>Browse All Movies</h1><p>Click "See Recs" to view recommendations for any movie.</p></div>', unsafe_allow_html=True)

    all_g = set()
    for _, row in movies.iterrows():
        all_g.update(extract_genres(str(row.get("tags", ""))))
    gf = st.selectbox("Filter by genre", ["All"] + sorted(all_g), key="gf")

    filtered = movies
    if gf != "All":
        mask = movies["tags"].apply(lambda t: gf.lower().replace("-","").replace(" ","") in str(t).lower())
        filtered = movies[mask]

    st.markdown(f'<div class="section-title">{len(filtered)} movies</div>', unsafe_allow_html=True)

    # Pre-compute top-1 recommendation for each movie in the subset
    COLS = 6
    subset = filtered.head(60).reset_index(drop=True)

    def get_top1_recommendations(sim, movie_titles, all_titles):
        results = {}
        title_to_idx = {t: i for i, t in enumerate(all_titles)}
        for title in movie_titles:
            if title not in title_to_idx:
                results[title] = ("--", 0)
                continue
            idx = title_to_idx[title]
            scores = sim[idx]
            ranked = sorted(list(enumerate(scores)), reverse=True, key=lambda x: x[1])
            for ri, sc in ranked:
                if ri != idx:
                    results[title] = (all_titles[ri], round(sc * 100, 1))
                    break
        return results

    top1_map = get_top1_recommendations(similarity, subset["title"].tolist(), movies["title"].tolist())

    for rs in range(0, len(subset), COLS):
        cols = st.columns(COLS, gap="small")
        for j, col in enumerate(cols):
            idx = rs + j
            if idx >= len(subset): break
            row = subset.iloc[idx]
            title = row["title"]
            genres = extract_genres(str(row.get("tags", "")))
            genre_str = " / ".join(genres[:3]) if genres else ""
            rec_title, rec_conf = top1_map.get(title, ("—", 0))
            with col:
                st.markdown(f"""
<div class="flip-container browse-card">
    <div class="flip-inner">
        <div class="flip-front">
            <div class="flip-front-title">{title}</div>
            <div class="flip-front-genres">{genre_str}</div>
        </div>
        <div class="flip-back">
            <div class="flip-back-label">Top Pick</div>
            <div class="flip-back-title">{rec_title}</div>
            <div class="flip-back-match">{rec_conf:.0f}% match</div>
        </div>
    </div>
</div>""", unsafe_allow_html=True)
                if st.button(" ", key=f"btn_see_{title}", use_container_width=True):
                    show_movie_recommendations(title)

    st.markdown('<div class="footer-text">CineMatch  /  Content-Based Recommendation Engine</div>', unsafe_allow_html=True)


# ━━━  TAB 3 — Watchlist  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_watchlist:
    st.markdown('<div class="hero" style="padding-bottom:14px"><h1>My Watchlist</h1><p>Movies saved for later.</p></div>', unsafe_allow_html=True)

    prefs = st.session_state.prefs
    wl = prefs.get("watchlist", [])

    if not wl:
        st.info("Your watchlist is empty. Use the '+ List' button on recommendations to add movies.")
    else:
        st.markdown(f'<div class="section-title">{len(wl)} movies saved</div>', unsafe_allow_html=True)
        for i, title in enumerate(wl):
            match = movies[movies["title"] == title]
            genres = extract_genres(str(match.iloc[0].get("tags", ""))) if len(match) > 0 else []
            genre_str = " / ".join(genres[:4]) if genres else ""
            is_w = title in prefs.get("watched", [])
            watched_mark = "  [Watched]" if is_w else ""

            c1, c2, c3 = st.columns([5, 1, 1])
            with c1:
                st.markdown(f'<div class="wl-card"><div class="wl-title">{title}{watched_mark}</div><div class="wl-genres">{genre_str}</div></div>', unsafe_allow_html=True)
            with c2:
                st.button("Mark unseen" if is_w else "Mark seen", key=f"wlw_{i}", on_click=do_watched, args=(title,))
            with c3:
                st.button("Remove", key=f"wlr_{i}", on_click=do_watchlist, args=(title,))

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1: st.metric("Liked", len(prefs.get("liked", [])))
    with s2: st.metric("Disliked", len(prefs.get("disliked", [])))
    with s3: st.metric("Watched", len(prefs.get("watched", [])))

    if st.button("Reset All Preferences", key="reset"):
        st.session_state.prefs = {"liked": [], "disliked": [], "watched": [], "watchlist": []}
        save_prefs()
        st.rerun()

    st.markdown('<div class="footer-text">CineMatch  /  Your Personal Movie Tracker</div>', unsafe_allow_html=True)


# ━━━  TAB 4 — How It Works  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_about:
    st.markdown('<div class="hero" style="padding-bottom:14px"><h1>How CineMatch Works</h1><p>A look under the hood of the recommendation engine.</p></div>', unsafe_allow_html=True)

    # ── Live Metrics ──
    st.markdown('<div class="section-title">Live Model Performance</div>', unsafe_allow_html=True)
    metrics_data = None
    if os.path.exists("metrics.pkl"):
        metrics_data = pickle.load(open("metrics.pkl", "rb"))
    if metrics_data:
        mc1, mc2, mc3, mc4 = st.columns(4)
        metric_info = [
            (mc1, "Accuracy", metrics_data["accuracy"]),
            (mc2, "Precision", metrics_data["precision"]),
            (mc3, "Recall", metrics_data["recall"]),
            (mc4, "F1 Score", metrics_data["f1_score"]),
        ]
        for col, label, val in metric_info:
            with col:
                st.markdown(f"""
<div style="background:#1c1c1c; border:1px solid #333; border-radius:10px; padding:20px; text-align:center">
    <div style="font-family:'Space Grotesk',sans-serif; font-size:1.8rem; font-weight:700; color:#c0392b">{val*100:.1f}%</div>
    <div style="font-size:.68rem; color:#666; text-transform:uppercase; letter-spacing:1px; margin-top:4px; font-weight:600">{label}</div>
</div>""", unsafe_allow_html=True)
    else:
        st.info("Run evaluate_model.py to generate metrics.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Snake Flowchart ──
    st.markdown('<div class="section-title">Recommendation Pipeline</div>', unsafe_allow_html=True)

    steps = [
        ("#c0392b", "1", "Data Collection", "TMDB dataset: 4800+ movies with title, overview, genres, keywords, credits"),
        ("#e67e22", "2", "Data Cleaning", "Drop nulls, remove duplicates, lowercase text, parse JSON genre/keyword columns"),
        ("#f39c12", "3", "Feature Engineering", "Merge overview + genres + keywords into a single tags column per movie"),
        ("#27ae60", "4", "Text Processing", "Porter Stemmer reduces words to roots: loved to love, running to run"),
        ("#2980b9", "5", "Vectorization", "CountVectorizer converts tags into 5000-dimensional word frequency vectors"),
        ("#8e44ad", "6", "Cosine Similarity", "Compute pairwise similarity between all 1493 movies (1493 x 1493 matrix)"),
        ("#c0392b", "7", "Recommendation", "Rank all movies by similarity to the query, return top 10 with confidence scores"),
        ("#27ae60", "8", "Dynamic Learning", "Likes boost similar movies +15%, dislikes reduce -10%, watched movies greyed out"),
    ]

    st.markdown("""
<style>
.snake { max-width: 700px; margin: 0 auto; }
.snake-row { display: flex; align-items: center; margin-bottom: 2px; }
.snake-row.right { flex-direction: row-reverse; }
.snake-step { flex: 1; display: flex; align-items: center; gap: 14px; background: #1c1c1c; border: 1px solid #333; border-radius: 10px; padding: 14px 18px; }
.snake-row.right .snake-step { flex-direction: row-reverse; text-align: right; }
.snake-num { width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-family: 'Space Grotesk', sans-serif; font-size: .85rem; font-weight: 700; color: #fff; flex-shrink: 0; }
.snake-text h4 { font-family: 'Space Grotesk', sans-serif; font-size: .85rem; font-weight: 600; color: #e5e5e5; margin: 0 0 3px; }
.snake-text p { font-size: .72rem; color: #808080; margin: 0; line-height: 1.4; }
.snake-connector { text-align: center; padding: 0; color: #333; font-size: .7rem; line-height: 1; }
.snake-connector.left-bend { text-align: left; padding-left: 40px; }
.snake-connector.right-bend { text-align: right; padding-right: 40px; }
</style>
""", unsafe_allow_html=True)

    html = '<div class="snake">'
    for i, (color, num, title, desc) in enumerate(steps):
        direction = "right" if i % 2 == 1 else ""
        html += f'''
<div class="snake-row {direction}">
    <div class="snake-step">
        <div class="snake-num" style="background:{color}">{num}</div>
        <div class="snake-text"><h4>{title}</h4><p>{desc}</p></div>
    </div>
</div>'''
        if i < len(steps) - 1:
            bend = "right-bend" if i % 2 == 0 else "left-bend"
            html += f'<div class="snake-connector {bend}">|</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Technical Details ──
    st.markdown('<div class="section-title">Technical Details</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
<div style="background:#1c1c1c; border-radius:10px; padding:20px; border:1px solid #333">
<h4 style="font-family:'Space Grotesk',sans-serif; color:#fff; margin:0 0 12px; font-size:.9rem">Cosine Similarity</h4>
<p style="color:#999; font-size:.78rem; line-height:1.6; margin:0">
Measures the cosine of the angle between two vectors. Movies with similar
tags point in similar directions, yielding a score close to 1.
</p>
<div style="background:#252525; border-radius:6px; padding:10px; margin-top:10px; font-family:monospace; font-size:.72rem; color:#ccc; text-align:center">
cos(A, B) = (A . B) / (||A|| x ||B||)
</div>
</div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div style="background:#1c1c1c; border-radius:10px; padding:20px; border:1px solid #333">
<h4 style="font-family:'Space Grotesk',sans-serif; color:#fff; margin:0 0 12px; font-size:.9rem">CountVectorizer</h4>
<p style="color:#999; font-size:.78rem; line-height:1.6; margin:0">
Converts text into a matrix of word counts. Each movie becomes a
5000-feature vector. Stop words (the, is, a) are removed automatically.
</p>
<div style="background:#252525; border-radius:6px; padding:10px; margin-top:10px; font-family:monospace; font-size:.72rem; color:#ccc; text-align:center">
[0, 1, 0, 3, 0, ...] (5000 dims)
</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
<div style="background:#1c1c1c; border-radius:10px; padding:20px; border:1px solid #333">
<h4 style="font-family:'Space Grotesk',sans-serif; color:#fff; margin:0 0 12px; font-size:.9rem">Evaluation Strategy</h4>
<p style="color:#999; font-size:.78rem; line-height:1.6; margin:0">
Genre overlap serves as ground truth. A recommendation is "relevant" if it
shares at least one genre with the query movie. Evaluated on 500 random samples.
</p>
</div>""", unsafe_allow_html=True)

    with col4:
        st.markdown("""
<div style="background:#1c1c1c; border-radius:10px; padding:20px; border:1px solid #333">
<h4 style="font-family:'Space Grotesk',sans-serif; color:#fff; margin:0 0 12px; font-size:.9rem">Dynamic Feedback</h4>
<p style="color:#999; font-size:.78rem; line-height:1.6; margin:0">
Likes add 15% of the liked movie's similarity vector to the query scores.
Dislikes subtract 10%. Preferences persist to JSON between sessions.
</p>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div class="divider"></div>
<div style="text-align:center; padding: 0 0 20px">
    <div style="font-family:'Space Grotesk',sans-serif; font-size:.95rem; font-weight:600; color:#fff; margin-bottom:8px">Tech Stack</div>
    <div style="display:flex; justify-content:center; gap:10px; flex-wrap:wrap">
        <span class="gt" style="font-size:.68rem; padding:4px 12px">Python</span>
        <span class="gt" style="font-size:.68rem; padding:4px 12px">Pandas</span>
        <span class="gt" style="font-size:.68rem; padding:4px 12px">scikit-learn</span>
        <span class="gt" style="font-size:.68rem; padding:4px 12px">NLTK</span>
        <span class="gt" style="font-size:.68rem; padding:4px 12px">NumPy</span>
        <span class="gt" style="font-size:.68rem; padding:4px 12px">Streamlit</span>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="footer-text">CineMatch  /  Data Science and Machine Learning Project</div>', unsafe_allow_html=True)

# ━━━  TAB 5 — Movie Map  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_map:
    st.markdown('<div class="hero" style="padding-bottom:14px"><h1>Movie Universe Map</h1><p>Explore all 1490+ movies clustered by content similarity.</p></div>', unsafe_allow_html=True)

    @st.cache_data
    def get_movie_coordinates(_sim, _movies_df):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(_sim)
        
        primary_genres = []
        for tags in _movies_df["tags"]:
            genres = extract_genres(str(tags))
            primary_genres.append(genres[0].title() if genres else "Unknown")
            
        df_map = pd.DataFrame({
            "x": coords[:, 0],
            "y": coords[:, 1],
            "Title": _movies_df["title"],
            "Genre": primary_genres
        })
        return df_map

    with st.spinner("Generating 2D projection using PCA..."):
        df_map = get_movie_coordinates(similarity, movies)
        
        import altair as alt
        
        # We use a dark theme suitable configuration
        chart = alt.Chart(df_map).mark_circle(size=70, opacity=0.8).encode(
            x=alt.X('x', axis=None),
            y=alt.Y('y', axis=None),
            color=alt.Color('Genre', scale=alt.Scale(scheme='tableau20'), legend=alt.Legend(title="Primary Genre")),
            tooltip=['Title', 'Genre']
        ).properties(
            height=650
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
        
    st.markdown("""
<div style="background:#1c1c1c; border-radius:10px; padding:20px; border:1px solid #333; margin-top:20px">
<h4 style="font-family:'Space Grotesk',sans-serif; color:#fff; margin:0 0 12px; font-size:.95rem">How this map is generated</h4>
<p style="color:#999; font-size:.8rem; line-height:1.6; margin:0">
We use <b>Principal Component Analysis (PCA)</b> to reduce the 1493-dimensional cosine similarity matrix down to 2 dimensions. 
Movies plotted closer together share more similar content (genres, keywords, plot themes). 
You can pan and zoom to explore clusters of similar movies!
</p>
</div>
""", unsafe_allow_html=True)