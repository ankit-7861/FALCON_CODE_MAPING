from __future__ import annotations

import io
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import streamlit as st
except Exception:  # pragma: no cover
    class _DummyStreamlit:
        def __getattr__(self, name):
            def _noop(*args, **kwargs):
                return None
            return _noop

        @staticmethod
        def cache_resource(show_spinner=False):
            def decorator(func):
                return func
            return decorator

        @staticmethod
        def cache_data(show_spinner=False):
            def decorator(func):
                return func
            return decorator

    st = _DummyStreamlit()


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MAPPING_FILE = DATA_DIR / "Extracted VIOLATION CODE Mapping to FALCON CODES.xlsx"
DEFAULT_STANDARD_FILE = DATA_DIR / "Target Fields - Standard Falcon Driver-Violations Code.xlsx"
DEFAULT_FEEDBACK_DB = DATA_DIR / "falcon_feedback.db"


ABBREVIATIONS = {
    "w/o": "without",
    "w/": "with",
    "dl": "driver license",
    "lic": "license",
    "veh": "vehicle",
    "mv": "motor vehicle",
    "opr": "operating",
    "improp": "improper",
    "obstruct/impede": "obstruct impede",
    "accdt": "accident",
    "dui": "driving under the influence",
    "dwi": "driving while intoxicated",
    "susp": "suspended",
    "rev": "revoked",
    "spd": "speed",
    "headlamps": "head lamps",
    "headlamp": "head lamp",
}

SYNONYMS = {
    "overspeed": "speeding",
    "unsafe lane changes": "improper lane change",
    "erratic lane changes": "improper lane change",
    "stop stand park obstruct impede traffic": "obstruct traffic",
    "operating without equipment as required by law": "without required equipment",
    "alcohol": "driving under the influence",
    "intoxicated": "driving under the influence",
    "no head lamps": "defective no head lamps",
    "defective headlights": "defective no head lamps",
}


@dataclass
class MappingAssets:
    known_map: Dict[str, Dict[str, str]]
    standards_df: pd.DataFrame
    standard_texts: List[str]
    encoder_name: str
    encoder: object
    standard_embeddings: object


def normalize_text(text: object) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text).lower().strip()
    s = s.replace("&", " and ")
    for k, v in ABBREVIATIONS.items():
        s = s.replace(k, v)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for k, v in SYNONYMS.items():
        if k in s:
            s = s.replace(k, v)
    return s


def normalize_col(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(col).lower()).strip()


def find_column(columns: List[str], candidates: List[str], required: bool = True) -> Optional[str]:
    normalized = {col: normalize_col(col) for col in columns}
    for candidate in candidates:
        cand = normalize_col(candidate)
        for col, norm in normalized.items():
            if cand == norm or cand in norm:
                return col
    if required:
        raise KeyError(f"Could not find column matching any of: {candidates}. Available: {columns}")
    return None


def build_record_key(violation: object, description: object, svccode: object) -> str:
    return f"{str(violation).strip()}||{str(description).strip()}||{str(svccode).strip()}"


def format_label(svccode: object, description: object, violation: object = None) -> str:
    code = str(svccode).strip() if svccode is not None else ""
    desc = str(description).strip() if description is not None else ""
    viol = str(violation).strip() if violation is not None else ""
    primary = code if code else viol
    return f"{primary} - {desc}" if primary else desc


@st.cache_resource(show_spinner=False)
def build_encoder_and_embeddings(standard_texts: Tuple[str, ...]):
    try:
        from sentence_transformers import SentenceTransformer

        model_name = "all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)
        emb = model.encode(list(standard_texts), normalize_embeddings=True)
        return ("sentence-transformers/" + model_name, model, np.asarray(emb, dtype="float32"))
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform(list(standard_texts))
        matrix = normalize(matrix)
        return ("tfidf-fallback", vectorizer, matrix)


@st.cache_data(show_spinner=False)
def load_assets(mapping_bytes: bytes, standard_bytes: bytes) -> MappingAssets:
    mapping_df = pd.read_excel(io.BytesIO(mapping_bytes), sheet_name="Violation Description")
    standards_df = pd.read_excel(io.BytesIO(standard_bytes))

    map_desc_col = find_column(
        list(mapping_df.columns),
        [
            "LLM Violations Convictions Failure To Appear Or Accidents Description",
            "description",
            "violation description",
        ],
    )
    falcon_col = find_column(
        list(mapping_df.columns),
        [
            "FALCON VIOLATIONS",
            "falcon violation",
            "falcon code",
        ],
    )

    std_violation_col = find_column(list(standards_df.columns), ["violation"])
    std_desc_col = find_column(list(standards_df.columns), ["description"])
    std_svccode_col = find_column(list(standards_df.columns), ["SVCCODE", "svc code"], required=False)
    std_points_col = find_column(
        list(standards_df.columns),
        ["points", "trucking_1_7_2026_points"],
        required=False,
    )

    mapping_df = mapping_df[[map_desc_col, falcon_col]].copy()
    mapping_df = mapping_df.dropna(subset=[map_desc_col, falcon_col])
    mapping_df["normalized_description"] = mapping_df[map_desc_col].map(normalize_text)
    mapping_df["falcon_violation"] = mapping_df[falcon_col].astype(str).str.strip()
    mapping_df = mapping_df[mapping_df["normalized_description"] != ""]
    mapping_df = mapping_df.drop_duplicates(subset=["normalized_description"], keep="first")

    known_map = {
        row["normalized_description"]: {
            "falcon_violation": row["falcon_violation"],
            "source_method": "existing mapping file",
        }
        for _, row in mapping_df.iterrows()
    }

    keep_cols = [c for c in [std_violation_col, std_desc_col, std_svccode_col, std_points_col] if c is not None]
    standards_df = standards_df[keep_cols].copy()

    rename_map = {
        std_violation_col: "violation",
        std_desc_col: "description",
    }
    if std_svccode_col is not None:
        rename_map[std_svccode_col] = "SVCCODE"
    if std_points_col is not None:
        rename_map[std_points_col] = "points"

    standards_df = standards_df.rename(columns=rename_map)

    if "SVCCODE" not in standards_df.columns:
        standards_df["SVCCODE"] = None
    if "points" not in standards_df.columns:
        standards_df["points"] = None

    standards_df = standards_df.dropna(subset=["violation", "description"]).copy()
    standards_df["violation"] = standards_df["violation"].astype(str).str.strip()
    standards_df["description"] = standards_df["description"].astype(str).str.strip()
    standards_df["SVCCODE"] = standards_df["SVCCODE"].fillna("").astype(str).str.strip()

    standards_df["search_text"] = (
        standards_df["violation"].fillna("")
        + " | "
        + standards_df["description"].fillna("")
        + " | "
        + standards_df["SVCCODE"].fillna("").astype(str)
    ).map(normalize_text)

    standards_df["record_key"] = standards_df.apply(
        lambda r: build_record_key(r["violation"], r["description"], r["SVCCODE"]),
        axis=1,
    )
    standards_df["display_label"] = standards_df.apply(
        lambda r: format_label(r["SVCCODE"], r["description"], r["violation"]),
        axis=1,
    )
    standards_df = standards_df.drop_duplicates(subset=["record_key"]).reset_index(drop=True)

    encoder_name, encoder, standard_embeddings = build_encoder_and_embeddings(
        tuple(standards_df["search_text"].tolist())
    )

    return MappingAssets(
        known_map=known_map,
        standards_df=standards_df,
        standard_texts=standards_df["search_text"].tolist(),
        encoder_name=encoder_name,
        encoder=encoder,
        standard_embeddings=standard_embeddings,
    )


def get_connection(db_path: Path = DEFAULT_FEEDBACK_DB):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback_aliases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            normalized_input TEXT NOT NULL,
            raw_input TEXT,
            record_key TEXT NOT NULL,
            feedback_type TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(normalized_input, record_key, feedback_type)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            normalized_input TEXT NOT NULL,
            raw_input TEXT,
            candidate_record_key TEXT,
            feedback_type TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()
    return conn


def get_learned_exact_match(normalized_input: str, assets: MappingAssets, conn) -> Optional[pd.Series]:
    sql = """
        SELECT fa.record_key, COUNT(*) AS cnt
        FROM feedback_aliases fa
        WHERE fa.normalized_input = ? AND fa.feedback_type = 'accepted'
        GROUP BY fa.record_key
        ORDER BY cnt DESC, fa.record_key ASC
        LIMIT 1
    """
    row = conn.execute(sql, (normalized_input,)).fetchone()
    if not row:
        return None

    match = assets.standards_df[assets.standards_df["record_key"] == row[0]]
    if match.empty:
        return None
    return match.iloc[0]


def encode_query(query: str, assets: MappingAssets):
    if assets.encoder_name.startswith("sentence-transformers/"):
        vec = assets.encoder.encode([query], normalize_embeddings=True)
        return np.asarray(vec, dtype="float32"), assets.standard_embeddings

    query_vec = assets.encoder.transform([query])
    return query_vec, assets.standard_embeddings


def get_feedback_scores(conn, normalized_input: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    accepted_rows = conn.execute(
        """
        SELECT record_key, COUNT(*) AS cnt
        FROM feedback_aliases
        WHERE feedback_type = 'accepted'
        GROUP BY record_key
        """
    ).fetchall()

    rejected_rows = conn.execute(
        """
        SELECT candidate_record_key, COUNT(*) AS cnt
        FROM feedback_events
        WHERE feedback_type = 'rejected'
          AND normalized_input = ?
          AND candidate_record_key IS NOT NULL
        GROUP BY candidate_record_key
        """,
        (normalized_input,),
    ).fetchall()

    accepted = {record_key: cnt for record_key, cnt in accepted_rows}
    rejected = {record_key: cnt for record_key, cnt in rejected_rows}
    return accepted, rejected


def get_top_candidates(
    description: str,
    assets: MappingAssets,
    conn,
    top_k: int = 2,
    cosine_weight: float = 0.95,
    feedback_weight: float = 0.05,
) -> pd.DataFrame:
    query = normalize_text(description)
    if not query:
        return pd.DataFrame(columns=["violation", "description", "SVCCODE", "points"])

    query_vec, std_matrix = encode_query(query, assets)

    if assets.encoder_name.startswith("sentence-transformers/"):
        cosine_scores = (query_vec @ std_matrix.T).flatten()
    else:
        cosine_scores = (query_vec @ std_matrix.T).toarray().flatten()

    scored = assets.standards_df[
        ["violation", "description", "SVCCODE", "points", "record_key", "display_label"]
    ].copy()

    scored["cosine_score"] = cosine_scores

    accepted_boosts, rejected_penalties = get_feedback_scores(conn, query)

    max_accept = max(accepted_boosts.values()) if accepted_boosts else 0
    max_reject = max(rejected_penalties.values()) if rejected_penalties else 0

    scored["feedback_score"] = scored["record_key"].map(
        lambda k: (accepted_boosts.get(k, 0) / max_accept) if max_accept > 0 else 0.0
    )
    scored["rejection_penalty"] = scored["record_key"].map(
        lambda k: (rejected_penalties.get(k, 0) / max_reject) if max_reject > 0 else 0.0
    )

    scored["final_score"] = (
        cosine_weight * scored["cosine_score"]
        + feedback_weight * scored["feedback_score"]
        - 0.10 * scored["rejection_penalty"]
    )

    scored = scored.sort_values(
        ["final_score", "cosine_score"],
        ascending=False,
    )

    return scored.head(top_k).reset_index(drop=True)


def map_description(description: str, assets: MappingAssets, conn, threshold: float, top_k: int = 2) -> Dict[str, object]:
    normalized = normalize_text(description)

    if not normalized:
        return {
            "input_description": description,
            "normalized_description": normalized,
            "status": "manual review",
            "method": "empty input",
            "confidence": 0.0,
            "candidate_table": pd.DataFrame(),
        }

    learned_match = get_learned_exact_match(normalized, assets, conn)
    if learned_match is not None:
        candidate_table = pd.DataFrame(
            [
                {
                    **learned_match.to_dict(),
                    "cosine_score": 1.0,
                    "feedback_score": 1.0,
                    "rejection_penalty": 0.0,
                    "final_score": 1.0,
                }
            ]
        )
        return {
            "input_description": description,
            "normalized_description": normalized,
            "status": "mapped",
            "method": "learned from user feedback",
            "confidence": 1.0,
            "candidate_table": candidate_table,
            "best_label": learned_match["display_label"],
        }

    if normalized in assets.known_map:
        known_text = assets.known_map[normalized]["falcon_violation"]
        exact_rows = assets.standards_df[
            assets.standards_df["violation"].astype(str).str.upper().eq(str(known_text).upper())
            | assets.standards_df["description"].astype(str).str.upper().eq(str(known_text).upper())
            | assets.standards_df["display_label"].astype(str).str.upper().eq(str(known_text).upper())
        ].copy()

        if not exact_rows.empty:
            exact_rows["cosine_score"] = 1.0
            exact_rows["feedback_score"] = 0.0
            exact_rows["rejection_penalty"] = 0.0
            exact_rows["final_score"] = 1.0
            exact_rows = exact_rows.head(max(1, top_k))

            return {
                "input_description": description,
                "normalized_description": normalized,
                "status": "mapped",
                "method": "existing mapping file",
                "confidence": 1.0,
                "candidate_table": exact_rows,
                "best_label": exact_rows.iloc[0]["display_label"],
            }

    candidates = get_top_candidates(description, assets, conn, top_k=top_k)

    if candidates.empty:
        return {
            "input_description": description,
            "normalized_description": normalized,
            "status": "manual review",
            "method": "no candidates",
            "confidence": 0.0,
            "candidate_table": candidates,
        }

    best = candidates.iloc[0]
    confidence = float(best["final_score"])
    status = "mapped" if confidence >= threshold else "manual review"

    return {
        "input_description": description,
        "normalized_description": normalized,
        "status": status,
        "method": "cosine similarity + feedback",
        "confidence": confidence,
        "candidate_table": candidates,
        "best_label": best["display_label"],
    }


def save_feedback(conn, raw_input: str, normalized_input: str, record_key: Optional[str], feedback_type: str):
    conn.execute(
        """
        INSERT INTO feedback_events (normalized_input, raw_input, candidate_record_key, feedback_type)
        VALUES (?, ?, ?, ?)
        """,
        (normalized_input, raw_input, record_key, feedback_type),
    )

    if feedback_type == "accepted" and record_key:
        conn.execute(
            """
            INSERT OR IGNORE INTO feedback_aliases (normalized_input, raw_input, record_key, feedback_type)
            VALUES (?, ?, ?, ?)
            """,
            (normalized_input, raw_input, record_key, feedback_type),
        )

    conn.commit()


def feedback_summary(conn) -> Tuple[int, int]:
    accepted = conn.execute(
        "SELECT COUNT(*) FROM feedback_aliases WHERE feedback_type = 'accepted'"
    ).fetchone()[0]
    events = conn.execute("SELECT COUNT(*) FROM feedback_events").fetchone()[0]
    return int(accepted), int(events)


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
    buf.seek(0)
    return buf.read()


def inject_ui_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Source+Serif+4:wght@600;700&display=swap');

        html, body, [class*="st-"] {
            font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
            font-size: 15px;
        }

        h1, h2, h3 {
            font-family: 'Source Serif 4', Georgia, serif;
            letter-spacing: -0.02em;
        }

        h1 {
            font-weight: 700;
            font-size: 2.0rem;
        }
        h2 {
            font-size: 1.4rem;
        }
        h3 {
            font-size: 1.1rem;
        }

        div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {
            font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
            font-size: 1.0rem;
        }

        div[data-testid="stTextArea"] textarea,
        div[data-testid="stSelectbox"] label,
        div[data-testid="stFileUploader"] label,
        .stButton > button {
            font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
            font-size: 1.0rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def read_input_descriptions(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    desc_col = find_column(list(df.columns), ["description", "violation description", "mvr description"])
    out = df.copy()
    out = out.rename(columns={desc_col: "input_description"})
    return out


def render_candidate_card(row: pd.Series, rank: int, input_text: str, normalized_text: str, conn):
    toggle_state_key = f"show_match_{rank}_{normalized_text}_{row['record_key']}"
    toggle_button_key = f"toggle_match_{rank}_{normalized_text}_{row['record_key']}"

    if toggle_state_key not in st.session_state:
        st.session_state[toggle_state_key] = True

    title_col, toggle_col = st.columns([5, 1])
    title_col.markdown(f"### Match {rank}: {row['display_label']}")

    toggle_label = "Hide" if st.session_state[toggle_state_key] else "Unhide"
    if toggle_col.button(toggle_label, key=toggle_button_key):
        st.session_state[toggle_state_key] = not st.session_state[toggle_state_key]

    if not st.session_state[toggle_state_key]:
        st.caption("Details hidden for this match.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Final", f"{row['final_score']:.3f}")
    c2.metric("Cosine", f"{row['cosine_score']:.3f}")
    c3.metric("Feedback", f"{row['feedback_score']:.3f}")

    st.write(
        {
            "violation": row["violation"],
            "description": row["description"],
            "SVCCODE": row["SVCCODE"],
            "points": row["points"],
        }
    )

    b1, b2 = st.columns(2)
    accept_key = f"accept_{rank}_{normalized_text}_{row['record_key']}"
    reject_key = f"reject_{rank}_{normalized_text}_{row['record_key']}"

    if b1.button("Correct", key=accept_key, type="primary"):
        save_feedback(conn, input_text, normalized_text, row["record_key"], "accepted")
        st.success(f"Saved feedback. Learned mapping: {row['display_label']}")

    if b2.button("Incorrect", key=reject_key):
        save_feedback(conn, input_text, normalized_text, row["record_key"], "rejected")
        st.info("Saved negative feedback for this candidate.")


def main():
    st.set_page_config(page_title="Falcon Violation Learning Mapper", layout="wide")
    inject_ui_styles()
    st.title("Falcon Violation Learning Mapper")
    st.caption("Known mapping first, then cosine similarity fallback, with feedback learning.")

    conn = get_connection()
    has_local_mapping = DEFAULT_MAPPING_FILE.exists()
    has_local_standard = DEFAULT_STANDARD_FILE.exists()

    with st.sidebar:
        st.header("Configuration")
        threshold = st.slider("High-confidence auto-map threshold", min_value=0.20, max_value=0.99, value=0.85, step=0.01)
        if has_local_mapping:
            st.caption(f"Using local mapping file: {DEFAULT_MAPPING_FILE.name}")
            mapping_upload = None
        else:
            mapping_upload = st.file_uploader("Existing mapping file (.xlsx)", type=["xlsx"], key="mapping")

        if has_local_standard:
            st.caption(f"Using local standard file: {DEFAULT_STANDARD_FILE.name}")
            standard_upload = None
        else:
            standard_upload = st.file_uploader("Falcon standard file (.xlsx)", type=["xlsx"], key="standard")

        st.caption(f"Feedback DB: {DEFAULT_FEEDBACK_DB}")
        accepted_count, event_count = feedback_summary(conn)
        st.metric("Learned aliases", accepted_count)
        st.metric("Feedback events", event_count)

    mapping_bytes = None
    standard_bytes = None

    if mapping_upload is not None:
        mapping_bytes = mapping_upload.read()
    elif DEFAULT_MAPPING_FILE.exists():
        mapping_bytes = DEFAULT_MAPPING_FILE.read_bytes()

    if standard_upload is not None:
        standard_bytes = standard_upload.read()
    elif DEFAULT_STANDARD_FILE.exists():
        standard_bytes = DEFAULT_STANDARD_FILE.read_bytes()

    if not mapping_bytes or not standard_bytes:
        missing_files = []
        if not mapping_bytes:
            missing_files.append(DEFAULT_MAPPING_FILE.name)
        if not standard_bytes:
            missing_files.append(DEFAULT_STANDARD_FILE.name)

        st.warning(
            "Upload the missing Excel file(s), or place them in the local data folder with these names:\n\n"
            + "\n".join(f"- {name}" for name in missing_files)
        )
        st.stop()

    assets = load_assets(mapping_bytes, standard_bytes)

    st.success(
        f"Known mappings: {len(assets.known_map):,} | "
        f"Falcon standards: {len(assets.standards_df):,} | "
        f"Model: {assets.encoder_name}"
    )

    tab1, tab2, tab3 = st.tabs(["Single search", "Batch search", "Learning view"])

    with tab1:
        sample = "DEFECTIVE/NO HEADLAMPS"
        input_text = st.text_area("Search new violation description", value=sample, height=100)

        if st.button("Find top 2 matches", type="primary"):
            result = map_description(input_text, assets, conn, threshold=threshold, top_k=2)

            c1, c2, c3 = st.columns(3)
            c1.metric("Status", result["status"])
            c2.metric("Method", result["method"])
            c3.metric("Top confidence", f"{result['confidence']:.3f}")

            st.markdown(
                """
                <div style='font-size:0.98rem; margin-bottom:0.5em; color:#444;'>
                <b>Score meanings:</b><br>
                <b>Final</b>: Weighted score combining Cosine similarity and Feedback.<br>
                <b>Cosine</b>: Similarity between your input and Falcon standard text.<br>
                <b>Feedback</b>: Boost from prior user acceptance of this match.
                </div>
                """,
                unsafe_allow_html=True,
            )

            if result.get("best_label"):
                st.subheader("Best formatted output")
                st.code(result["best_label"])

            candidates = result["candidate_table"]
            if candidates.empty:
                st.warning("No candidates found.")
            else:
                st.subheader("Top 2 matched Falcon fields")
                for idx, row in candidates.head(2).iterrows():
                    render_candidate_card(row, idx + 1, input_text, result["normalized_description"], conn)

                st.subheader("Manual correction")
                options_df = assets.standards_df[["display_label", "record_key"]].copy()
                selected_label = st.selectbox(
                    "Choose another Falcon standard if none of the top 2 is correct",
                    options_df["display_label"].tolist(),
                )

                if st.button("Save manual correction"):
                    selected_row = options_df[options_df["display_label"] == selected_label].iloc[0]
                    save_feedback(conn, input_text, result["normalized_description"], selected_row["record_key"], "accepted")
                    st.success(f"Manual correction saved: {selected_label}")

                if st.button("None of these are close"):
                    save_feedback(conn, input_text, result["normalized_description"], None, "manual_review")
                    st.info("Marked for manual review.")

    with tab2:
        batch_file = st.file_uploader(
            "Upload CSV/XLSX with a description column",
            type=["csv", "xlsx", "xls"],
            key="batch",
        )

        if batch_file is not None:
            batch_df = read_input_descriptions(batch_file)
            st.dataframe(batch_df.head(10), use_container_width=True)

            if st.button("Run batch top-2 search", type="primary"):
                rows = []

                for desc in batch_df["input_description"].fillna(""):
                    result = map_description(desc, assets, conn, threshold=threshold, top_k=2)
                    candidates = result["candidate_table"]

                    row = {
                        "input_description": desc,
                        "normalized_description": result["normalized_description"],
                        "status": result["status"],
                        "method": result["method"],
                        "confidence": result["confidence"],
                    }

                    for i in range(2):
                        if i < len(candidates):
                            cand = candidates.iloc[i]
                            row[f"top{i+1}_output"] = cand["display_label"]
                            row[f"top{i+1}_cosine"] = cand["cosine_score"]
                            row[f"top{i+1}_feedback"] = cand["feedback_score"]
                            row[f"top{i+1}_final"] = cand["final_score"]
                            row[f"top{i+1}_record_key"] = cand["record_key"]
                        else:
                            row[f"top{i+1}_output"] = None
                            row[f"top{i+1}_cosine"] = None
                            row[f"top{i+1}_feedback"] = None
                            row[f"top{i+1}_final"] = None
                            row[f"top{i+1}_record_key"] = None

                    rows.append(row)

                result_df = pd.DataFrame(rows)
                st.dataframe(result_df, use_container_width=True)

                st.download_button(
                    "Download results as CSV",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="falcon_top2_results.csv",
                    mime="text/csv",
                )
                st.download_button(
                    "Download results as Excel",
                    data=to_excel_bytes(result_df),
                    file_name="falcon_top2_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    with tab3:
        st.subheader("What improves over time")
        st.markdown(
            """
            - First, the system checks the known mapping file.
            - If not found there, it uses cosine similarity on Falcon standard records.
            - Accepted user feedback becomes learned aliases.
            - Rejected candidates get penalty for the same input next time.
            """
        )

        accepted = pd.read_sql_query(
            "SELECT * FROM feedback_aliases ORDER BY created_at DESC LIMIT 100",
            conn,
        )
        events = pd.read_sql_query(
            "SELECT * FROM feedback_events ORDER BY created_at DESC LIMIT 100",
            conn,
        )

        st.write("Recent accepted aliases")
        st.dataframe(accepted, use_container_width=True)

        st.write("Recent feedback events")
        st.dataframe(events, use_container_width=True)

    with st.expander("Expected output example"):
        st.code("61190 - DEFECTIVE/NO HEADLAMPS")
        st.markdown("<div style='margin-top:0.5em'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            The app formats each candidate as <code>code - description</code>.<br>
            If <b>SVCCODE</b> is present it is used as the code.<br>
            Otherwise the <b>violation</b> field is used.
            """,
            unsafe_allow_html=True
        )

    conn.close()


if __name__ == "__main__":
    main()