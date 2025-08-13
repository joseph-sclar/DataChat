"""
DataChat — explore any CSV with types, quick EDA, and a chat that answers with tables and plots.
Built by Joseph Sclar · joseph.sclar2@gmail.com
"""

import os
import io
import re
import json
import traceback
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------- state ----------
if "chat" not in st.session_state:
    st.session_state.chat = []
if "df" not in st.session_state:
    st.session_state.df = None
if "schema" not in st.session_state:
    st.session_state.schema = {}
if "last_result" not in st.session_state:
    st.session_state.last_result = None

APP_TITLE = "DataChat"
SYSTEM_RULES = (
    "You answer by emitting ONLY Python code. A pandas DataFrame named df is provided.\n"
    "Use only pandas, numpy, matplotlib (pd, np, plt). No I/O or other imports.\n"
    "ALWAYS create a pandas DataFrame named `answer_df` that directly answers the question "
    "(even if it's one value -> one-row DataFrame with clear column names).\n"
    "Optionally set `answer_text` (str) for a short explanation.\n"
    "If plotting helps, create a matplotlib figure (use plt.figure(); fig = plt.gcf()). "
    "Prefer calling the helper `_nice_bar(answer_df, x=<col>, y=<col>, title=..., top_n=..., fmt=...)` "
    "for bar charts with long labels.\n"
    "Do not print Streamlit code or any other UI code."
)

# ---------- OpenAI client ----------
try:
    from openai import OpenAI
    _openai_ok = True
except Exception:
    OpenAI = None
    _openai_ok = False

def get_openai_client(api_key: str):
    if not _openai_ok:
        raise RuntimeError("Install 'openai' package")
    return OpenAI(api_key=api_key)

# ---------- helpers ----------

def build_schema_prompt(df: pd.DataFrame) -> str:
    # small, JSON-serializable preview
    head = df.head(8).copy()

    def _ser(x):
        if hasattr(x, "isoformat"):
            try:
                return x.isoformat()
            except Exception:
                return str(x)
        if hasattr(x, "item"):
            try:
                return x.item()
            except Exception:
                return str(x)
        return x

    head_serializable = head.applymap(_ser)
    head_json = head_serializable.to_dict(orient="list")
    schema_lines = [f"- {c}: {str(df[c].dtype)}" for c in df.columns]
    return (
        "Columns and dtypes:\n" + "\n".join(schema_lines) +
        "\n\nSample (first rows, truncated):\n" + json.dumps(head_json)[:4000]
    )


def ask_model_for_code(api_key: str, model: str, df: pd.DataFrame, question: str) -> str:
    client = get_openai_client(api_key)
    messages = [
        {"role": "system", "content": SYSTEM_RULES},
        {"role": "user", "content": f"Here is the dataset schema and sample.\n{build_schema_prompt(df)}"},
        {"role": "user", "content": f"Question: {question}\nReturn ONLY executable Python code, no markdown fences."},
    ]
    out = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=1200,
    )
    return out.choices[0].message.content.strip()


CODE_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL | re.IGNORECASE)

def extract_code(text: str) -> str:
    m = CODE_BLOCK_RE.search(text)
    return m.group(1).strip() if m else text.strip()

FORBIDDEN_PATTERNS = [
    r"__import__",
    r"import\s+(?!numpy|matplotlib\.pyplot)",
    r"open\(",
    r"exec\(",
    r"eval\(",
    r"subprocess",
    r"os\.system",
    r"sys\.modules",
    r"pickle",
    r"pathlib",
    r"requests",
    r"urllib",
]

def run_user_code(code: str, df: "pd.DataFrame"):
    import pandas as pd
    import matplotlib.pyplot as plt
    import io, contextlib, numpy as np, textwrap

    # namespace the user code runs in
    ns = {"pd": pd, "plt": plt, "np": np, "df": df.copy() if df is not None else None}

    # --- chart helper available to generated code ---
    def _nice_bar(data, x, y, title=None, top_n=None, fmt="{:,.0f}"):
        import matplotlib.pyplot as _plt
        import numpy as _np
        import textwrap as _tw

        def _fmt_val(v):
            # Accept ":,.0f", "{:,.0f}", or even bad inputs like "{x:,.0f}"
            try:
                if fmt is None:
                    return str(v)
                if "{" in fmt:
                    # if named field like "{x:,.0f}", strip the name to positional
                    _clean = _tw.re.sub(r"\{[^:}]*:", "{:", fmt)
                    return _clean.format(v)
                else:
                    # plain format spec like ":,.0f"
                    return format(v, fmt)
            except Exception:
                # last-ditch fallback
                try:
                    return "{:,.0f}".format(v)
                except Exception:
                    return str(v)

        dfp = data.copy()
        if top_n:
            dfp = dfp.sort_values(y, ascending=False).head(top_n)

        # wrap long labels
        def _wrap(s):
            return "\n".join(_tw.wrap(str(s), width=14)) if isinstance(s, str) else s
        dfp[x] = dfp[x].map(_wrap)

        horizontal = len(dfp) > 10
        _plt.figure(figsize=(11, 6) if not horizontal else (10, 8))

        if horizontal:
            _plt.barh(dfp[x], dfp[y])
            _plt.xlabel(y); _plt.ylabel(x)
        else:
            _plt.bar(dfp[x], dfp[y])
            _plt.ylabel(y); _plt.xlabel(x)
            _plt.xticks(rotation=30, ha="right")

        # value labels
        ax = _plt.gca()
        for b in ax.patches:
            val = b.get_width() if horizontal else b.get_height()
            if val is None or (isinstance(val, float) and _np.isnan(val)):
                continue
            txt = _fmt_val(val)
            if horizontal:
                ax.text(val, b.get_y() + b.get_height()/2, " " + txt, va="center")
            else:
                ax.text(b.get_x() + b.get_width()/2, val, txt, ha="center", va="bottom")

        if title:
            _plt.title(title)
        _plt.tight_layout()
        return _plt.gcf()


    # expose helper to the exec namespace
    ns["_nice_bar"] = _nice_bar

    DF_CANDIDATE_NAMES = ["answer_df","result_df","output_df","df_out","table","result","output"]

    figure = None
    table = None
    text = ""
    stdout = ""

    # run user code with stdout capture
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, ns, ns)
    finally:
        stdout = buf.getvalue()

    # capture a figure if any
    if plt.get_fignums():
        figure = plt.gcf()

    # choose a table (prefer answer_df, else biggest DF in namespace, else coerce)
    for name in DF_CANDIDATE_NAMES:
        if isinstance(ns.get(name), pd.DataFrame):
            table = ns[name]
            break
    if table is None:
        dfs = [v for v in ns.values() if isinstance(v, pd.DataFrame)]
        if dfs:
            table = max(dfs, key=lambda d: (len(d), d.shape[1]))
    if table is None:
        for name in DF_CANDIDATE_NAMES:
            obj = ns.get(name)
            if isinstance(obj, (pd.Series, list, tuple)):
                table = pd.DataFrame(obj)
                break

    text = ns.get("answer_text", "")

    return {
        "text": text,
        "table": table,
        "figure": figure,
        "code": code,      
        "stdout": stdout,  
        "error": None
    }

def render_result(result: dict):
    if result.get("text"):
        st.markdown(result["text"])

    if result.get("table") is not None:
        st.markdown("**Result table:**")
        st.dataframe(result["table"].head(100), use_container_width=True)

    if result.get("figure") is not None:
        st.markdown("**Chart:**")
        st.pyplot(result["figure"], clear_figure=False, use_container_width=True)

    if result.get("stdout"):
        with st.expander("Console output"):
            st.code(result["stdout"])

# ---------- page ----------
st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="wide")
st.title("🗂️ DataChat")

with st.sidebar:
    st.subheader("Settings")
    # auto-fill API key from env or Streamlit secrets
    default_key = os.getenv("OPENAI_API_KEY", "")
    try:
        default_key = st.secrets.get("OPENAI_API_KEY", default_key)  
    except Exception:
        pass
    api_key = st.text_input("OpenAI API Key", type="password", value=default_key)
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1"], index=0)

    st.markdown("---")
    st.subheader("Load data")
    file = st.file_uploader("Upload CSV", type=["csv"]) 

    with st.expander("CSV read options (optional)"):
        sep = st.text_input("Separator (blank = auto)", value="")
        enc = st.text_input("Encoding (blank = auto)", value="")
        read_kwargs = {}
        if sep:
            read_kwargs["sep"] = sep
        if enc:
            read_kwargs["encoding"] = enc

    st.markdown("---")
    st.caption("Or use a built-in sample dataset")
    sample_rows = st.select_slider("Rows", options=[100, 500, 1000, 2000], value=1000)

    def make_sample_df(n=1000, seed=42):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        didx = rng.integers(0, len(dates), size=n)
        date_col = dates[didx]
        countries = np.array(["USA","Canada","UK","Germany","Spain","France","Italy","Mexico"])
        country = rng.choice(countries, size=n, p=[0.28,0.12,0.12,0.1,0.1,0.1,0.1,0.08])
        customers = [f"Cust_{i:03d}" for i in range(1,181)]
        customer = rng.choice(customers, size=n)
        catalog = [
            ("Wireless Mouse","Electronics",16.0),
            ("Laptop Stand","Electronics",45.0),
            ("Mechanical Keyboard","Electronics",85.0),
            ("HD Monitor","Electronics",200.0),
            ("Desk Lamp","Home",23.0),
            ("Coffee Mug","Home",7.5),
            ("Notebook","Stationery",2.5),
            ("Pen Set","Stationery",4.2),
            ("Sticky Notes","Stationery",1.2),
            ("Office Chair","Furniture",120.0),
            ("Standing Desk","Furniture",320.0),
            ("USB-C Hub","Electronics",29.0),
            ("Webcam","Electronics",55.0),
        ]
        idx = rng.integers(0, len(catalog), size=n)
        prod, cat, base_price = zip(*[catalog[i] for i in idx])
        prod = np.array(prod); cat = np.array(cat); base_price = np.array(base_price, dtype=float)
        price = base_price * (1 + rng.normal(0, 0.12, size=n)).clip(-0.2, 0.35)
        months = np.array([d.month for d in date_col])
        promo_factor = np.where(np.isin(months, [3,6,11]), 0.9, 1.0)
        price = (price * promo_factor).round(2)
        quantity = rng.choice([1,2,3,4,5], size=n, p=[0.55,0.22,0.12,0.07,0.04])
        order_id = np.arange(1, n+1)
        return pd.DataFrame({
            "OrderID": order_id,
            "Date": pd.to_datetime(date_col),
            "Product": prod,
            "Category": cat,
            "Quantity": quantity,
            "Price": price,
            "Customer": customer,
            "Country": country,
        })

    if st.button("📊 Use sample dataset"):
        st.session_state.df = make_sample_df(sample_rows)
        st.success(f"Loaded sample dataset (rows={sample_rows}) ✔")

# load uploaded
if 'file' in locals() and file is not None:
    try:
        df_raw = pd.read_csv(file, **read_kwargs)
    except Exception:
        file.seek(0)
        df_raw = pd.read_csv(file, sep=None, engine="python")
    st.session_state.df = df_raw.copy()
    st.success(f"Loaded {df_raw.shape[0]:,} rows × {df_raw.shape[1]:,} columns")

# ---------- typing controls ----------
TYPE_OPTIONS = [("auto","Auto-detect"),("int","Integer"),("float","Float"),("bool","Boolean"),("category","Category"),("datetime","Datetime"),("text","Text")]

def infer_kind(s: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(s):
        return "int"
    if pd.api.types.is_float_dtype(s):
        return "float"
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if s.dtype == object and s.nunique(dropna=True) <= max(20, int(0.05 * len(s))):
        return "category"
    return "text"

def coerce_series(s: pd.Series, kind: str) -> pd.Series:
    if kind == "auto":
        return coerce_series(s, infer_kind(s))
    if kind == "int":
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    if kind == "float":
        return pd.to_numeric(s, errors="coerce").astype(float)
    if kind == "bool":
        truthy = {"true","t","yes","y","1"}
        falsey = {"false","f","no","n","0"}
        def _b(x):
            if pd.isna(x):
                return pd.NA
            if isinstance(x, (int, float)):
                return bool(x)
            s_ = str(x).strip().lower()
            if s_ in truthy:
                return True
            if s_ in falsey:
                return False
            return pd.NA
        return s.map(_b).astype("boolean")
    if kind == "category":
        return s.astype("category")
    if kind == "datetime":
        return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if kind == "text":
        return s.astype(str)
    return s

# ---------- tabs ----------
if st.session_state.df is None:
    st.info("Upload a CSV or use the sample dataset to start.")
else:
    tabs = st.tabs(["🏠 Home", "🧭 Column Mapping", "📊 EDA & Insights", "💬 Chatbot"])

    # Home
    with tabs[0]:
        st.subheader("What this app does")
        st.markdown(
            "**DataChat** lets you:\n"
            "- Upload a CSV (or load a built-in sample)\n"
            "- Map columns to the right data types\n"
            "- Get a quick EDA with automatic takeaways\n"
            "- Ask questions in natural language and get answers, tables, and charts"
        )
        st.markdown("### How to use")
        st.markdown(
            "1. In the sidebar, upload a CSV or click **Use sample dataset**.\n"
            "2. Go to **Column Mapping** to set types (or keep auto).\n"
            "3. Check **EDA & Insights** for a quick overview.\n"
            "4. Open **Chatbot** and ask questions like *Top products by revenue* or *Monthly trend by country*."
        )
        st.markdown("### Privacy")
        st.markdown("Your data stays in your session. API calls only send a compact schema preview and a few head rows.")
        st.caption("Built by Joseph Sclar · joseph.sclar2@gmail.com")

    # Column Mapping
    with tabs[1]:
        st.subheader("Map your column types")
        schema = st.session_state.schema or {}
        for c in st.session_state.df.columns:
            suggested = infer_kind(st.session_state.df[c])
            current = schema.get(c, "auto")
            idx = [k for k, _ in TYPE_OPTIONS].index(current if current else "auto")
            label = f"{c}  ·  suggested: {suggested}"
            choice = st.selectbox(label, [t[0] for t in TYPE_OPTIONS], format_func=lambda k: dict(TYPE_OPTIONS)[k], key=f"type_{c}", index=idx)
            schema[c] = choice
        st.session_state.schema = schema
        if st.button("Apply mapping", use_container_width=True):
            df2 = st.session_state.df.copy()
            for col, kind in schema.items():
                df2[col] = coerce_series(df2[col], kind)
            st.session_state.df = df2
            st.success("Types updated ✔")
        with st.expander("Typed Data Preview"):
            info = pd.DataFrame({
                "column": st.session_state.df.columns,
                "dtype": [str(st.session_state.df[c].dtype) for c in st.session_state.df.columns],
                "non_null": [int(st.session_state.df[c].notna().sum()) for c in st.session_state.df.columns],
                "n_unique": [int(st.session_state.df[c].nunique(dropna=True)) for c in st.session_state.df.columns],
            })
            st.dataframe(info, use_container_width=True)
            st.dataframe(st.session_state.df.head(50), use_container_width=True)

    # EDA
    with tabs[2]:
        st.subheader("Quick EDA & Insights")
        df = st.session_state.df
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
        c3.metric("Memory (MB)", f"{mem_mb:.2f}")
        dt_cols = list(df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns)
        if dt_cols:
            rng = (pd.to_datetime(df[dt_cols[0]].min(), errors='coerce'), pd.to_datetime(df[dt_cols[0]].max(), errors='coerce'))
            c4.metric("Time range", f"{str(rng[0])[:10]} → {str(rng[1])[:10]}")
        else:
            rng = (None, None)
            c4.metric("Time range", "—")
        st.markdown("#### Columns")
        col_info = pd.DataFrame({
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "non_null": [int(df[c].notna().sum()) for c in df.columns],
            "n_unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
            "pct_missing": [float(100*(1 - df[c].notna().mean())) for c in df.columns],
        })
        st.dataframe(col_info, use_container_width=True)
        miss = col_info.set_index("column")["pct_missing"].sort_values(ascending=False)
        if (miss > 0).any():
            st.markdown("#### Missing values (%) by column")
            st.bar_chart(miss)
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if num_cols:
            st.markdown("#### Numeric summary (describe)")
            st.dataframe(df[num_cols].describe().transpose(), use_container_width=True)
        cat_cols = [c for c in df.columns if str(df[c].dtype) in ["category", "object"]]
        if cat_cols:
            st.markdown("#### Top categories (first 5)")
            for c in cat_cols[:5]:
                st.write(f"**{c}** — top 5")
                vc = df[c].astype(str).value_counts().head(5)
                st.dataframe(vc.rename("count"), use_container_width=True)
        if len(num_cols) >= 2:
            st.markdown("#### Correlation (numeric)")
            corr = df[num_cols].corr(numeric_only=True)
            st.dataframe(corr, use_container_width=True)
        insights = []
        if not miss.empty and miss.iloc[0] > 0:
            insights.append(f"Highest missingness: **{miss.index[0]}** at **{miss.iloc[0]:.1f}%**.")
        most_unique = col_info.sort_values("n_unique", ascending=False).iloc[0]
        insights.append(f"Most unique values: **{most_unique['column']}** (n={int(most_unique['n_unique'])}).")
        if num_cols:
            means = df[num_cols].mean(numeric_only=True).sort_values(ascending=False)
            insights.append(f"Largest mean: **{means.index[0]} = {means.iloc[0]:.3g}**.")
        if dt_cols and rng[0] is not None:
            insights.append(f"Datetime detected: **{dt_cols[0]}** spanning **{str(rng[0])[:10]} → {str(rng[1])[:10]}**.")
        st.markdown("#### Takeaways")
        if insights:
            for it in insights:
                st.markdown(f"- {it}")
        else:
            st.markdown("- No obvious takeaways detected.")

    with tabs[3]:
        st.subheader("Chat with your data")

        # Compact schema header
        header_info = pd.DataFrame({
            "Column": st.session_state.df.columns if st.session_state.df is not None else [],
            "Type": st.session_state.df.dtypes.astype(str) if st.session_state.df is not None else []
        })
        if not header_info.empty:
            st.dataframe(header_info, use_container_width=True)

        # Persistently show the last successful result without re-appending to chat
        lr = st.session_state.get("last_result")
        if lr and (lr.get("text") or lr.get("table") is not None or lr.get("figure") is not None):
            st.markdown("**Last result**")
            if lr.get("text"):
                st.markdown(lr["text"])  # show text only (no chat append)
            if lr.get("table") is not None:
                st.markdown("**Result table:**")
                st.dataframe(lr["table"].head(100), use_container_width=True)
            if lr.get("figure") is not None:
                st.markdown("**Chart:**")
                st.pyplot(lr["figure"], clear_figure=False, use_container_width=True)
            if lr.get("stdout"):
                with st.expander("Console output"):
                    st.code(lr["stdout"])

        with st.container(border=True):
            # Chat history (messages only)
            for m in st.session_state.chat:
                if m["role"] == "user":
                    st.markdown(f"**You:** {m['content']}")
                else:
                    st.markdown(m["content"])  

            # Submit form with debounce & disabled state to avoid double execution
            with st.form("chat_form", clear_on_submit=False):
                q = st.text_input("Ask a question...", key="chat_input")
                cA, cB = st.columns([1, 0.25])
                submitted = cA.form_submit_button(
                    "Ask", type="primary", disabled=st.session_state.get('busy', False)
                )
                clear = cB.form_submit_button("Clear chat")

            if clear:
                st.session_state.chat = []
                st.session_state.last_result = None
                st.session_state['busy'] = False
                st.rerun()

            if submitted and q.strip():
                if not api_key:
                    st.error("Enter your OpenAI API Key in the sidebar.")
                else:
                    # Time-based debounce for accidental double submits of the same prompt
                    import time as _time
                    now = _time.time()
                    last_q = st.session_state.get("last_prompt")
                    last_t = st.session_state.get("last_submit_at", 0.0)
                    if last_q == q and (now - last_t) < 1.2:
                        st.info("Ignored duplicate click.")
                    else:
                        st.session_state['busy'] = True
                        st.session_state['last_prompt'] = q
                        st.session_state['last_submit_at'] = now
                        st.session_state.chat.append({"role": "user", "content": q})
                        try:
                            with st.spinner("Thinking..."):
                                resp = ask_model_for_code(api_key, model, st.session_state.df, q)
                                code = extract_code(resp)
                                result = run_user_code(code, st.session_state.df)
                            # Only save and display if we actually got something
                            if result and (result.get("text") or result.get("table") is not None or result.get("figure") is not None):
                                st.session_state.last_result = result
                                # Render via the unified renderer
                                render_result(result)
                            else:
                                st.warning("No output generated for this request.")
                        except Exception:
                            st.session_state.chat.append({
                                "role": "assistant",
                                "content": "❌ Error\n\n````text\n" + traceback.format_exc() + "\n````"
                            })
                        finally:
                            st.session_state['busy'] = False

# footer
st.caption("© 2025 Joseph Sclar · joseph.sclar2@gmail.com")
