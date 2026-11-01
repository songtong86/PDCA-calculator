# -*- coding: utf-8 -*-
"""
PDAC Risk Calculator — Fully optimized for Top-Journal style & Streamlit Cloud deployment.
"""
import streamlit as st
from pathlib import Path
import joblib
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Plot style
# ============================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 120,
})

# ============================================================
# 动态路径配置 (适配云端)
# ============================================================
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR / "data"

EXPECTED_FEATURES = ["CEA", "DBIL", "Age", "SurgeryHistory", "CA19-9"]

BASE_UNITS = {
    "CEA": "ng/mL",
    "DBIL": "μmol/L",
    "Age": "years",
    "SurgeryHistory": "",
    "CA19-9": "U/mL",
}
BINARY_FEATURES = {"SurgeryHistory"}

FEATURE_META = {
    "CEA": {"default": 2.5, "min": 0.0, "max": 1000.0},
    "DBIL": {"default": 4.0, "min": 0.0, "max": 500.0},
    "Age": {"default": 60.0, "min": 18.0, "max": 100.0},
    "SurgeryHistory": {"default": 0, "min": 0, "max": 1},
    "CA19-9": {"default": 15.0, "min": 0.0, "max": 1000.0},
}

LOCKED_THRESHOLD = 0.50

# ============================================================
# 内置的标准化参数 (代替巨大的 training_parameters_mice.pkl)
# ============================================================
MEAN_DICT = {
    "Age": 56.82857142857143,
    "SurgeryHistory": 0.36,
    "CA19-9": 11.651157142857143,
    "CEA": 3.3454857142857146,
    "DBIL": 5.3337142857142865
}

STD_DICT = {
    "Age": 12.706209312853336,
    "SurgeryHistory": 0.48,
    "CA19-9": 8.803011427729416,
    "CEA": 2.822665365232236,
    "DBIL": 3.0664181857333026
}


# ============================================================
# Helpers
# ============================================================
def _resolve_mean_std(name, mean_dict, std_dict, fallback_mean=0.0, fallback_std=1.0):
    if name in mean_dict and name in std_dict:
        return mean_dict[name], std_dict[name]
    alias_map = {"CA19-9": ["CA19-9", "CA199", "CA19_9"]}
    for alt in alias_map.get(name, []):
        if alt in mean_dict and alt in std_dict:
            return mean_dict[alt], std_dict[alt]
    return fallback_mean, fallback_std


class _CalibratedPredictor:
    def __init__(self, base, cal):
        self.base = base
        self.cal = cal
        self._cal_cls = type(cal).__name__.lower()

    def predict_proba(self, X):
        p = self.base.predict_proba(X)[:, 1].reshape(-1, 1)
        if "isotonic" in self._cal_cls or hasattr(self.cal, "out_of_bounds"):
            p_cal = self.cal.predict(p.ravel())
        else:
            p_cal = self.cal.predict_proba(p)[:, 1]
        p_cal = np.asarray(p_cal).reshape(-1)
        return np.column_stack([1 - p_cal, p_cal])


def _is_full_classifier(model, n):
    try:
        model.predict_proba(np.zeros((1, n)))
        return True
    except Exception:
        return False


@st.cache_resource
def load_artifacts():
    # 移除 absolute paths，使用动态生成的 BASE_DIR
    raw_model_path = BASE_DIR / "best_model.pkl"
    if not raw_model_path.exists():
        st.error(f"Model file not found: {raw_model_path}. Please ensure it is in the `data` folder.")
        st.stop()

    raw_model = joblib.load(raw_model_path)
    predict_model = raw_model
    calib_msg = "uncalibrated"

    cal_path = BASE_DIR / "calibrated_model.pkl"
    if cal_path.exists():
        cal_obj = joblib.load(cal_path)
        if _is_full_classifier(cal_obj, len(EXPECTED_FEATURES)):
            predict_model = cal_obj
            calib_msg = "calibrated"
        else:
            predict_model = _CalibratedPredictor(raw_model, cal_obj)
            calib_msg = "post-hoc calibrated"

    # 不再加载 training_parameters_mice.pkl，直接使用硬编码字典
    X_train_path = BASE_DIR / "X_train.npy"
    if not X_train_path.exists():
        st.error(f"Background data not found: {X_train_path}")
        st.stop()

    X_train = np.load(X_train_path)
    if X_train.shape[1] != len(EXPECTED_FEATURES):
        X_train = X_train[:, :len(EXPECTED_FEATURES)]

    return predict_model, raw_model, EXPECTED_FEATURES, X_train, MEAN_DICT, STD_DICT, calib_msg


@st.cache_resource
def build_explainer(_raw_model, _X_train):
    bg = _X_train[:min(200, len(_X_train))]
    return shap.TreeExplainer(_raw_model, data=bg)


def compute_shap(explainer, x_norm):
    sv = explainer.shap_values(x_norm)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    shap_vals = sv.reshape(-1).astype(float)

    ev = np.asarray(explainer.expected_value).reshape(-1)
    base_val = float(ev[1]) if ev.size > 1 else float(ev[0])
    return shap_vals, base_val


# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="PDAC Risk Calculator",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    "<h2 style='text-align:center; margin-bottom:2px; font-weight:600;'>"
    "PDAC Risk Calculator for Patients with Non-Elevated CA19-9"
    "</h2>"
    "<p style='text-align:center; color:#555; margin-top:0; font-size:14px;'>"
    "Differentiating imaging-suspected early-stage PDAC from non-PDAC "
    "pancreatic lesions in patients with serum CA19-9 &lt; 35 U/mL"
    "</p>",
    unsafe_allow_html=True,
)

with st.spinner("Loading model..."):
    (predict_model, raw_model, feature_names,
     X_train, mean_dict, std_dict, calib_msg) = load_artifacts()
    explainer = build_explainer(raw_model, X_train)

st.caption(
    f"Gradient Boosting classifier with Platt calibration · "
    f"5 routine indicators (CEA, DBIL, Age, Surgery history, CA19-9) · "
    f"TreeSHAP attribution in log-odds space · locked threshold = {LOCKED_THRESHOLD:.2f}"
)

# ============================================================
# 输入表单
# ============================================================
with st.container(border=True):
    with st.form("single_form"):
        cols = st.columns(2)
        raw_values = {}
        features_ordered = ["CEA", "CA19-9", "DBIL", "Age", "SurgeryHistory"]

        for i, f in enumerate(features_ordered):
            meta = FEATURE_META[f]
            unit = BASE_UNITS.get(f, "")
            label = f if not unit else f"{f} ({unit})"

            with cols[i % 2]:
                if f in BINARY_FEATURES:
                    raw_values[f] = st.selectbox(
                        label, options=[0, 1], index=int(meta["default"]),
                        format_func=lambda x: "Yes" if x == 1 else "No",
                    )
                else:
                    fmt = "%.1f" if f == "Age" else "%.2f"
                    raw_values[f] = st.number_input(
                        label, value=float(meta["default"]),
                        min_value=float(meta["min"]), max_value=float(meta["max"]),
                        step=1.0 if f == "Age" else 0.1, format=fmt,
                    )

        st.write("")
        col_btn_left, col_btn_mid, col_btn_right = st.columns([1, 1, 1])
        with col_btn_mid:
            submitted = st.form_submit_button(
                "Predict", use_container_width=True, type="primary"
            )

# 计算逻辑
if submitted:
    x_raw = np.array([[raw_values[f] for f in feature_names]], dtype=float)
    x_norm = np.zeros_like(x_raw)
    for i, f in enumerate(feature_names):
        m, s = _resolve_mean_std(f, mean_dict, std_dict)
        x_norm[0, i] = (x_raw[0, i] - m) / (s if s != 0 else 1.0)

    prob = float(predict_model.predict_proba(x_norm)[0, 1])
    st.session_state["x_norm"] = x_norm
    st.session_state["prob"] = prob

# ============================================================
# 结果展示区
# ============================================================
if "prob" in st.session_state:
    st.divider()
    prob = st.session_state["prob"]
    x_norm = st.session_state["x_norm"]

    risk_level = "High" if prob >= LOCKED_THRESHOLD else "Low"
    color = "#d9534f" if prob >= LOCKED_THRESHOLD else "#5cb85c"

    st.markdown(
        f"""
        <div style='text-align: center; margin-bottom: 30px;'>
            <p style='font-size: 24px; color: #333; margin-bottom: 5px;'>
                Based on feature values, predicted possibility of PDAC is:
            </p>
            <p style='font-size: 42px; font-weight: bold; color: {color}; margin: 0; line-height: 1.2;'>
                {prob * 100:.1f}%
            </p>
            <p style='font-size: 16px; color: {color}; margin-top: 5px;'>
                ({risk_level} risk relative to 0.50 threshold)
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### Feature attribution (SHAP)")
    shap_vals, base_val = compute_shap(explainer, x_norm)
    display_names = [f.replace("SurgeryHistory", "Surgery history") for f in feature_names]

    expl = shap.Explanation(
        values=shap_vals,
        base_values=base_val,
        data=x_norm[0],
        feature_names=display_names,
    )

    shap.plots.waterfall(expl, max_display=len(feature_names), show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 4)
    plt.tight_layout(pad=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    order = np.argsort(np.abs(shap_vals))[::-1]
    top_i = int(order[0])
    top_name = display_names[top_i]
    top_dir = "increases" if shap_vals[top_i] > 0 else "decreases"
    pos_sum = float(shap_vals[shap_vals > 0].sum())
    neg_sum = float(shap_vals[shap_vals < 0].sum())
    net = pos_sum + neg_sum

    st.markdown(
        f"The baseline log-odds ({base_val:+.2f}) is shifted to "
        f"**{prob * 100:.1f}%** probability. "
        f"The largest contributor is **{top_name}** "
        f"(SHAP = {shap_vals[top_i]:+.3f}), which {top_dir} the risk. "
        f"Cumulative positive contributions = {pos_sum:+.3f}; "
        f"negative = {neg_sum:+.3f}; net = {net:+.3f}."
    )

else:
    st.divider()
    st.info(
        "👈 Please fill in the patient measurements above and click **Predict** to view the risk estimation and SHAP explanation.")

# ============================================================
# Usage notes & Footer
# ============================================================
with st.expander("Usage notes and model limitations", expanded=False):
    st.markdown(
        "**1. Intended population.** Adults (≥18 years) with a pancreatic lesion "
        "on imaging, no definite advanced features, and serum CA19-9 &lt; 35 U/mL "
        "at initial presentation. \"Non-elevated CA19-9\" captures both true Lewis "
        "antigen-negative nonproducers and patients with low tumor burden.\n\n"
        "**2. Decision threshold.** Primary threshold **0.50**, locked *before* "
        "external validation. External temporal validation (n = 112): AUC 0.7955 "
        "(95% CI 0.6945–0.8727), sensitivity 63.04%, specificity 84.85%. "
        "At this threshold, 56 of 66 non-PDAC patients would have been correctly "
        "assigned a negative result. Output is for clinical reference only.\n\n"
        "**3. Limitations.** Single-center retrospective development. Lewis antigen "
        "genotyping was not available. External validation in independent cohorts is "
        "required before routine clinical adoption."
    )

st.divider()
st.caption(
    "Research use and clinical decision support only. This tool does not "
    "constitute a medical diagnosis. External validation in independent "
    "multicenter cohorts is required before routine clinical adoption. "
    "No input data are uploaded or stored."
)