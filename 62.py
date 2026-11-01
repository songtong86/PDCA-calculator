# -*- coding: utf-8 -*-
"""
PDAC Risk Calculator — Publication Figure Generator.
Generates a high-resolution, uncropped full-page mockup of the web calculator.
"""
import streamlit as st
from pathlib import Path
import joblib
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ============================================================
# Plot style (journal-compatible)
# ============================================================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ============================================================
# Paths and constants
# ============================================================
BASE_DIR = Path(r"C:/python/1/data")
AA2_OUTPUT_DIR = Path(r"C:/python/1/output_matched_direct/bootstrap_data")

EXPECTED_FEATURES = ["CEA", "DBIL", "Age", "SurgeryHistory", "CA19-9"]
LOCKED_THRESHOLD = 0.50


# ============================================================
# 兼容后验校准器的包装类
# ============================================================
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


# ============================================================
# Helpers (Model Loading)
# ============================================================
@st.cache_resource
def load_artifacts():
    raw_model = joblib.load(BASE_DIR / "best_model.pkl")
    predict_model = raw_model

    cal_path = BASE_DIR / "calibrated_model.pkl"
    if cal_path.exists():
        cal_obj = joblib.load(cal_path)
        if _is_full_classifier(cal_obj, len(EXPECTED_FEATURES)):
            predict_model = cal_obj
        else:
            predict_model = _CalibratedPredictor(raw_model, cal_obj)

    params_path = AA2_OUTPUT_DIR / "training_parameters_mice.pkl"
    std_param = joblib.load(params_path)["standardize_param"]
    mean_dict = dict(zip(std_param["num_cols"], std_param["means"]))
    std_dict = dict(zip(std_param["num_cols"], std_param["stds"]))

    X_train = np.load(BASE_DIR / "X_train.npy")
    if X_train.shape[1] != len(EXPECTED_FEATURES):
        X_train = X_train[:, :len(EXPECTED_FEATURES)]

    return predict_model, raw_model, EXPECTED_FEATURES, X_train, mean_dict, std_dict


@st.cache_resource
def build_explainer(_raw_model, _X_train):
    return shap.TreeExplainer(_raw_model, data=_X_train[:200])


def _resolve_mean_std(name, mean_dict, std_dict):
    return mean_dict.get(name, 0.0), std_dict.get(name, 1.0)


# ============================================================
# 手写高定版瀑布图 (彻底告别 shap 的版本 Bug)
# ============================================================
def draw_custom_waterfall(ax, shap_vals, base_val, feature_names, feature_values):
    # 按照绝对值从小到大排序（这样最大的影响会在最上面）
    order = np.argsort(np.abs(shap_vals))
    sv = shap_vals[order]
    names = [feature_names[i] for i in order]
    vals = [feature_values[i] for i in order]

    n = len(sv)
    y_pos = np.arange(n)

    # 累计值计算
    cum_vals = base_val + np.cumsum(sv)
    prev_cums = np.concatenate(([base_val], cum_vals[:-1]))

    # 绘制柱状图
    for i in range(n):
        val = sv[i]
        start = prev_cums[i]
        width = val
        color = '#ff0051' if val > 0 else '#008bfb'  # 经典 SHAP 红蓝配色

        ax.barh(i, width, left=start, height=0.6, color=color, edgecolor='none')

        # 在柱子旁边写上具体的 SHAP 值
        text_x = start + width + (0.05 if width > 0 else -0.05)
        ha = 'left' if width > 0 else 'right'
        ax.text(text_x, i, f"{val:+.2f}", va='center', ha=ha, fontsize=9, color='#333')

    # 标注左侧的特征和具体的输入值
    y_labels = [f"{names[i]} = {vals[i]:.2f}" for i in range(n)]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=10)

    # 添加基准线
    ax.axvline(base_val, color='#888', linestyle='--', linewidth=1)
    ax.text(base_val, n - 0.2, f"E[f(X)] = {base_val:.2f}", ha='center', va='bottom', fontsize=9, color='#555')

    # 添加最终预测线
    final_val = cum_vals[-1]
    ax.axvline(final_val, color='#333', linestyle=':', linewidth=1)
    ax.text(final_val, n - 0.2, f"f(x) = {final_val:.2f}", ha='center', va='bottom', fontsize=9, color='#333')

    # 坐标轴美化
    ax.set_xlabel("Model output (log-odds)", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)  # 隐藏 y 轴刻度线


# ============================================================
# Main Figure Generation Logic
# ============================================================
def generate_publication_figure(predict_model, explainer, mean_dict, std_dict, input_data):
    # 1. 数据预处理
    x_raw = np.array([[input_data[f] for f in EXPECTED_FEATURES]], dtype=float)
    x_norm = np.zeros_like(x_raw)
    for i, f in enumerate(EXPECTED_FEATURES):
        m, s = _resolve_mean_std(f, mean_dict, std_dict)
        x_norm[0, i] = (x_raw[0, i] - m) / (s if s != 0 else 1.0)

    # 2. 预测与 SHAP
    prob = float(predict_model.predict_proba(x_norm)[0, 1])

    sv = explainer.shap_values(x_norm)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    shap_vals = sv.reshape(-1).astype(float)

    ev = explainer.expected_value
    base_val = float(ev[1]) if isinstance(ev, (list, np.ndarray)) and len(ev) > 1 else float(ev)

    # 3. 开始绘制画布 (长图 10x12)
    fig = plt.figure(figsize=(10, 12), facecolor='white')
    # 重新分配比例：标题 0.8，输入 1.5，结论 1.2，瀑布图 3.0，底部 0.5
    gs = fig.add_gridspec(5, 1, height_ratios=[0.8, 1.5, 1.2, 3.0, 0.5], hspace=0.5)

    # ------------------ 1. 标题区 ------------------
    ax_title = fig.add_subplot(gs[0])
    ax_title.axis('off')
    ax_title.text(0.5, 0.8, "PDAC Risk Calculator for Patients with Non-Elevated CA19-9",
                  ha='center', va='center', fontsize=18, fontweight='bold', color='#111')
    ax_title.text(0.5, 0.3, "Differentiating imaging-suspected early-stage PDAC from non-PDAC pancreatic lesions",
                  ha='center', va='center', fontsize=12, color='#555')

    # ------------------ 2. 输入区 ------------------
    ax_input = fig.add_subplot(gs[1])
    ax_input.axis('off')
    rect = patches.FancyBboxPatch((0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.02",
                                  edgecolor="#ccc", facecolor="#fcfcfc", lw=1.5)
    ax_input.add_patch(rect)

    labels = [
        f"CEA (ng/mL): {input_data['CEA']:.2f}",
        f"CA19-9 (U/mL): {input_data['CA19-9']:.2f}",
        f"DBIL (μmol/L): {input_data['DBIL']:.2f}",
        f"Age (years): {input_data['Age']:.1f}",
        f"Surgery History: {'Yes' if input_data['SurgeryHistory'] == 1 else 'No'}"
    ]

    for i, text in enumerate(labels):
        y_pos = 0.75 - (i % 3) * 0.25
        x_pos = 0.1 if i < 3 else 0.55
        ax_input.text(x_pos, y_pos, text, fontsize=11, va='center',
                      bbox=dict(facecolor='#f0f2f6', edgecolor='none', boxstyle='round,pad=0.5'))

    ax_input.text(0.5, 0.05, "Predict", ha='center', va='center',
                  fontsize=12, fontweight='bold', color='white',
                  bbox=dict(facecolor='#ff4b4b', edgecolor='none', boxstyle='round,pad=0.6', alpha=0.9))

    # ------------------ 3. 结论区 ------------------
    ax_result = fig.add_subplot(gs[2])
    ax_result.axis('off')
    risk_level = "High" if prob >= LOCKED_THRESHOLD else "Low"
    color = "#d9534f" if prob >= LOCKED_THRESHOLD else "#5cb85c"

    ax_result.text(0.5, 0.8, "Based on feature values, predicted possibility of PDAC is:",
                   ha='center', va='center', fontsize=14, color='#333')
    ax_result.text(0.5, 0.4, f"{prob * 100:.1f}%",
                   ha='center', va='center', fontsize=38, fontweight='bold', color=color)
    ax_result.text(0.5, 0.05, f"({risk_level} risk relative to 0.50 threshold)",
                   ha='center', va='center', fontsize=12, color=color)

    # ------------------ 4. SHAP 瀑布图区 (使用手写函数) ------------------
    ax_shap = fig.add_subplot(gs[3])
    display_names = [f.replace("SurgeryHistory", "Surgery history") for f in EXPECTED_FEATURES]
    feature_values = x_raw[0]  # 原始输入值用于显示

    # 调用我们手写的瀑布图
    draw_custom_waterfall(ax_shap, shap_vals, base_val, display_names, feature_values)
    ax_shap.set_title("Feature attribution (SHAP)", loc='left', fontsize=14, fontweight='bold', pad=15)

    # ------------------ 5. 底部免责声明 ------------------
    ax_footer = fig.add_subplot(gs[4])
    ax_footer.axis('off')
    ax_footer.text(0.5, 0.5, "Research use and clinical decision support only. Not a medical diagnosis.",
                   ha='center', va='center', fontsize=10, color='#888', style='italic')

    return fig, prob, shap_vals


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Figure Generator", layout="centered")

st.title("📊 Web Calculator Figure Generator")
st.markdown(
    "This page directly generates a **publication-ready, uncropped figure** of the web calculator for your manuscript.")

with st.spinner("Loading models..."):
    predict_model, raw_model, features, X_train, mean_dict, std_dict = load_artifacts()
    explainer = build_explainer(raw_model, X_train)

st.write("---")

st.subheader("1. Select patient case")
case = st.radio("Choose a case to plot:",
                ["High-risk example", "Low-risk example", "Custom"])

if case == "High-risk example":
    input_data = {"CEA": 3.50, "DBIL": 20.00, "Age": 60.0, "SurgeryHistory": 1, "CA19-9": 24.00}
elif case == "Low-risk example":
    input_data = {"CEA": 2.50, "DBIL": 4.00, "Age": 60.0, "SurgeryHistory": 0, "CA19-9": 15.00}
else:
    col1, col2 = st.columns(2)
    input_data = {
        "CEA": col1.number_input("CEA", value=2.5),
        "CA19-9": col2.number_input("CA19-9", value=15.0),
        "DBIL": col1.number_input("DBIL", value=4.0),
        "Age": col2.number_input("Age", value=60.0),
        "SurgeryHistory": col1.selectbox("Surgery History", options=[0, 1],
                                         format_func=lambda x: "Yes" if x == 1 else "No")
    }

st.write("---")
st.subheader("2. Generate and Download")

if st.button("Generate Publication Figure", type="primary"):
    fig, prob, shap_vals = generate_publication_figure(predict_model, explainer, mean_dict, std_dict, input_data)

    st.pyplot(fig, use_container_width=True)

    tif_path = "Figure_PDAC_Calculator.tif"
    pdf_path = "Figure_PDAC_Calculator.pdf"

    fig.savefig(tif_path, dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, format="pdf", bbox_inches='tight')

    st.success(f"Figure generated successfully! (Predicted Probability: {prob * 100:.1f}%)")

    with open(tif_path, "rb") as f_tif, open(pdf_path, "rb") as f_pdf:
        col1, col2 = st.columns(2)
        col1.download_button("⬇️ Download TIFF (300 DPI)", f_tif, file_name=tif_path, mime="image/tiff")
        col2.download_button("⬇️ Download PDF (Vector)", f_pdf, file_name=pdf_path, mime="application/pdf")