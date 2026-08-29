# -*- coding: utf-8 -*-
"""
تطبيق ويب مفتوح المصدر لتحليل PLS-SEM
يقرأ بيانات من ملف إكسل، يبني نموذج القياس والنموذج البنائي تفاعليًا،
وينفّذ كل الحسابات والرسوم البيانية المعتادة في PLS-SEM — بما في ذلك:
النماذج التكوينية مع اختبار VIF، تحليل الوساطة، متغيرات التعديل،
تصدير تقرير PDF جاهز، وحفظ/تحميل مشاريع متعددة.
"""

import io
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from pls_engine import PLSModel
import project_store
import auth
import licensing
from pdf_report import build_report

st.set_page_config(page_title="PLS-SEM", layout="wide")


def _admin_password() -> str:
    """كلمة مرور لوحة تحكم المشرفة. على Streamlit Cloud تُضبط من Settings → Secrets
    باسم ADMIN_PASSWORD. محليًا بدون إعداد، القيمة الافتراضية هي admin123 (غيّريها!)."""
    try:
        return st.secrets.get("ADMIN_PASSWORD", "admin123")
    except Exception:
        return "admin123"


# ---------------------------------------------------------------- تنسيق RTL
st.markdown("""
<style>
html, body, [class*="css"]  { direction: rtl; text-align: right; font-family: "Tahoma","Arial",sans-serif; }
.stButton>button { direction: rtl; }
[data-testid="stMetricValue"] { direction: ltr; }
table { direction: rtl; }
</style>
""", unsafe_allow_html=True)

st.title("🧮 أداة تحليل PLS-SEM")

# ======================================================= بوابة تسجيل الدخول
if "user_email" not in st.session_state:
    st.session_state.user_email = None

if st.session_state.user_email is None:
    st.caption("سجّلي دخولك، أو أنشئي حسابًا جديدًا باستخدام كود الترخيص الذي استلمتِه.")
    tab_login, tab_register = st.tabs(["🔑 تسجيل الدخول", "🆕 إنشاء حساب بكود ترخيص"])

    with tab_login:
        with st.form("login_form"):
            login_email = st.text_input("الإيميل", key="login_email")
            login_pwd = st.text_input("كلمة المرور", type="password", key="login_pwd")
            submitted = st.form_submit_button("دخول")
            if submitted:
                ok, msg = auth.authenticate(login_email, login_pwd)
                if ok:
                    st.session_state.user_email = login_email.strip().lower()
                    st.rerun()
                else:
                    st.error(msg)

    with tab_register:
        st.caption("التسجيل يتطلب كود ترخيص صالح (تحصلين عليه بعد الدفع من صاحبة التطبيق).")
        with st.form("register_form"):
            reg_key = st.text_input("كود الترخيص (مثال: ABCD-1234-EFGH-5678)", key="reg_key")
            reg_email = st.text_input("الإيميل", key="reg_email")
            reg_pwd = st.text_input("كلمة المرور (6 أحرف على الأقل)", type="password", key="reg_pwd")
            reg_pwd2 = st.text_input("تأكيد كلمة المرور", type="password", key="reg_pwd2")
            submitted2 = st.form_submit_button("إنشاء الحساب")
            if submitted2:
                key_ok, key_msg = licensing.validate_key(reg_key)
                if not key_ok:
                    st.error(key_msg)
                elif reg_pwd != reg_pwd2:
                    st.error("كلمتا المرور غير متطابقتين.")
                else:
                    ok, msg = auth.register_user(reg_email, reg_pwd)
                    if ok:
                        licensing.redeem_key(reg_key, reg_email)
                        st.success(msg + " يمكنك الآن تسجيل الدخول من التبويب المجاور.")
                    else:
                        st.error(msg)

    with st.expander("🔐 لوحة تحكم المشرفة (توليد أكواد الترخيص)"):
        admin_pwd_input = st.text_input("كلمة مرور المشرفة", type="password", key="admin_pwd_input")
        if admin_pwd_input == _admin_password() and admin_pwd_input != "":
            st.success("تم التحقق — مرحبًا بك في لوحة التحكم.")

            c1, c2 = st.columns(2)
            with c1:
                note = st.text_input("ملاحظة (اسم الزبون مثلاً — اختياري)", key="key_note")
            with c2:
                valid_days = st.number_input("صلاحية الكود بالأيام (0 = بلا انتهاء)", min_value=0, value=0, step=1)
            if st.button("➕ توليد كود ترخيص جديد"):
                new_key = licensing.generate_key(note=note, valid_days=valid_days or None)
                st.success(f"الكود الجديد: `{new_key}` — انسخيه وأرسليه للزبون.")

            st.markdown("#### كل الأكواد")
            keys = licensing.list_keys()
            if keys:
                st.dataframe(pd.DataFrame(keys), use_container_width=True)
            else:
                st.caption("لا توجد أكواد بعد.")
        elif admin_pwd_input != "":
            st.error("كلمة مرور خاطئة.")

    st.stop()

# ------------------------------------------------- المستخدم مسجَّل دخوله من هنا
with st.sidebar:
    st.success(f"👤 {st.session_state.user_email}")
    if st.button("🚪 تسجيل خروج"):
        for k in ["user_email", "model", "spec", "paths", "interactions",
                  "spec_final", "paths_final", "project_df", "current_df"]:
            st.session_state.pop(k, None)
        st.rerun()

st.caption("ارفعي ملف إكسل يحوي بياناتك، حدّدي نموذج القياس والعلاقات البنائية، واحصلي على كل الحسابات والرسوم.")




def _is_significant(p_val):
    """يتحقق من المعنوية دون الوقوع في فخ اعتبار 0.0 قيمة فارغة (falsy) في بايثون."""
    return (p_val is not None) and (not np.isnan(p_val)) and (p_val < 0.05)


for key, default in [("spec", {}), ("paths", []), ("interactions", [])]:
    if key not in st.session_state:
        st.session_state[key] = default

# ======================================================= الشريط الجانبي: المشاريع
with st.sidebar:
    st.divider()
    st.header("📁 مشاريعي المحفوظة")
    st.caption("احفظي تعريف النموذج (والبيانات اختياريًا) لتُكملي العمل لاحقًا دون إعادة البناء.")

    user_email = st.session_state.user_email
    projects = project_store.list_projects(user_email)
    if projects:
        names = [p["name"] for p in projects]
        sel = st.selectbox("اختاري مشروعًا محفوظًا", names, key="sel_project")
        c1, c2 = st.columns(2)
        if c1.button("📂 تحميل"):
            proj = project_store.load_project(user_email, sel)
            if proj:
                st.session_state.spec = proj["model_spec"]
                st.session_state.paths = [tuple(p) for p in proj["structural_paths"]]
                st.session_state.interactions = [tuple(i) for i in proj.get("interactions", [])]
                if proj["data"] is not None:
                    st.session_state["project_df"] = proj["data"]
                    st.success(f"تم تحميل المشروع «{sel}» مع بياناته.")
                else:
                    st.session_state.pop("project_df", None)
                    st.success(f"تم تحميل تعريف المشروع «{sel}» — ارفعي ملف الإكسل المطابق له.")
                st.session_state.pop("model", None)
                st.rerun()
        if c2.button("🗑️ حذف"):
            project_store.delete_project(user_email, sel)
            st.rerun()
    else:
        st.caption("لا توجد مشاريع محفوظة بعد.")

    st.divider()
    st.subheader("💾 حفظ المشروع الحالي")
    proj_name = st.text_input("اسم المشروع", key="proj_name_input")
    save_data_too = st.checkbox("احفظي البيانات أيضًا (تُخزَّن داخل قاعدة بيانات محلية على الخادم)")
    if st.button("💾 حفظ"):
        if not proj_name:
            st.error("اكتبي اسمًا للمشروع.")
        elif not st.session_state.spec:
            st.error("لا يوجد نموذج مُعرَّف بعد لحفظه.")
        else:
            df_to_save = st.session_state.get("current_df") if save_data_too else None
            project_store.save_project(
                user_email, proj_name, st.session_state.spec, st.session_state.paths,
                st.session_state.interactions, df_to_save, include_data=save_data_too,
            )
            st.success(f"تم حفظ المشروع «{proj_name}».")

# ======================================================= 1) رفع الملف
st.header("1️⃣ رفع بيانات الإكسل")
uploaded = st.file_uploader("اختر ملف Excel (.xlsx) — كل عمود = مؤشر/سؤال، كل صف = مستجيب", type=["xlsx", "xls"])

if uploaded is not None:
    df = pd.read_excel(uploaded)
elif "project_df" in st.session_state:
    df = st.session_state["project_df"]
    st.info("تُستخدم حاليًا البيانات المحمَّلة من المشروع المحفوظ. يمكنك رفع ملف آخر لاستبدالها.")
else:
    st.info("⬆️ ارفعي ملف إكسل للبدء. يمكنك أيضًا تجربة التطبيق ببيانات عيّنة.")
    if st.button("تحميل بيانات عيّنة للتجربة"):
        st.session_state["use_sample"] = True
    if not st.session_state.get("use_sample"):
        st.stop()
    df = pd.read_excel("sample_data.xlsx")

st.session_state["current_df"] = df
st.success(f"تم تحميل البيانات: {df.shape[0]} صف × {df.shape[1]} عمود")
with st.expander("👁️ معاينة البيانات"):
    st.dataframe(df.head(20), use_container_width=True)

numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if len(numeric_cols) < len(df.columns):
    st.warning("بعض الأعمدة غير رقمية وسيتم تجاهلها من قائمة المؤشرات المتاحة.")

# ======================================================= 2) بناء نموذج القياس
st.header("2️⃣ نموذج القياس — تحديد المتغيرات الكامنة ومؤشراتها")
st.write("أضيفي متغيرًا كامنًا (Construct) واختاري الأعمدة (المؤشرات) التي تنتمي إليه. "
         "النوع A = انعكاسي (الأكثر شيوعًا)، النوع B = تكويني (يُفعّل معه اختبار VIF تلقائيًا).")

col1, col2, col3, col4 = st.columns([2, 3, 1, 1])
with col1:
    new_lv_name = st.text_input("اسم المتغير الكامن الجديد", key="new_lv_name")
with col2:
    new_lv_inds = st.multiselect("المؤشرات (أعمدة الإكسل)", options=numeric_cols, key="new_lv_inds")
with col3:
    new_lv_mode = st.selectbox("النوع", ["A (انعكاسي)", "B (تكويني)"], key="new_lv_mode")
with col4:
    st.write("")
    st.write("")
    if st.button("➕ إضافة"):
        if new_lv_name and new_lv_inds:
            st.session_state.spec[new_lv_name] = {
                "indicators": new_lv_inds,
                "mode": "A" if "A" in new_lv_mode else "B",
            }
            st.rerun()
        else:
            st.error("اكتبي اسمًا واختاري مؤشرًا واحدًا على الأقل.")

if st.session_state.spec:
    st.subheader("المتغيرات الكامنة المُعرَّفة")
    for lv, s in list(st.session_state.spec.items()):
        c1, c2, c3 = st.columns([2, 5, 1])
        c1.markdown(f"**{lv}** (النوع {s['mode']})")
        c2.write(", ".join(s["indicators"]))
        if c3.button("🗑️ حذف", key=f"del_{lv}"):
            del st.session_state.spec[lv]
            st.session_state.paths = [p for p in st.session_state.paths if lv not in p]
            st.session_state.interactions = [i for i in st.session_state.interactions if lv not in i]
            st.rerun()

# ======================================================= 3) النموذج البنائي
st.header("3️⃣ النموذج البنائي — العلاقات بين المتغيرات الكامنة")
lv_list = list(st.session_state.spec.keys())

if len(lv_list) < 2:
    st.info("عرّفي متغيرين كامنين على الأقل لإضافة علاقات بينهما.")
else:
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        src = st.selectbox("من (المستقل)", lv_list, key="src_lv")
    with c2:
        tgt = st.selectbox("إلى (التابع)", lv_list, key="tgt_lv")
    with c3:
        st.write("")
        st.write("")
        if st.button("➕ إضافة علاقة"):
            if src != tgt and (src, tgt) not in st.session_state.paths:
                st.session_state.paths.append((src, tgt))
                st.rerun()

    if st.session_state.paths:
        st.subheader("العلاقات البنائية المُعرَّفة")
        for i, (s, t) in enumerate(st.session_state.paths):
            c1, c2 = st.columns([6, 1])
            c1.markdown(f"**{s} → {t}**")
            if c2.button("🗑️", key=f"delp_{i}"):
                st.session_state.paths.pop(i)
                st.rerun()

# ======================================================= 3.5) متغيرات التعديل (Moderation)
st.header("3️⃣.5️⃣ متغيرات التعديل (Moderation) — اختياري")
st.caption("يُبنى متغير التفاعل بأسلوب المرحلتين: حاصل ضرب درجات المتغيرين × إضافة مسار جديد نحو المتغير التابع.")

if len(lv_list) < 2 or not st.session_state.paths:
    st.info("عرّفي النموذج البنائي أولًا (خطوة 3) قبل إضافة تعديل.")
else:
    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
    with c1:
        mod_pred = st.selectbox("المتغير المستقل", lv_list, key="mod_pred")
    with c2:
        mod_mod = st.selectbox("المتغير المُعدِّل (Moderator)", lv_list, key="mod_mod")
    with c3:
        mod_tgt = st.selectbox("المتغير التابع المتأثر", lv_list, key="mod_tgt")
    with c4:
        st.write("")
        st.write("")
        if st.button("➕ إضافة تعديل"):
            triple = (mod_pred, mod_mod, mod_tgt)
            if len({mod_pred, mod_mod, mod_tgt}) == 3 and triple not in st.session_state.interactions:
                st.session_state.interactions.append(triple)
                st.rerun()
            else:
                st.error("اختاري ثلاثة متغيرات مختلفة، والتأكد من عدم تكرار نفس التعديل.")

    if st.session_state.interactions:
        st.subheader("متغيرات التعديل المُعرَّفة")
        for i, (p, m, t) in enumerate(st.session_state.interactions):
            c1, c2 = st.columns([6, 1])
            c1.markdown(f"**{p} × {m} → {t}**")
            if c2.button("🗑️", key=f"deli_{i}"):
                st.session_state.interactions.pop(i)
                st.rerun()

# ======================================================= 4) التشغيل
st.header("4️⃣ تشغيل التحليل")

n_boot = st.slider("عدد عيّنات Bootstrapping (لاختبار المعنوية الإحصائية)", 100, 2000, 500, step=100)
run = st.button("🚀 نفّذ تحليل PLS-SEM", type="primary")

if run:
    if len(st.session_state.spec) < 2:
        st.error("عرّفي متغيرين كامنين على الأقل.")
        st.stop()
    if not st.session_state.paths:
        st.error("أضيفي علاقة بنائية واحدة على الأقل.")
        st.stop()

    spec0 = st.session_state.spec
    paths0 = st.session_state.paths
    interactions = st.session_state.interactions

    try:
        with st.spinner("جارٍ تقدير النموذج الأساسي..."):
            base_model = PLSModel(df, spec0, paths0)

        # إضافة متغيرات التعديل إن وُجدت: تُبنى من درجات النموذج الأساسي ثم يُعاد التقدير على النموذج الموسَّع
        if interactions:
            df_aug = df.copy()
            spec_aug = dict(spec0)
            paths_aug = list(paths0)
            for (pred, mod, tgt) in interactions:
                name, vals = base_model.add_interaction_term(pred, mod, tgt)
                df_aug[name] = vals
                spec_aug[name] = {"indicators": [name], "mode": "A"}
                paths_aug.append((name, tgt))
            with st.spinner("جارٍ تقدير النموذج مع متغيرات التعديل..."):
                model = PLSModel(df_aug, spec_aug, paths_aug)
            spec_final, paths_final, df_final = spec_aug, paths_aug, df_aug
        else:
            model = base_model
            spec_final, paths_final, df_final = spec0, paths0, df

    except Exception as e:
        st.error(f"خطأ أثناء التقدير: {e}")
        st.stop()

    with st.spinner(f"جارٍ تنفيذ {n_boot} عيّنة Bootstrap لاختبار المعنوية..."):
        model.bootstrap(n_boot=n_boot)
    model.compute_vif()
    model.mediation_analysis(n_boot=min(n_boot, 500))

    st.session_state["model"] = model
    st.session_state["spec_final"] = spec_final
    st.session_state["paths_final"] = paths_final

    st.success(f"✅ اكتمل التقدير (تقارب الخوارزمية خلال {model.n_iterations} تكرار)")

# ======================================================= عرض النتائج
if "model" in st.session_state:
    model = st.session_state["model"]
    spec = st.session_state["spec_final"]
    paths = st.session_state["paths_final"]

    st.header("📊 النتائج")

    tab_diagram, tab_measure, tab_struct, tab_discrim, tab_vif, tab_med, tab_export = st.tabs(
        ["🗺️ مخطط المسار", "📏 نموذج القياس", "🔗 النموذج البنائي", "🧪 الصلاحية التمييزية",
         "📐 VIF", "🔀 الوساطة", "📥 تصدير"]
    )

    # ---------------- دالة رسم مخطط المسار (تُستخدم في العرض وفي PDF) ----------------
    def draw_path_diagram():
        fig, ax = plt.subplots(figsize=(9, 6))
        G = nx.DiGraph()
        for lv in spec:
            G.add_node(lv)
        for s, t in paths:
            G.add_edge(s, t, weight=model.path_coefficients.get((s, t), 0))

        pos = nx.spring_layout(G, seed=7, k=1.6)
        nx.draw_networkx_nodes(G, pos, node_size=3200, node_color="#4C78A8", ax=ax)
        nx.draw_networkx_labels(G, pos, font_color="white", font_weight="bold", ax=ax)
        edge_labels = {(s, t): f"{model.path_coefficients.get((s, t), 0):.3f}" for s, t in G.edges()}
        nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=22, edge_color="#555",
                                connectionstyle="arc3,rad=0.08", ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="#B23", ax=ax)

        for lv in model.endogenous:
            if lv in pos:
                x, y = pos[lv]
                r2 = model.r_squared.get(lv, np.nan)
                ax.text(x, y - 0.13, f"R²={r2:.3f}", ha="center", color="#0a5", fontsize=10, fontweight="bold")

        ax.set_axis_off()
        return fig

    # ---------------- مخطط المسار ----------------
    with tab_diagram:
        st.subheader("مخطط المسار (Path Diagram)")
        fig = draw_path_diagram()
        st.pyplot(fig)
        diagram_buf = io.BytesIO()
        fig.savefig(diagram_buf, format="png", dpi=150, bbox_inches="tight")
        diagram_buf.seek(0)
        st.session_state["diagram_png"] = diagram_buf.getvalue()

        st.markdown("**المؤشرات وتوزيعاتها (Outer Loadings) لكل متغير كامن:**")
        cols = st.columns(len(spec))
        for col, lv in zip(cols, spec):
            with col:
                st.markdown(f"**{lv}**")
                loadings_df = model.loadings[lv].round(3).sort_values(ascending=False)
                st.dataframe(loadings_df.rename("التوزيع"), use_container_width=True)

    # ---------------- نموذج القياس ----------------
    with tab_measure:
        st.subheader("جودة نموذج القياس (Measurement Model)")

        quality_df = pd.DataFrame({
            "Cronbach's Alpha": model.cronbach_alpha,
            "rho_A (تقريبي)": model.rho_a,
            "الثبات المركّب CR": model.composite_reliability,
            "AVE": model.ave,
        }).round(3)
        st.dataframe(quality_df, use_container_width=True)
        st.caption("القيم المرجعية الشائعة: Alpha/CR > 0.70 مقبول (>0.60 استكشافي) — AVE > 0.50")

        st.markdown("### التوزيعات الخارجية (Outer Loadings) التفصيلية")
        for lv in spec:
            st.markdown(f"**{lv}**")
            fig2, ax2 = plt.subplots(figsize=(6, 2 + 0.3 * len(spec[lv]["indicators"])))
            l = model.loadings[lv].sort_values()
            bar_colors = ["#d62728" if v < 0.7 else "#2ca02c" for v in l.values]
            ax2.barh(l.index, l.values, color=bar_colors)
            ax2.axvline(0.7, color="gray", linestyle="--", linewidth=1)
            ax2.set_xlim(0, 1)
            ax2.set_xlabel("Loading")
            st.pyplot(fig2)

    # ---------------- النموذج البنائي ----------------
    with tab_struct:
        st.subheader("معاملات المسار والمعنوية الإحصائية (Bootstrapping)")

        rows = []
        for (s, t), coef in model.path_coefficients.items():
            bt = model.bootstrap_results.get((s, t), {})
            p_val = bt.get("p", np.nan)
            rows.append({
                "العلاقة": f"{s} → {t}",
                "المعامل (β)": round(coef, 3),
                "الانحراف المعياري": round(bt.get("std", np.nan), 3),
                "قيمة t": round(bt.get("t", np.nan), 3),
                "قيمة p": round(p_val, 4) if not np.isnan(p_val) else np.nan,
                "معنوي (p<0.05)؟": "✅ نعم" if _is_significant(p_val) else "❌ لا",
                "الحد الأدنى CI 95%": round(bt.get("ci_low", np.nan), 3),
                "الحد الأعلى CI 95%": round(bt.get("ci_high", np.nan), 3),
                "f²": round(model.f_squared.get((s, t), np.nan), 3),
            })
        st.session_state["path_rows"] = rows
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption("f²: 0.02 صغير، 0.15 متوسط، 0.35 كبير (Cohen 1988)")

        interaction_rows = [r for r in rows if "_x_" in r["العلاقة"].split(" → ")[0]]
        if interaction_rows:
            st.markdown("### أثر متغيرات التعديل (Moderation)")
            st.dataframe(pd.DataFrame(interaction_rows), use_container_width=True)
            st.caption("إن كان معامل مسار التفاعل (Predictor_x_Moderator) معنويًا، فهذا دليل على وجود أثر تعديل حقيقي.")

        st.markdown("### معامل التحديد R² و R² المعدَّل")
        r2_df = pd.DataFrame({
            "R²": model.r_squared,
            "R² المعدَّل": model.r_squared_adj,
        }).round(3)
        st.session_state["r2_df"] = r2_df
        st.dataframe(r2_df, use_container_width=True)
        st.caption("القيم المرجعية الشائعة (Chin 1998): 0.67 قوي، 0.33 متوسط، 0.19 ضعيف")

    # ---------------- الصلاحية التمييزية ----------------
    with tab_discrim:
        st.subheader("معيار Fornell-Larcker")
        st.write("القطر = الجذر التربيعي لـ AVE، ويجب أن يكون أكبر من الارتباطات في نفس الصف/العمود.")
        st.dataframe(model.fornell_larcker.round(3), use_container_width=True)

        st.subheader("نسبة HTMT (Heterotrait-Monotrait Ratio)")
        st.write("القيم المرجعية: يُفضَّل أن تكون أقل من 0.90 (أو 0.85 للمفاهيم المتقاربة نظريًا).")
        htmt_display = model.htmt.round(3)
        st.dataframe(htmt_display, use_container_width=True)

        fig3, ax3 = plt.subplots(figsize=(5, 4))
        im = ax3.imshow(model.htmt.astype(float).fillna(0), cmap="RdYlGn_r", vmin=0, vmax=1)
        ax3.set_xticks(range(len(model.lv_names)))
        ax3.set_yticks(range(len(model.lv_names)))
        ax3.set_xticklabels(model.lv_names, rotation=45, ha="right")
        ax3.set_yticklabels(model.lv_names)
        for i in range(len(model.lv_names)):
            for j in range(len(model.lv_names)):
                v = model.htmt.iloc[i, j]
                if not np.isnan(v):
                    ax3.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)
        fig3.colorbar(im, ax=ax3, shrink=0.8)
        st.pyplot(fig3)

    # ---------------- VIF ----------------
    with tab_vif:
        st.subheader("اختبار التعدد الخطي البنائي (Structural VIF)")
        st.caption("يُحسب بين المتغيرات المستقلة التي تؤثر في نفس المتغير التابع. القيمة المرجعية: VIF < 5 (يُفضَّل < 3.3).")
        if model.inner_vif:
            for tgt, s in model.inner_vif.items():
                st.markdown(f"**التأثير على {tgt}**")
                st.dataframe(s.round(3).rename("VIF"), use_container_width=True)
        else:
            st.info("لا توجد معادلة بنائية بأكثر من متغير مستقل واحد، فلا حاجة لاختبار VIF بنائي هنا.")

        st.subheader("اختبار التعدد الخطي للمؤشرات التكوينية (Outer VIF — Mode B)")
        if model.outer_vif:
            for lv, s in model.outer_vif.items():
                st.markdown(f"**{lv}**")
                vif_df = s.round(3).rename("VIF").to_frame()
                st.dataframe(vif_df, use_container_width=True)
                if (s > 5).any():
                    st.warning(f"⚠️ بعض مؤشرات {lv} لديها VIF > 5 — قد تحتاجين لدمج/حذف مؤشرات متشابهة.")
        else:
            st.info("لا يوجد متغير كامن مُعرَّف بالنوع B (تكويني) في هذا النموذج.")

    # ---------------- الوساطة ----------------
    with tab_med:
        st.subheader("تحليل الوساطة (Mediation Analysis)")
        st.caption("يُكتشف تلقائيًا كل سلسلة X → M → Y موجودة في نموذجك البنائي، ويُحسب الأثر المباشر وغير المباشر و VAF.")
        med = getattr(model, "mediation_results", [])
        if med:
            med_rows = []
            for r in med:
                med_rows.append({
                    "X": r["x"], "M (الوسيط)": r["m"], "Y": r["y"],
                    "a (X→M)": round(r["a"], 3), "b (M→Y)": round(r["b"], 3),
                    "الأثر المباشر": round(r["direct"], 3),
                    "الأثر غير المباشر": round(r["indirect"], 3),
                    "الأثر الكلي": round(r["total"], 3),
                    "VAF": f"{r['vaf']*100:.1f}%" if not np.isnan(r["vaf"]) else "-",
                    "قيمة t": round(r["t"], 3) if not np.isnan(r["t"]) else "-",
                    "قيمة p": round(r["p"], 4) if not np.isnan(r["p"]) else "-",
                    "معنوي؟": "✅ نعم" if _is_significant(r["p"]) else "❌ لا",
                })
            st.dataframe(pd.DataFrame(med_rows), use_container_width=True)
            st.caption("VAF: أقل من 20% = لا وساطة تقريبًا | 20–80% = وساطة جزئية | أكثر من 80% = وساطة كاملة (Hair et al.)")
        else:
            st.info("لم يُكتشف أي سلسلة وساطة (X → M → Y) في نموذجك البنائي الحالي.")

    # ---------------- التصدير ----------------
    with tab_export:
        st.subheader("تصدير النتائج")

        rows = st.session_state.get("path_rows", [])
        r2_df = st.session_state.get("r2_df", pd.DataFrame())
        quality_df = pd.DataFrame({
            "Cronbach's Alpha": model.cronbach_alpha,
            "rho_A (تقريبي)": model.rho_a,
            "الثبات المركّب CR": model.composite_reliability,
            "AVE": model.ave,
        }).round(3)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            pd.DataFrame(rows).to_excel(writer, sheet_name="Path Coefficients", index=False)
            r2_df.to_excel(writer, sheet_name="R2")
            quality_df.to_excel(writer, sheet_name="Measurement Quality")
            model.fornell_larcker.to_excel(writer, sheet_name="Fornell-Larcker")
            model.htmt.to_excel(writer, sheet_name="HTMT")
            for lv in spec:
                model.loadings[lv].to_frame("Loading").to_excel(writer, sheet_name=f"Loadings_{lv}"[:31])
            if getattr(model, "mediation_results", None):
                pd.DataFrame(model.mediation_results).to_excel(writer, sheet_name="Mediation", index=False)
            for lv, s in getattr(model, "outer_vif", {}).items():
                s.to_frame("VIF").to_excel(writer, sheet_name=f"VIF_{lv}"[:31])
            model.scores.to_excel(writer, sheet_name="LV Scores", index=False)

        st.download_button(
            "⬇️ تحميل كل النتائج كملف Excel",
            data=output.getvalue(),
            file_name="PLS_SEM_Results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # ------------- PDF -------------
        if st.button("📄 توليد تقرير PDF"):
            with st.spinner("جارٍ توليد التقرير..."):
                diagram_png = st.session_state.get("diagram_png")
                pdf_buf = io.BytesIO()
                build_report(
                    pdf_buf, model, spec, paths, quality_df, r2_df, rows,
                    model.htmt, model.fornell_larcker,
                    diagram_png_bytes=io.BytesIO(diagram_png) if diagram_png else None,
                    mediation_rows=getattr(model, "mediation_results", None),
                    vif_outer=getattr(model, "outer_vif", None),
                    vif_inner=getattr(model, "inner_vif", None),
                )
                st.session_state["pdf_bytes"] = pdf_buf.getvalue()

        if "pdf_bytes" in st.session_state:
            st.download_button(
                "⬇️ تحميل تقرير PDF",
                data=st.session_state["pdf_bytes"],
                file_name="PLS_SEM_Report.pdf",
                mime="application/pdf",
            )

        model_json = {
            "model_spec": spec,
            "structural_paths": paths,
            "interactions": st.session_state.interactions,
        }
        st.download_button(
            "⬇️ تحميل تعريف النموذج (JSON) لإعادة الاستخدام لاحقًا",
            data=json.dumps(model_json, ensure_ascii=False, indent=2),
            file_name="model_definition.json",
            mime="application/json",
        )

st.markdown("---")
st.caption("أداة مفتوحة المصدر لتحليل PLS-SEM — مبنية بلغة Python (خوارزمية Lohmöller / path weighting scheme). "
           "للاستخدام الأكاديمي، يُنصح بمراجعة النتائج مقابل مرجع Hair et al. (A Primer on PLS-SEM).")
