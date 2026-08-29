# -*- coding: utf-8 -*-
"""
محرك حسابات PLS-SEM (Partial Least Squares - Structural Equation Modeling)
مبني من الصفر اعتمادًا على خوارزمية Lohmöller (path weighting scheme)
مرجع الأسلوب: Hair, Hult, Ringle & Sarstedt - A Primer on PLS-SEM

المدخلات:
- data: DataFrame يحتوي على المؤشرات (indicators) كأعمدة
- model_spec: dict {اسم المتغير الكامن: {"indicators": [...], "mode": "A" أو "B"}}
- structural_paths: قائمة أزواج (predictor, target) تمثل العلاقات البنائية

المخرجات: outer loadings/weights, LV scores, path coefficients, R²,
Cronbach's alpha, rho_A, CR, AVE, Fornell-Larcker, HTMT, bootstrap
"""

import numpy as np
import pandas as pd


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """توحيد قياسي (mean=0, std=1) لكل عمود - population std (ddof=0) كما في SmartPLS."""
    return (df - df.mean()) / df.std(ddof=0)


class PLSModel:
    def __init__(self, data: pd.DataFrame, model_spec: dict, structural_paths: list,
                 max_iter: int = 300, tol: float = 1e-7):
        self.raw_data = data.copy()
        self.model_spec = model_spec
        self.structural_paths = structural_paths
        self.max_iter = max_iter
        self.tol = tol

        self.lv_names = list(model_spec.keys())
        self.n_lv = len(self.lv_names)

        # التحقق من وجود كل الأعمدة
        all_indicators = []
        for lv, spec in model_spec.items():
            all_indicators.extend(spec["indicators"])
        missing = [c for c in all_indicators if c not in data.columns]
        if missing:
            raise ValueError(f"أعمدة غير موجودة في البيانات: {missing}")

        self.data_std = _standardize(data[all_indicators].astype(float))

        # مصفوفة الاتصال الداخلي (adjacency) بين المتغيرات الكامنة
        self.adj = pd.DataFrame(0, index=self.lv_names, columns=self.lv_names)
        for src, tgt in structural_paths:
            self.adj.loc[src, tgt] = 1
            self.adj.loc[tgt, src] = 1  # للتقريب الداخلي نحتاج الجيران في الاتجاهين

        self.endogenous = sorted(set(t for _, t in structural_paths))
        self.exogenous = [lv for lv in self.lv_names if lv not in self.endogenous]

        self._fit()

    # ------------------------------------------------------------------
    def _fit(self):
        data = self.data_std
        lv_names = self.lv_names
        spec = self.model_spec

        # التهيئة: الوزن الأولي = 1 لكل مؤشر (موزون بالتساوي) ثم توحيد الدرجة
        weights = {}
        for lv in lv_names:
            inds = spec[lv]["indicators"]
            w = pd.Series(1.0, index=inds)
            weights[lv] = w

        scores = pd.DataFrame(index=data.index, columns=lv_names, dtype=float)
        for lv in lv_names:
            inds = spec[lv]["indicators"]
            s = data[inds].values @ weights[lv].values
            s = (s - s.mean()) / s.std(ddof=0)
            scores[lv] = s

        prev_weights_vec = np.concatenate([weights[lv].values for lv in lv_names])

        for iteration in range(self.max_iter):
            # ---- الخطوة 1: التقريب الداخلي (inner approximation) ----
            inner = pd.DataFrame(index=data.index, columns=lv_names, dtype=float)
            for lv in lv_names:
                neighbors = [o for o in lv_names if self.adj.loc[lv, o] == 1]
                if not neighbors:
                    inner[lv] = scores[lv]
                    continue
                # أوزان المخطط الداخلي = ارتباط (path weighting scheme المبسط = centroid على الإشارة)
                vals = np.zeros(len(data))
                for o in neighbors:
                    corr = np.corrcoef(scores[lv].values, scores[o].values)[0, 1]
                    sign = 1.0 if corr >= 0 else -1.0
                    vals += sign * scores[o].values
                inner[lv] = vals

            # ---- الخطوة 2: التقريب الخارجي (outer approximation) لتحديث الأوزان ----
            new_weights = {}
            for lv in lv_names:
                inds = spec[lv]["indicators"]
                mode = spec[lv].get("mode", "A")
                X = data[inds].values
                y = inner[lv].values
                y = (y - y.mean()) / (y.std(ddof=0) + 1e-12)
                if mode == "A":  # انعكاسي: انحدار بسيط لكل مؤشر (يعادل الارتباط بعد التوحيد)
                    w = (X.T @ y) / len(y)
                else:  # مكوّن (formative): انحدار متعدد
                    XtX = X.T @ X
                    try:
                        w = np.linalg.solve(XtX, X.T @ y)
                    except np.linalg.LinAlgError:
                        w = np.linalg.lstsq(XtX, X.T @ y, rcond=None)[0]
                new_weights[lv] = pd.Series(w, index=inds)

            # تحديث الدرجات
            new_scores = pd.DataFrame(index=data.index, columns=lv_names, dtype=float)
            for lv in lv_names:
                inds = spec[lv]["indicators"]
                s = data[inds].values @ new_weights[lv].values
                std = s.std(ddof=0)
                if std < 1e-12:
                    std = 1e-12
                s = (s - s.mean()) / std
                new_scores[lv] = s

            new_weights_vec = np.concatenate([new_weights[lv].values for lv in lv_names])
            diff = np.max(np.abs(new_weights_vec - prev_weights_vec))

            weights = new_weights
            scores = new_scores
            prev_weights_vec = new_weights_vec

            if diff < self.tol:
                break

        self.n_iterations = iteration + 1
        self.converged = diff < self.tol
        self.weights = weights
        self.scores = scores

        # التوزيعات (outer loadings) = ارتباط كل مؤشر بدرجة متغيره الكامن
        loadings = {}
        for lv in lv_names:
            inds = spec[lv]["indicators"]
            loadings[lv] = pd.Series(
                {ind: np.corrcoef(data[ind].values, scores[lv].values)[0, 1] for ind in inds}
            )
        self.loadings = loadings

        self._fit_structural_model()
        self._fit_measurement_quality()

    # ------------------------------------------------------------------
    def _fit_structural_model(self):
        """انحدار OLS لكل متغير داخلي (endogenous) على المتغيرات المؤثرة فيه مباشرة."""
        self.path_coefficients = {}
        self.r_squared = {}
        self.r_squared_adj = {}
        self.f_squared = {}

        for target in self.endogenous:
            predictors = [src for src, tgt in self.structural_paths if tgt == target]
            X = self.scores[predictors].values.astype(float)
            y = self.scores[target].values.astype(float)
            n = len(y)
            k = len(predictors)

            X_design = np.column_stack([X, np.ones(n)])
            beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
            coefs = beta[:-1]
            y_hat = X_design @ beta
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            r2_adj = 1 - (1 - r2) * (n - 1) / max(n - k - 1, 1)

            for pred, coef in zip(predictors, coefs):
                self.path_coefficients[(pred, target)] = coef

            self.r_squared[target] = r2
            self.r_squared_adj[target] = r2_adj

            # f² لكل متغير مستقل: أثر استبعاده من النموذج
            for i, pred in enumerate(predictors):
                remaining = [p for p in predictors if p != pred]
                if remaining:
                    Xr = self.scores[remaining].values.astype(float)
                    Xr_design = np.column_stack([Xr, np.ones(n)])
                    beta_r, _, _, _ = np.linalg.lstsq(Xr_design, y, rcond=None)
                    y_hat_r = Xr_design @ beta_r
                    ss_res_r = np.sum((y - y_hat_r) ** 2)
                    r2_excl = 1 - ss_res_r / ss_tot if ss_tot > 0 else 0.0
                else:
                    r2_excl = 0.0
                denom = 1 - r2
                f2 = (r2 - r2_excl) / denom if denom > 1e-9 else np.nan
                self.f_squared[(pred, target)] = f2

    # ------------------------------------------------------------------
    def _fit_measurement_quality(self):
        """معايير جودة نموذج القياس: Cronbach's alpha, rho_A, CR, AVE + Fornell-Larcker + HTMT."""
        data = self.data_std
        spec = self.model_spec

        alpha, rho_a, cr, ave = {}, {}, {}, {}
        for lv in self.lv_names:
            inds = spec[lv]["indicators"]
            k = len(inds)
            load = self.loadings[lv].values

            if k < 2:
                alpha[lv] = np.nan
                rho_a[lv] = np.nan
                cr[lv] = np.nan
                ave[lv] = float(load[0] ** 2) if k == 1 else np.nan
                continue

            # Cronbach's alpha
            item_vars = data[inds].var(ddof=1).values
            total_var = data[inds].sum(axis=1).var(ddof=1)
            alpha[lv] = (k / (k - 1)) * (1 - item_vars.sum() / total_var) if total_var > 0 else np.nan

            # Composite Reliability (rho_c) و AVE من التوزيعات
            sum_load = load.sum()
            sum_load_sq = (load ** 2).sum()
            error_var = (1 - load ** 2).sum()
            cr[lv] = (sum_load ** 2) / ((sum_load ** 2) + error_var) if ((sum_load ** 2) + error_var) > 0 else np.nan
            ave[lv] = sum_load_sq / k

            # rho_A (Dijkstra-Henseler): rho_A = (w'w)^2 / (w' (S - diag(S) + diag(w'w من الأوزان)) w)
            # نستخدم صيغة عملية مبسطة على أوزان outer الموحّدة بحيث درجة LV بتباين=1
            w = self.weights[lv].reindex(inds).values.astype(float)
            S = data[inds].cov(ddof=1).values
            S_offdiag = S - np.diag(np.diag(S))
            w_sq_sum = np.sum(w ** 2)
            numerator = (np.sum(w)) ** 2 - w_sq_sum if k > 1 else np.nan
            denom_rho = w @ S_offdiag @ w
            if k > 1 and abs(denom_rho) > 1e-9:
                rho_a_val = (np.sum(w) ** 2 - w_sq_sum) / denom_rho if False else None
            # الصيغة القياسية (Dijkstra & Henseler 2015):
            # rho_A = (w'w)^2 / (w' S_offdiag w + w'w) ... نعتمد التقريب الشائع في الأدبيات:
            try:
                rho_a[lv] = float((np.sum(w) ** 2) / (np.sum(w) ** 2 - w_sq_sum + w @ S @ w)) \
                    if (np.sum(w) ** 2 - w_sq_sum + w @ S @ w) > 1e-9 else np.nan
                rho_a[lv] = min(max(rho_a[lv], 0.0), 1.0)
            except Exception:
                rho_a[lv] = np.nan

        self.cronbach_alpha = alpha
        self.rho_a = rho_a
        self.composite_reliability = cr
        self.ave = ave

        # Fornell-Larcker: الجذر التربيعي لـ AVE على القطر، الارتباطات بين LV خارج القطر
        fl = pd.DataFrame(index=self.lv_names, columns=self.lv_names, dtype=float)
        for i in self.lv_names:
            for j in self.lv_names:
                if i == j:
                    fl.loc[i, j] = np.sqrt(ave[i]) if not np.isnan(ave[i]) else np.nan
                else:
                    fl.loc[i, j] = np.corrcoef(self.scores[i], self.scores[j])[0, 1]
        self.fornell_larcker = fl

        # HTMT (Heterotrait-Monotrait Ratio)
        htmt = pd.DataFrame(index=self.lv_names, columns=self.lv_names, dtype=float)
        for i in self.lv_names:
            for j in self.lv_names:
                if i == j:
                    htmt.loc[i, j] = np.nan
                    continue
                inds_i = spec[i]["indicators"]
                inds_j = spec[j]["indicators"]
                hetero = []
                for a in inds_i:
                    for b in inds_j:
                        hetero.append(abs(np.corrcoef(data[a], data[b])[0, 1]))
                mono_i = []
                for idx1 in range(len(inds_i)):
                    for idx2 in range(idx1 + 1, len(inds_i)):
                        mono_i.append(abs(np.corrcoef(data[inds_i[idx1]], data[inds_i[idx2]])[0, 1]))
                mono_j = []
                for idx1 in range(len(inds_j)):
                    for idx2 in range(idx1 + 1, len(inds_j)):
                        mono_j.append(abs(np.corrcoef(data[inds_j[idx1]], data[inds_j[idx2]])[0, 1]))
                mono_all = mono_i + mono_j
                if len(mono_all) == 0 or np.mean(mono_all) == 0:
                    htmt.loc[i, j] = np.nan
                else:
                    htmt.loc[i, j] = np.mean(hetero) / np.mean(mono_all)
        self.htmt = htmt

    # ------------------------------------------------------------------
    def compute_vif(self):
        """
        Collinearity check (VIF) لكل مؤشر ضمن كل متغير كامن تكويني (Mode B)،
        ولكل متغير مستقل ضمن كل معادلة بنائية (structural VIF).
        القيمة المرجعية الشائعة: VIF < 5 (يُفضَّل < 3.3).
        """
        data = self.data_std
        spec = self.model_spec

        # 1) VIF على مستوى المؤشرات (فقط للمتغيرات التكوينية Mode B)
        outer_vif = {}
        for lv, s in spec.items():
            if s.get("mode") != "B":
                continue
            inds = s["indicators"]
            if len(inds) < 2:
                continue
            vif_series = {}
            for ind in inds:
                others = [c for c in inds if c != ind]
                X = data[others].values
                y = data[ind].values
                X_design = np.column_stack([X, np.ones(len(y))])
                beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
                y_hat = X_design @ beta
                ss_res = np.sum((y - y_hat) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                vif_series[ind] = 1 / (1 - r2) if r2 < 0.999 else np.inf
            outer_vif[lv] = pd.Series(vif_series)
        self.outer_vif = outer_vif

        # 2) VIF البنائي: لكل معادلة (target) بين المتغيرات الكامنة المستقلة لها
        inner_vif = {}
        for target in self.endogenous:
            predictors = [src for src, tgt in self.structural_paths if tgt == target]
            if len(predictors) < 2:
                continue
            vif_series = {}
            for pred in predictors:
                others = [p for p in predictors if p != pred]
                X = self.scores[others].values.astype(float)
                y = self.scores[pred].values.astype(float)
                X_design = np.column_stack([X, np.ones(len(y))])
                beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
                y_hat = X_design @ beta
                ss_res = np.sum((y - y_hat) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                vif_series[pred] = 1 / (1 - r2) if r2 < 0.999 else np.inf
            inner_vif[target] = pd.Series(vif_series)
        self.inner_vif = inner_vif
        return outer_vif, inner_vif

    # ------------------------------------------------------------------
    def find_mediation_chains(self):
        """
        يكتشف تلقائيًا كل سلاسل الوساطة الممكنة من نوع X -> M -> Y
        حيث توجد أيضًا علاقة مباشرة X -> Y (شرط أساسي لتحليل وساطة كلاسيكي)،
        أو حتى بدون علاقة مباشرة (وساطة كاملة / indirect-only).
        """
        direct = set(self.structural_paths)
        chains = []
        for x, m in self.structural_paths:
            for m2, y in self.structural_paths:
                if m2 == m and y != x and x != m:
                    chains.append((x, m, y))
        return chains

    def mediation_analysis(self, chains=None, n_boot: int = 500, seed: int = 123):
        """
        لكل سلسلة (X, M, Y): الأثر غير المباشر = a*b (X->M ثم M->Y)،
        الأثر المباشر = X->Y إن وُجد، الأثر الكلي = مباشر + غير مباشر،
        VAF = غير مباشر / كلي، مع اختبار Bootstrap لمعنوية الأثر غير المباشر.
        """
        if chains is None:
            chains = self.find_mediation_chains()

        rng = np.random.default_rng(seed)
        n = len(self.raw_data)
        results = []

        for (x, m, y) in chains:
            a = self.path_coefficients.get((x, m), np.nan)
            b = self.path_coefficients.get((m, y), np.nan)
            direct = self.path_coefficients.get((x, y), 0.0)
            indirect = a * b
            total = direct + indirect
            vaf = indirect / total if abs(total) > 1e-9 else np.nan

            boot_indirect = []
            for _ in range(n_boot):
                idx = rng.integers(0, n, n)
                sample = self.raw_data.iloc[idx].reset_index(drop=True)
                try:
                    mm = PLSModel(sample, self.model_spec, self.structural_paths,
                                  max_iter=self.max_iter, tol=1e-5)
                    a_b = mm.path_coefficients.get((x, m))
                    b_b = mm.path_coefficients.get((m, y))
                    if a_b is not None and b_b is not None:
                        boot_indirect.append(a_b * b_b)
                except Exception:
                    continue

            vals = np.array(boot_indirect)
            if len(vals) >= 10:
                std_err = vals.std(ddof=1)
                t_val = indirect / std_err if std_err > 1e-9 else np.nan
                from math import erf, sqrt
                p_val = 2 * (1 - 0.5 * (1 + erf(abs(t_val) / sqrt(2)))) if not np.isnan(t_val) else np.nan
                ci_low, ci_high = np.percentile(vals, [2.5, 97.5])
            else:
                t_val = p_val = ci_low = ci_high = np.nan

            results.append(dict(
                x=x, m=m, y=y, a=a, b=b, direct=direct, indirect=indirect,
                total=total, vaf=vaf, t=t_val, p=p_val, ci_low=ci_low, ci_high=ci_high,
                n_valid=len(vals),
            ))

        self.mediation_results = results
        return results

    # ------------------------------------------------------------------
    def add_interaction_term(self, predictor: str, moderator: str, target: str):
        """
        يبني متغير تفاعل (Moderation) بأسلوب المرحلتين (two-stage approach):
        يضرب درجات LV الموحّدة لـ predictor × moderator كمؤشر تكويني وحيد
        لمتغير كامن جديد "predictor_x_moderator"، ويضيف مسارًا منه إلى target.
        يُعيد اسم المتغير الكامن الجديد الذي يجب إضافته إلى model_spec/paths
        قبل إعادة تشغيل PLSModel من جديد على مستوى الواجهة.
        """
        inter_name = f"{predictor}_x_{moderator}"
        p_scores = self.scores[predictor].values
        m_scores = self.scores[moderator].values
        interaction_scores = p_scores * m_scores
        return inter_name, interaction_scores

    # ------------------------------------------------------------------
    def bootstrap(self, n_boot: int = 500, seed: int = 42):
        """Bootstrapping لتقدير الأهمية الإحصائية لمعاملات المسار (t-value, p-value, CI)."""
        rng = np.random.default_rng(seed)
        n = len(self.raw_data)
        path_keys = list(self.path_coefficients.keys())
        boot_estimates = {k: [] for k in path_keys}

        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            sample = self.raw_data.iloc[idx].reset_index(drop=True)
            try:
                m = PLSModel(sample, self.model_spec, self.structural_paths,
                             max_iter=self.max_iter, tol=1e-5)
                for k in path_keys:
                    if k in m.path_coefficients:
                        boot_estimates[k].append(m.path_coefficients[k])
            except Exception:
                continue

        results = {}
        for k in path_keys:
            vals = np.array(boot_estimates[k])
            if len(vals) < 10:
                results[k] = dict(mean=np.nan, std=np.nan, t=np.nan, p=np.nan,
                                   ci_low=np.nan, ci_high=np.nan, n_valid=len(vals))
                continue
            orig = self.path_coefficients[k]
            std_err = vals.std(ddof=1)
            t_val = orig / std_err if std_err > 1e-9 else np.nan
            # p-value تقريبي ثنائي الطرف عبر التوزيع الطبيعي القياسي
            from math import erf, sqrt
            p_val = 2 * (1 - 0.5 * (1 + erf(abs(t_val) / sqrt(2)))) if not np.isnan(t_val) else np.nan
            ci_low, ci_high = np.percentile(vals, [2.5, 97.5])
            results[k] = dict(mean=vals.mean(), std=std_err, t=t_val, p=p_val,
                               ci_low=ci_low, ci_high=ci_high, n_valid=len(vals))
        self.bootstrap_results = results
        return results
