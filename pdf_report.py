# -*- coding: utf-8 -*-
"""
توليد تقرير PDF جاهز بنتائج تحليل PLS-SEM (نص عربي + جداول + الرسم البياني لمخطط المسار).
يستخدم reportlab + arabic_reshaper + python-bidi لعرض النص العربي بشكل صحيح (RTL + الأشكال المتصلة).
"""

import io
import os

import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts", "NotoNaskhArabic.ttf")
FONT_NAME = "NotoNaskh"

_registered = False


def _ensure_font():
    global _registered
    if not _registered:
        pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))
        _registered = True


def rtl(text) -> str:
    """يهيّئ النص العربي (يربط الحروف ويعكس اتجاهه) ليُعرض بشكل صحيح في PDF.
    النصوص/الأرقام اللاتينية داخل النص تبقى بترتيبها الصحيح تلقائيًا عبر bidi."""
    text = str(text)
    return get_display(arabic_reshaper.reshape(text))


class ArabicPDFReport:
    def __init__(self, path_or_buffer, title="تقرير تحليل PLS-SEM"):
        _ensure_font()
        self.c = canvas.Canvas(path_or_buffer, pagesize=A4)
        self.width, self.height = A4
        self.margin = 2 * cm
        self.y = self.height - self.margin
        self.title = title
        self._new_page(first=True)

    # ---------------------------------------------------------- helpers
    def _new_page(self, first=False):
        if not first:
            self.c.showPage()
        self.y = self.height - self.margin
        self.c.setFont(FONT_NAME, 10)

    def _check_space(self, needed):
        if self.y - needed < self.margin:
            self._new_page()

    def heading(self, text, size=16, space_after=0.9 * cm):
        self._check_space(size + space_after)
        self.c.setFont(FONT_NAME, size)
        self.c.setFillColor(colors.HexColor("#1f3864"))
        self.c.drawRightString(self.width - self.margin, self.y, rtl(text))
        self.c.setFillColor(colors.black)
        self.y -= size + space_after

    def subheading(self, text, size=12.5, space_after=0.6 * cm):
        self._check_space(size + space_after)
        self.c.setFont(FONT_NAME, size)
        self.c.setFillColor(colors.HexColor("#4C78A8"))
        self.c.drawRightString(self.width - self.margin, self.y, rtl(text))
        self.c.setFillColor(colors.black)
        self.y -= size + space_after

    def paragraph(self, text, size=10, leading=15):
        self._check_space(leading)
        self.c.setFont(FONT_NAME, size)
        self.c.drawRightString(self.width - self.margin, self.y, rtl(text))
        self.y -= leading

    def spacer(self, h=0.4 * cm):
        self.y -= h

    def table(self, headers, rows, col_widths=None, size=9, row_h=0.62 * cm):
        """جدول بسيط RTL: العمود الأول من اليمين. headers/rows نصوص جاهزة (سيتم تهيئتها عربيًا تلقائيًا)."""
        n_cols = len(headers)
        total_w = self.width - 2 * self.margin
        if col_widths is None:
            col_widths = [total_w / n_cols] * n_cols
        self._check_space(row_h * (len(rows) + 1) if len(rows) < 15 else row_h * 3)

        def draw_row(values, y, bold=False, fill=None):
            x = self.width - self.margin
            if fill:
                self.c.setFillColor(fill)
                self.c.rect(self.width - self.margin - sum(col_widths), y - row_h + 4,
                            sum(col_widths), row_h, fill=1, stroke=0)
                self.c.setFillColor(colors.black)
            self.c.setFont(FONT_NAME, size)
            for val, w in zip(values, col_widths):
                x -= w
                self.c.drawCentredString(x + w / 2, y - row_h + 12, rtl(val))
            self.c.setStrokeColor(colors.HexColor("#cccccc"))
            self.c.line(self.width - self.margin, y - row_h, self.width - self.margin - sum(col_widths), y - row_h)

        draw_row(headers, self.y, fill=colors.HexColor("#DDE6F0"))
        self.y -= row_h
        for i, r in enumerate(rows):
            self._check_space(row_h)
            draw_row(r, self.y, fill=colors.HexColor("#F7F9FC") if i % 2 == 0 else None)
            self.y -= row_h
        self.spacer(0.3 * cm)

    def image(self, img_bytes_or_path, max_w=None, max_h=9 * cm):
        from reportlab.lib.utils import ImageReader
        img = ImageReader(img_bytes_or_path)
        iw, ih = img.getSize()
        max_w = max_w or (self.width - 2 * self.margin)
        scale = min(max_w / iw, max_h / ih)
        w, h = iw * scale, ih * scale
        self._check_space(h + 0.5 * cm)
        x = self.width - self.margin - w
        self.c.drawImage(img, x, self.y - h, width=w, height=h, preserveAspectRatio=True, mask="auto")
        self.y -= h + 0.5 * cm

    def save(self):
        self.c.showPage()
        self.c.save()


def build_report(buffer, model, spec, paths, quality_df, r2_df, path_rows,
                  htmt_df, fl_df, diagram_png_bytes=None, mediation_rows=None,
                  vif_outer=None, vif_inner=None):
    report = ArabicPDFReport(buffer)
    report.heading("تقرير تحليل PLS-SEM", size=18)
    report.paragraph(f"عدد المشاهدات: {len(model.raw_data)}    |    عدد المتغيرات الكامنة: {len(spec)}    |    عدد العلاقات البنائية: {len(paths)}")
    report.spacer(0.5 * cm)

    if diagram_png_bytes is not None:
        report.subheading("مخطط المسار")
        report.image(diagram_png_bytes)

    # ملاحظة على ترتيب الأعمدة: الدالة table() ترسم أول عنصر في القائمة في أقصى
    # يمين الجدول (لأن القراءة العربية تبدأ من اليمين)، لذا نضع عمود "التعريف"
    # (اسم المتغير/العلاقة) أولًا في كل قائمة headers/rows هنا.

    report.subheading("جودة نموذج القياس")
    headers = ["المتغير", "Cronbach's Alpha", "rho_A", "الثبات المركّب CR", "AVE"]
    rows = []
    for lv in quality_df.index:
        r = quality_df.loc[lv]
        alpha_val = r["Cronbach's Alpha"]
        rows.append([lv, f"{alpha_val:.3f}", f"{r['rho_A (تقريبي)']:.3f}",
                     f"{r['الثبات المركّب CR']:.3f}", f"{r['AVE']:.3f}"])
    report.table(headers, rows)

    report.subheading("معاملات المسار والمعنوية الإحصائية")
    headers2 = ["العلاقة", "المعامل β", "قيمة t", "قيمة p", "معنوي؟", "f²"]
    rows2 = [[r["العلاقة"], str(r["المعامل (β)"]), str(r["قيمة t"]), str(r["قيمة p"]),
              r["معنوي (p<0.05)؟"], str(r["f²"])] for r in path_rows]
    report.table(headers2, rows2)

    report.subheading("معامل التحديد R²")
    headers3 = ["المتغير التابع", "R²", "R² المعدَّل"]
    rows3 = [[t, f"{r2_df.loc[t, 'R²']:.3f}", f"{r2_df.loc[t, 'R² المعدَّل']:.3f}"] for t in r2_df.index]
    report.table(headers3, rows3)

    report.subheading("معيار Fornell-Larcker للصلاحية التمييزية")
    fl_headers = [""] + list(fl_df.columns)
    fl_rows = []
    for lv in fl_df.index:
        row_vals = [lv] + [f"{fl_df.loc[lv, c]:.3f}" for c in fl_df.columns]
        fl_rows.append(row_vals)
    report.table(fl_headers, fl_rows, size=8)

    report.subheading("نسبة HTMT")
    ht_headers = [""] + list(htmt_df.columns)
    ht_rows = []
    for lv in htmt_df.index:
        row_vals = [lv] + [("-" if np.isnan(htmt_df.loc[lv, c]) else f"{htmt_df.loc[lv, c]:.3f}") for c in htmt_df.columns]
        ht_rows.append(row_vals)
    report.table(ht_headers, ht_rows, size=8)

    if vif_outer:
        report.subheading("اختبار التعدد الخطي (VIF) — المؤشرات التكوينية")
        for lv, s in vif_outer.items():
            report.paragraph(f"{lv}:")
            rows_v = [[i, f"{v:.2f}"] for i, v in s.items()]
            report.table(["المؤشر", "VIF"], rows_v, size=9)

    if vif_inner:
        report.subheading("اختبار التعدد الخطي (VIF) البنائي")
        for tgt, s in vif_inner.items():
            report.paragraph(f"التأثير على {tgt}:")
            rows_v = [[i, f"{v:.2f}"] for i, v in s.items()]
            report.table(["المتغير المستقل", "VIF"], rows_v, size=9)

    if mediation_rows:
        report.subheading("تحليل الوساطة (Mediation)")
        headers_m = ["X", "Y", "M", "مباشر", "غير مباشر", "VAF"]
        rows_m = [[
            r["x"], r["y"], r["m"], f"{r['direct']:.3f}", f"{r['indirect']:.3f}",
            ("-" if np.isnan(r["vaf"]) else f"{r['vaf']*100:.1f}%"),
        ] for r in mediation_rows]
        report.table(headers_m, rows_m, size=8)

    report.spacer(1 * cm)
    report.paragraph("تم إنشاء هذا التقرير تلقائيًا بواسطة أداة PLS-SEM مفتوحة المصدر.", size=8)

    report.save()
    return buffer
