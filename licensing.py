# -*- coding: utf-8 -*-
"""
نظام أكواد الترخيص (License Keys) — يحوّل التطبيق من "تسجيل حر بالإيميل"
إلى "وصول مدفوع": لا يمكن إنشاء حساب جديد إلا بكود ترخيص صالح غير مُستخدَم.

سير العمل المقترح:
1. المُشرفة (صاحبة التطبيق) تولّد كودًا من لوحة التحكم (سرّية بكلمة مرور).
2. تستلم الدفع من الزبون بأي وسيلة تراها (تحويل بنكي، رابط دفع، نقدًا...) —
   هذا خارج نطاق الكود نفسه ويتم يدويًا من طرفك.
3. تُرسلين الكود للزبون.
4. الزبون يُدخل الكود + إيميله + كلمة مرور يختارها بنفسه عند "إنشاء حساب".
5. الكود يُصبح غير قابل للاستخدام مرة أخرى (مربوط بذلك الحساب فقط).
"""

import secrets
import sqlite3
import string
from datetime import datetime, timedelta

DB_PATH = "licenses.db"


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS license_keys (
            code TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            expires_at TEXT,
            note TEXT,
            used_by TEXT,
            used_at TEXT
        )
    """)
    return conn


def _generate_code() -> str:
    alphabet = string.ascii_uppercase + string.digits
    parts = ["".join(secrets.choice(alphabet) for _ in range(4)) for _ in range(4)]
    return "-".join(parts)


def generate_key(note: str = "", valid_days: int = None) -> str:
    """تولّد كود ترخيص جديد. valid_days=None يعني بلا تاريخ انتهاء."""
    code = _generate_code()
    expires_at = None
    if valid_days:
        expires_at = (datetime.now() + timedelta(days=valid_days)).isoformat(timespec="seconds")
    conn = _get_conn()
    conn.execute(
        "INSERT INTO license_keys (code, created_at, expires_at, note) VALUES (?, ?, ?, ?)",
        (code, datetime.now().isoformat(timespec="seconds"), expires_at, note),
    )
    conn.commit()
    conn.close()
    return code


def list_keys():
    conn = _get_conn()
    rows = conn.execute(
        "SELECT code, created_at, expires_at, note, used_by, used_at FROM license_keys ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [
        {"code": r[0], "created_at": r[1], "expires_at": r[2], "note": r[3], "used_by": r[4], "used_at": r[5]}
        for r in rows
    ]


def validate_key(code: str):
    """يتحقق من أن الكود موجود وغير مُستخدَم وغير منتهي الصلاحية، دون استهلاكه."""
    code = (code or "").strip().upper()
    conn = _get_conn()
    row = conn.execute(
        "SELECT expires_at, used_by FROM license_keys WHERE code=?", (code,)
    ).fetchone()
    conn.close()
    if row is None:
        return False, "كود الترخيص غير صحيح."
    expires_at, used_by = row
    if used_by:
        return False, "هذا الكود مُستخدَم بالفعل."
    if expires_at and datetime.fromisoformat(expires_at) < datetime.now():
        return False, "انتهت صلاحية هذا الكود."
    return True, "الكود صالح."


def redeem_key(code: str, email: str):
    """يربط الكود بحساب المستخدم بعد نجاح التسجيل به (يجعله غير قابل لإعادة الاستخدام)."""
    code = (code or "").strip().upper()
    conn = _get_conn()
    conn.execute(
        "UPDATE license_keys SET used_by=?, used_at=? WHERE code=?",
        (email.strip().lower(), datetime.now().isoformat(timespec="seconds"), code),
    )
    conn.commit()
    conn.close()


def revoke_key(code: str):
    """يحذف كودًا غير مُستخدَم (لإلغائه قبل بيعه فعليًا مثلاً)."""
    code = (code or "").strip().upper()
    conn = _get_conn()
    conn.execute("DELETE FROM license_keys WHERE code=? AND used_by IS NULL", (code,))
    conn.commit()
    conn.close()
