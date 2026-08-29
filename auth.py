# -*- coding: utf-8 -*-
"""
نظام حسابات بسيط (تسجيل + دخول) بالإيميل وكلمة المرور — بدون أي خدمة خارجية.
يُخزِّن الحسابات في قاعدة بيانات SQLite محلية على نفس خادم التطبيق.
كلمات المرور لا تُخزَّن أبدًا كنص صريح؛ تُخزَّن فقط كـ hash (PBKDF2-HMAC-SHA256) مع ملح عشوائي (salt).

ملاحظة: هذه النسخة لا تُرسل بريد تحقق فعليًا (ذلك يتطلب ربط التطبيق بخدمة بريد
مثل SendGrid/SMTP وإعداد مفاتيح API) — الإيميل هنا يُستخدم كمعرّف فريد لكل
حساب لفصل بيانات المستخدمين عن بعضهم، وهذا يكفي لتحقيق هدف "كل مستخدم يرى
مشاريعه فقط". يمكن إضافة تحقق بريد حقيقي لاحقًا إذا احتجتِه.
"""

import hashlib
import os
import re
import sqlite3
from datetime import datetime

DB_PATH = "users.db"
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    return conn


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000).hex()


def is_valid_email(email: str) -> bool:
    return bool(EMAIL_RE.match(email or ""))


def register_user(email: str, password: str):
    email = email.strip().lower()
    if not is_valid_email(email):
        return False, "صيغة الإيميل غير صحيحة."
    if len(password) < 6:
        return False, "كلمة المرور يجب أن تكون 6 أحرف على الأقل."

    conn = _get_conn()
    existing = conn.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone()
    if existing:
        conn.close()
        return False, "هذا الإيميل مسجَّل بالفعل. جرّبي تسجيل الدخول بدلاً من ذلك."

    salt = os.urandom(16)
    pwd_hash = _hash_password(password, salt)
    conn.execute(
        "INSERT INTO users (email, salt, password_hash, created_at) VALUES (?, ?, ?, ?)",
        (email, salt.hex(), pwd_hash, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()
    return True, "تم إنشاء الحساب بنجاح."


def authenticate(email: str, password: str):
    email = email.strip().lower()
    conn = _get_conn()
    row = conn.execute("SELECT salt, password_hash FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    if row is None:
        return False, "لا يوجد حساب بهذا الإيميل."
    salt = bytes.fromhex(row[0])
    expected_hash = row[1]
    actual_hash = _hash_password(password, salt)
    if actual_hash != expected_hash:
        return False, "كلمة المرور غير صحيحة."
    return True, "تم تسجيل الدخول بنجاح."
