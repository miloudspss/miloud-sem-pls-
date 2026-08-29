# -*- coding: utf-8 -*-
"""
حفظ وتحميل "مشاريع" التحليل (تعريف النموذج + العلاقات + إعدادات التعديل/الوساطة)
باستخدام قاعدة بيانات SQLite محلية بسيطة — بدون الحاجة لأي خادم خارجي.

كل مشروع مرتبط بإيميل صاحبه (user_email) بحيث يرى كل مستخدم مشاريعه فقط،
بالاعتماد على نظام الحسابات في auth.py.
"""

import json
import sqlite3
import base64
import io
from datetime import datetime

import pandas as pd

DB_PATH = "projects.db"


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            model_spec TEXT NOT NULL,
            structural_paths TEXT NOT NULL,
            interactions TEXT,
            data_excel_b64 TEXT,
            UNIQUE(user_email, name)
        )
    """)
    return conn


def save_project(user_email: str, name: str, model_spec: dict, structural_paths: list,
                  interactions: list = None, data: pd.DataFrame = None,
                  include_data: bool = False):
    conn = _get_conn()
    data_b64 = None
    if include_data and data is not None:
        buf = io.BytesIO()
        data.to_excel(buf, index=False)
        data_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    conn.execute(
        """INSERT INTO projects (user_email, name, created_at, model_spec, structural_paths, interactions, data_excel_b64)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(user_email, name) DO UPDATE SET
                created_at=excluded.created_at,
                model_spec=excluded.model_spec,
                structural_paths=excluded.structural_paths,
                interactions=excluded.interactions,
                data_excel_b64=excluded.data_excel_b64
        """,
        (
            user_email.strip().lower(),
            name,
            datetime.now().isoformat(timespec="seconds"),
            json.dumps(model_spec, ensure_ascii=False),
            json.dumps(structural_paths, ensure_ascii=False),
            json.dumps(interactions or [], ensure_ascii=False),
            data_b64,
        ),
    )
    conn.commit()
    conn.close()


def list_projects(user_email: str):
    conn = _get_conn()
    rows = conn.execute(
        "SELECT name, created_at, (data_excel_b64 IS NOT NULL) FROM projects WHERE user_email=? ORDER BY created_at DESC",
        (user_email.strip().lower(),),
    ).fetchall()
    conn.close()
    return [{"name": r[0], "created_at": r[1], "has_data": bool(r[2])} for r in rows]


def load_project(user_email: str, name: str):
    conn = _get_conn()
    row = conn.execute(
        "SELECT model_spec, structural_paths, interactions, data_excel_b64 FROM projects WHERE user_email=? AND name=?",
        (user_email.strip().lower(), name),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    model_spec = json.loads(row[0])
    structural_paths = [tuple(p) for p in json.loads(row[1])]
    interactions = json.loads(row[2]) if row[2] else []
    data = None
    if row[3]:
        data = pd.read_excel(io.BytesIO(base64.b64decode(row[3])))
    return {
        "model_spec": model_spec,
        "structural_paths": structural_paths,
        "interactions": interactions,
        "data": data,
    }


def delete_project(user_email: str, name: str):
    conn = _get_conn()
    conn.execute("DELETE FROM projects WHERE user_email=? AND name=?", (user_email.strip().lower(), name))
    conn.commit()
    conn.close()
