# YTshield — Minimal & Pro KPIs (SQLite) [UTC-fix + period fix]
# -------------------------------------------------------------
# Запуск: python -m streamlit run app.py

import os, sqlite3, tempfile
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from pandas.api import types as ptypes

st.set_page_config(page_title="YTshield — Minimal & Pro KPIs", layout="wide", page_icon="📊")

# ---------- utils

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols, seen = [], {}
    for i, c in enumerate(df.columns):
        name = str(c).strip()
        if not name or name.lower().startswith("unnamed"):
            name = f"col_{i}"
        if name in seen:
            k = seen[name]
            seen[name] = k + 1
            name = f"{name}_{k}"
        else:
            seen[name] = 1
        new_cols.append(name)
    df.columns = new_cols
    return df

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = sanitize_columns(df)
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ["date","time","created","updated","expire","at","ts"]):
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                pass
    return df

def norm_to_utc_naive(s: pd.Series) -> pd.Series:
    """tz-aware → UTC → drop tz (везде одинаковый тип даты)."""
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_localize(None)

def pick_numeric(df: pd.DataFrame, preferred: list[str]) -> Optional[str]:
    for c in preferred:
        if c in df.columns and ptypes.is_numeric_dtype(df[c]):
            return c
    for c in df.columns:
        if ptypes.is_numeric_dtype(df[c]):
            return c
    return None

def pick_datetime(df: pd.DataFrame, preferred: list[str]) -> Optional[str]:
    for c in preferred:
        if c in df.columns and ptypes.is_datetime64_any_dtype(df[c]):
            return c
    for c in df.columns:
        if ptypes.is_datetime64_any_dtype(df[c]):
            return c
    return None

def resolve_db_path(uploaded_file, manual_path: str | None):
    if uploaded_file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        tmp.write(uploaded_file.getbuffer())
        tmp.flush(); tmp.close()
        return tmp.name, True
    if manual_path:
        return manual_path, False
    return None, False

@st.cache_data(ttl=300)
def load_sqlite(db_path: str):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    tables = pd.read_sql_query("SELECT name, type FROM sqlite_master WHERE type IN ('table','view')", con)

    def get(name_like: str) -> Optional[str]:
        low = name_like.lower()
        exact = tables.loc[tables["name"].str.lower() == low, "name"]
        if not exact.empty: return exact.iloc[0]
        part = tables.loc[tables["name"].str.lower().str.contains(low), "name"]
        return part.iloc[0] if not part.empty else None

    t_users = get("users")
    t_paid  = get("paid_transactions")
    t_bonus = get("bonus_transactions")

    users = pd.read_sql_query(f'SELECT * FROM "{t_users}"', con) if t_users else pd.DataFrame()
    paid  = pd.read_sql_query(f'SELECT * FROM "{t_paid}"', con) if t_paid else pd.DataFrame()
    bonus = pd.read_sql_query(f'SELECT * FROM "{t_bonus}"', con) if t_bonus else pd.DataFrame()
    con.close()

    return parse_dates(users), parse_dates(paid), parse_dates(bonus), tables

# ---------- sidebar

with st.sidebar:
    st.header("Источник данных")
    up = st.file_uploader("Загрузить .db", type=["db","sqlite","sqlite3"])
    path = st.text_input("Или путь к .db", value="")
    db_path, is_temp = resolve_db_path(up, path.strip() or None)

    st.divider()
    st.caption("Транзакции учитывать:")
    tx_scope = st.radio(
        "Диапазон для бонусов/оплат",
        ["по когорте (date_reg)", "по датам транзакций"],
        index=0,
        help="По умолчанию считаем по пользователям, стартовавшим в выбранный период. "
             "Если выбрать 'по датам транзакций' — фильтруем записи транзакций по их timestamp."
    )

st.title("📊 YTshield — Minimal & Pro KPIs")

if not db_path:
    st.info("Загрузи SQLite .db или укажи путь в сайдбаре.")
    st.stop()
if not os.path.exists(db_path):
    st.error(f"Файл не найден: {db_path}")
    st.stop()

users, paid_df, bonus_df, meta = load_sqlite(db_path)
if users.empty:
    st.error("Таблица users не найдена или пуста.")
    st.stop()

# ---------- prepare users

u = sanitize_columns(users.copy())
for col in ["telegram_id","user_login","full_name","refer_id","count_refer","sub_url",
            "coins_count","expire_date","date_reg","discount","url_ref","promo_code"]:
    if col not in u.columns: u[col] = np.nan

# даты -> naive UTC
u["date_reg"]    = norm_to_utc_naive(u["date_reg"])
u["expire_date"] = norm_to_utc_naive(u["expire_date"])

u["count_refer"] = pd.to_numeric(u["count_refer"], errors="coerce")
u["telegram_id_str"] = u["telegram_id"].astype(str)

# ---------- period picker (safe for single-date selection)

dmin, dmax = u["date_reg"].min(), u["date_reg"].max()
if pd.isna(dmin) or pd.isna(dmax):
    st.warning("Нет валидных дат в users.date_reg — период ограничить нельзя, показываю всё.")
    start_date, end_date = None, None
    cohort_mask = pd.Series([True]*len(u), index=u.index)
else:
    sel = st.date_input(
        "Период (по users.date_reg)",
        value=(dmin.date(), dmax.date()),
        help="Выбери две даты (начало и конец периода)."
    )
    if isinstance(sel, (list, tuple)) and len(sel) == 2:
        start_date, end_date = sel
        if start_date > end_date:
            start_date, end_date = end_date, start_date
        cohort_mask = u["date_reg"].dt.date.between(start_date, end_date)
    else:
        st.info("Выбери две даты (начало и конец периода), чтобы продолжить.")
        st.stop()

# >>> ВАЖНО: сформируем когорту и список id (раньше их не было, из-за этого NameError)
u_cohort = u.loc[cohort_mask].copy()
cohort_ids = set(u_cohort["telegram_id_str"])

# ---------- USERS KPIs (минимум)

starts = int(u_cohort["telegram_id"].notna().sum())
via_ref = int(u_cohort["refer_id"].notna().sum())
with_promo = int(u_cohort["promo_code"].notna().sum())
with_expire = int(u_cohort["expire_date"].notna().sum())

count_refer_sum = int(pd.to_numeric(u_cohort["count_refer"], errors="coerce").fillna(0).sum())
count_refer_inviter_cnt = int((pd.to_numeric(u_cohort["count_refer"], errors="coerce").fillna(0) > 0).sum())

# ---------- BONUS TX KPIs

bonus_uid = next((c for c in ["telegram_id","user_id","uid","tg_id","chat_id"] if c in bonus_df.columns), None) if not bonus_df.empty else None
bonus_time_col = pick_datetime(bonus_df, ["created_at","date","time","ts"]) if not bonus_df.empty else None
if bonus_time_col and bonus_time_col in bonus_df.columns:
    bonus_df[bonus_time_col] = norm_to_utc_naive(bonus_df[bonus_time_col])

if bonus_df.empty or not bonus_uid:
    bonus_taken = 0
else:
    bb = bonus_df.copy()
    bb["__uid__"] = bb[bonus_uid].astype(str)
    if tx_scope == "по когорте (date_reg)":
        bb = bb[bb["__uid__"].isin(cohort_ids)]
    else:
        if start_date and end_date and bonus_time_col and bonus_time_col in bb.columns and ptypes.is_datetime64_any_dtype(bb[bonus_time_col]):
            bb = bb[bb[bonus_time_col].dt.date.between(start_date, end_date)]
    bonus_taken = int(bb["__uid__"].notna().sum())

# ---------- PAID TX KPIs

paid_uid = next((c for c in ["telegram_id","user_id","uid","tg_id","chat_id"] if c in paid_df.columns), None) if not paid_df.empty else None
paid_time_col = pick_datetime(paid_df, ["paid_at","created_at","date","time","ts"]) if not paid_df.empty else None
if paid_time_col and paid_time_col in paid_df.columns:
    paid_df[paid_time_col] = norm_to_utc_naive(paid_df[paid_time_col])

amount_col = None
if not paid_df.empty:
    if "amount_rub" in paid_df.columns and ptypes.is_numeric_dtype(paid_df["amount_rub"]):
        amount_col = "amount_rub"
    else:
        amount_col = pick_numeric(paid_df, ["amount","sum","value","total","price","rub","usd"])

if paid_df.empty or not paid_uid:
    paid_records = 0
    paid_users_unique = 0
    paid_sum = 0.0
else:
    pp = paid_df.copy()
    pp["__uid__"] = pp[paid_uid].astype(str)
    if tx_scope == "по когорте (date_reg)":
        pp = pp[pp["__uid__"].isin(cohort_ids)]
    else:
        if start_date and end_date and paid_time_col and paid_time_col in pp.columns and ptypes.is_datetime64_any_dtype(pp[paid_time_col]):
            pp = pp[pp[paid_time_col].dt.date.between(start_date, end_date)]
    paid_records = int(pp["__uid__"].notna().sum())
    paid_users_unique = int(pp["__uid__"].nunique())
    paid_sum = float(pd.to_numeric(pp[amount_col], errors="coerce").fillna(0).sum()) if amount_col else 0.0

# ---------- OUTPUT (минимум)

st.subheader("Итоговые показатели (минимум)")
c1,c2,c3,c4 = st.columns(4)
c1.metric("START (users.telegram_id)", starts)
c2.metric("Пришли по рефералке (refer_id)", via_ref)
c3.metric("С промо (promo_code)", with_promo)
c4.metric("Активировали конфиг (expire_date)", with_expire)

c5,c6,c7 = st.columns(3)
c5.metric("Сумма count_refer", count_refer_sum)
c6.metric("Рефереров с count_refer>0", count_refer_inviter_cnt)
c7.metric("Бонус взяли (строк в bonus_transactions)", bonus_taken)

c8,c9,c10 = st.columns(3)
c8.metric("Оплатили (уникальные telegram_id)", paid_users_unique)
c9.metric("Всего оплат (строк)", paid_records)
c10.metric("Сумма оплат (amount_rub)", round(paid_sum, 2))

# ---------- KPI Pro

st.header("KPI Pro (что возможно сейчас)")

# LTV (упрощённый)
ltv_avg_all = 0.0
ltv_avg_payers = 0.0
if not paid_df.empty and paid_uid and amount_col:
    pp_all = paid_df.copy()
    pp_all["__uid__"] = pp_all[paid_uid].astype(str)

    if tx_scope == "по когорте (date_reg)":
        pp_all = pp_all[pp_all["__uid__"].isin(cohort_ids)]
    else:
        if start_date and end_date and paid_time_col and paid_time_col in pp_all.columns and ptypes.is_datetime64_any_dtype(pp_all[paid_time_col]):
            pp_all = pp_all[pp_all[paid_time_col].dt.date.between(start_date, end_date)]

    revenue_sum = float(pd.to_numeric(pp_all[amount_col], errors="coerce").fillna(0).sum())
    users_count = max(1, len(u_cohort))
    ltv_avg_all = revenue_sum / users_count

    by_user = pp_all.groupby("__uid__")[amount_col].sum()
    ltv_avg_payers = float(by_user.mean()) if not by_user.empty else 0.0

# Retention / Churn по expire_date
def retention_rate(u_df: pd.DataFrame, days: int) -> float:
    if u_df.empty: return 0.0
    have = u_df.dropna(subset=["date_reg"]).copy()
    if have.empty: return 0.0
    cutoff = have["date_reg"] + pd.to_timedelta(days, unit="D")   # все naive UTC
    retained = have["expire_date"].notna() & (have["expire_date"] >= cutoff)
    return float(retained.mean()) * 100.0

ret_d30  = retention_rate(u_cohort, 30)
ret_d90  = retention_rate(u_cohort, 90)
ret_d180 = retention_rate(u_cohort, 180)
churn_d30, churn_d90, churn_d180 = 100-ret_d30, 100-ret_d90, 100-ret_d180

# Помесячная выручка (по дате платежа)
rev_by_month = None
if not paid_df.empty and paid_time_col and amount_col and ptypes.is_datetime64_any_dtype(paid_df[paid_time_col]):
    tmp = paid_df.dropna(subset=[paid_time_col]).copy()
    tmp["amount___"] = pd.to_numeric(tmp[amount_col], errors="coerce")
    tmp = tmp.dropna(subset=["amount___"])
    rev_by_month = tmp.set_index(paid_time_col)["amount___"].resample("MS").sum()

# Активные по месяцам
active_by_month = None
arpu_by_month = None
if u["date_reg"].notna().any():
    m_start = u["date_reg"].min().to_period("M").start_time
    m_end_candidates = [u["date_reg"].max()]
    if u["expire_date"].notna().any():
        m_end_candidates.append(u["expire_date"].max())
    m_end = max([d for d in m_end_candidates if pd.notna(d)])
    months = pd.period_range(m_start, m_end, freq="M")
    if len(months) > 0:
        active_counts = []
        for p in months:
            ms, me = p.start_time, p.end_time
            mask = u["date_reg"].fillna(pd.Timestamp.min) <= me
            mask &= u["expire_date"].fillna(pd.Timestamp.min) >= ms
            active_counts.append(int(mask.sum()))
        active_by_month = pd.Series(active_counts, index=months.to_timestamp())
        if rev_by_month is not None:
            idx = active_by_month.index.union(rev_by_month.index)
            aa = active_by_month.reindex(idx).fillna(method="ffill").fillna(0)
            rr = rev_by_month.reindex(idx).fillna(0)
            arpu_by_month = rr / aa.replace(0, np.nan)

# ------- вывод KPI Pro

c1,c2,c3 = st.columns(3)
c1.metric("LTV (ср. на пользователя, cohort)", f"{ltv_avg_all:,.2f}".replace(",", " "))
c2.metric("LTV (ср. на платящего в cohort)", f"{ltv_avg_payers:,.2f}".replace(",", " "))
c3.metric("ARPU (посл. месяц)", f"{(arpu_by_month.dropna().iloc[-1] if arpu_by_month is not None and not arpu_by_month.dropna().empty else 0):,.2f}".replace(",", " "))

c4,c5,c6 = st.columns(3)
c4.metric("Retention D30", f"{ret_d30:.1f}%")
c5.metric("Retention D90", f"{ret_d90:.1f}%")
c6.metric("Retention D180", f"{ret_d180:.1f}%")

c7,c8,c9 = st.columns(3)
c7.metric("Churn D30",  f"{churn_d30:.1f}%")
c8.metric("Churn D90",  f"{churn_d90:.1f}%")
c9.metric("Churn D180", f"{churn_d180:.1f}%")

st.subheader("Помесячная выручка / активные / ARPU")
if rev_by_month is not None:
    st.line_chart(rev_by_month.rename("Revenue"), height=220)
else:
    st.info("Нет корректной даты платежа в paid_transactions — график выручки недоступен.")

if active_by_month is not None:
    st.bar_chart(active_by_month.rename("Active users"), height=220)
else:
    st.info("Недостаточно данных для расчёта активных по месяцам.")

if arpu_by_month is not None and not arpu_by_month.dropna().empty:
    st.line_chart(arpu_by_month.rename("ARPU"), height=220)
else:
    st.info("Недостаточно данных для ARPU по месяцам.")

# ---------- Разрез по промокодам

st.subheader("Разрез по промокодам (только users за период)")
promo_breakdown = (
    u_cohort["promo_code"]
    .fillna("—")
    .astype(str)
    .value_counts()
    .rename_axis("promo_code")
    .reset_index(name="users_count")
)
st.dataframe(promo_breakdown, use_container_width=True)

with st.expander("Диагностика источников"):
    st.write({
        "users.shape": u.shape,
        "paid.shape": paid_df.shape,
        "bonus.shape": bonus_df.shape,
        "paid_uid": paid_uid if 'paid_uid' in locals() else None,
        "paid_time_col": paid_time_col if 'paid_time_col' in locals() else None,
        "amount_col": amount_col if 'amount_col' in locals() else None,
        "bonus_uid": bonus_uid if 'bonus_uid' in locals() else None,
        "bonus_time_col": bonus_time_col if 'bonus_time_col' in locals() else None,
        "tx_scope": tx_scope
    })
