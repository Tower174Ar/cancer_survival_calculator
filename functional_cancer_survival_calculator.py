import pandas as pd
import numpy as np

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import streamlit as st
import matplotlib.pyplot as plt


CAT_COLS = [
    'age', 'sex', 'race', 'ethnicity', 'income', 'residency',
    'laterality', 'location', 'tumor_grade', 'surgery_status',
    'radiotherapy', 'chemotherapy'
]


raw = pd.read_csv("extended_synthetic_survival_data.csv")


_dummy = pd.get_dummies(raw, columns=CAT_COLS, drop_first=True)
MODEL_COLUMNS = _dummy.drop(columns=["survival_time", "event_status"]).columns.tolist()

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) One‑hot encode CAT_COLS (drop_first=True)
    2) Ensure all MODEL_COLUMNS are present (fill missing with zeros)
    3) Impute all numeric columns (except event_status) via SimpleImputer(median)
    4) Preserve survival_time and event_status
    """
    df_enc = pd.get_dummies(df, columns=CAT_COLS, drop_first=True)
    for col in MODEL_COLUMNS:
        if col not in df_enc.columns:
            df_enc[col] = 0
    df_enc = df_enc[MODEL_COLUMNS + ["survival_time", "event_status"]]
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(df_enc.drop(columns=["event_status"]))
    df_imp = pd.DataFrame(X, columns=df_enc.drop(columns=["event_status"]).columns, index=df_enc.index)
    df_imp["event_status"] = df_enc["event_status"].astype(bool)
    return df_imp


data = preprocess(raw)
df_train, df_val = train_test_split(data, test_size=0.3, random_state=42)

cph = CoxPHFitter()
cph.fit(df_train, duration_col="survival_time", event_col="event_status")


c_index_train = concordance_index(
    df_train["survival_time"],
    -cph.predict_partial_hazard(df_train),
    df_train["event_status"]
)
c_index_val = concordance_index(
    df_val["survival_time"],
    -cph.predict_partial_hazard(df_val),
    df_val["event_status"]
)

times = np.arange(6, 61, 6)
times = times[times <= df_val["survival_time"].max()]

auc_times, auc_scores = cumulative_dynamic_auc(
    Surv.from_dataframe("event_status","survival_time",df_train),
    Surv.from_dataframe("event_status","survival_time",df_val),
    cph.predict_partial_hazard(df_val),
    times=times
)


if isinstance(auc_scores, (float, np.floating)):
    auc_scores = [auc_scores]
    auc_times  = [times[0]]


st.title("Cancer Survival Calculator")

mode = st.radio("Mode:", ["Batch CSV analysis", "Single‑patient lookup"])

if mode == "Batch CSV analysis":
    st.header("Batch CSV analysis")
    uploaded = st.file_uploader("Upload cohort CSV", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        df_proc = preprocess(df)
     
        ci = concordance_index(
            df_proc["survival_time"],
            -cph.predict_partial_hazard(df_proc),
            df_proc["event_status"]
        )
        st.subheader("Global C‑index")
        st.write(f"- Validation C‑index: **{ci:.3f}**")

        st.subheader("Time‑dependent AUCs")
        with np.errstate(divide='ignore', invalid='ignore'):
            atimes, ascores = cumulative_dynamic_auc(
                Surv.from_dataframe("event_status","survival_time",df_train),
                Surv.from_dataframe("event_status","survival_time",df_proc),
                cph.predict_partial_hazard(df_proc),
                times=times
            )
        
        if isinstance(ascores, (float, np.floating)):
            ascores = [ascores]
            atimes  = [times[0]]
        for t_val, auc in zip(atimes, ascores):
            val = "nan" if np.isnan(auc) else f"{auc:.3f}"
            st.write(f"- AUC at {int(t_val)} months: **{val}**")
    
        df_proc["predicted_risk"] = cph.predict_partial_hazard(df_proc)
        df_proc["decile"] = pd.qcut(df_proc["predicted_risk"], 10, labels=False) + 1
        agg = df_proc.groupby("decile")[["survival_time","event_status"]].apply(
            lambda d: pd.Series({
                "n": len(d),
                "events": d["event_status"].sum(),
                "median_time": d["survival_time"].median()
            })
        ).reset_index()
        st.subheader("Decile Summary")
        st.dataframe(agg)
      
        st.subheader("Calibration Curves")
        plt.figure()
        for t in times:
            sf = cph.predict_survival_function(df_proc, times=[t]).T
            plt.plot(sf, label=f"{int(t)} mo")
        plt.xlabel("Predicted survival")
        plt.ylabel("Observed survival")
        plt.legend()
        st.pyplot(plt)

else:
    st.header("Single‑patient lookup")
    user_input = {}
    for col in CAT_COLS:
        opts = raw[col].dropna().unique().tolist()
        user_input[col] = st.selectbox(f"{col.capitalize()}", opts)
    if st.button("Calculate risk & survival"):
        user_df = pd.DataFrame([user_input])
        user_df["survival_time"] = df_val["survival_time"].median()
        user_df["event_status"]  = False
        user_proc = preprocess(user_df)
        risk = cph.predict_partial_hazard(user_proc).iloc[0]
        surv = cph.predict_survival_function(user_proc, times=times)
        st.subheader("Predicted relative risk")
        st.write(f"**{risk:.2f}** (higher → worse prognosis)")
        st.subheader("Survival curve")
        st.line_chart(surv.T)

