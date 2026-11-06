"""Демо: streamlit run app.py"""
import os

import pandas as pd
import streamlit as st

from src.demo_backend import (SI_THRESHOLD, load_comparison, load_package,
                              load_test_predictions, predict, predict_many)
from src.viz import descriptor_importance_plot, draw_mol, parity_plot

st.set_page_config(page_title="Прогноз SI против VACV", page_icon="🧪", layout="wide")


@st.cache_resource
def get_package():
    return load_package()


package = get_package()
st.title("🧪 Прогноз индекса селективности (SI) против вируса оспы VACV")
if package is None:
    st.warning("Модель не найдена (models/best_model.pkl). Показаны ЗАГЛУШКА-значения: запустите train.py.")

EXAMPLES = {
    "Цидофовир": "NC1=NC(=O)N(CC(CO)OCP(O)(O)=O)C=C1",
    "Рибавирин": "NC(=O)c1ncn(n1)C1OC(CO)C(O)C1O",
    "Ацикловир": "NC1=NC2=C(N=CN2COCCO)C(=O)N1",
    "Аспирин (контроль)": "CC(=O)Oc1ccccc1C(=O)O",
}
st.session_state.setdefault("smiles", EXAMPLES["Цидофовир"])

tab_pred, tab_batch, tab_about = st.tabs(["Прогноз", "Пакетный CSV", "О модели"])

with tab_pred:
    cols = st.columns(len(EXAMPLES))
    for col, (name, smi) in zip(cols, EXAMPLES.items()):
        col.button(name, on_click=lambda s=smi: st.session_state.update(smiles=s), use_container_width=True)
    st.text_input("SMILES молекулы", key="smiles")

    res = predict(st.session_state["smiles"], package)
    if not res["valid"]:
        st.error(res["error"])
    else:
        left, right = st.columns([1, 1])
        with left:
            st.image(draw_mol(res["mol"]), caption="Структура молекулы (RDKit)")
        with right:
            m1, m2 = st.columns(2)
            m1.metric("Прогноз SI", f"{res['SI']:.1f}")
            m2.metric("log10(SI)", f"{res['log10_SI']:.2f}")
            if res["std_SI"] is not None:
                st.caption(f"Разброс деревьев ансамбля: ±{res['std_SI']:.1f} SI. Чем больше, тем менее уверена модель.")
            if res["active"]:
                st.success(f"SI ≥ {SI_THRESHOLD:g}: молекула выглядит перспективной")
            else:
                st.info(f"SI < {SI_THRESHOLD:g}: селективность низкая")
            st.caption("Статус «активно/неактивно» получен порогом по регрессии, отдельного классификатора нет.")
        st.subheader("Дескрипторы молекулы")
        st.table(pd.DataFrame([res["descriptors"]]).T.rename(columns={0: "Значение"}))

with tab_batch:
    st.write("Загрузите CSV со столбцом `smiles` (до 500 молекул).")
    up = st.file_uploader("CSV", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        if "smiles" not in df.columns:
            st.error("Нет столбца smiles")
        else:
            out = predict_many(df["smiles"].head(500).tolist(), package)
            st.dataframe(out, use_container_width=True)
            st.download_button("Скачать результат", out.to_csv(index=False), "predictions.csv", "text/csv")

with tab_about:
    if package is not None:
        st.write(f"Лучшая модель: **{package.get('model_name', '?')}**, цель: `{package.get('target', '?')}`")
        st.caption(f"Признаки: {package.get('features', 'Morgan fingerprint + дескрипторы')}")
    comp = load_comparison()
    if comp is None:
        st.info("Таблицы сравнения нет: появится после запуска train.py (results/model_comparison.csv).")
    else:
        comp = comp.copy()
        comp["RMSE≈√MSE_mean"] = comp["MSE_mean"] ** 0.5
        st.write("Кросс-валидация (5 фолдов):")
        st.dataframe(comp, use_container_width=True)
    test_df = load_test_predictions()
    g1, g2 = st.columns(2)
    if test_df is not None:
        g1.pyplot(parity_plot(test_df))
    if package is not None:
        fig = descriptor_importance_plot(package)
        if fig is not None:
            g2.pyplot(fig)
    if os.path.exists("experiments/results.csv"):
        st.subheader("Таблица экспериментов")
        st.dataframe(pd.read_csv("experiments/results.csv"), use_container_width=True)
    st.markdown("[GitHub репозиторий](https://github.com/HighBlueJameveu/NuclearITHackathon)")