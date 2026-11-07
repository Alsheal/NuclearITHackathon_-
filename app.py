"""Демо: streamlit run app.py
Приложение только показывает результаты общей работы: прогнозы, графики, состояние проекта.
"""
import os

import pandas as pd
import streamlit as st

from src import demo_backend as B
from src import viz

st.set_page_config(page_title="Прогноз SI против VACV", page_icon="🧪", layout="wide")


@st.cache_resource
def get_model(mtime):  # mtime в ключе: после переобучения модель подхватится сама
    return B.load_model()


model, model_error = get_model(os.path.getmtime(B.MODEL_PATH) if os.path.exists(B.MODEL_PATH) else 0)
st.title("🧪 Прогноз индекса селективности (SI) против вируса осповакцины VACV")
if model is None:
    st.warning(f"{model_error}. Показаны ЗАГЛУШКА-значения, это не настоящие прогнозы.")

EXAMPLES = {
    "Цидофовир": "NC1=NC(=O)N(CC(CO)OCP(O)(O)=O)C=C1",
    "Рибавирин": "NC(=O)c1ncn(n1)C1OC(CO)C(O)C1O",
    "Ацикловир": "NC1=NC2=C(N=CN2COCCO)C(=O)N1",
    "Аспирин (контроль)": "CC(=O)Oc1ccccc1C(=O)O",
}
st.session_state.setdefault("smiles", EXAMPLES["Цидофовир"])
tab_pred, tab_batch, tab_res = st.tabs(["Прогноз", "Кандидаты (CSV)", "Результаты проекта"])

with tab_pred:
    cols = st.columns(len(EXAMPLES))
    for col, (name, smi) in zip(cols, EXAMPLES.items()):
        col.button(name, on_click=lambda s=smi: st.session_state.update(smiles=s))
    st.text_input("SMILES молекулы", key="smiles")
    try:
        res = B.predict(st.session_state["smiles"], model)
    except Exception as e:
        res = {"valid": False, "error": f"Ошибка модели: {e}"}
    if not res["valid"]:
        st.error(res["error"])
    else:
        left, right = st.columns(2)
        left.image(viz.draw_mol(res["mol"]), caption="Структура молекулы (RDKit)")
        with right:
            m1, m2 = st.columns(2)
            m1.metric("Прогноз SI", f"{res['SI']:.1f}")
            m2.metric("log10(SI)", f"{res['log10_SI']:.2f}")
            if res["low"] is not None:
                st.caption(f"Вероятный диапазон: {res['low']:.1f} – {res['high']:.1f} "
                           "(разброс деревьев ансамбля, грубая оценка уверенности).")
            if res["active"]:
                st.success(f"SI ≥ {B.SI_THRESHOLD:g}: молекула выглядит перспективной")
            else:
                st.info(f"SI < {B.SI_THRESHOLD:g}: селективность низкая")
            if res["lipinski"]:
                st.warning("Правило пяти Липински нарушено: " + ", ".join(res["lipinski"]))
            else:
                st.caption("Правило пяти Липински выполнено (подсказка о «лекарственности», не часть модели).")
        st.subheader("Дескрипторы молекулы")
        st.table(pd.DataFrame([res["descriptors"]]).T.rename(columns={0: "Значение"}))

with tab_batch:
    st.write("Загрузите CSV со столбцом `smiles` (например, список придуманных соединений). "
             "Молекулы будут ранжированы по прогнозу SI.")
    up = st.file_uploader("CSV", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        if "smiles" not in df.columns:
            st.error("Нет столбца smiles")
        else:
            try:
                out = B.predict_many(df["smiles"].head(1000).tolist(), model).sort_values("SI", ascending=False)
            except Exception as e:
                st.error(f"Ошибка модели: {e}")
                st.stop()
            st.dataframe(out)
            st.pyplot(viz.top_plot(out))
            st.download_button("Скачать результат", out.to_csv(index=False), "candidates.csv", "text/csv")

with tab_res:
    st.subheader("Состояние проекта")
    for label, ok, path in B.project_status(model):
        st.write(f"{'✅' if ok else '⬜'} {label} (`{path}`)")

    rep = B.read_report()
    if rep:
        st.subheader("Качество модели на проверочной выборке")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE (log10 SI)", f"{rep['validation_MAE_log10']:.3f}")
        c2.metric("RMSE (log10 SI)", f"{rep['validation_RMSE_log10']:.3f}")
        c3.metric("R²", f"{rep['validation_R2']:.3f}")
        c4.metric("Обучающих молекул", rep["training_samples"])
        st.caption(f"{rep['model']}, признаков: {rep['features']}, цель: {rep['target']}")
        with st.expander("Параметры модели"):
            st.json(rep["parameters"])

    exp = B.read_csv(B.EXPERIMENTS_PATH)
    if exp is not None:
        st.subheader("Эксперименты")
        st.dataframe(exp)

    g1, g2 = st.columns(2)
    sub = B.read_submission()
    if sub is not None:
        g1.pyplot(viz.hist_si(sub, f"Прогнозы для теста Kaggle ({len(sub)} молекул)"))
        g1.caption(f"Доля с SI ≥ {B.SI_THRESHOLD:g}: {(sub >= B.SI_THRESHOLD).mean():.0%}, медиана: {sub.median():.1f}")
    if model is not None:
        fig = viz.importance_plot(model)
        if fig is not None:
            g2.pyplot(fig)

    st.subheader("Данные")
    d1, d2 = st.columns(2)
    train = B.read_csv(B.TRAIN_PATH)
    if train is not None and "SI VACV" in train.columns:
        d1.pyplot(viz.hist_si(train["SI VACV"], f"Данные организаторов ({len(train)} строк)"))
    ext_path, ext = B.read_external()
    if ext is not None:
        d2.write(f"Внешний датасет `{ext_path}`: **{len(ext)}** строк")
        if "doi" in ext.columns:
            d2.caption(f"Со ссылкой на публикацию (DOI): {ext['doi'].notna().mean():.0%}")
        si_col = next((c for c in ("SI_VACV", "SI VACV", "SI") if c in ext.columns), None)
        if si_col:
            d2.pyplot(viz.hist_si(ext[si_col], "Внешние данные"))
    elif train is None:
        st.info("Данных пока нет: положите data/train.csv и внешний датасет, графики появятся сами.")
    st.markdown("[GitHub репозиторий](https://github.com/HighBlueJameveu/NuclearITHackathon)")