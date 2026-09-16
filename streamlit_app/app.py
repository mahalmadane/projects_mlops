import requests
import streamlit as st

API_URL = st.sidebar.text_input("URL de l'API", "http://127.0.0.1:8000")

st.title("🚢 Titanic Prediction")
st.write("Renseigne les informations du passager puis lance la prédiction.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Classe (Pclass)", [1, 2, 3], index=2)
        sex = st.selectbox("Sexe", ["male", "female"])
        age = st.number_input("Âge", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
        sibsp = st.number_input(
            "Frères/soeurs/époux à bord (SibSp)", min_value=0.0, max_value=10.0, value=0.0, step=1.0
        )

    with col2:
        parch = st.number_input(
            "Parents/enfants à bord (Parch)", min_value=0, max_value=10, value=0, step=1
        )
        fare = st.number_input("Prix du billet (Fare)", min_value=0.0, value=32.0, step=1.0)
        embarked = st.selectbox("Port d'embarquement", ["S", "C", "Q"])

    submitted = st.form_submit_button("Prédire")

if submitted:
    payload = {
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        prediction = result.get("prediction")
        model_name = result.get("model")

        if prediction == 1:
            st.success(f"✅ Survit (modèle utilisé : {model_name})")
        else:
            st.error(f"❌ Ne survit pas (modèle utilisé : {model_name})")

        st.json(result)

    except requests.exceptions.ConnectionError:
        st.error(f"Impossible de contacter l'API sur {API_URL}. Est-elle bien lancée ?")
    except requests.exceptions.HTTPError as e:
        st.error(f"Erreur de l'API : {e}")
        st.code(response.text)
    except Exception as e:
        st.error(f"Erreur inattendue : {e}")
