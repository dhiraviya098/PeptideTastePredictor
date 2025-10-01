

import streamlit as st
import pandas as pd
import numpy as np
from modlamp.descriptors import GlobalDescriptor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# NEW imports for properties and sequence conversion
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import seq3


# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Peptide Taste Predictor", layout="wide")

st.title("Peptide Taste Predictor")
st.write("Predict peptide taste (Sweet / Salty / Sour / Bitter), basic properties, and structural hints.")

# -----------------------------
# 1. Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("AIML.xlsx")        # <-- Ensure AIML.xlsx is in the repo root
    df.columns = df.columns.str.strip()
    return df

data = load_data()
st.subheader("📂 Dataset Preview")
st.dataframe(data.head())

# -----------------------------
# 2. Feature Extraction
# -----------------------------
def compute_features(seq_list):
    """Compute biochemical descriptors using modlamp."""
    if not seq_list:
        return pd.DataFrame()
    try:
        gd = GlobalDescriptor(seq_list)
        gd.calculate_all()
        return pd.DataFrame(gd.descriptor)
    except Exception as e:
        st.error(f"Feature calculation failed: {e}")
        return pd.DataFrame()

# -----------------------------
# 3. Peptide Property Calculation
# -----------------------------
def compute_peptide_properties(seq: str) -> dict:
    """Compute basic peptide properties using Biopython."""
    analysis = ProteinAnalysis(seq)

    props = {
        "Molecular Weight": round(analysis.molecular_weight(), 2),
        "Isoelectric Point": round(analysis.isoelectric_point(), 2),
        "Aromaticity": round(analysis.aromaticity(), 3),
        "Instability Index": round(analysis.instability_index(), 2),
        "Gravy (Hydrophobicity)": round(analysis.gravy(), 3)
    }
    return props

# -----------------------------
# 4. Train / Load Model
# -----------------------------
MODEL_PATH = "taste_model.pkl"

@st.cache_resource
def train_model():
    X = compute_features(data["peptide"].tolist())
    y = data["Taste"]

    if X.empty:
        st.error("Feature computation failed or returned empty DataFrame.")
        return None, None, None, None, None

    # Remove classes with <2 samples
    counts = y.value_counts()
    valid_classes = counts[counts >= 2].index
    valid_indices = y[y.isin(valid_classes)].index
    X = X.loc[valid_indices]
    y = y[valid_indices]

    if len(y.unique()) < 2:
        st.error("Not enough classes with sufficient samples to train a model.")
        return None, None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42
    )
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, X, y, acc, report

# Load or train model
model, X, y, acc, report = train_model()

# -----------------------------
# 5. Show Model Metrics
# -----------------------------
if model:
    if acc is not None:
        st.success(f"✅ Model trained with accuracy: {acc:.2f}")
        st.write("📊 Classification Report")
        st.json(report)

        # Feature Importance
        st.subheader("🔎 Top 20 Feature Importances")
        feat_importances = pd.Series(model.feature_importances_, index=X.columns)
        feat_importances = feat_importances.sort_values(ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(x=feat_importances.values, y=feat_importances.index, ax=ax)
        ax.set_title("Top 20 Feature Importances")
        st.pyplot(fig)

        # Confusion Matrix
        st.subheader("🧪 Confusion Matrix")
        y_pred_all = model.predict(X)
        cm = confusion_matrix(y, y_pred_all, labels=model.classes_)
        cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)

        fig2, ax2 = plt.subplots(figsize=(6,5))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax2)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        ax2.set_title("Confusion Matrix")
        st.pyplot(fig2)

# -----------------------------
# 6. User Input for Prediction
# -----------------------------
st.subheader("🔬 Predict Taste and Properties of a Peptide")
user_seq = st.text_input("Enter peptide sequence (e.g., EDEGEQPRPF)")

if st.button("Predict"):
    if user_seq:
        feats = compute_features([user_seq])
        if not feats.empty:
            # Taste Prediction
            prediction = model.predict(feats)[0]
            probs = model.predict_proba(feats)[0]

            st.success(f"Predicted Taste: **{prediction}**")
            st.write("Class Probabilities:")
            prob_df = pd.DataFrame({
                "Taste Class": model.classes_,
                "Probability": probs
            })
            st.dataframe(prob_df)

            # Peptide Properties
            st.subheader("🧪 Peptide Properties")
            try:
                props = compute_peptide_properties(user_seq)
                st.json(props)
            except Exception as e:
                st.error(f"Property calculation failed: {e}")

            # Predicted Structure (3-letter representation)
            st.subheader("🧬 Predicted Peptide Structure")
            try:
                aa3_list = [seq3(res) for res in user_seq]
                st.write("3-letter amino acid sequence:")
                st.write(" - ".join(aa3_list))
                st.info("A true 3D structure requires external tools like AlphaFold.")
            except Exception as e:
                st.error(f"Could not generate structure: {e}")
        else:
            st.warning("Could not compute features for the entered sequence.")
    else:
        st.warning("Please enter a peptide sequence.")

# -----------------------------
# 7. Batch Prediction
# -----------------------------
st.subheader("📤 Batch Prediction from File")
uploaded = st.file_uploader("Upload a CSV or Excel file with a 'peptide' column")

if uploaded:
    if uploaded.name.endswith(".csv"):
        df_up = pd.read_csv(uploaded)
    else:
        df_up = pd.read_excel(uploaded)

    df_up.columns = df_up.columns.str.strip()

    if "peptide" not in df_up.columns:
        st.error("File must contain a 'peptide' column.")
    else:
        feats_up = compute_features(df_up["peptide"].tolist())
        if not feats_up.empty:
            preds = model.predict(feats_up)
            df_up["Predicted_Taste"] = preds

            st.write("Batch Predictions:")
            st.dataframe(df_up)

            st.download_button(
                "Download Results",
                df_up.to_csv(index=False).encode(),
                file_name="predictions.csv",
                mime="text/csv"
            )
        else:
            st.error("Could not compute features for the sequences in the uploaded file.")
import py3Dmol

# -----------------------------
# 8. AlphaFold / ColabFold Section
# -----------------------------
st.subheader("🧬 AlphaFold/ColabFold Predicted Structure")

st.markdown("""
You can generate a predicted 3D structure for your peptide using **[ColabFold](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2_mmseqs2_advanced.ipynb)**:
1. Open the link above in Google Colab.
2. Paste your peptide sequence in FASTA format.
3. Run the notebook (needs a GPU runtime).
4. Download the predicted PDB file.
5. Upload the PDB file below to visualize it here.
""")

uploaded_pdb = st.file_uploader("Upload ColabFold/AlphaFold PDB file", type=["pdb"])

def show_structure(pdb_text):
    """Render a 3D structure from a PDB string using py3Dmol."""
    view = py3Dmol.view(width=600, height=500)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    return view

if uploaded_pdb is not None:
    pdb_content = uploaded_pdb.read().decode("utf-8")
    viewer = show_structure(pdb_content)
    # Embed the 3D viewer into Streamlit
    st.components.v1.html(viewer._make_html(), height=550)
else:
    st.info("No PDB uploaded yet. Use ColabFold to generate one.")
