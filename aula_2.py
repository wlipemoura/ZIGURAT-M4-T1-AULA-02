import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#Configurações visuais
st.set_page_config(page_title="Análise Exploratória de Dados - Aula 2", layout="wide")

st.image("logo.png", width=320)

st.title("Análise Exploratória de Dados - Aula 2")


# Upload do dataset
st.header("Upload do Dataset")

uploaded_file = st.file_uploader(
    "Envie o arquivo CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Faça o upload do arquivo para continuar.")
    st.stop()

df = pd.read_csv(uploaded_file, delimiter=";")
st.success("Dataset carregado com sucesso!")

# Visualização inicial
st.header("Visualização dos dados")

num_linhas = st.slider(
    "Quantidade de linhas para visualizar",
    min_value=5,
    max_value=50,
    value=10
)

st.dataframe(df.head(num_linhas))


hue_col = st.selectbox("Escolha a coluna para hue (pairplot)", df.columns)

if st.button("Gerar Pairplot"):
    fig = sns.pairplot(df, hue=hue_col)
    st.pyplot(fig)
    plt.clf()


# Gráfico outlook x status
st.subheader("Distribuição de outlook por status")

fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.countplot(y="outlook", hue="status", data=df, ax=ax1)
st.pyplot(fig1)
plt.clf()


# Gráfico temperature x status
st.subheader("Distribuição de temperature por status")

fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.countplot(y="temperature", hue="status", data=df, ax=ax2)
st.pyplot(fig2)
plt.clf()


# Separando inputs e outputs
st.header("Preparação dos Dados")

X = df.drop("status", axis=1)
y = df["status"]

st.write("**Inputs (X):**")
st.dataframe(X.head())


# Normalização
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)


# Configuração do modelo
st.header("Configuração do Decision Tree")

max_depth = st.slider("Max depth", 1, 10, 6)
max_leaf_nodes = st.slider("Max leaf nodes", 2, 10, 4)

dt = DecisionTreeClassifier(
    max_depth=max_depth,
    max_leaf_nodes=max_leaf_nodes
)


# Treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, train_size=2/3, random_state=42
)

dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)


# Avaliação
st.header("Avaliação do Modelo")

accuracy = accuracy_score(y_test, y_pred)
st.metric("Acurácia", f"{accuracy:.2f}")

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))


# Matriz de confusão
st.subheader("Matriz de Confusão")

cm = confusion_matrix(y_test, y_pred)

fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
ax_cm.set_xlabel("Previsto")
ax_cm.set_ylabel("Real")
st.pyplot(fig_cm)
plt.clf()


# Visualização da árvore
st.header("Visualização da Árvore de Decisão")

fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
plot_tree(
    dt,
    feature_names=X.columns,
    class_names=[str(c) for c in dt.classes_],
    filled=True,
    rounded=True,
    ax=ax_tree
)
st.pyplot(fig_tree)
plt.clf()


#Previsão
st.header("Previsão de novo cenário")

st.write("Informe os valores para gerar uma previsão:")

col1, col2, col3, col4 = st.columns(4)

with col1:
    v1 = st.number_input("Atributo 1 (outlook)", value=10)
with col2:
    v2 = st.number_input("Atributo 2 (temperature)", value=20)
with col3:
    v3 = st.number_input("Atributo 3 (humidity)", value=30)
with col4:
    v4 = st.number_input("Atributo 4 (windy)", value=40)

if st.button("Prever"):
    novo_cenario = [[v1, v2, v3, v4]]

    X_new = scaler.transform(novo_cenario)

    previsao = dt.predict(X_new)

    st.success(f"Previsão do modelo: **{previsao[0]}**")