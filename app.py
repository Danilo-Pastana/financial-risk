import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import re
import io

# ---------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------
st.set_page_config(
    page_title="Analisador de Risco Financeiro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Estilo visual
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2rem;
        }
        [data-testid="stMetric"] {
            background: #1A1D24;
            border: 1px solid #30343B;
            padding: 18px;
            border-radius: 12px;
        }
        [data-testid="stMetricLabel"] {
            color: #A7AAB3;
        }
        [data-testid="stMetricValue"] {
            font-size: clamp(1.1rem, 1.8vw, 1.7rem);
            white-space: nowrap;
        }
        h1, h2, h3 {
            letter-spacing: -0.3px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

DATA_PATH = Path("data/dataset.csv")
VALID_STATUSES = ["alive", "failed"]


# ---------------------------------------------------------
# Funções de carregamento
# ---------------------------------------------------------
@st.cache_data
def load_csv(file_or_path):
    return pd.read_csv(file_or_path)


@st.cache_data
def load_arff(file_bytes):
    """Lê os arquivos ARFF da UCI sem exigir biblioteca adicional."""
    content = file_bytes.decode("utf-8", errors="replace")
    lines = content.splitlines()

    attributes = []
    data_start = None

    for index, line in enumerate(lines):
        stripped = line.strip()

        if stripped.lower().startswith("@attribute"):
            match = re.match(
                r"@attribute\s+(['\"]?)(.*?)\1\s+(.+)",
                stripped,
                flags=re.IGNORECASE
            )
            if match:
                attributes.append(match.group(2))

        if stripped.lower() == "@data":
            data_start = index + 1
            break

    if data_start is None or not attributes:
        raise ValueError("O arquivo não possui um cabeçalho ARFF válido.")

    data_text = "\n".join(
        line for line in lines[data_start:]
        if line.strip() and not line.strip().startswith("%")
    )

    df = pd.read_csv(
        io.StringIO(data_text),
        header=None,
        names=attributes,
        na_values=["?", " ?"]
    )

    return df


def format_number(value):
    return f"{value:,.0f}".replace(",", ".")


# ---------------------------------------------------------
# Cabeçalho e navegação
# ---------------------------------------------------------
st.title("Analisador de Risco Financeiro")
st.caption(
    "Projeto acadêmico de Machine Learning para análise "
    "de dados financeiros e estudo de falência empresarial."
)

st.sidebar.title("Navegação")
st.sidebar.caption("Projeto acadêmico de Machine Learning")

dataset_choice = st.sidebar.radio(
    "Base de dados",
    [
        "Dataset original",
        "Polish Companies Bankruptcy (UCI)",
        "Taiwanese Bankruptcy Prediction (UCI)"
    ]
)

# ---------------------------------------------------------
# Modo 1: Dataset original
# ---------------------------------------------------------
if dataset_choice == "Dataset original":
    uploaded_file = st.sidebar.file_uploader(
        "Substituir dataset original (CSV)",
        type=["csv"]
    )

    try:
        if uploaded_file is not None:
            df = load_csv(uploaded_file)
            source_label = "Arquivo CSV enviado"
        elif DATA_PATH.exists():
            df = load_csv(DATA_PATH)
            source_label = "Dataset local do projeto"
        else:
            st.info(
                "Não encontrei data/dataset.csv. Coloque o arquivo nessa "
                "pasta ou envie um CSV pela barra lateral."
            )
            st.stop()
    except Exception as error:
        st.error(f"Não foi possível carregar o CSV: {error}")
        st.stop()

    required_columns = {"company_name", "status_label", "year"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        st.error(
            "O arquivo não contém todas as colunas esperadas. "
            f"Colunas ausentes: {', '.join(sorted(missing_columns))}"
        )
        st.stop()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["status_label"] = (
        df["status_label"].astype(str).str.strip().str.lower()
    )
    df = df[df["status_label"].isin(VALID_STATUSES)].copy()

    if df.empty:
        st.warning("Não há registros com status 'alive' ou 'failed'.")
        st.stop()

    st.sidebar.subheader("Filtros")
    years = sorted(df["year"].dropna().astype(int).unique().tolist())

    if years:
        selected_years = st.sidebar.slider(
            "Período",
            min_value=int(min(years)),
            max_value=int(max(years)),
            value=(int(min(years)), int(max(years)))
        )
        filtered_df = df[
            df["year"].between(selected_years[0], selected_years[1])
        ].copy()
    else:
        filtered_df = df.copy()

    selected_status = st.sidebar.multiselect(
        "Situação da empresa",
        options=VALID_STATUSES,
        default=VALID_STATUSES
    )
    filtered_df = filtered_df[
        filtered_df["status_label"].isin(selected_status)
    ]

    if filtered_df.empty:
        st.warning("Nenhum registro encontrado para os filtros selecionados.")
        st.stop()

    st.subheader("Visão geral")

    total_records = len(filtered_df)
    total_companies = filtered_df["company_name"].nunique()
    period_start = (
        int(filtered_df["year"].min())
        if filtered_df["year"].notna().any() else "—"
    )
    period_end = (
        int(filtered_df["year"].max())
        if filtered_df["year"].notna().any() else "—"
    )
    failed_percentage = (
        filtered_df["status_label"].eq("failed").mean() * 100
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de registros", format_number(total_records))
    col2.metric("Empresas distintas", format_number(total_companies))
    col3.metric("Período analisado", f"{period_start}–{period_end}")
    col4.metric("Registros failed", f"{failed_percentage:.2f}%")

    st.caption(
        f"Fonte: {source_label}. Os indicadores respeitam os filtros."
    )
    st.divider()

    st.subheader("Distribuição da situação empresarial")
    distribution = (
        filtered_df["status_label"]
        .value_counts()
        .reindex(VALID_STATUSES, fill_value=0)
        .rename_axis("Situação")
        .reset_index(name="Quantidade")
    )
    distribution["Percentual"] = (
        distribution["Quantidade"] / distribution["Quantidade"].sum() * 100
    ).round(2)

    chart_col, table_col = st.columns([1.7, 1])

    with chart_col:
        fig = px.pie(
            distribution,
            names="Situação",
            values="Quantidade",
            hole=0.62,
            color="Situação",
            color_discrete_map={"alive": "#22C55E", "failed": "#EF4444"}
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=25, b=15, l=10, r=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with table_col:
        st.markdown("#### Dados por categoria")
        st.dataframe(distribution, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Registros ao longo dos anos")

    temporal = (
        filtered_df.dropna(subset=["year"])
        .groupby(["year", "status_label"])
        .size()
        .reset_index(name="Quantidade")
    )

    if not temporal.empty:
        temporal["year"] = temporal["year"].astype(int)
        fig_year = px.line(
            temporal,
            x="year",
            y="Quantidade",
            color="status_label",
            markers=True,
            color_discrete_map={"alive": "#22C55E", "failed": "#EF4444"},
            labels={
                "year": "Ano",
                "Quantidade": "Quantidade de registros",
                "status_label": "Situação"
            }
        )
        fig_year.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_year, use_container_width=True)

    with st.expander("Visualizar registros do dataset"):
        st.dataframe(filtered_df.head(100), use_container_width=True)
        st.caption("Exibindo no máximo 100 registros.")

# ---------------------------------------------------------
# Modo 2: Dataset Polish Companies Bankruptcy (UCI)
# ---------------------------------------------------------
elif dataset_choice == "Polish Companies Bankruptcy (UCI)":
    st.subheader("Polish Companies Bankruptcy Prediction — UCI")
    st.write(
        "Envie um dos arquivos ARFF da UCI. Os arquivos representam "
        "horizontes de previsão diferentes: de 1 a 5 anos."
    )

    arff_file = st.sidebar.file_uploader(
        "Enviar arquivo ARFF",
        type=["arff"],
        help="Selecione 1year.arff, 2year.arff, 3year.arff, 4year.arff ou 5year.arff."
    )

    if arff_file is None:
        st.info(
            "Selecione um dos cinco arquivos ARFF na barra lateral para "
            "visualizar os dados da UCI. O dataset original permanece preservado."
        )
        st.stop()

    try:
        uci_df = load_arff(arff_file.getvalue())
    except Exception as error:
        st.error(f"Não foi possível interpretar o arquivo ARFF: {error}")
        st.stop()

    if "class" not in uci_df.columns:
        st.error("O arquivo ARFF não possui a coluna-alvo 'class'.")
        st.stop()

    feature_columns = [
        column for column in uci_df.columns if column != "class"
    ]

    uci_df["class"] = pd.to_numeric(uci_df["class"], errors="coerce")
    uci_df = uci_df.dropna(subset=["class"]).copy()

    if uci_df.empty:
        st.warning("O arquivo não contém registros válidos.")
        st.stop()

    class_counts = (
        uci_df["class"].value_counts().sort_index()
        .rename_axis("Classe")
        .reset_index(name="Quantidade")
    )
    class_counts["Percentual"] = (
        class_counts["Quantidade"] / class_counts["Quantidade"].sum() * 100
    ).round(2)

    st.caption(
        f"Arquivo carregado: {arff_file.name}. "
        "Esta base possui 64 atributos financeiros e uma coluna-alvo class."
    )

    total_records = len(uci_df)
    total_features = len(feature_columns)
    missing_values = int(uci_df[feature_columns].isna().sum().sum())
    class_one_percentage = (
        uci_df["class"].eq(1).mean() * 100
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de registros", format_number(total_records))
    col2.metric("Variáveis financeiras", total_features)
    col3.metric("Valores ausentes", format_number(missing_values))
    col4.metric("Registros da classe 1", f"{class_one_percentage:.2f}%")

    st.divider()
    st.subheader("Distribuição da variável-alvo")

    chart_col, table_col = st.columns([1.7, 1])

    with chart_col:
        fig_class = px.pie(
            class_counts,
            names="Classe",
            values="Quantidade",
            hole=0.62,
            color="Classe",
            color_discrete_map={0: "#22C55E", 1: "#EF4444"}
        )
        fig_class.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=25, b=15, l=10, r=10)
        )
        st.plotly_chart(fig_class, use_container_width=True)

    with table_col:
        st.markdown("#### Registros por classe")
        st.dataframe(class_counts, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Análise dos indicadores financeiros")

    selected_feature = st.selectbox(
        "Selecione uma variável financeira",
        options=feature_columns
    )

    feature_values = pd.to_numeric(
        uci_df[selected_feature], errors="coerce"
    ).dropna()

    if not feature_values.empty:
        fig_hist = px.histogram(
            uci_df,
            x=selected_feature,
            color="class",
            nbins=40,
            barmode="overlay",
            opacity=0.7,
            color_discrete_map={0: "#22C55E", 1: "#EF4444"}
        )
        fig_hist.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title=selected_feature,
            yaxis_title="Quantidade"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### Estatísticas descritivas")
        stats = uci_df[feature_columns].describe().T
        st.dataframe(
            stats.loc[[selected_feature]],
            use_container_width=True
        )

    with st.expander("Visualizar registros do arquivo UCI"):
        st.dataframe(uci_df.head(100), use_container_width=True)
        st.caption("Exibindo no máximo 100 registros.")

    st.info(
        "Observação: esta visualização é descritiva. A coluna 'class' é "
        "a variável-alvo fornecida no arquivo. Nenhum modelo de previsão "
        "está sendo treinado ou executado nesta versão."
    )



# ---------------------------------------------------------
# Modo 3: Dataset Taiwanese Bankruptcy Prediction (UCI)
# ---------------------------------------------------------
else:
    st.subheader("Taiwanese Bankruptcy Prediction — UCI")
    st.write(
        "Envie o arquivo data.csv da UCI. Esta base reúne registros "
        "financeiros de empresas taiwanesas e uma variável-alvo binária."
    )

    taiwan_file = st.sidebar.file_uploader(
        "Enviar dataset taiwanês (CSV)",
        type=["csv"],
        help="Selecione o arquivo data.csv baixado na página da UCI."
    )

    if taiwan_file is None:
        st.info(
            "Baixe o arquivo data.csv na página oficial da UCI e envie-o "
            "pela barra lateral. O dataset original e o polonês permanecem disponíveis."
        )
        st.markdown(
            "[Abrir Taiwanese Bankruptcy Prediction na UCI]"
            "(https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction)"
        )
        st.stop()

    try:
        taiwan_df = load_csv(taiwan_file)
    except Exception as error:
        st.error(f"Não foi possível carregar o CSV: {error}")
        st.stop()

    # Remove espaços extras dos nomes das colunas.
    taiwan_df.columns = taiwan_df.columns.astype(str).str.strip()

    # A UCI disponibiliza a variável-alvo com o nome 'Bankrupt?'.
    target_column = next(
        (column for column in taiwan_df.columns
         if column.strip().lower() == "bankrupt?"),
        None
    )

    if target_column is None:
        st.error(
            "Não encontrei a coluna-alvo 'Bankrupt?'. "
            "Verifique se você enviou o data.csv correto da UCI."
        )
        st.stop()

    taiwan_df[target_column] = pd.to_numeric(
        taiwan_df[target_column], errors="coerce"
    )
    taiwan_df = taiwan_df.dropna(subset=[target_column]).copy()
    taiwan_df = taiwan_df[
        taiwan_df[target_column].isin([0, 1])
    ].copy()

    if taiwan_df.empty:
        st.warning("O arquivo não contém registros válidos.")
        st.stop()

    feature_columns = [
        column for column in taiwan_df.columns
        if column != target_column
        and pd.api.types.is_numeric_dtype(taiwan_df[column])
    ]

    if not feature_columns:
        st.error("Não encontrei indicadores financeiros numéricos no arquivo.")
        st.stop()

    class_counts = (
        taiwan_df[target_column]
        .value_counts()
        .reindex([0, 1], fill_value=0)
        .rename_axis("Classe")
        .reset_index(name="Quantidade")
    )
    class_counts["Situação"] = class_counts["Classe"].map({
        0: "Não falida",
        1: "Falida"
    })
    class_counts["Percentual"] = (
        class_counts["Quantidade"] / class_counts["Quantidade"].sum() * 100
    ).round(2)

    total_records = len(taiwan_df)
    total_features = len(feature_columns)
    missing_values = int(taiwan_df[feature_columns].isna().sum().sum())
    bankrupt_percentage = taiwan_df[target_column].eq(1).mean() * 100

    st.caption(
        f"Arquivo carregado: {taiwan_file.name}. "
        "Fonte: Taiwanese Bankruptcy Prediction — UCI Machine Learning Repository."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de registros", format_number(total_records))
    col2.metric("Indicadores financeiros", total_features)
    col3.metric("Valores ausentes", format_number(missing_values))
    col4.metric("Registros de empresas falidas", f"{bankrupt_percentage:.2f}%")

    st.divider()
    st.subheader("Distribuição da situação empresarial")

    chart_col, table_col = st.columns([1.7, 1])

    with chart_col:
        fig_taiwan = px.pie(
            class_counts,
            names="Situação",
            values="Quantidade",
            hole=0.62,
            color="Situação",
            color_discrete_map={
                "Não falida": "#22C55E",
                "Falida": "#EF4444"
            }
        )
        fig_taiwan.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=25, b=15, l=10, r=10)
        )
        st.plotly_chart(fig_taiwan, use_container_width=True)

    with table_col:
        st.markdown("#### Registros por situação")
        st.dataframe(
            class_counts[["Situação", "Quantidade", "Percentual"]],
            use_container_width=True,
            hide_index=True
        )

    st.divider()
    st.subheader("Análise dos indicadores financeiros")

    selected_feature = st.selectbox(
        "Selecione uma variável financeira",
        options=feature_columns
    )

    feature_values = pd.to_numeric(
        taiwan_df[selected_feature], errors="coerce"
    ).dropna()

    if not feature_values.empty:
        plot_df = taiwan_df.copy()
        plot_df["Situação"] = plot_df[target_column].map({
            0: "Não falida",
            1: "Falida"
        })

        fig_hist = px.histogram(
            plot_df,
            x=selected_feature,
            color="Situação",
            nbins=40,
            barmode="overlay",
            opacity=0.7,
            color_discrete_map={
                "Não falida": "#22C55E",
                "Falida": "#EF4444"
            }
        )
        fig_hist.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title=selected_feature,
            yaxis_title="Quantidade"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### Estatísticas descritivas")
        stats = taiwan_df[feature_columns].describe().T
        st.dataframe(
            stats.loc[[selected_feature]],
            use_container_width=True
        )

    with st.expander("Visualizar registros do dataset"):
        st.dataframe(taiwan_df.head(100), use_container_width=True)
        st.caption("Exibindo no máximo 100 registros.")

    st.info(
        "Nesta base, a variável 'Bankrupt?' indica a situação empresarial: "
        "0 = não falida e 1 = falida. Esta visualização é descritiva; "
        "nenhum modelo de Machine Learning está sendo treinado ou executado."
    )

# ---------------------------------------------------------
# Rodapé
# ---------------------------------------------------------
st.divider()
st.caption(
    "Projeto acadêmico — Previsão de Falência Empresarial utilizando "
    "Machine Learning. Esta versão realiza análises descritivas e "
    "ainda não executa previsões."
)
