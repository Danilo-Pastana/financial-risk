# Financial Risk Analyzer

Dashboard acadêmico para exploração descritiva do projeto de previsão de falência empresarial.

## Estrutura

```text
financial-risk-dashboard/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   └── dataset.csv
└── assets/
```

## Requisitos

- Python 3.10 ou superior
- pip

## Como executar no Windows (PowerShell)

Abra o terminal dentro da pasta do projeto e execute:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

O Streamlit abrirá o navegador. Normalmente, o endereço local será:

`http://localhost:8501`

## Dataset

A aplicação já procura automaticamente o arquivo `data/dataset.csv`.
Também é possível substituir o arquivo pela barra lateral, enviando outro CSV.

O CSV precisa conter as colunas:

- `company_name`
- `status_label`
- `year`

As variáveis financeiras `X1` até `X18` permanecem no dataset para análises futuras.

## O que esta versão faz

- Mostra a quantidade de registros;
- conta empresas distintas;
- apresenta o período analisado;
- calcula a proporção de registros `failed`;
- exibe a distribuição entre `alive` e `failed`;
- mostra a quantidade de registros por ano e situação;
- permite filtrar por período e situação;
- permite visualizar uma amostra dos dados.

## Limite desta versão

Esta é uma interface de análise exploratória. Ela **não treina modelos e não prevê falência**.
Uma funcionalidade de previsão só deverá ser adicionada depois que o modelo for desenvolvido e avaliado adequadamente.
