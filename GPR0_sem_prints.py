# Databricks notebook source
# MAGIC %pip install openpyxl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC Leitura do arquivo com índices SPI acumulados para cidade de Catalão: [ic_lamfo_spi_commodities/spi.xlsx](https://github.com/Pesquisa-UFCAT/ic_lamfo_spi_commodities/blob/main/spi.xlsx)
# MAGIC

# COMMAND ----------

import pandas as pd

df = pd.read_excel(
    "/Workspace/Users/lucas.cusinato@bb.com.br/spi/spi.xlsx",
    sheet_name="Sheet1"
)
df["AnoMes"] = pd.to_datetime(df["AnoMes"], format="%Y-%m")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Leitura do arquivo tratado com preços de milho em Goiás: [ic_lamfo_spi_commodities/PrecosGraos/Precos_Produtor_Tratado.xlsx](https://github.com/Pesquisa-UFCAT/ic_lamfo_spi_commodities/blob/main/PrecosGraos/Precos_Produtor_Tratado.xlsx)

# COMMAND ----------

df_milho = pd.read_excel(
    "/Workspace/Users/lucas.cusinato@bb.com.br/spi/Precos_Produtor_Tratado.xlsx",
    sheet_name="Milho_GO"
)
df_milho["Data"] = pd.to_datetime(df_milho["Data"], format="%m/%Y")
display(df_milho)

# COMMAND ----------

df_merged = df_milho.merge(df, left_on="Data", right_on="AnoMes", how="inner")
display(df_merged)

# COMMAND ----------

# MAGIC %md
# MAGIC Visualização da série de preços de milho em Goiás por Kg em dólares.

# COMMAND ----------

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(df_merged["Data"], df_merged["Preco_Kg_Dolar"])
plt.xlabel("Data")
plt.ylabel("Preço do Kg em dólares")

# COMMAND ----------

# MAGIC %md
# MAGIC Criação de variáveis com preço n meses antes, para auxiliar na predição.

# COMMAND ----------

import numpy as np

df_merged["Preco_Kg_Dolar_shift1"] = df_merged["Preco_Kg_Dolar"].shift(1)
df_merged["Preco_Kg_Dolar_shift2"] = df_merged["Preco_Kg_Dolar"].shift(2)
df_merged["Preco_Kg_Dolar_shift3"] = df_merged["Preco_Kg_Dolar"].shift(3)
df_merged["Preco_Kg_Dolar_shift6"] = df_merged["Preco_Kg_Dolar"].shift(6)
df_merged["Preco_Kg_Dolar_shift12"] = df_merged["Preco_Kg_Dolar"].shift(12)

df_merged["log_retorno_Preco_Kg_Dolar"] = np.log(df_merged["Preco_Kg_Dolar"] / df_merged["Preco_Kg_Dolar"].shift(1))
df_merged["log_retorno_Preco_Kg_Dolar_shift1"] = df_merged["log_retorno_Preco_Kg_Dolar"].shift(1)
df_merged["log_retorno_Preco_Kg_Dolar_shift2"] = df_merged["log_retorno_Preco_Kg_Dolar"].shift(2)
df_merged["log_retorno_Preco_Kg_Dolar_shift3"] = df_merged["log_retorno_Preco_Kg_Dolar"].shift(3)
df_merged["log_retorno_Preco_Kg_Dolar_shift6"] = df_merged["log_retorno_Preco_Kg_Dolar"].shift(6)
df_merged["log_retorno_Preco_Kg_Dolar_shift12"] = df_merged["log_retorno_Preco_Kg_Dolar"].shift(12)
display(df_merged)

# COMMAND ----------

# MAGIC %md
# MAGIC Ajuste de regressão com processo gaussiano para prever o preço do milho utilizando as variáveis de preços nos meses anteriores
# MAGIC
# MAGIC Medidas de erros e previsões do modelo foram apuradas para os 20% dos dados mais recentes, não utilizados no treino do modelo.

# COMMAND ----------

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Seleciona as features e target
features = ["Preco_Kg_Dolar_shift1", "Preco_Kg_Dolar_shift2", "Preco_Kg_Dolar_shift3", "Preco_Kg_Dolar_shift6", "Preco_Kg_Dolar_shift12"]
X = df_merged[features].values
y = df_merged["Preco_Kg_Dolar"].values

# Remove linhas com NaN
mask = ~pd.isnull(X).any(axis=1) & ~pd.isnull(y)
X = X[mask]
y = y[mask]

# Separa treino e teste (20% para teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define kernels conforme exemplo do scikit-learn
long_term_trend_kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=50.0, length_scale_bounds=(1.0, 1e3))
seasonal_kernel = C(1.0, (1e-2, 1e2)) * ExpSineSquared(length_scale=1.0, periodicity=12.0, length_scale_bounds=(1e-2, 1e2), periodicity_bounds=(1.0, 100.0))
irregularities_kernel = C(0.1, (1e-3, 1e1)) * RationalQuadratic(length_scale=1.0, alpha=1.0)
noise_kernel = WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))

kernel = long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel

# Ajusta o modelo
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
gpr.fit(X_train, y_train)

# Previsão
y_pred, y_std = gpr.predict(X_test, return_std=True)

# Avaliação
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}, R2: {r2:.4f}")

# Visualização
plt.figure(figsize=(10,5))
plt.plot(range(len(y)), y, 'k-', label="Observado")
plt.plot(range(len(X_train), len(y)), y_pred, 'b-', label="Previsto (GPR)")
plt.fill_between(range(len(X_train), len(y)), y_pred - 1.96*y_std, y_pred + 1.96*y_std, color='blue', alpha=0.2, label="95% IC")
plt.xlabel("Tempo")
plt.ylabel("Log-retorno do preço do Kg em dólares")
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC O modelo apresenta bom ajuste, mas sempre necessita do preço no mês imediatamente anterior (Preco_Kg_Dolar_shift2) para prever o seguinte.
# MAGIC
# MAGIC Um horizonte de predição de dois meses já traria muito mais utilidade para o modelo, visto que leva-se um tempo até consolidar os preços de cada mês (por UF). Além disso, ampliar o horizonte de predição auxiliaria nos processos de negociação desses grãos. Nesse sentido, tentamos ajustar esse modelo sem a variável Preco_Kg_Dolar_shift2 do mês imediatamente anterior, mas a qualidade das predições reduz significativamente:

# COMMAND ----------

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Seleciona as features e target
features = ["Preco_Kg_Dolar_shift2", "Preco_Kg_Dolar_shift3", "Preco_Kg_Dolar_shift6", "Preco_Kg_Dolar_shift12"]
X = df_merged[features].values
y = df_merged["Preco_Kg_Dolar"].values

# Remove linhas com NaN
mask = ~pd.isnull(X).any(axis=1) & ~pd.isnull(y)
X = X[mask]
y = y[mask]

# Separa treino e teste (20% para teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define kernels conforme exemplo do scikit-learn
long_term_trend_kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=50.0, length_scale_bounds=(1.0, 1e3))
seasonal_kernel = C(1.0, (1e-2, 1e2)) * ExpSineSquared(length_scale=1.0, periodicity=12.0, length_scale_bounds=(1e-2, 1e2), periodicity_bounds=(1.0, 100.0))
irregularities_kernel = C(0.1, (1e-3, 1e1)) * RationalQuadratic(length_scale=1.0, alpha=1.0)
noise_kernel = WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))

kernel = long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel

# Ajusta o modelo
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
gpr.fit(X_train, y_train)

# Previsão
y_pred, y_std = gpr.predict(X_test, return_std=True)

# Avaliação
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}, R2: {r2:.4f}")

# Visualização
plt.figure(figsize=(10,5))
plt.plot(range(len(y)), y, 'k-', label="Observado")
plt.plot(range(len(X_train), len(y)), y_pred, 'b-', label="Previsto (GPR)")
plt.fill_between(range(len(X_train), len(y)), y_pred - 1.96*y_std, y_pred + 1.96*y_std, color='blue', alpha=0.2, label="95% IC")
plt.xlabel("Tempo")
plt.ylabel("Log-retorno do preço do Kg em dólares")
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Para compensar a redução na capacidade preditiva do modelo ao se retirar os preços do mês imediatamente anterior, incluímos como covariáveis na regressão os índices de precipitação SPI acumulados em diferentes períodos. 
# MAGIC
# MAGIC Tentamos modelar os preços incluindo os índices de precipitação SPI acumulados em diferentes períodos, mas a inclusão direta desses índices como covariáveis não aprimorou a qualidade do ajuste do modelo. Todas as tentativas de ajuste incluindo o histórico SPI tiveram performance preditiva inferior no período de teste futuro - 20% dos dados mais recentes.
# MAGIC
# MAGIC A seguir ajustamos trocando a variável do preço do mês anterior pelo índice acumulado SPI do mesmo mês:

# COMMAND ----------

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Seleciona as features e target
features = ["SPI_1_shift", "Preco_Kg_Dolar_shift3", "Preco_Kg_Dolar_shift6", "Preco_Kg_Dolar_shift12"]
X = df_merged[features].values
y = df_merged["Preco_Kg_Dolar"].values

# Remove linhas com NaN
mask = ~pd.isnull(X).any(axis=1) & ~pd.isnull(y)
X = X[mask]
y = y[mask]

# Separa treino e teste (20% para teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define kernels conforme exemplo do scikit-learn
long_term_trend_kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=50.0, length_scale_bounds=(1.0, 1e3))
seasonal_kernel = C(1.0, (1e-2, 1e2)) * ExpSineSquared(length_scale=1.0, periodicity=12.0, length_scale_bounds=(1e-2, 1e2), periodicity_bounds=(1.0, 100.0))
irregularities_kernel = C(0.1, (1e-3, 1e1)) * RationalQuadratic(length_scale=1.0, alpha=1.0)
noise_kernel = WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))

kernel = long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel

# Ajusta o modelo
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
gpr.fit(X_train, y_train)

# Previsão
y_pred, y_std = gpr.predict(X_test, return_std=True)

# Avaliação
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}, R2: {r2:.4f}")

# Visualização
plt.figure(figsize=(10,5))
plt.plot(range(len(y)), y, 'k-', label="Observado")
plt.plot(range(len(X_train), len(y)), y_pred, 'b-', label="Previsto (GPR)")
plt.fill_between(range(len(X_train), len(y)), y_pred - 1.96*y_std, y_pred + 1.96*y_std, color='blue', alpha=0.2, label="95% IC")
plt.xlabel("Tempo")
plt.ylabel("Preço do Kg em dólares")
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Novos modelos incluindo, uma a uma, todas as colunas de índice SPI acumulado disponíveis na planilha inicial, mantendo todas as variáveis de preços nos meses anteriores. As medidas de erro não tiveram alteração significativa em nenhuma das tentativas de inclusão de índice SPI acumulado:

# COMMAND ----------

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

for ii_SPI in [1, 2, 3, 6, 12, 24]:
    # Seleciona as features e target
    features = [f"SPI_{ii_SPI}", "Preco_Kg_Dolar_shift1", "Preco_Kg_Dolar_shift2", "Preco_Kg_Dolar_shift3", "Preco_Kg_Dolar_shift6", "Preco_Kg_Dolar_shift12"]
    X = df_merged[features].values
    y = df_merged["Preco_Kg_Dolar"].values

    # Remove linhas com NaN
    mask = ~pd.isnull(X).any(axis=1) & ~pd.isnull(y)
    X = X[mask]
    y = y[mask]

    # Separa treino e teste (20% para teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Define kernels conforme exemplo do scikit-learn
    long_term_trend_kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=50.0, length_scale_bounds=(1.0, 1e3))
    seasonal_kernel = C(1.0, (1e-2, 1e2)) * ExpSineSquared(length_scale=1.0, periodicity=12.0, length_scale_bounds=(1e-2, 1e2), periodicity_bounds=(1.0, 100.0))
    irregularities_kernel = C(0.1, (1e-3, 1e1)) * RationalQuadratic(length_scale=1.0, alpha=1.0)
    noise_kernel = WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))

    kernel = long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel

    # Ajusta o modelo
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
    gpr.fit(X_train, y_train)

    # Previsão
    y_pred, y_std = gpr.predict(X_test, return_std=True)

    # Avaliação
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.4f}, R2: {r2:.4f}")

    # Visualização
    plt.figure(figsize=(10,5))
    plt.plot(range(len(y)), y, 'k-', label="Observado")
    plt.plot(range(len(X_train), len(y)), y_pred, 'b-', label="Previsto (GPR)")
    plt.fill_between(range(len(X_train), len(y)), y_pred - 1.96*y_std, y_pred + 1.96*y_std, color='blue', alpha=0.2, label="95% IC")
    plt.xlabel("Tempo")
    plt.ylabel(f"Preço do Kg em dólares - SPI_{ii_SPI}")
    plt.legend()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Repetição dos ajustes acima, mas considerando os índices SPI acumulados até um mês anterior - na variável SPI_shifted. Os resultados também são semelhantes ao ajuste inicial sem os índices SPI:

# COMMAND ----------

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

for ii_SPI in [1, 2, 3, 6, 12, 24]:
    # Seleciona as features e target
    features = ["SPI_shifted", "Preco_Kg_Dolar_shift1", "Preco_Kg_Dolar_shift2", "Preco_Kg_Dolar_shift3", "Preco_Kg_Dolar_shift6", "Preco_Kg_Dolar_shift12"]
    df_merged["SPI_shifted"] = df_merged[f"SPI_{ii_SPI}"].shift(1)
    X = df_merged[features].values
    y = df_merged["Preco_Kg_Dolar"].values

    # Remove linhas com NaN
    mask = ~pd.isnull(X).any(axis=1) & ~pd.isnull(y)
    X = X[mask]
    y = y[mask]

    # Separa treino e teste (20% para teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Define kernels conforme exemplo do scikit-learn
    long_term_trend_kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=50.0, length_scale_bounds=(1.0, 1e3))
    seasonal_kernel = C(1.0, (1e-2, 1e2)) * ExpSineSquared(length_scale=1.0, periodicity=12.0, length_scale_bounds=(1e-2, 1e2), periodicity_bounds=(1.0, 100.0))
    irregularities_kernel = C(0.1, (1e-3, 1e1)) * RationalQuadratic(length_scale=1.0, alpha=1.0)
    noise_kernel = WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))

    kernel = long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel

    # Ajusta o modelo
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
    gpr.fit(X_train, y_train)

    # Previsão
    y_pred, y_std = gpr.predict(X_test, return_std=True)

    # Avaliação
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.4f}, R2: {r2:.4f}")

    # Visualização
    plt.figure(figsize=(10,5))
    plt.plot(range(len(y)), y, 'k-', label="Observado")
    plt.plot(range(len(X_train), len(y)), y_pred, 'b-', label="Previsto (GPR)")
    plt.fill_between(range(len(X_train), len(y)), y_pred - 1.96*y_std, y_pred + 1.96*y_std, color='blue', alpha=0.2, label="95% IC")
    plt.xlabel("Tempo")
    plt.ylabel(f"Preço do Kg em dólares - SPI_{ii_SPI}(lag 1)")
    plt.legend()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Em uma última tentativa, procuramos identificar se há melhoria ao incluir índices SPI acumulados por diferentes meses e para alguns meses anteriores ao mês da predição.

# COMMAND ----------

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

for ii_shift in [1, 2, 3, 6]:
    for ii_acumulado in [1, 2, 3, 6, 12, 24]:
    # Seleciona as features e target
        features = ["SPI_{ii_acumulado}_shifted_{ii_shift}", "Preco_Kg_Dolar_shift3", "Preco_Kg_Dolar_shift6", "Preco_Kg_Dolar_shift12"]
        df_merged["SPI_{ii_acumulado}_shifted_{ii_shift}"] = df_merged[f"SPI_{ii_acumulado}"].shift(ii_shift)
        X = df_merged[features].values
        y = df_merged["Preco_Kg_Dolar"].values

        # Remove linhas com NaN
        mask = ~pd.isnull(X).any(axis=1) & ~pd.isnull(y)
        X = X[mask]
        y = y[mask]

        # Separa treino e teste (20% para teste)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Define kernels conforme exemplo do scikit-learn
        long_term_trend_kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=50.0, length_scale_bounds=(1.0, 1e3))
        seasonal_kernel = C(1.0, (1e-2, 1e2)) * ExpSineSquared(length_scale=1.0, periodicity=12.0, length_scale_bounds=(1e-2, 1e2), periodicity_bounds=(1.0, 100.0))
        irregularities_kernel = C(0.1, (1e-3, 1e1)) * RationalQuadratic(length_scale=1.0, alpha=1.0)
        noise_kernel = WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))

        kernel = long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel

        # Ajusta o modelo
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
        gpr.fit(X_train, y_train)

        # Previsão
        y_pred, y_std = gpr.predict(X_test, return_std=True)

        # Avaliação
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"MSE: {mse:.4f}, R2: {r2:.4f}")

        # Visualização
        plt.figure(figsize=(10,5))
        plt.plot(range(len(y)), y, 'k-', label="Observado")
        plt.plot(range(len(X_train), len(y)), y_pred, 'b-', label="Previsto (GPR)")
        plt.fill_between(range(len(X_train), len(y)), y_pred - 1.96*y_std, y_pred + 1.96*y_std, color='blue', alpha=0.2, label="95% IC")
        plt.xlabel("Tempo")
        plt.ylabel(f"Preço do Kg em dólares - SPI_{ii_acumulado} shifted {ii_shift} meses")
        plt.legend()
        plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Agora tudo em log-retorno
# MAGIC
# MAGIC Nenhum prestou então essa parte do notebook está pouco comentada.

# COMMAND ----------

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(df_merged["Data"], df_merged["log_retorno_Preco_Kg_Dolar"])
plt.xlabel("Data")
plt.ylabel("Log-retorno do preço do Kg em dólares")

# COMMAND ----------

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Seleciona as features e target
features = ["log_retorno_Preco_Kg_Dolar_shift1", "log_retorno_Preco_Kg_Dolar_shift2", "log_retorno_Preco_Kg_Dolar_shift3", "log_retorno_Preco_Kg_Dolar_shift6", "log_retorno_Preco_Kg_Dolar_shift12"]
X = df_merged[features].values
y = df_merged["log_retorno_Preco_Kg_Dolar"].values

# Remove linhas com NaN
mask = ~pd.isnull(X).any(axis=1) & ~pd.isnull(y)
X = X[mask]
y = y[mask]

# Separa treino e teste (20% para teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define kernels conforme exemplo do scikit-learn
long_term_trend_kernel = C(1.0, (1e-9, 1e2)) * RBF(length_scale=50.0, length_scale_bounds=(1.0, 1e3))
seasonal_kernel = C(1.0, (2e-3, 1e2)) * ExpSineSquared(length_scale=1.0, periodicity=12.0, length_scale_bounds=(1e-2, 1e2), periodicity_bounds=(1.0, 10000.0))
irregularities_kernel = C(0.1, (3e-4, 1e1)) * RationalQuadratic(length_scale=1.0, alpha=1.0, alpha_bounds=(1e-05, 100_000_000.0))
noise_kernel = WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))

kernel = long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel

# Ajusta o modelo
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
gpr.fit(X_train, y_train)

# Previsão
y_pred, y_std = gpr.predict(X_test, return_std=True)

# Avaliação
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}, R2: {r2:.4f}")

# Visualização
plt.figure(figsize=(10,5))
plt.plot(range(len(y)), y, 'k-', label="Observado")
plt.plot(range(len(X_train), len(y)), y_pred, 'b-', label="Previsto (GPR)")
plt.fill_between(range(len(X_train), len(y)), y_pred - 1.96*y_std, y_pred + 1.96*y_std, color='blue', alpha=0.2, label="95% IC")
plt.xlabel("Tempo")
plt.ylabel("Log-retorno do preço do Kg em dólares")
plt.legend()
plt.show()

# COMMAND ----------

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

for ii_SPI in [1, 2, 3, 6, 12, 24]:
    # Seleciona as features e target
    features = [f"SPI_{ii_SPI}", "log_retorno_Preco_Kg_Dolar_shift1", "log_retorno_Preco_Kg_Dolar_shift2", "log_retorno_Preco_Kg_Dolar_shift3", "log_retorno_Preco_Kg_Dolar_shift6", "log_retorno_Preco_Kg_Dolar_shift12"]
    X = df_merged[features].values
    y = df_merged["log_retorno_Preco_Kg_Dolar"].values

    # Remove linhas com NaN
    mask = ~pd.isnull(X).any(axis=1) & ~pd.isnull(y)
    X = X[mask]
    y = y[mask]

    # Separa treino e teste (20% para teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Define kernels conforme exemplo do scikit-learn
    long_term_trend_kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=50.0, length_scale_bounds=(1.0, 1e3))
    seasonal_kernel = C(1.0, (1e-2, 1e2)) * ExpSineSquared(length_scale=1.0, periodicity=12.0, length_scale_bounds=(1e-2, 1e2), periodicity_bounds=(1.0, 100.0))
    irregularities_kernel = C(0.1, (1e-3, 1e1)) * RationalQuadratic(length_scale=1.0, alpha=1.0)
    noise_kernel = WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))

    kernel = long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel

    # Ajusta o modelo
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
    gpr.fit(X_train, y_train)

    # Previsão
    y_pred, y_std = gpr.predict(X_test, return_std=True)

    # Avaliação
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.4f}, R2: {r2:.4f}")

    # Visualização
    plt.figure(figsize=(10,5))
    plt.plot(range(len(y)), y, 'k-', label="Observado")
    plt.plot(range(len(X_train), len(y)), y_pred, 'b-', label="Previsto (GPR)")
    plt.fill_between(range(len(X_train), len(y)), y_pred - 1.96*y_std, y_pred + 1.96*y_std, color='blue', alpha=0.2, label="95% IC")
    plt.xlabel("Tempo")
    plt.ylabel(f"Log-retorno do preço do Kg em dólares - SPI_{ii_SPI}")
    plt.legend()
    plt.show()

# COMMAND ----------

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

for ii_SPI in [1, 2, 3, 6, 12, 24]:
    # Seleciona as features e target
    features = ["SPI_shifted", "log_retorno_Preco_Kg_Dolar_shift1", "log_retorno_Preco_Kg_Dolar_shift2", "log_retorno_Preco_Kg_Dolar_shift3", "log_retorno_Preco_Kg_Dolar_shift6", "log_retorno_Preco_Kg_Dolar_shift12"]
    df_merged["SPI_shifted"] = df_merged[f"SPI_{ii_SPI}"].shift(1)
    X = df_merged[features].values
    y = df_merged["log_retorno_Preco_Kg_Dolar"].values

    # Remove linhas com NaN
    mask = ~pd.isnull(X).any(axis=1) & ~pd.isnull(y)
    X = X[mask]
    y = y[mask]

    # Separa treino e teste (20% para teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Define kernels conforme exemplo do scikit-learn
    long_term_trend_kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=50.0, length_scale_bounds=(1.0, 1e3))
    seasonal_kernel = C(1.0, (1e-2, 1e2)) * ExpSineSquared(length_scale=1.0, periodicity=12.0, length_scale_bounds=(1e-2, 1e2), periodicity_bounds=(1.0, 100.0))
    irregularities_kernel = C(0.1, (1e-3, 1e1)) * RationalQuadratic(length_scale=1.0, alpha=1.0)
    noise_kernel = WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))

    kernel = long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel

    # Ajusta o modelo
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
    gpr.fit(X_train, y_train)

    # Previsão
    y_pred, y_std = gpr.predict(X_test, return_std=True)

    # Avaliação
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.4f}, R2: {r2:.4f}")

    # Visualização
    plt.figure(figsize=(10,5))
    plt.plot(range(len(y)), y, 'k-', label="Observado")
    plt.plot(range(len(X_train), len(y)), y_pred, 'b-', label="Previsto (GPR)")
    plt.fill_between(range(len(X_train), len(y)), y_pred - 1.96*y_std, y_pred + 1.96*y_std, color='blue', alpha=0.2, label="95% IC")
    plt.xlabel("Tempo")
    plt.ylabel(f"Log-retorno do preço do Kg em dólares - SPI_{ii_SPI}(lag 1)")
    plt.legend()
    plt.show()