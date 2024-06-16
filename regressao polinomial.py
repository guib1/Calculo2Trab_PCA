import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('steam-games.csv')

print("Cabe�alho do DataFrame:")
print(df.head())

print("\nTipos de dados das colunas:")
print(df.dtypes)

print("\nResumo estat�stico do DataFrame:")
print(df.describe(include='all'))

print("\nIniciando o pr�-processamento dos dados...")
print("Aplicando a codifica��o one-hot a 'overall_review'...")

df = pd.get_dummies(df, columns=['overall_review'])

print("Limpando e convertendo colunas num�ricas...")

df['original_price'] = pd.to_numeric(df['original_price'].str.replace('[^0-9.]', '', regex=True), errors='coerce')
df['discount_percentage'] = pd.to_numeric(df['discount_percentage'].str.replace('[^0-9.]', '', regex=True), errors='coerce')

print("Preenchendo valores ausentes nas colunas num�ricas com a m�dia...")

num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

print("Preenchendo valores ausentes nas colunas categ�ricas com a moda...")

cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

print("Selecionando vari�veis independentes e a vari�vel dependente...")

review_columns = ['overall_review_Very Positive', 'overall_review_Mostly Positive', 'overall_review_Mixed', 'overall_review_Mostly Negative', 'overall_review_Overwhelmingly Positive']
X = df.drop(review_columns, axis=1, errors='ignore')
y = df['overall_review_Very Positive']

print("\nCabe�alho ap�s o pr�-processamento:")
print(X.head())

numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
X_numeric = X[numeric_cols]

print("Dividindo os dados em conjuntos de treinamento e teste...")

X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

print("Normalizando os dados...")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Aplicando PCA...")

pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print("Treinando o modelo de regress�o log�stica...")

model = LogisticRegression()
model.fit(X_train, y_train)

print("Fazendo previs�es no conjunto de teste...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAvalia��o do modelo:")
print(f"Acur�cia: {accuracy:.4f}")
print(f"Acur�cia (em porcentagem): {accuracy * 100:.2f}%")
print("\nRelat�rio de Classifica��o:")
print(classification_report(y_test, y_pred))
print("Gerando visualiza��es...")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['original_price'], bins=30, kde=True)
plt.title('Distribui��o dos Pre�os Originais')
plt.xlabel('Pre�o Original')
plt.ylabel('Frequ�ncia')

plt.subplot(1, 2, 2)
sns.histplot(df['discount_percentage'], bins=30, kde=True)
plt.title('Distribui��o dos Descontos')
plt.xlabel('Desconto (%)')
plt.ylabel('Frequ�ncia')

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x='overall_review_Very Positive', y='original_price', data=df, palette='coolwarm')
plt.title('Pre�o Original por Avalia��o Muito Positiva')
plt.xlabel('Avalia��o Muito Positiva')
plt.ylabel('Pre�o Original')

plt.subplot(1, 2, 2)
sns.boxplot(x='overall_review_Very Positive', y='discount_percentage', data=df, palette='coolwarm')
plt.title('Desconto por Avalia��o Muito Positiva')
plt.xlabel('Avalia��o Muito Positiva')
plt.ylabel('Desconto (%)')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='original_price', y='discount_percentage', hue='overall_review_Very Positive', palette='coolwarm', s=50)
plt.title('Pre�o Original vs. Desconto vs. Avalia��o Muito Positiva')
plt.xlabel('Pre�o Original')
plt.ylabel('Desconto (%)')
plt.legend(title='Avalia��o Muito Positiva', loc='upper right')
plt.show()
