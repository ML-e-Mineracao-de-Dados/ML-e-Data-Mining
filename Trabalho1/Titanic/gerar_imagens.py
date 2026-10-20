import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Create resultados directory if it doesn't exist
import os
os.makedirs('resultados', exist_ok=True)

# Load dataset
train_path = 'train.csv'
df = pd.read_csv(train_path)

# Separate X and y
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocess: encode categorical, impute missing
le_sex = LabelEncoder()
X_train['Sex'] = le_sex.fit_transform(X_train['Sex'])
X_val['Sex'] = le_sex.transform(X_val['Sex'])

le_embarked = LabelEncoder()
X_train['Embarked'] = le_embarked.fit_transform(X_train['Embarked'].fillna('S'))
X_val['Embarked'] = le_embarked.transform(X_val['Embarked'].fillna('S'))

age_median = X_train['Age'].median()
fare_median = X_train['Fare'].median()
X_train['Age'] = X_train['Age'].fillna(age_median)
X_val['Age'] = X_val['Age'].fillna(age_median)
X_train['Fare'] = X_train['Fare'].fillna(fare_median)
X_val['Fare'] = X_val['Fare'].fillna(fare_median)

# Train Decision Tree
clf = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
f1_survived = f1_score(y_val, y_pred, pos_label=1)

# Save metrics table
metrics_df = pd.DataFrame({
    'Modelo': ['Árvore de Decisão'],
    'Conjunto avaliado': ['Validação'],
    'Acurácia': [accuracy],
    'F1-score (Survived = 1)': [f1_survived]
})
metrics_df.to_csv('metrics.csv', index=False)

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Não sobreviveu (0)', 'Sobreviveu (1)'], yticklabels=['Não sobreviveu (0)', 'Sobreviveu (1)'])
plt.title('Matriz de Confusão')
plt.xlabel('Predição')
plt.ylabel('Real')
plt.tight_layout()
plt.savefig('resultados/matriz_confusao.png', dpi=300, bbox_inches='tight')
plt.close()

# Tree plot
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'], class_names=['Não sobreviveu', 'Sobreviveu'], filled=True, rounded=True, fontsize=10)
plt.title('Árvore de Decisão - Titanic')
plt.tight_layout()
plt.savefig('resultados/arvore_decisao.png', dpi=300, bbox_inches='tight')
plt.close()

print('Imagens geradas em resultados/')