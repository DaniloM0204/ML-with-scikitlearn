from sklearn.datasets import load_breast_cancer

#Carregar o dataset
data = load_breast_cancer()

# Criando novas váriaveis para os dados

#Organizando dados
label_names = data['target_names']
laels = data['target']
feature_names = data['feature_names']
features = data['data']

# Olhando para os dados
print(label_names)
print(labels[0])
print(feature_names[0])
print(features[0])

from sklearn.model_selection import train_test_split

#Dividir os dados
train, test, train_labels, test_labels = train_test_split(features,labels,test_size = 0.33, random_state=42)

#Construindo e avaliando o modelo

from sklearn.naive_bayes import GaussianNB

#Inicializar nosso classificador
gnb = GaussianNB()

# Treinar nosso classificador
model = gnb.fit(train, train_labels)

# Fazer previsões
preds = gnb.predict(test)
print(preds)

from sklearn.metrics import accuracy_score

# Avaliar a precisão
print(accuracy_score(test_labels,preds))


