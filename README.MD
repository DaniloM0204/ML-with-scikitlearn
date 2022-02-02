Passo 1 — Importando o Scikit-learn

Crie um ambiente env

No terminal - use esses códigos

'my_env/bin/activate' - para ativar o diretório

-Verifique se o Scikit-learn está instalado
'python -c "import sklearn"'

Caso esteja, correra normal o código, caso contrário aparecer
úma mensagem de erro e sera necessário utilizar este comando.

'pip install scikit-learn[alldeps]'

Eu farei o código inteiramente no python(uma forma de gambiarra, com quase clean code)

Passo 2 — Importando o Dataset do Scikit-learn

Neste módulo será feito a inserção do sklearn para carregar 
um de seus datasets que será utilizado, para prever se um tumor é
maligno ou benigno

'from sklearn.datasets import load_breast_cancer

# Carregar o dataset
data = load_breast_cancer()'

A variável data representa um objeto Python que funciona como um dicionário. 
As chaves importantes do dicionário a considerar são os nomes dos rótulos de classificação (target_names), 
os rótulos reais (target),os nomes de atributo/característica (feature_names), e os atributos (data).

Criaremos agora novas varáveis para cada conjunto

'
# Organizar nossos dados
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

'Agora temos listas para cada conjunto de informações.
Para entender melhor nosso conjunto de dados, 
vamos dar uma olhada em nossos dados imprimindo nossos rótulos de classe, 
o primeiro rótulo da instância de dados, nossos nomes de características, 
e os valores das características para a primeira instância de dados.

'

# Olhando para os nossos dados
print(label_names)
print(labels[0])
print(feature_names[0])
print(features[0])
'

Passo 3 — Organizando Dados em Conjuntos

Você usa o conjunto de testes para treinar e avaliar o modelo durante o estágio de desenvolvimento. 
Então você usa o modelo treinado para fazer previsões no conjunto de testes não visualizado.
Essa abordagem lhe dá uma noção do desempenho e robustez do modelo.

Felizmente, o sklearn tem uma função chamada train_test_split(), que divide seus dados nesses conjuntos. Importe a função e em seguida utilize-a para dividir os dados:

'from sklearn.model_selection import train_test_split

# Dividir nossos dados
train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.33,
                                                          random_state=42)'

Passo 4 — Construindo e Avaliando o Modelo


Existem muitos modelos para machine learning, e cada modelo tem seus pontos fortes e fracos. Neste tutorial,
vamos nos concentrar em um algoritmo simples que geralmente funciona bem em tarefas de classificação binária, a saber Naive Bayes (NB).


Primeiro, importe o módulo GaussianNB. 
Em seguida inicialize o modelo com a função GaussianNB(), 
depois treine o modelo, ajustando-o aos dados usando gnb.fit():


'from sklearn.naive_bayes import GaussianNB

# Inicializar nosso classificador
gnb = GaussianNB()

# Treinar nosso classificador
model = gnb.fit(train, train_labels)'

# Fazer previsões
preds = gnb.predict(test)
print(preds)

Como você vê na saída do Jupyter Notebook, a função predict() 
retornou uma matriz de 0s e 1s que representa nossos 
valores previstos para a classe tumor (maligno vs. benigno).


Passo 5 — Avaliando a Precisão do Modelo

