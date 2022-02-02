Passo 1 � Importando o Scikit-learn

Crie um ambiente env

No terminal - use esses c�digos

'my_env/bin/activate' - para ativar o diret�rio

-Verifique se o Scikit-learn est� instalado
'python -c "import sklearn"'

Caso esteja, correra normal o c�digo, caso contr�rio aparecer
�ma mensagem de erro e sera necess�rio utilizar este comando.

'pip install scikit-learn[alldeps]'

Eu farei o c�digo inteiramente no python(uma forma de gambiarra, com quase clean code)

Passo 2 � Importando o Dataset do Scikit-learn

Neste m�dulo ser� feito a inser��o do sklearn para carregar 
um de seus datasets que ser� utilizado, para prever se um tumor �
maligno ou benigno

'from sklearn.datasets import load_breast_cancer

# Carregar o dataset
data = load_breast_cancer()'

A vari�vel data representa um objeto Python que funciona como um dicion�rio. 
As chaves importantes do dicion�rio a considerar s�o os nomes dos r�tulos de classifica��o (target_names), 
os r�tulos reais (target),os nomes de atributo/caracter�stica (feature_names), e os atributos (data).

Criaremos agora novas var�veis para cada conjunto

'
# Organizar nossos dados
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

'Agora temos listas para cada conjunto de informa��es.
Para entender melhor nosso conjunto de dados, 
vamos dar uma olhada em nossos dados imprimindo nossos r�tulos de classe, 
o primeiro r�tulo da inst�ncia de dados, nossos nomes de caracter�sticas, 
e os valores das caracter�sticas para a primeira inst�ncia de dados.

'

# Olhando para os nossos dados
print(label_names)
print(labels[0])
print(feature_names[0])
print(features[0])
'

Passo 3 � Organizando Dados em Conjuntos

Voc� usa o conjunto de testes para treinar e avaliar o modelo durante o est�gio de desenvolvimento. 
Ent�o voc� usa o modelo treinado para fazer previs�es no conjunto de testes n�o visualizado.
Essa abordagem lhe d� uma no��o do desempenho e robustez do modelo.

Felizmente, o sklearn tem uma fun��o chamada train_test_split(), que divide seus dados nesses conjuntos. Importe a fun��o e em seguida utilize-a para dividir os dados:

'from sklearn.model_selection import train_test_split

# Dividir nossos dados
train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.33,
                                                          random_state=42)'

Passo 4 � Construindo e Avaliando o Modelo


Existem muitos modelos para machine learning, e cada modelo tem seus pontos fortes e fracos. Neste tutorial,
vamos nos concentrar em um algoritmo simples que geralmente funciona bem em tarefas de classifica��o bin�ria, a saber Naive Bayes (NB).


Primeiro, importe o m�dulo GaussianNB. 
Em seguida inicialize o modelo com a fun��o GaussianNB(), 
depois treine o modelo, ajustando-o aos dados usando gnb.fit():


'from sklearn.naive_bayes import GaussianNB

# Inicializar nosso classificador
gnb = GaussianNB()

# Treinar nosso classificador
model = gnb.fit(train, train_labels)'

# Fazer previs�es
preds = gnb.predict(test)
print(preds)

Como voc� v� na sa�da do Jupyter Notebook, a fun��o predict() 
retornou uma matriz de 0s e 1s que representa nossos 
valores previstos para a classe tumor (maligno vs. benigno).


Passo 5 � Avaliando a Precis�o do Modelo
