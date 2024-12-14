#!/usr/bin/env python
# coding: utf-8

# # **Caso Practìco: Support Vector Marchine**

# ## **Description**
# The Web has long become a major platform for online criminal activities. URLs are used as the main vehicle in this domain. To counter this issues security community focused its efforts on developing techniques for mostly blacklisting of malicious URLs.
# 
# While successful in protecting users from known malicious domains, this approach only solves part of the problem. The new malicious URLs that sprang up all over the web in masses commonly get a head start in this race. Besides that, Alexa ranked, trusted websites may convey compromised fraudulent URLs called defacement URL.
# 
# We study mainly five different types of URLs:
# 
# Benign URLs: Over 35,300 benign URLs were collected from Alexa top websites. The domains have been passed through a Heritrix web crawler to extract the URLs. Around half a million unique URLs are crawled initially and then passed to remove duplicate and domain only URLs. Later the extracted URLs have been checked through Virustotal to filter the benign URLs.
# 
# * **Spam URLs:** Around 12,000 spam URLs were collected from the publicly available WEBSPAM-UK2007 dataset.
# 
# * **Phishing URLs:** Around 10,000 phishing URLs were taken from OpenPhish which is a repository of active phishing sites.
# 
# * **Malware URLs:** More than 11,500 URLs related to malware websites were obtained from DNS-BH which is a project that maintain list of malware sites.
# 
# * **Defacement URLs:** More than 45,450 URLs belong to Defacement URL category. They are Alexa ranked trusted websites hosting fraudulent or hidden URL that contains both malicious web pages.

# ## **Imports**

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline


# ## **Funciones Auxiliares**

# In[2]:


# Construccion de una funcion que realice el particionado completo

def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size= 0.4, random_state=rstate, shuffle=shuffle, stratify = strat
    )
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size = 0.5, random_state = rstate, shuffle= shuffle, stratify= strat
    )
    return (train_set, val_set, test_set)


# In[3]:


# Representacion grafica del limite de decision.
def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    #At the decision boundary, w0*x0 + w1*x1 + b = 0 
    # => x1 = -w0/w1 * x0 -b/w1

    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 -b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s = 180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)
    


# # **1.-Lectura del DataSet**

# In[4]:


df = pd.read_csv("dataset/Phishing.csv")


# ## 2.- **Visualizacion preliminar de la informacion**

# In[5]:


df.head(10)


# In[6]:


# Describe las caracteristicas
df.describe()


# In[7]:


# Obtener la info. de nuestro dataset
df.info()


# In[8]:


df["URL_Type_obf_Type"].value_counts()


# In[9]:


# Comprobacion si existen vaalores nulos
is_null = df.isna().any()
is_null[is_null]


# In[10]:


# Comprobacion de la existencia de valores infinitos
is_inf = df.isin([np.inf, -np.inf]).any()
is_inf[is_inf]


# In[11]:


# Representacion grafica de dos caracteristicas
plt.figure(figsize=(12 , 6))
plt.scatter(df["domainUrlRatio"][df['URL_Type_obf_Type'] == "phishing"], df["domainlength"][df['URL_Type_obf_Type']== "phishing"], c="r", marker=".") # grafico de puntos
plt.scatter(df["domainUrlRatio"][df['URL_Type_obf_Type'] == "benign"], df["domainlength"][df['URL_Type_obf_Type']== "benign"], c="g", marker="x")
plt.xlabel("domainUrlRatio", fontsize= 13)
plt.ylabel("domainlength", fontsize = 13)
plt.show()


# ## **3.- Division del cojunto de datos del DataSet**

# In[12]:


# Division del DataSet
train_set, val_set, test_set = train_val_test_split(df)


# In[13]:


X_train = train_set.drop("URL_Type_obf_Type", axis=1)
y_train = train_set["URL_Type_obf_Type"].copy()

X_val = val_set.drop("URL_Type_obf_Type", axis=1)
y_val = val_set["URL_Type_obf_Type"].copy()

X_test = test_set.drop("URL_Type_obf_Type", axis=1)
y_test = test_set["URL_Type_obf_Type"].copy()


# ## **4.- Preparacion del DataSet**

# In[14]:


# Eliminar el atributo que tiene valores infinitos.
X_train = X_train.drop("argPathRatio", axis=1)
X_val = X_val.drop("argPathRatio", axis=1)
X_test = X_test.drop("argPathRatio", axis=1)


# In[15]:


# Rellenamos los valores nulos con la mediana 
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")


# In[16]:


# Rellenar los valores nulos
X_train_prep = imputer.fit_transform(X_train)
X_val_prep = imputer.fit_transform(X_val)
X_test_prep = imputer.fit_transform(X_test)


# In[17]:


# Transformar el resultado a un DataFrame de pandas
X_train_prep = pd.DataFrame(X_train_prep, columns = X_train.columns, index=y_train.index)
X_val_prep = pd.DataFrame(X_val_prep, columns = X_val.columns, index=y_val.index)
X_test_prep = pd.DataFrame(X_test_prep, columns = X_test.columns, index=y_test.index)


# In[18]:


X_train_prep.head(10)


# In[19]:


# Comprobar si hay valores nulos en el DataSet de entrenamiento
is_null = X_train_prep.isna().any()
is_null[is_null]


# ## **5.-SVM: Kernel Lineal**
# ### **5.1:Conjunto de datos reducido**
# * Entrenamiento del algoritmo con un conjunto de datos reducidos.

# In[20]:


# Reducimos el conjunto de datos para representarlo graficamente
X_train_reduced = X_train_prep[["domainUrlRatio", "domainlength"]].copy()
X_val_reduced = X_val_prep[["domainUrlRatio", "domainlength"]].copy()


# In[21]:


X_train_reduced


# In[22]:


from sklearn.svm import SVC

# SVM Large Margin Classification
svm_clf = SVC(kernel = "linear", C = 50)
svm_clf.fit(X_train_reduced, y_train)


# ## **Representacion del limite de decision**

# In[23]:


def plot_svm_decision_boundary(svm_clf, xmin, xmax):
    w = svm.clf.coef_[0]
    b = svm.clf.intercept_[0]

    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)


# In[24]:


plt.figure(figsize=(12,6))
plt.plot(X_train_reduced.values[:, 0][y_train=="phishing"], X_train_reduced.values[:, 1][y_train=="phishing"], "g^")
plt.plot(X_train_reduced.values[:, 0][y_train=="benign"], X_train_reduced.values[:, 1][y_train=="benign"], "bs")
plot_svc_decision_boundary(svm_clf, 0, 1)
plt.title("$C = {}$".format(svm_clf.C), fontsize = 16)
plt.axis([0, 1, -100, 250]) # xmin, xmax, ymin, ymax
plt.xlabel("domainUrlRatio", fontsize = 13)
plt.ylabel("domainlength", fontsize = 13)
plt.show()


# #### **Predicciòn con un DataSet reducido.**

# In[25]:


y_pred = svm_clf.predict(X_val_reduced)


# In[26]:


print("F1 Score:", f1_score(y_pred, y_val, pos_label='phishing'))


# ##### **Como se verà màs adelante, para determinados kernels es muy importante escalar el DataSet. En este caso, para el kernel de Linux, no es tan relevante, aunque es posible obtener mejores resultados**

# In[27]:


svm_clf_sc = Pipeline([
    ("scaler", RobustScaler()),
    ("linear_svc", SVC(kernel = "linear", C = 50))
])

svm_clf_sc.fit(X_train_reduced, y_train)


# In[28]:


# Comprobar
y_pred = svm_clf_sc.predict(X_val_reduced)


# In[29]:


print("F1 Score:", f1_score(y_pred, y_val, pos_label='phishing'))


# ### **5.2 DataSet Completo**

# In[30]:


# Entrenamiento con el DataSet
from sklearn.svm import SVC

svm_clf = SVC(kernel="linear", C = 1)
svm_clf.fit(X_train_prep, y_train)


# In[31]:


# Sacar el f1 score
y_pred = svm_clf.predict(X_val_prep)


# In[32]:


# imprimir el f1 score
print("F1 Score:", f1_score(y_pred, y_val, pos_label='phishing'))


# ## Maquinas de soporte NO lineales
# Las máquinas de soporte vectorial no lineales (o *Nonlinear Support Vector Machines*, SVM no lineales) son una extensión de las máquinas de soporte vectorial (SVM) que permiten clasificar datos no linealmente separables. Mientras que las SVM lineales buscan un hiperplano que separe los datos en dos clases en un espacio de características, las SVM no lineales logran separar datos utilizando una técnica conocida como *kernel trick*.
# 
# ### Principales características:
# 
# 1. **Kernel Trick (Truco del Núcleo):**
#    Las SVM no lineales aplican un "truco del núcleo" que transforma los datos a un espacio de mayor dimensión donde es más probable que sean linealmente separables. Este proceso se lleva a cabo sin necesidad de calcular explícitamente la transformación, lo que hace el cálculo más eficiente.
# 
# 2. **Funciones de Kernel más comunes:**
#    - **Kernel lineal**: Ideal cuando los datos son aproximadamente lineales.
#    - **Kernel polinómico**: Introduce relaciones polinómicas entre características.
#    - **Kernel de base radial (RBF)**: Es uno de los más populares y permite manejar patrones complejos en los datos, incluyendo relaciones no lineales.
#    - **Kernel sigmoidal**: Similar a una función de activación en redes neuronales.
# 
# 3. **Optimización:**
#    El objetivo de una SVM no lineal sigue siendo encontrar el hiperplano óptimo en el espacio transformado que maximice el margen entre las dos clases. Esto se resuelve a través de técnicas de optimización convexa.
# 
# 4. **Ventajas:**
#    - Excelente rendimiento con datos complejos no linealmente separables.
#    - Generalización efectiva a nuevos datos debido a la maximización del margen.
#    - Flexible, gracias a la variedad de funciones de kernel disponibles.
# 
# 5. **Desventajas:**
#    - Puede ser computacionalmente costosa en comparación con las SVM lineales.
#    - Requiere elegir y ajustar adecuadamente el kernel y sus parámetros (como el parámetro `C` y `gamma` en el caso del kernel RBF).

# ## **6.0 SVM: Kernel no lineal**

# ### **6.1 Polynomial Kernel(I)**
# 
# Entrenamiento del algoritmo con un conjunto de datos reducidos

# In[33]:


# Para representar el limite de decisiòn, tenemos que pasar la variable objetivo a numèrica
y_train_num = y_train.factorize()[0]
y_val_num = y_val.factorize()[0]


# In[34]:


from sklearn.datasets import make_moons
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures

polynomial_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree = 3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=20, loss="hinge", random_state=42, max_iter=100000))
])

polynomial_svm_clf.fit(X_train_reduced, y_train_num)


# ## **Representaciòn de lìmite de decisiòn**

# In[35]:


def plot_dataset(X, y):
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g.")
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "b.")


# In[36]:


def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

fig, axes = plt.subplots(ncols=2, figsize=(15,5), sharey=True)
plt.sca(axes[0])
plot_dataset(X_train_reduced.values, y_train_num)
plot_predictions(polynomial_svm_clf, [0, 1, -100, 250])
plt.xlabel("domainUrlRatio", fontsize=12)
plt.ylabel("domainlength", fontsize=12)
plt.sca(axes[1])
plot_predictions(polynomial_svm_clf, [0, 1, -100, 250])
plt.xlabel("domainUrlRatio", fontsize=12)
plt.ylabel("domainlength", fontsize=12)
plt.show()


# ### **Prediccion con el DataSet reducido**

# In[37]:


y_pred = polynomial_svm_clf.predict(X_val_reduced)


# In[38]:


print("F1 Score:", f1_score(y_pred, y_val_num))


# # **6.2 Polynomial Kernel (II)**
# 
# * **Existes una forma mas sencilla de entrenar un algoritmo SVM que utilice polynomial kernel, utilizado el parametro kernel de la propia funcion implementada con Sklearn**
# * **Entrenamiento del algoritmo co un DataSet reducido**

# In[39]:


svm_clf =SVC(kernel = "poly", degree = 3, coef0 = 10, C = 20)
svm_clf.fit(X_train_reduced, y_train_num)


# In[40]:


def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

fig, axes = plt.subplots(ncols=2, figsize=(15,5), sharey=True)
plt.sca(axes[0])
plot_dataset(X_train_reduced.values, y_train_num)
plot_predictions(polynomial_svm_clf, [0, 1, -100, 250])
plt.xlabel("domainUrlRatio", fontsize=12)
plt.ylabel("domainlength", fontsize=12)
plt.sca(axes[1])
plot_predictions(polynomial_svm_clf, [0, 1, -100, 250])
plt.xlabel("domainUrlRatio", fontsize=12)
plt.ylabel("domainlength", fontsize=12)
plt.show()


# ##### Prediccion de un DataSet reducido

# In[41]:


y_pred = svm_clf.predict(X_val_reduced)
print("F1 Score:", f1_score(y_pred, y_val_num))


# ##### Prediccion del DataSet completo

# In[42]:


sv_clf= SVC(kernel="poly", degree= 3, coef0 = 10, C=40)
svm_clf.fit(X_train_prep, y_train_num)


# In[43]:


y_pred = svm_clf.predict(X_val_prep)


# In[44]:


print("F1 Score:", f1_score(y_pred, y_val_num))


# # **6.2.1 Gaussian kernel**
# 
# Entrenamiento del algoritmo de un DataSet reducido

# In[45]:


rbf_kernel_svm_clf = Pipeline([
    ("scaler", RobustScaler()),
    ("svm_clf", SVC(kernel = "rbf", gamma = 0.5, C = 1000))
])
rbf_kernel_svm_clf.fit(X_train_reduced, y_train_num)


# ##### Representacion del limite de decision

# In[46]:


fig, axes = plt.subplots(ncols=2, figsize=(15,5), sharey=True)
plt.sca(axes[0])
plot_dataset(X_train_reduced.values, y_train_num)
plot_predictions(rbf_kernel_svm_clf, [0, 1, -100, 250])
plt.xlabel("domainUrlRatio", fontsize=12)
plt.ylabel("domainlength", fontsize=12)
plt.sca(axes[1])
plot_predictions(rbf_kernel_svm_clf, [0, 1, -100, 250])
plt.xlabel("domainUrlRatio", fontsize=12)
plt.ylabel("domainlength", fontsize=12)
plt.show()


# # Preccion del conjunto de datos reducido

# In[47]:


y_pred = rbf_kernel_svm_clf.predict(X_val_reduced)
print("F1 Score:", f1_score(y_pred, y_val_num))


# # Prediccion del dataset completo

# In[48]:


rbf_kernel_svm_clf= SVC(kernel="rbf", degree= 3, coef0 = 10, C=40)
rbf_kernel_svm_clf.fit(X_train_prep, y_train_num)


# In[49]:


y_pred = rbf_kernel_svm_clf.predict(X_val_prep)
print("F1 Score:", f1_score(y_pred, y_val_num))

