#From NLTK we import a function that splits the text into words (tokens)
from nltk.tokenize import word_tokenize
import nltk.stem
from unidecode import unidecode
from lxml import etree
from nltk.corpus import stopwords
import gensim
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import csv
import pandas as pd
from scipy import spatial

#from ipywidgets import interact




def articles_preprocessing(vacuum, physical):
    print ("\n Preprocesando articulos...\n")
    vacuum_tubes_data = etree.parse(vacuum)
    physical_quantities_data = etree.parse(physical)
    
    root_vacuum = vacuum_tubes_data.getroot()
    root_physical =  physical_quantities_data.getroot()
    
    
    
    texto_vacuum = []
    i = 0
    for child in root_vacuum:
        for child2 in child:
            if "title" in child2.tag:
                texto_vacuum.append(child2.text)
            if "revision" in child2.tag:
                for child3 in child2:
                    if "text" in child3.tag:
                        texto_vacuum[i] += " " + child3.text
                        i += 1     
                        break
                break
                        
    texto_physical = [] 
    i = 0
    for child in root_physical:
        for child2 in child:
            #Eliminamos articulos del fichero para que esten balanceados
            if len(texto_physical) == len(texto_vacuum):
                break
            if "title" in child2.tag:
                texto_physical.append(child2.text)
            if "revision" in child2.tag:
                for child3 in child2:
                    if "text" in child3.tag:
                        texto_physical[i] += " " + child3.text                   
                        i += 1     
                        break
                break        


    
    tokenize_text_vacuum = []
    s= nltk.stem.SnowballStemmer('english')
    eng_stopwords = stopwords.words('english')
    eng_stopwords.append('http') # eliminamos las palabras no deseadas
    eng_stopwords.append('ref')
    eng_stopwords.append('titl')
    
    for text in texto_vacuum:
        tokens = word_tokenize(text)
        tokens = [s.stem(token.lower()) for token in tokens if token.isalnum()]
        tokens = [token for token in tokens if token not in eng_stopwords]
        tokenize_text_vacuum.append(' '.join(tokens))
    
    #Next, we create a list keeping only alphanumeric tokens and removing capital letters
    tokenize_text_physical = []
    
    
    for text in texto_physical:
        tokens = word_tokenize(text)
        tokens = [s.stem(token.lower()) for token in tokens if token.isalnum()]
        tokens = [token for token in tokens if token not in eng_stopwords]
        tokenize_text_physical.append(' '.join(tokens))
    
    
    
          
    corpusname = 'vacuum'
    
    # We first sort the lists according to dates by zipping the lists together, and then unzipping after sorting
    #zipped_list = zip(tokenize_text_vacuum)
    #zipped_list.sort()
    
    #contenido_vacuum = [el[0] for el in zipped_list]
    contenido_vacuum = tokenize_text_vacuum
    contenido_physical = tokenize_text_physical
    
    
    # We create the file with the corpus
    f = open(corpusname+'_corpus.txt', 'wb')
    for contenido in contenido_vacuum:
        contenido = unidecode(contenido)
        f.write(contenido +'\n')
    f.close()
   
    
    # We create the file for the physical
    
    corpusname = 'physical'
    
    # We create the file with the corpus
    f = open(corpusname+'_corpus.txt', 'wb')
    for contenido in contenido_physical:
        contenido=unidecode(contenido)
        f.write(contenido +'\n')
    f.close()
    
        
    for i in range(len(contenido_physical)):
        contenido_physical[i] = contenido_physical[i].split()
        
    for i in  range(len(contenido_vacuum)):
        contenido_vacuum[i] = contenido_vacuum[i].split()

    return contenido_vacuum, contenido_physical







def LDA(contenido_vacuum, contenido_physical, N_topics):
    print ("\n Ejecutando LDA...\n")
    textos =  []
    for i in contenido_vacuum:
        textos.append(i)
    
    for i in contenido_physical:
        textos.append(i)

    #textos.append(contenido_vacuum)  
    dictionary = gensim.corpora.Dictionary(textos)
    
    corpus = [dictionary.doc2bow(aux) for aux in textos]
    
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=N_topics, id2word = dictionary, passes=20)
    
    corpus2 = [dictionary.doc2bow(aux) for aux in textos]
    a = ldamodel[corpus2]
    a = np.array(a)
    #print len([i for i in a])
    y = np.ones((len(textos),1))
    X=np.zeros((len(textos), N_topics))
    for i in range(len(textos)):
        for j in range(len(a[i])):
            X[i][a[i][j][0]] = a[i][j][1]
    y[len(contenido_vacuum):len(y)] = -1      
    
    return X, y
    
    
    
    
    
    
def data_preprocessing(X, y, k):
    print ("\n Preprocesando las matrices de datos...\n")
    permutation = np.random.permutation(len(X))
    X_train = []
    X_test = []
    
    
    y_train = []
    y_test = []
    X_aux = np.zeros((X.shape[0], X.shape[1]))
    y_aux = np.zeros((len(y),1))
    for i in range(len(X_aux)):   
        X_aux[permutation[i]] = X[i]
        y_aux[permutation[i]] = y[i]
    
    var=0
    for i in range(len(X_aux)):
        if i < k*len(X_aux):
            if i == 0:
                X_train = np.array([X_aux[i]])
                y_train = np.array([y_aux[i]])
            else:
                X_train = np.append(X_train, np.array([X_aux[i]]), axis=0)
                y_train = np.append(y_train, np.array([y_aux[i]]), axis=0)
        else:
            if var==0:
                X_test = np.array([X_aux[i]]) 
                y_test = np.array([y_aux[i]])
                var=1
            else:
                X_test = np.append(X_test, np.array([X_aux[i]]), axis = 0)
                y_test = np.append(y_test, np.array([y_aux[i]]), axis = 0)   
                
                
    medias_X_train = np.mean(X_train, axis = 0)   
    desvs_X_train = np.std(X_train, axis = 0)  
    X_train = (X_train - medias_X_train) / desvs_X_train
    X_test = (X_test - medias_X_train) / desvs_X_train
                
    return X_train, X_test, y_train, y_test
    
                
    
def SVM_classifier(X_train, X_test, y_train, C, gamma, kernel):
        
    
    # clf = svm.SVC()
    clf = svm.SVC(C = C, gamma = gamma, kernel = kernel, random_state = None)
    # np.random.seed(seed)
    clf.fit(X_train, y_train.flatten())
    
    
    return np.array([clf.predict(X_test)]).T






def validate_SVM_classifier(X_train, y_train, C, gamma, kernel, M):
    # This fragment of code runs SVM with M-fold cross validation
        
    # Obtain the indices for the different folds
    n_tr = X_train.shape[0]        
    """ Vector con elementos de 0 a n_tr - 1 sin repeticion """
    permutation = np.random.permutation(n_tr)
    
    set_indices = {}
    for k in range(M):
        set_indices[k] = []
    k = 0
    """ a cada capeta se le asigna unos indices de X_train """
    for pos in range(n_tr):
        set_indices[k].append(permutation[pos])
        k = (k + 1) % M
        
    # Now, we run the cross-validation process using the SVM method 
       
    # Obtain the validation errors
    error_val = np.zeros((len(C), len(gamma)))
    error_val_iter = np.zeros((len(C), len(gamma)))
    
    print (" Test mode: %i-fold cross-validation" % (M))    
    print (" Support Vector Machine")    
    print (" Kernel used: %s\n" % kernel)
    for k in range(M):
        print (" Building model for fold %i ..." % (k + 1))
        """ conjunto de validacion """
        val_indices = set_indices[k]
        train_indices = []
        for j in range(M):
            if not j == k:
                """ conjunto de entrenamiento """
                train_indices += set_indices[j]   
        
        for i in range(len(C)):
            for j in range(len(gamma)): 
                error_val_iter[i, j] =  sum(abs(y_train[val_indices] - SVM_classifier(X_train[train_indices, :],  X_train[val_indices, :], y_train[train_indices], C[i], gamma[j], kernel))) / (2 * len(y_train[val_indices]))     
                                                                             
        error_val += error_val_iter
     
    error_val /= M    
    """ elegimos el minimo de la matriz de errores obtenida, y sus parametros asociados """
    pos_min = np.where(error_val == np.min(error_val[:, :]))
    C_opt = np.mean(C[pos_min[0]])
    gamma_opt = np.mean(gamma[pos_min[1]])
    print ("\n La C optima optima es %f y la gamma optima es %f, que generan una probabilidad de error = %f%%" % (C_opt, gamma_opt, np.min(error_val[:, :]) * 100))
    return C_opt, gamma_opt 

def logistic(x):                                        
    p = 1.0 / (1 + np.exp(-x))
    return p


def logregFit(Z_tr, Y_tr, rho, n_it):

    # Data dimension
    n_dim = Z_tr.shape[1]

    # Initialize variables
    nll_tr = np.zeros(n_it)
    #pe_tr = np.zeros(n_it)
    w = np.random.randn(n_dim,1)

    # Running the gradient descent algorithm
    for n in range(n_it):
        
        # Compute posterior probabilities for weight w
        p1_tr = logistic(np.dot(Z_tr, w))
        p0_tr = logistic(-np.dot(Z_tr, w))

        # Compute negative log-likelihood
        nll_tr[n] = - np.dot(Y_tr.T, np.log(p1_tr)) - np.dot((1-Y_tr).T, np.log(p0_tr))

        # Update weights
        w += rho*np.dot(Z_tr.T, Y_tr - p1_tr)

    return w, nll_tr

def logregPredict(Z, w):

    # Compute posterior probability of class 1 for weights w.
    p = logistic(np.dot(Z, w))
    
    # Class
    #D = [int(pn.round()) for pn in p]
    D=p>0.5
    D=D*1
    return p, D


def knn_classifier(X1,Y1,X2,k):
    """ Compute the k-NN classification for the observations contained in
        the rows of X2, for the training set given by the rows in X1 and the
        components of S1. k is the number of neighbours.
    """
    if X1.ndim == 1:
        X1 = np.asmatrix(X1).T
    if X2.ndim == 1:
        X2 = np.asmatrix(X2).T
    distances = spatial.distance.cdist(X1,X2,'euclidean')
    neighbors = np.argsort(distances, axis=0, kind='quicksort', order=None)
    closest = neighbors[range(k),:]
    
    y_values = np.zeros([X2.shape[0],1])
    for idx in range(X2.shape[0]):
        y_values[idx] = np.median(Y1[closest[:,idx]])
        
    return y_values




""" Funcion principal del programa """   
def main():    
   
    # Contenido_vacuum = 1, Contenido_physical = -1    
    N_topics = 4 #Numero de topics    
    M = 10 # numero de folds  
    k = 0.7

    vacuum = 'Vacuum_tubes.xml'
    physical = 'Physical_quantities.xml'
    #saved_data = 'data.csv'    
    
    
    # SVM hyperparameters for classificating the y values
    C = np.linspace(25, 60, 40) 
    gamma = np.linspace(0.1, 0.5, 40)    
    kernel = 'rbf'
    
    contenido_physical, contenido_vacuum = articles_preprocessing(vacuum, physical)
    X, y = LDA(contenido_vacuum, contenido_physical, N_topics)

    
    # Save output file
    """with open(saved_data,'wb') as f:
            wtr = csv.writer(f, delimiter= ',')            
            for i, x in enumerate(X):
                wtr.writerow((np.append(X[i], y[i], axis = 1)).flatten())             
    print (" Fichero guardado") 
    
    
    data = pd.read_csv(saved_data, header = None)
    X = data.values[:, : -1]
    y = np.array([data.values[:, -1]]).T"""
    
    
    X_train, X_test, y_train, y_test = data_preprocessing(X, y, k)
    
    print ("\n Classification of X_train values\n")
    C_opt, gamma_opt = validate_SVM_classifier(X_train, y_train, C, gamma, kernel, M)
    
    ######################################################################
    ###########                  TEST                          ###########
    ######################################################################
    y = SVM_classifier(X_train, X_test, y_train, C_opt, gamma_opt, kernel)

    errores = (y - y_test) / 2 # hipotesis equiprobables
    error = np.sum(np.abs(errores))/len(y_test)        
    
    print ("\nProbabilidad de error de test: %f%%" % (error * 100)) 
    
    # ipdb.set_trace()
    P_FA = float(sum(errores == 1)) / len(y_test)
    P_M = float(sum(errores == -1)) / len(y_test)
    
    auc = roc_auc_score(y_test, y)
    # auc = metrics.auc(y, y_test)
    
    print (" \nProbabilidad de Falsa Alarma: %f%%" % (P_FA * 100))
    print (" Probabilidad de Perdidas: %f%%" % (P_M * 100))
    print (" AUC: %f" % auc)
    
    
    
    
    """plt.scatter(X_train[:, 1], X_train[:, 2], c=y_train, s=50, cmap='copper')
    
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X_train[:, 1], X_train[:, 2], X_train[:, 3], c=y_train, s=50, cmap='spring')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    #interact(plot_3D, elev=[-90, 90], azip=(-180, 180));"""
###################################################################
########################  EXTENSION  ##############################
###################################################################

#En la extension se probaran varios clasificadores para ver las prestaciones de cada uno y 
#ver cual es el mejor   

#Regresion logistica
    print ("\n\n-----REGRESION LOGISTICA-----\n")
# Parameters of the algorithms
    rho = float(1)/50    # Learning step
    n_it = 200   # Number of iterations
    
    # Compute Z's
    Z_tr = np.c_[np.ones(len(X_train)), X_train] 
    Z_tst = np.c_[np.ones(len(X_test)), X_test]
   # n_dim = Z_tr.shape[1]
    y_train=(y_train+1)/2   
    y_test=(y_test+1)/2
    # Convert target arrays to column vectors
    #Y_tr2 = [np.newaxis].T
    #Y_tst2 = Y_tst[np.newaxis].T

    
    # Running the gradient descent algorithm
    w, nll_tr = logregFit(Z_tr, y_train, rho, n_it)
    
    # Classify training and test data
    p_tr, D_tr = logregPredict(Z_tr, w)
    p_tst, D_tst = logregPredict(Z_tst, w)
        
    # Compute error rates
    E_tr = D_tr!=y_train
    E_tst = D_tst!=y_test
    
    # Error rates
    pe_tr = float(sum(E_tr)) / len(E_tr)
    pe_tst = float(sum(E_tst)) / len(E_tst)
    
    
    print ("Los pesos (w) optimos son:")
    print (w)
    print ("\nProbabilidad de error de train: %f%%" %(pe_tr*100))
    print ("Probabilidad de error de test: %f%%" %(pe_tst*100))
    #print ("The NLL after training is " + str(nll_tr[len(nll_tr)-1]))
    errores=(D_tst-y_test)
    P_FA = float(sum(errores == 1)) / len(y_test)
    P_M = float(sum(errores == -1)) / len(y_test)
    
    auc = roc_auc_score(y_test, D_tst)
    # auc = metrics.auc(y, y_test)
    
    print ("\nProbabilidad de Falsa Alarma: %f%%" % (P_FA * 100))
    print ("Probabilidad de Perdidas: %f%%" % (P_M * 100))
    print ("AUC: %f" % auc)
    
    
    
    
    ###########   KNN
    
    print ("\n\n-------------KNN-------------\n")
    M = 10 #X_train.shape[0]
    permutation = np.random.permutation(X_train.shape[0])
    
    # Initialize sets of indices
    set_indices = {n: [] for n in range(M)}
    
    # Distribute data amont M partitions
    n = 0
    for pos in range(X_train.shape[0]):
        set_indices[n].append(permutation[pos])
        n = (n+1) % M
    
    # Now, we run the cross-validation process using the k-nn method
    k_max = 40
    k_list = [2*j+1 for j in range(int(k_max/2))]
    
    # Obtain the validation errors
    pe_val = 0
    for n in range(M):
        i_val = set_indices[n]
        i_tr = []
        for kk in range(M):
            if not n==kk:
                i_tr += set_indices[kk]
        
        pe_val_iter = []
        for k in k_list:
            y_tr_iter = knn_classifier(X_train[i_tr], y_train[i_tr], X_train[i_val], k)
            pe_val_iter.append(np.mean(y_train[i_val] != y_tr_iter))
    
        pe_val = pe_val + np.asarray(pe_val_iter).T
    
    pe_val = pe_val / M
    
    # We compute now the train and test errors curves
    pe_tr = [np.mean(y_train != knn_classifier(X_train, y_train, X_train, k).T) for k in k_list]
    
    k_opt = k_list[np.argmin(pe_val)]
    print ("La k optima es: %d" %k_opt)
    y_out_train = knn_classifier(X_train, y_train, X_train, k_opt).T
    y_out_test = knn_classifier(X_train, y_train, X_test, k_opt).T
    pe_train = np.mean(y_train != y_out_train.T)
    pe_tst = np.mean(y_test != y_out_test.T)
    print ("\nProbabilidad de error de train: %f%%" %(pe_train*100))
    print ("Probabilidad de error de test: %f%%" %(pe_tst*100))
    
    errores=(y_out_test.T-y_test)
    P_FA = float(sum(errores == 1)) / len(y_test)
    P_M = float(sum(errores == -1)) / len(y_test)
    
    auc = roc_auc_score(y_test, y_out_test.T)
    # auc = metrics.auc(y, y_test)
    
    print ("\nProbabilidad de Falsa Alarma: %f%%" % (P_FA * 100))
    print ("Probabilidad de Perdidas: %f%%" % (P_M * 100))
    print ("AUC: %f" % auc)
    

   
######################################################################################################
 
if __name__ == "__main__":
    main()

   
    
            


