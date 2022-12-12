import numpy as np
from sklearn import tree
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix as matriz
from sklearn.neural_network import MLPClassifier as MPC
from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.ensemble import RandomForestClassifier as forest

# k-NN classifier
def k_nn(X_train, y_train, X_test, y_test, n_neighbors):
    file = open("./results/knn" + str(n_neighbors) + ".txt", 'w')

    neigh = KNeighborsClassifier(n_neighbors, metric='euclidean')
    neigh.fit(X_train, y_train)
    knn_probs = neigh.predict_proba(X_test)
    acuracia = neigh.score(np.array(X_test), y_test)
    y_pred = neigh.predict(X_test)
    # neigh.score(X_test, y_test)
    matrix = confusion_matrix(y_test, y_pred)

    print("Matriz de confusão para knn k = ", n_neighbors, "\n")
    file.writelines("Matriz de confusão para knn k = " +
                    str(n_neighbors) + "\n\n")

    a = []

    for x in range(len(matrix)):
        a.append([])
        for j in matrix[x]:
            a[x].append(j)
            for x in a:
                print(x)
                file.writelines(str(x) + "\n")

    print("\nAcurácia de: ", acuracia, "%.\n")
    file.writelines("\n\nAcurácia de: " + str(acuracia) + "%.")
    file.close()
    return knn_probs

# SVM com Grid search


def svm(X_train, y_train, X_test, y_test):
    file = open("./results/svm.txt", 'w')

    clf = SVC(probability=True)

    clf.fit(np.array(X_train), y_train)

    svm_probs = clf.predict_proba(X_test)

    acuracia = clf.score(np.array(X_test), y_test)

    y_pred = clf.predict(X_test)

    matrix = confusion_matrix(y_test, y_pred)

    print("Matriz de confusão para SVM:\n")
    file.writelines("Matriz de confusão para SVM:\n\n")

    a = []

    for x in range(len(matrix)):
        a.append([])
        for j in matrix[x]:
            a[x].append(j)

    for x in a:
        print(x)
        file.writelines(str(x) + "\n")

    print("\nAcurácia de: ", acuracia, "%.\n")
    file.writelines("\n\nAcurácia de: " + str(acuracia) + "%.")

    file.close()

    return svm_probs


# MLP
def mlp(X_train, X_test, y_train, y_test):
    file = open("./results/mlp.txt", 'w')
    mlp = MPC(random_state=0, max_iter=10000)
    mlp.fit(np.array(X_train), y_train)
    mpl_probs = mlp.predict_proba(X_test)
    acuracia = mlp.score(np.array(X_test), y_test)
    y_pred = mlp.predict(X_test)
    matrix = matriz(y_test, y_pred)

    print("Matriz de confusão para MLP:\n")
    file.writelines("Matriz de confusão para MLP:\n\n")

    a = []

    for x in range(len(matrix)):
        a.append([])
        for j in matrix[x]:
            a[x].append(j)

    for x in a:
        print(x)
        file.writelines(str(x) + "\n")

    print("\nAcurácia de: ", acuracia, "%.\n")
    file.writelines("\n\nAcurácia de: " + str(acuracia) + "%.")

    file.close()

# Random Forest Classifier


def randomForest(X_train, y_train, X_test, y_test):
    file = open("./results/randomForest.txt", 'w')
    rf = forest(n_estimators=100)
    rf.fit(np.array(X_train), y_train)
    rf_probs = rf.predict_proba(X_test)
    acuracia = rf.score(np.array(X_test), y_test)
    y_pred = rf.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)

    print("Matriz de confusão para Random Forest:\n")
    file.writelines("Matriz de confusão para Random Forest:\n\n")

    a = []

    for x in range(len(matrix)):
        a.append([])
        for j in matrix[x]:
            a[x].append(j)

    for x in a:
        print(x)
        file.writelines(str(x) + "\n")

    print("\nAcurácia de: ", acuracia, "%.\n")
    file.writelines("\n\nAcurácia de: " + str(acuracia) + "%.")

    file.close()


# Decision Tree
def decision_tree(X_train, y_train, X_test, y_test):
    file = open("./results/decisionTree.txt", 'w')
    dt = tree()
    dt.fit(np.array(X_train), y_train)
    dt_probs = dt.predict_proba(X_test)
    acuracia = dt.score(np.array(X_test), y_test)
    y_pred = dt.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)

    print("Matriz de confusão para Decision Tree:\n")
    file.writelines("Matriz de confusão para Decision Tree:\n\n")

    a = []

    for x in range(len(matrix)):
        a.append([])
        for j in matrix[x]:
            a[x].append(j)

    for x in a:
        print(x)
        file.writelines(str(x) + "\n")

    print("\nAcurácia de: ", acuracia, "%.\n")
    file.writelines("\n\nAcurácia de: " + str(acuracia) + "%.")

    file.close()

    return dt_probs
