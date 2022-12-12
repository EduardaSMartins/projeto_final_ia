import extractor
import classifiers

# extractor.redimention('./dataset')
X, y = extractor.pixels('./dataset')

# Definindo teste e treino
X_train, X_test, y_train, y_test = extractor.train_test(X, y)

# Normalização
X_train, X_test = extractor.minMax(X_train, X_test)

# Classificadores
classifiers.k_nn(X_train, y_train, X_test, y_test, 1)
classifiers.k_nn(X_train, y_train, X_test, y_test, 3)
classifiers.k_nn(X_train, y_train, X_test, y_test, 5)
classifiers.k_nn(X_train, y_train, X_test, y_test, 7)
classifiers.k_nn(X_train, y_train, X_test, y_test, 9)
classifiers.k_nn(X_train, y_train, X_test, y_test, 11)
classifiers.svm(X_train, X_test, y_train, y_test)
classifiers.decision_tree(X_train, X_test, y_train, y_test)
classifiers.mlp(X_train, X_test, y_train, y_test)