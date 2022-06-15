import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import balanced_accuracy_score


class ClassicModels:
    def __init__(self, model, space, x_train, x_test, y_train, y_test):
        self.model = model
        self.space = space
        self.X_train = x_train
        self.X_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def FindBestParams(self, model, space, x_train, x_test, y_train, y_test):
        res = []  # Return the saved arrays
        scores_test = []  # Saving the scores of the test -- 0
        best_models = []  # Saving the fitted best models --1
        conf_matrixs = []  # Saving the confussion matrix --2
        roc_graph = []  # Saving the ROC graph parameters  --3

        for m in range(0, len(model)):

            clf = GridSearchCV(model[m], space[m], scoring='roc_auc')
            clf.fit(x_train, y_train)

            y_hat = clf.predict(x_test)
            auc = balanced_accuracy_score(y_test, y_hat)

            x = range(len(y_hat))
            y_pred = y_hat
            y_true = y_test
            y_prob = clf.predict_proba(x_test)[::, 1]
            fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)

            scores_test.append(auc)
            best_models.append(clf)
            conf_matrixs.append(confusion_matrix(y_true=y_true, y_pred=y_pred))
            roc_graph.append([fpr, tpr])

            # plotting the tree
            # if(m == 2):
            #     fig = plt.figure(figsize=(25, 20))
            #     _ = tree.plot_tree(clf.best_estimator_,
            #                        filled=True)
            #     fig.show()

            print("Best parameters set found on development set: ", model[m])
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_["mean_test_score"]
            stds = clf.cv_results_["std_test_score"]
            for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
                print("%0.3f (+/-%0.03f) for %r" % (mean * -1, std * 2, params))

            res.append(scores_test)
            res.append(best_models)
            res.append(conf_matrixs)
            res.append(roc_graph)

        return res
