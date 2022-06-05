import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn import metrics



class ClassicModels:
    def __init__(self,model, space ,x_train, x_test, y_train, y_test):
        self.model = model
        self.space = space
        self.X_train = x_train
        self.X_test = x_test
        self.y_train = y_train
        self.y_test = y_test


    def FindBestParams(self, model, space ,x_train, x_test, y_train, y_test ):
        res = []  # Return the saved arrays
        scores_test = []  # Saving the scores of the test
        best_models = []  # Saving the fitted best models

        for m in range(0, len(model)):

            clf = GridSearchCV(model[m], space[m], scoring='roc_auc')
            clf.fit(x_train, y_train)

            y_hat = clf.predict(x_test)
            auc = metrics.roc_auc_score(y_test, y_hat)

            x = range(len(y_hat))
            y_pred = y_hat
            y_true = y_test

            #
            # plt.bar(x, y_pred, width=0.4)
            # plt.xlabel("x")
            # plt.ylabel("Total rain")
            # plt.title("results for model %f - Prediction" % (m))
            # plt.show()
            #
            # plt.bar(x, y_pred, width=0.4)
            # plt.xlabel("x")
            # plt.ylabel("Total rain")
            # plt.title("results for model %f - True values" % (m))
            # plt.show()

            scores_test.append(auc)
            best_models.append(clf)

            print("Best parameters set found on development set: " , model[m])
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_["mean_test_score"]
            # stds = clf.cv_results_["std_test_score"]
            # for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
            #     print("%0.3f (+/-%0.03f) for %r" % (mean * -1, std * 2, params))

        res.append(scores_test)
        res.append(best_models)

        return res
