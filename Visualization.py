from matplotlib import pyplot as plt
from sklearn.metrics import plot_confusion_matrix


class Visualization:
    def __init__(self, res_data, fig_size):
       self.figsiz = fig_size
       self.res_data = res_data
       self.models = ["LogisticRegression", "KNeighborsClassifier", "DecisionTreeClassifier", "RandomForestClassifier"]

    def ShowAUC(self, data):

        x_for_plot = self.models
        y_for_plot = self.res_data[0]

        plt.figure(figsize=self.figsiz)

        def addlabels(x, y):
            for i in range(len(x)):
                plt.text(i, round(y[i], 5), round(y[i], 5))

        addlabels(x_for_plot, y_for_plot)
        plt.bar(x_for_plot, y_for_plot, width=0.4)
        plt.xlabel("Model")
        plt.ylabel("AUC")
        plt.title(f"AUC for each model - TEST - {data}")

    def Show_Confussion_Matrix(self, X_test, y_test):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=self.figsiz)
        c = 0
        for cls, ax in zip(self.res_data[1], axes.flatten()):
            plot_confusion_matrix(cls, X_test, y_test, ax=ax, cmap='Blues')
            ax.set_title(self.models[c])
            c += 1
        plt.tight_layout()

    def ShowConfussionMatrix(self, data):

        plt.figure(self.figsiz)
        c = 0
        for conf_matrix in self.res_data[2]:
            c += 1
            fig, ax = plt.subplots(figsize=self.figsiz)
            ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

            plt.xlabel('Predictions', fontsize=18)
            plt.ylabel('Actuals', fontsize=18)
            plt.title(f'Confusion Matrix - {data}', fontsize=18)
            plt.show()

    def ShowRoc(self, data):
        counter = 0
        plt.figure(figsize=self.figsiz)
        for i in  self.res_data[3]:
            # create ROC curve
            plt.plot(i[0], i[1], label="Model " + self.models[counter] + " AUC = " + str(self.res_data[0][counter]))
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.title(f'ROC - {data}', fontsize=18)
            plt.legend(loc=4)
            counter += 1
