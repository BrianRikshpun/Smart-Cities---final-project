from matplotlib import pyplot as plt


class Visualization:
    def __init__(self, res_data):
       self.res_data = res_data


    def ShowAUC(self, res_data):

        x_for_plot = ["LogisticRegression", "KNeighborsClassifier", "DecisionTreeClassifier", "RandomForestClassifier"]
        y_for_plot = res_data[0]

        def addlabels(x, y):
            for i in range(len(x)):
                plt.text(i, round(y[i], 5), round(y[i], 5))

        addlabels(x_for_plot, y_for_plot)
        plt.bar(x_for_plot, y_for_plot, width=0.4)
        plt.xlabel("Model")
        plt.ylabel("AUC")
        plt.title("AUC for each model - TEST")
        plt.show()


    def ShowConfussionMatrix(self, res_data):

        for conf_matrix in res_data[2]:
            fig, ax = plt.subplots(figsize=(7.5, 7.5))
            ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

            plt.xlabel('Predictions', fontsize=18)
            plt.ylabel('Actuals', fontsize=18)
            plt.title('Confusion Matrix', fontsize=18)
            plt.show()