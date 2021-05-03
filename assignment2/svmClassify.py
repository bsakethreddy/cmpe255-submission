from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import seaborn as sns
sns.set_theme()


class Solution:
    def __init__(self) -> None:
        self.faces = fetch_lfw_people(min_faces_per_person=60)
        self.target_names = self.faces.target_names
        print("data loaded")
        print("faces.target_names = ", self.faces.target_names)
        print("faces.images.shape = ", self.faces.images.shape)
        pass

    def transformData(self):
        # normalize the data
        X = self.faces.data
        y = self.faces.target
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, y

    def splitData(self, X, y):

        # split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42000)
        return X_train, X_test, y_train, y_test

    def defineClassifer(self):
        # TODO
        # define svm classifier.
        pca = RandomizedPCA(n_components=150, whiten=True, random_state=42000)
        svc = SVC(kernel='rbf', class_weight='balanced', C=[1, 10])
        model = make_pipeline(pca, svc)
        parameters = {'C': [1, 5, 10, 25, 50, 70], 'gamma': ['scale', 'auto']}
        clf = GridSearchCV(svc, parameters)
        return clf

    def classifyImages(self, clf, X_train, X_test, y_train, y_test) -> None:
        # TODO fit the data
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return y_pred

    def calc_accuracy(self, y_test, y_pred) -> None:
        # TODO Print preciison score
        precisionScore = precision_score(y_test, y_pred, average='macro')
        recallScore = recall_score(y_test, y_pred, average='macro')
        f1Score = f1_score(y_test, y_pred, average='macro')
        support = precision_recall_fscore_support(
            y_test, y_pred, average=None)[3]

        print("precision score = ", precisionScore)
        print("recall score = ", recallScore)
        print("f1 score = ", precisionScore)
        print("support = ", precisionScore)
        pass

    def plot_images(self, X_test, y_test, y_pred):
        n_samples, h, w = self.faces.images.shape
        prediction_titles = []
        for i in range(y_pred.shape[0]):
            pred_name = self.target_names[y_pred[i]].rsplit(' ', 1)[-1]
            true_name = self.target_names[y_test[i]].rsplit(' ', 1)[-1]
            dummy = (pred_name, true_name)
            prediction_titles.append(dummy)
        print("prediction titles = ", prediction_titles)

        actual = []
        predicted = []
        for element in prediction_titles:
            predicted.append(element[0])
            actual.append(element[1])

        row = 4
        col = 6
        plt.figure(figsize=(1.8 * col, 2.4 * row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)

        # fig = plt.figure(figsize=(10, 7))

        for i in range(row * col):
            plt.subplot(row, col, i + 1)
            plt.imshow(X_test[i].reshape((h, w)), cmap=plt.cm.gray)
            if actual[i] == predicted[i]:
                plt.title(predicted[i], size=12, fontdict={'color': 'black'})
            else:
                plt.title(predicted[i], size=12, fontdict={'color': 'red'})
            plt.xticks(())
            plt.yticks(())
            # fig.add_subplot(4, 6, i)
        plt.show()

    def drawHeatMap(self, y_test, y_pred) -> None:
        # TODO
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        ax = plt.subplot()
        sns.heatmap(cm.T, annot=True, fmt='g', ax=ax)
        ax.xaxis.set_ticklabels(self.target_names, rotation=90)
        ax.yaxis.set_ticklabels(self.target_names, rotation=0)
        ax.set_ylabel('Predicted')
        ax.set_xlabel('Actual')
        plt.show()


def test() -> None:
    solution = Solution()
    X, y = solution.transformData()
    X_train, X_test, y_train, y_test = solution.splitData(X, y)
    clf = solution.defineClassifer()
    y_pred = solution.classifyImages(clf, X_train, X_test, y_train, y_test)
    solution.calc_accuracy(y_test, y_pred)

    solution.plot_images(X_test, y_test, y_pred)
    solution.drawHeatMap(y_test, y_pred)


if __name__ == "__main__":
    # execute only if run as a script
    test()
