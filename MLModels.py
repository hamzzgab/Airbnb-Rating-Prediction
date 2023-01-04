import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# MODEL
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

# PREPROCESSING
from sklearn.model_selection import train_test_split

# METRICS
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from yellowbrick.classifier import ROCAUC

FILE_PATH = ""

models_details = []

TARGET_LIST = ['review_scores_rating', 'review_scores_accuracy',
               'review_scores_cleanliness', 'review_scores_checkin',
               'review_scores_communication', 'review_scores_location',
               'review_scores_value']

Y_set = set(TARGET_LIST)
id_columns = ['Unnamed: 0.1', 'Unnamed: 0', 'id', 'listing_id', 'phone', 'has_identity_verified']
sentiment_remove = ['neu', 'pos', 'neg', 'compound']


class TrainingData:
    def __init__(self, path="final_train_data.csv", target=None):
        (self.X, self.y) = (None, None)
        (self.Xtrain, self.Xtest), (self.ytrain, self.ytest) = (None, None), (None, None)
        self.model = None

        self.listings = pd.read_csv(path)

        self.needed_columns = list(
            set(self.listings.keys()) - set(id_columns)
            - set(sentiment_remove) - Y_set)

        self.target = target
        self.predicting = TARGET_LIST[self.target].split('scores_')[1].upper()

        if target is None:
            raise ValueError("TARGET CANNOT BE NONE!")

        self.return_data()

    def return_data(self):
        self.X = self.listings[self.needed_columns]
        self.y = self.listings[TARGET_LIST[self.target]]

        self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(self.X, self.y, test_size=0.2,
                                                                            random_state=1)
        return (self.Xtrain, self.Xtest), (self.ytrain, self.ytest)

    def train_data(self):
        print(f"TRAINING: {self.model}")
        print(f"PREDICTING: {self.predicting}")
        self.model.fit(self.Xtrain, self.ytrain)

    def print_metrics(self):
        model_score = self.model.score(self.Xtest, self.ytest)

        ypred = self.model.predict(self.Xtest)
        model_mean_error = mean_squared_error(self.ytest, ypred)
        
        models_details.append([f"{self.model} - {self.predicting}", model_score, model_mean_error])

        print("-" * 10 + "CONFUSION-MATRIX" + "-" * 10)
        print(confusion_matrix(self.ytest, ypred))

        print("-" * 10 + "CLASSIFICATION-REPORT" + "-" * 10)
        print(classification_report(self.ytest, ypred))

    def plot_ROC_curve(self):
        visualizer = ROCAUC(self.model, encoder={0: '0',
                                                 1: '1',
                                                 2: '2', },
                            macro=False, micro=False)
        visualizer.fit(self.Xtrain, self.ytrain)
        visualizer.score(self.Xtest, self.ytest)

        self.model_name = str(self.model).split('(')[0]

        visualizer.show(outpath=f"{self.model_name}/ROCAUC/{self.predicting}.png")
        return visualizer

    def feature_importance(self):
        feature_importance = abs(self.model.coef_[0])
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        

        top_features = pd.DataFrame({'feature_imp': feature_importance,
                                     'features': self.X.columns},
                                    columns=['feature_imp', 'features'])
        

        top_features = top_features.sort_values(by='feature_imp',
                                                ascending=False)

        top_features.to_csv(f"{self.model_name}/FeatureImportance/{self.predicting}.csv", index=False)
        top_features = top_features.head(15)
        
        col = []
        for val in top_features.feature_imp:
            if val < 10:
                col.append('red')
            elif 10 < val < 40:
                col.append('orange')
            elif 40 < val < 80:
                col.append('blue')
            else:
                col.append('green')
        

        # fig = plt.figure()
        # plt.barh(top_features.features, top_features.feature_imp, color=col)
        # plt.xlabel('Importance', fontsize=18)
        # plt.ylabel('Features', fontsize=18)
        # plt.xticks(fontsize=18)
        # plt.yticks(fontsize=18)
        # plt.title(f'{self.predicting.title()} Feature Importance', fontsize=18)
        # plt.tight_layout()
        # plt.savefig(f"{self.model_name}/FeatureImportance/{self.predicting}.png", 
        #             dpi=120)
        # plt.show();
        # plt.close();


def display_models_details():
    print(tabulate(models_details, headers=['Model', 'Score', 'Mean Error']))


class LogisticRegressionModel(TrainingData):
    def __init__(self, path=None, target=None, penalty='l2', C=1.0):
        super().__init__(path=path, target=target)
        self.model = LogisticRegression(penalty=penalty, C=C)
        self.train_data()
        self.print_metrics()
        self.plot_ROC_curve()
        self.feature_importance()


class KNeighborsModel(TrainingData):
    def __init__(self, path=None, target=None, n_neighbors=5):
        super().__init__(path=path, target=target)
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors,
                                          weights='uniform')
        self.train_data()
        self.print_metrics()
        self.plot_ROC_curve()


class DummyClassifierModel(TrainingData):
    def __init__(self, path=None, target=None, strategy='most_frequent'):
        super().__init__(path=path, target=target)
        self.model = DummyClassifier(strategy=strategy)
        self.train_data()
        self.print_metrics()


"""

print(tabulate(models_details, headers=['Model', 'Score', 'Mean Error']))


dfreq = DummyClassifier(strategy='most_frequent').fit(Xtrain, ytrain)
ypred = dfreq.predict(Xtest)
print_metrics(dfreq, Xtest, ytest, ypred)

dunif = DummyClassifier(strategy='uniform').fit(Xtrain, ytrain)
ypred = dunif.predict(Xtest)
print_metrics(dunif, Xtest, ytest, ypred)

dstrut = DummyClassifier(strategy='stratified').fit(Xtrain, ytrain)
ypred = dstrut.predict(Xtest)
print_metrics(dstrut, Xtest, ytest, ypred)


print(tabulate(models_details, headers=['Model', 'Score', 'Mean Error']))
# print(tabulate(np.column_stack((list(X.keys()), list(model.coef_[0]))), headers=['Feature', 'Score']))
"""
