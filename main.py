import numpy as np
import matplotlib.pyplot as plt

from MLModels import LogisticRegressionModel, KNeighborsModel, DummyClassifierModel
from MLModels import display_models_details

from sklearn.model_selection import cross_val_score

max_features = 25
name = "final_train_data"
if max_features != 25:
    name += f"_features-{max_features}"


def create_data():
    # LISTING DATA PRE-PROCESSING START
    # ---------------------------------
    from ListingsPreprocessing import ListingsPreprocessing
    listings_data = ListingsPreprocessing(auto_process=True, dropNaN=True)
    # -------------------------------
    # LISTING DATA PRE-PROCESSING END

    # REVIEWS DATA PRE-PROCESSING START
    # ---------------------------------
    from ReviewsPreprocessing import ReviewsPreprocessing

    reviews_data = ReviewsPreprocessing(auto_process=True, max_features=max_features)
    # -------------------------------
    # REVIEWS DATA PRE-PROCESSING END

    # MERGING DATA START
    # ------------------
    import pandas as pd
    FILE_PATH = ""

    reviews_polarity = pd.read_csv(f"{FILE_PATH}reviews_data.csv")
    listings_preprocessed = pd.read_csv(f"{FILE_PATH}listings_data.csv")
    merged_inner = pd.merge(left=listings_preprocessed, right=reviews_polarity,
                            left_on='id', right_on='listing_id')

    merged_inner.to_csv(f"{name}.csv")
    # ----------------
    # MERGING DATA END


create_new_data = False

if create_new_data:
    create_data()

# -----TARGET-----|
# rating---------0|
# accuracy-------1|
# cleanliness----2|
# checkin--------3|
# communication--4|
# location-------5|
# value----------6|
# ----------------|

# FINAL SELECTIONS
params = {
    # target
    0: {'C': 5,
        'n_neighbors': 7},  # rating
    1: {'C': 1,
        'n_neighbors': 3},  # accuracy
    2: {'C': 2,
        'n_neighbors': 3},  # cleanliness
    3: {'C': 10,
        'n_neighbors': 5},  # checkin
    4: {'C': 5,
        'n_neighbors': 7},  # communication
    5: {'C': 2,
        'n_neighbors': 7},  # location
    6: {'C': 5,
        'n_neighbors': 7},  # value
}

# stratified is a better approach for this problem statement
dummy_strats = ['most_frequent', 'stratified', 'uniform']

# PREDICTING 7 REVIEW SCORES
for target in [0, 1, 2, 3, 4, 5, 6]:
    Logi = LogisticRegressionModel(path=f"{name}.csv", target=target, C=params[target]['C'])
    KNN = KNeighborsModel(path=f"{name}.csv", target=target, n_neighbors=params[target]['n_neighbors'])
    Dummy = DummyClassifierModel(path=f"{name}.csv", target=target, strategy=dummy_strats[2])
display_models_details()


import pandas as pd
rating = pd.read_csv("LogisticRegression/FeatureImportance/rating.csv")
accuracy = pd.read_csv("LogisticRegression/FeatureImportance/accuracy.csv")
cleanliness = pd.read_csv("LogisticRegression/FeatureImportance/cleanliness.csv")
checkin = pd.read_csv("LogisticRegression/FeatureImportance/checkin.csv")
communication = pd.read_csv("LogisticRegression/FeatureImportance/communication.csv")
location = pd.read_csv("LogisticRegression/FeatureImportance/location.csv")
value = pd.read_csv("LogisticRegression/FeatureImportance/value.csv")

# CROSS VALIDATION
"""
for target in [0, 1, 2, 3, 4, 5, 6]:
    # LOGISTIC REGRESSION START
    # -------------------------
    print('-'*10+'LOGISTIC REGRESSION'+'-'*10)
    C = [0.01, 1.0, 2.0, 5.0, 10.0, 15.0]
    
    mean_error = []
    std_error = []
    
    for c in C:
        Logi = LogisticRegressionModel(path=f"{name}.csv", target=target, C=c)
        
        score = cross_val_score(Logi.model, Logi.Xtest, Logi.ytest, cv=5, scoring='accuracy')
        mean_error.append(np.array(score).mean())
        std_error.append(np.array(score).std())
    
    fig = plt.figure()
    plt.errorbar(C, mean_error, yerr=std_error)
    plt.title(f'Logistic Regression - {Logi.predicting.capitalize()}', fontsize=20)
    plt.xlim((min(C) - 1, max(C) + 1))
    plt.locator_params(axis='x', nbins=10)
    plt.xlabel('C', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'LogisticRegression/CrossVal/{Logi.predicting}.png', dpi=120)
    plt.show();
    # -----------------------
    # LOGISTIC REGRESSION END
    
    
    # KNN START
    # ---------
    print('-'*10+'KNN'+'-'*10)
    N = [1, 3, 5, 7, 9, 11]
    
    mean_error = []
    std_error = []
    
    for n in N:
        KNN = KNeighborsModel(path=f"{name}.csv", target=target, n_neighbors=n)
    
        score = cross_val_score(KNN.model, KNN.Xtest, KNN.ytest, cv=5, scoring='accuracy')
        mean_error.append(np.array(score).mean())
        std_error.append(np.array(score).std())
    
    fig = plt.figure()
    plt.rc('font', size=20)
    plt.errorbar(N, mean_error, yerr=std_error)
    plt.title(f'KNeighbors Classifier - {KNN.predicting.capitalize()}', fontsize=20)
    plt.xlim((min(N) - 1, max(N) + 1))
    plt.locator_params(axis='x', nbins=10)
    plt.xlabel('n_neighbors', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'KNeighborsCLassifier/CrossVal/{KNN.predicting}.png', dpi=120)
    plt.show();
    # -------
    # KNN END


Dummy = DummyClassifierModel(target=target, strategy='uniform')
display_models_details()
"""
