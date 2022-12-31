import numpy as np
import matplotlib.pyplot as plt

from MLModels import LogisticRegressionModel, KNeighborsModel, DummyClassifierModel
from MLModels import display_models_details

from sklearn.model_selection import cross_val_score

def create_data():
    # LISTING DATA PRE-PROCESSING START
    # ---------------------------------
    from ListingsPreprocessing import ListingsPreprocessing
    listings_data = ListingsPreprocessing(auto_process=True, dropNaN=True)

    # IMPUTING DATA
    """
    listings_data_imputed = ListingsPreprocessing(auto_process=False)
    listings_data_imputed.label_encode(columns=ENCODE_COLUMNS)
    listings_data_imputed.count_vars('amenities')
    listings_data_imputed.count_vars('host_verifications')
    
    listings_data_imputed.txt_to_numeric('$', ',', column='price')
    listings_data_imputed.txt_to_numeric('%', column='host_response_rate')
    listings_data_imputed.txt_to_numeric('%', column='host_acceptance_rate')
    
    
    # NOT NEEDED IF WE'RE DOING DROPNA
    imputation_data = ['host_response_time', 'host_response_rate', 'host_acceptance_rate',
                       'bedrooms', 'beds'
                       ,'review_scores_rating', 'review_scores_accuracy',
                       'review_scores_cleanliness', 'review_scores_checkin',
                       'review_scores_communication', 'review_scores_location',
                       'review_scores_value']
    for data in imputation_data:
        mean = listings[data].mean()
        if 'host' and 'bed' in data:
            mean = int(mean)
        listings[data].fillna(mean, inplace = True)
    
    scaling_data = set(listings_data_imputed.listings.keys()) - set(ENCODE_COLUMNS) - set(ID_FEATURE) - set(PRED_FEATURES)
    listings_data_imputed.scale_values(columns=scaling_data)
    
    listings_data_imputed.bin_values()
    
    listings = listings_data_imputed.listings
    
    listings_data_imputed.save_file(file_name='listings_imputed_data.csv')
    """
    # -------------------------------
    # LISTING DATA PRE-PROCESSING END

    # REVIEWS DATA PRE-PROCESSING START
    # ---------------------------------
    from ReviewsPreprocessing import ReviewsPreprocessing
    reviews_data = ReviewsPreprocessing(auto_process=True)
    # -------------------------------
    # REVIEWS DATA PRE-PROCESSING END

    # MERGING DATA START
    # ------------------
    import pandas as pd
    FILE_PATH = ""

    reviews_polarity = pd.read_csv(f"{FILE_PATH}reviews_data.csv")
    listings_preprocessed = pd.read_csv(f"{FILE_PATH}listings_data.csv")
    merged_inner = pd.merge(left=listings_preprocessed, right=reviews_polarity, left_on='id', right_on='listing_id')
    merged_inner.to_csv("final_train_data.csv")
    # ----------------
    # MERGING DATA END


create_new_data = True

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

target = 1

# for target in [0, 1, 2, 3, 4, 5, 6]:
#     Logi = LogisticRegressionModel(target=target, C=1.0)
#     KNN = KNeighborsModel(target=target, n_neighbors=5)
#     Dummy = DummyClassifierModel(target=target, strategy='most_frequent')
# display_models_details()


for target in [0, 1, 2, 3, 4, 5, 6]:
    # LOGISTIC REGRESSION START
    # -------------------------
    print('-'*10+'LOGISTIC REGRESSION'+'-'*10)
    C = [0.001, 0.01, 1.0, 2.0, 3.0, 4.0, 10]
    
    mean_error = []
    std_error = []
    
    for c in C:
        Logi = LogisticRegressionModel(target=target, C=c)
        
        score = cross_val_score(Logi.model, Logi.Xtest, Logi.ytest, cv=5, scoring='accuracy')
        mean_error.append(np.array(score).mean())
        std_error.append(np.array(score).std())
    
    plt.rc('font', size=12)
    plt.errorbar(C, mean_error, yerr=std_error)
    plt.title(f'Logistic Regression - {Logi.predicting}')
    plt.xlim((min(C) - 1, max(C) + 1))
    plt.locator_params(axis='x', nbins=10)
    plt.xlabel('C')
    plt.ylabel('Accuracy')
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
        KNN = KNeighborsModel(target=target, n_neighbors=n)
    
        score = cross_val_score(KNN.model, KNN.Xtest, KNN.ytest, cv=5, scoring='accuracy')
        mean_error.append(np.array(score).mean())
        std_error.append(np.array(score).std())
    
    plt.rc('font', size=12)
    plt.errorbar(N, mean_error, yerr=std_error)
    plt.title('KNeighbors Classifier - {KNN.predicting}')
    plt.xlim((min(N) - 1, max(N) + 1))
    plt.locator_params(axis='x', nbins=10)
    plt.xlabel('n_neighbors')
    plt.ylabel('Accuracy')
    plt.savefig(f'KNeighborsCLassifier/CrossVal/{KNN.predicting}.png', dpi=120)
    plt.show();
    # -------
    # KNN END


Dummy = DummyClassifierModel(target=target, strategy='uniform')
display_models_details()

