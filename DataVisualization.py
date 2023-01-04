import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

FILE_PATH = ""

listings = pd.read_csv(f"{FILE_PATH}listings.csv")
reviews = pd.read_csv(f"{FILE_PATH}reviews.csv")

print(f"Total listings: {len(listings.axes[0])}")
print(f"Features in listings: {len(listings.axes[1])}")

print(f"\nTotal reviews: {len(reviews.axes[0])}")
print(f"Features in reviews: {len(reviews.axes[1])}")


def get_nan_values(df, cert=None, drop_NaN=True):
    store_NaN = {}
    if cert != None:
        store_NaN[cert] = [df[cert].isna().sum()]
    else:
        for key in df.keys():
            store_NaN[key] = [df[key].isna().sum()]
    store_NaN = pd.DataFrame(store_NaN)

    if drop_NaN:
        store_NaN = store_NaN[store_NaN != 0]
        store_NaN = store_NaN.T.dropna()
    else:
        store_NaN = store_NaN.T
    return store_NaN


df = get_nan_values(listings, drop_NaN=True)
df = df.sort_values(by=[0], ascending=False)

col = []
for val in df[0]:
    if val < 1000:
        col.append('green')
    elif 1000 < val < 3000:
        col.append('blue')
    elif 3000 < val < 4500:
        col.append('orange')
    else:
        col.append('red')

fig = plt.figure(figsize=(20, 10))
plt.xticks(rotation=90, fontsize=14)
plt.yticks(fontsize=14)
plt.bar(list(df.T.keys()), df[0], color=col)
plt.show();

df = get_nan_values(reviews, drop_NaN=False)
df = df.sort_values(by=[0], ascending=False)

col = []
for val in df[0]:
    if val < 1000:
        col.append('green')
    elif 1000 < val < 3000:
        col.append('blue')
    elif 3000 < val < 4500:
        col.append('orange')
    else:
        col.append('red')

fig = plt.figure(figsize=(20, 10))
plt.xticks(rotation=90)
plt.bar(list(df.T.keys()), df[0], color=col, width=0.1)
plt.show();

host_about_NaN = get_nan_values(listings, cert='host_about')
print(host_about_NaN.T)

print(listings.host_response_time.value_counts())

listings['host_response_rate'] = listings["host_response_rate"].str.replace("%", "")
listings['host_response_rate'] = pd.to_numeric(listings['host_response_rate'])
listings['host_acceptance_rate'] = listings["host_acceptance_rate"].str.replace("%", "")
listings['host_acceptance_rate'] = pd.to_numeric(listings['host_acceptance_rate'])

keys = ['host_response_rate', 'host_acceptance_rate']

tabular_format = []

for key in keys:
    tabular_format.append([key, listings[key].min(), listings[key].max(), listings[key].mean(), listings[key].median()])

print(tabulate(tabular_format, headers=['Name', 'Min', 'Max', 'Mean', 'Median']))

keys = ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication',
        'review_scores_location', 'review_scores_value']

tabular_format = []

for key in keys:
    tabular_format.append([key, listings[key].min(), listings[key].max(), listings[key].mean(), listings[key].median()])

print(tabulate(tabular_format, headers=['Name', 'Min', 'Max', 'Mean', 'Median']))


import matplotlib.pyplot as plt
import pandas as pd
listings = pd.read_csv("final_train_data.csv")
y = listings[[
'review_scores_rating', 
'review_scores_accuracy', 
'review_scores_cleanliness', 
'review_scores_checkin', 
'review_scores_communication', 
'review_scores_location', 
'review_scores_value' 
]]
index = [
'rating', 
'accuracy', 
'cleanliness', 
'checkin', 
'communication', 
'location', 
'value' 
]
# plt.figure()
plt.rcParams.update({'font.size': 12}) # must set in top
# fig = plt.figure(figsize=(20, 10))
df = pd.DataFrame({'0': [1052, 1058, 1042, 1593, 1620, 1080, 1094], 
                   '1': [1125, 1108, 1128, 1557, 1530, 1071, 1030], 
                   '2': [973, 984, 980, None, None, 999, 1026]}, index=index)
ax = df.plot.bar(rot=90, title='Review Score Values after binning')
ax.set_ylabel("value counts")


import pandas as pd
from ListingsPreprocessing import ListingsPreprocessing

listings_data = ListingsPreprocessing(auto_process=False, dropNaN=True)

ENCODE_COLUMNS = [
    'host_is_superhost', 
    'host_identity_verified', 'instant_bookable',
]


listings_data.label_encode(columns=ENCODE_COLUMNS)
listings_data.count_vars('amenities', drop_column=True)

listings_data.txt_to_numeric('$', ',', column='price')
listings_data.txt_to_numeric('%', column='host_response_rate')
listings_data.txt_to_numeric('%', column='host_acceptance_rate')

training_data = pd.read_csv('final_train_data.csv')

new_store = pd.DataFrame([])
new_store['accommodates_not_normalized'] = listings_data.listings['accommodates']
new_store['accommodates'] = training_data['accommodates']

new_store['price_not_normalized'] = listings_data.listings['price']
new_store['price'] = training_data['price']

new_store['price_not_normalized'] = listings_data.listings['price']
new_store['price'] = training_data['price']

new_store['host_response_rate_not_normalized'] = listings_data.listings['host_response_rate']
new_store['host_response_rate'] = training_data['host_response_rate']


new_store['host_acceptance_rate_not_normalized'] = listings_data.listings['host_acceptance_rate']
new_store['host_acceptance_rate'] = training_data['host_acceptance_rate']

new_store['review_scores_rating'] = listings_data.listings['review_scores_rating']

import matplotlib.pyplot as plt

col = ['host_acceptance_rate', 'price', 'host_response_rate']
i = 1

fig = plt.figure()
plt.scatter(new_store[col[i]], new_store.review_scores_rating)
name = " ".join(col[i].split('_')).title()
plt.xlabel(name)
plt.ylabel('Review Scores Rating')
plt.title('Normalised Data')
plt.savefig(f'{col[i]}.png', dpi=300)
plt.close(fig)

fig = plt.figure()
plt.scatter(new_store[f"{col[i]}_not_normalized"], new_store.review_scores_rating)
plt.xlabel(name)
plt.ylabel('Review Scores Rating')
plt.title('Not Normalised Data')
plt.savefig(f'{col[i]}_not_normalised.png', dpi=300)
plt.close(fig)

import seaborn as sns

# START 3D PLOT
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(new_store.price, new_store.maximum_nights, new_store.review_scores_rating_class, color='blue')
ax.set_xlabel('price')
ax.set_ylabel('maximum')
ax.set_zlabel('rating')



import pandas as pd
vectors = pd.read_csv('vectors_tfidf_25.csv')
new_store['tfidf_would'] = training_data['tfidf_would']
fig = plt.figure()
plt.scatter(new_store['tfidf_would'], new_store.review_scores_rating)
plt.show()


listings_data = ListingsPreprocessing(auto_process=True)


speed = [0.1, 17.5, 40, 48, 52, 69, 88]
lifespan = [2, 8, 70, 1.5, 25, 12, 28]
index = ['snail', 'pig', 'elephant',
         'rabbit', 'giraffe', 'coyote', 'horse']

import pandas as pd
acc1 = [0.349206,
       0.346032,
       0.369841,
       0.515873,
       0.51746,
       0.322222,
       0.333333,]

acc2 = [0.347619,
        0.301587,
        0.350794,
        0.487302,
        0.468254,
        0.322222,
        0.328571]

term = ['Rating', 'Accuracy', 'Cleanliness', 'Checkin', 
        'Communication', 'Location', 'Value']

df = pd.DataFrame({'most_frequent': acc1, 
                   'strategy': acc2}, index=term)
ax = df.plot.bar(rot=15, fontsize=13)
ax.legend(fontsize=16)
ax.set_ylabel('Accuracy', fontsize=16)
plt.title("Dummy Classifier Comparison")
plt.grid(False)
