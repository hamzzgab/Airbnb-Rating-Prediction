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

"""
import matplotlib.pyplot as plt
import pandas as pd
listings = pd.read_csv("merged_data_tfidf_100.csv")


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
plt.savefig('bruh.jpg')


from ListingsPreprocessing import ListingsPreprocessing
ENCODE_COLUMNS = [
    'host_is_superhost', 
    'host_identity_verified', 'instant_bookable',
]
listings_data = ListingsPreprocessing(auto_process=False, dropNaN=True)
listings_data.label_encode(columns=ENCODE_COLUMNS)
listings_data.count_vars('amenities', drop_column=True)
# self.count_vars('host_verifications', drop_column=False)

listings_data.txt_to_numeric('$', ',', column='price')
listings_data.txt_to_numeric('%', column='host_response_rate')
listings_data.txt_to_numeric('%', column='host_acceptance_rate')

listings = listings_data.listings

cat, bins = pd.qcut(listings.review_scores_accuracy, 3, duplicates='drop', retbins=True)

self.listings.review_scores_accuracy = pd.qcut(self.listings.review_scores_accuracy, 3, duplicates='drop', labels=[0,1,2])
"""
