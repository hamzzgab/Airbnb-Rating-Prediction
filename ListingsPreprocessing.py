import pandas as pd
from ast import literal_eval
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer(sparse_output=True)

FILE_PATH = ""

ID_FEATURE = ['id']

IMP_FEATURES = [
    'host_is_superhost','host_verifications','host_identity_verified',
    'accommodates','minimum_nights','maximum_nights','instant_bookable','price',
    'host_response_time','host_response_rate','host_acceptance_rate',
    'host_listings_count','amenities',
    # 'latitude','longitude',

    # 'host_location', 
    'bedrooms','beds', 
    # 'host_about', 'host_neighborhood', 'neighborhood_overview',

    # 'property_type',
    'room_type',
    # 'bathrooms_text', 'description',
    # 'host_total_listings_count',
    # 'has_availability', # Most of them are having availability data imbalance

    # 'neighbourhood',
    'neighbourhood_cleansed',
]

PRED_FEATURES = [
    # PREDICTIONS
    'review_scores_rating', 'review_scores_accuracy', 
    'review_scores_cleanliness', 'review_scores_checkin', 
    'review_scores_communication', 'review_scores_location', 
    'review_scores_value', 
]


FEATURES = ID_FEATURE + IMP_FEATURES + PRED_FEATURES


ENCODE_COLUMNS = [
    'host_is_superhost', 
    'host_identity_verified', 'instant_bookable',
]

BINARIZE_COLUMNS = [
    'room_type', 'neighbourhood_cleansed',
    'host_response_time', 'host_verifications'
]


label_encoder = LabelEncoder()

# SCALING LISTINGS DATA START
# ---------------------------
def z_score(x):
  return (x - x.mean())/(x.std())

def minmax(x):
  return (x - x.min()) / (x.max() - x.min())


"""
for f in listings.keys():
    new_listings[f] = z_score(listings[f])
"""
# -------------------------
# SCALING LISTINGS DATA END


class ListingsPreprocessing:
  def __init__(self, feature_columns=None, auto_process=False, dropNaN=False):
    if feature_columns is None:
        feature_columns = FEATURES
    self.listings = pd.read_csv(f"{FILE_PATH}listings.csv",  converters={'amenities': literal_eval, 
                                                            'host_verifications': literal_eval})
    self.feature_columns = feature_columns
    
    self.dropNaN = dropNaN
    self.drop_columns()
    
    if auto_process:
      self.label_encode(columns=ENCODE_COLUMNS)
      self.count_vars('amenities', drop_column=True)
      # self.count_vars('host_verifications', drop_column=False)

      self.txt_to_numeric('$', ',', column='price')
      self.txt_to_numeric('%', column='host_response_rate')
      self.txt_to_numeric('%', column='host_acceptance_rate')

      self.bin_values()
     
      self.scaling_data = set(self.listings.keys()) - set(ENCODE_COLUMNS) - set(BINARIZE_COLUMNS) - set(ID_FEATURE) - set(PRED_FEATURES)
      self.scale_values(columns=self.scaling_data)
      
      self.one_hot_encode('room_type', 'neighbourhood_cleansed', 'host_response_time', 'host_verifications', pop=True)
      
      self.save_file()


  def drop_columns(self):
      self.listings.drop(columns=self.listings.columns.difference(self.feature_columns), axis=1, inplace=True)
      if self.dropNaN:
          self.drop_NaN()
      
  def drop_NaN(self):
      self.listings = self.listings.dropna()
    
  def label_encode(self, columns):
    for column in columns:
        if column == 'host_response_time':
            self.listings = self.listings.replace('within an hour', 3)
            self.listings = self.listings.replace('within a few hours', 2)
            self.listings = self.listings.replace('within a day', 1)
            self.listings = self.listings.replace('a few days or more', 0)
        else:
            self.listings[column] = label_encoder.fit_transform(self.listings[column])

  def count_vars(self, column, drop_column=True):
    self.listings[f'{column}_count'] = range(0, len(self.listings))
    for index, row in self.listings.iterrows():
      self.listings[f'{column}_count'][index] = len(row[f'{column}'])
    if drop_column:
        self.listings.drop([f'{column}'], axis=1, inplace=True)

  def txt_to_numeric(self, *argv, **kwargs):
    for arg in argv:
      self.listings[kwargs['column']] = self.listings[kwargs['column']].str.replace(arg, '')      
    self.listings[kwargs['column']] = pd.to_numeric(self.listings[kwargs['column']])

  def one_hot_encode(self, *argv, pop=False):
      for arg in argv:
          if type(self.listings[arg][0]) == type('str'):
              df = pd.get_dummies(self.listings[arg])
              self.listings = pd.concat([self.listings, df], axis=1)
              if pop:
                  self.listings.pop(arg)
          else:
              self.listings = self.listings.join(
                  pd.DataFrame.sparse.from_spmatrix(
                      mlb.fit_transform(self.listings.pop('host_verifications')),
                      index=self.listings.index,
                      columns=mlb.classes_))
  
  def bin_values(self):
    self.listings.review_scores_rating = pd.qcut(self.listings.review_scores_rating, 3, duplicates='drop', labels=[0,1,2])
    self.listings.review_scores_cleanliness = pd.qcut(self.listings.review_scores_cleanliness, 3, duplicates='drop', labels=[0,1,2])
    self.listings.review_scores_checkin = pd.qcut(self.listings.review_scores_checkin, 2, duplicates='drop', labels=[0, 1])
    self.listings.review_scores_communication = pd.qcut(self.listings.review_scores_communication, 2, duplicates='drop', labels=[0,1])
    self.listings.review_scores_location = pd.qcut(self.listings.review_scores_location, 3, duplicates='drop', labels=[0,1,2])
    self.listings.review_scores_value = pd.qcut(self.listings.review_scores_value, 3, duplicates='drop', labels=[0,1,2])
    self.listings.review_scores_accuracy = pd.qcut(self.listings.review_scores_accuracy, 3, duplicates='drop', labels=[0,1,2])

  def scale_values(self, columns=None, func=z_score):
       for f in columns:
           if f not in ['amenities', 'host_verifications', 'room_type', 'neighbourhood_cleansed', 'host_response_time']:
               self.listings[f] = func(self.listings[f])

  def save_file(self, file_name='listings_data.csv'):
    self.listings.to_csv(file_name)