# Data Preprocessing

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


train_data = pd.read_csv("D:/SoftDevelopement/Hackathon/trainingData1.csv")

auction_indicator_map = {'OpenAuction':1,'CloseAuction':2,np.NaN:3}
train_data['auctionIndicator'] = train_data['auctionIndicator'].map(auction_indicator_map)


#num_obs = len(train_data)
#num_1 = len(train_data.loc[train_data['output_closePriceDirection'] == 1])
#num_0 = len(train_data.loc[train_data['output_closePriceDirection'] == 0])
#num_minus_1 = len(train_data.loc[train_data['output_closePriceDirection'] == -1])
#print("Number of True cases:  {0} ({1:2.2f}%)".format(num_1, (num_1/num_obs) * 100))
#print("Number of False cases: {0} ({1:2.2f}%)".format(num_0, (num_0/num_obs) * 100))
#print("Number of False cases: {0} ({1:2.2f}%)".format(num_minus_1, (num_minus_1/num_obs) * 100))

list_of_col = list(train_data.columns.values)

print("# rows in dataframe {0}".format(len(train_data)))

# to define plot
def plot_corr(df, size=11):
    corr = df.corr()    # data frame correlation function
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)   # color code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)), corr.columns)  # draw x tick marks
    plt.yticks(range(len(corr.columns)), corr.columns)  # draw y tick marks      
 
plot = plot_corr(train_data)
cor_data = train_data.corr()

train_data['date'] = train_data['date'].str.replace('d','',case = False)
train_data['binStartTime'] = train_data['binStartTime'].str.replace(':00.000','',case = False)
train_data['binStartTime'] = train_data['binStartTime'].str.replace(':','',case = False)
train_data['binEndTime'] = train_data['binEndTime'].str.replace(':00.000','',case = False)
train_data['binEndTime'] = train_data['binEndTime'].str.replace(':','',case = False)
train_data['stock'] = train_data['stock'].str.replace('stock','',case = False)

x_train = train_data.iloc[:,0:12]
y_train = train_data.iloc[:,14]

#x_train.drop(x_train.columns[[3,4]],axis=1)
#del x_train['binEndTime']
#del x_train['binStartTime']
#x_train.dtypes

for col in x_train:
    x_train[col] = pd.to_numeric(x_train[col],errors='coerce')
    

#interpolate_test = x_train.head(100)

x_train = x_train.interpolate(method='nearest')


#Prepare test data
test_data_volume = pd.read_csv("D:/SoftDevelopement/Hackathon/testingVolume1.csv")
test_data_volume['auctionIndicator'] = test_data_volume['auctionIndicator'].map(auction_indicator_map)

test_data_volume['date'] = test_data_volume['date'].str.replace('d','',case = False)
test_data_volume['binStartTime'] = test_data_volume['binStartTime'].str.replace(':00.000','',case = False)
test_data_volume['binStartTime'] = test_data_volume['binStartTime'].str.replace(':','',case = False)
test_data_volume['binEndTime'] = test_data_volume['binEndTime'].str.replace(':00.000','',case = False)
test_data_volume['binEndTime'] = test_data_volume['binEndTime'].str.replace(':','',case = False)
test_data_volume['stock'] = test_data_volume['stock'].str.replace('stock','',case = False)












nb_model = GaussianNB()

nb_model.fit(x_train, y_train.ravel())
nb_predict_train = nb_model.predict(x_train)

#print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train, nb_predict_train)))
#print()

rf_model = RandomForestClassifier(random_state=42)      # Create random forest object
rf_model.fit(x_train, y_train.ravel())

     

     
#for col in list_of_col:
#    print("# rows missing {0}: {1}".format(col,len(train_data.loc[train_data[col] == np.nan])))

#imp = Imputer(missing_values ="NaN",strategy = "mean",axis = 0)
#train_data['volume'] = imp.fit_transform(train_data['volume'].reshape(-1,1))





















