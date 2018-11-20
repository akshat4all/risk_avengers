# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:53:43 2018

@author: kedar
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


train_data = pd.read_csv("D:/SoftDevelopement/Hackathon/mydata/trainingData1.csv")
auction_indicator_map = {'OpenAuction':1,'CloseAuction':2,np.NaN:3}
train_data['auctionIndicator'] = train_data['auctionIndicator'].map(auction_indicator_map)
train_data.fillna(method='ffill',inplace=True)



#average of binPrice
train_data['binPrice'] = (train_data.binStartPrice+train_data.binEndPrice)/2
train_data['date'] = train_data['date'].str.replace('d','',case = False)
train_data['stock'] = train_data['stock'].str.replace('stock','',case = False)


#binFrame = train_data.groupby(['date','stock','binNum'],as_index=False).agg({'volume':'sum'})

X = train_data[['date','stock','binNum','volume','auctionIndicator','binPrice']]

#train_data = train_data.set_index(train_data['binStartTime'])
#train_data.loc['09:00:00.000':'10:05:00.000']

def getframeid(binNum):
    frameId = binNum/10;
    if frameId < 1:
        return 1
    else:
        return frameId
    

X3 = X[X.auctionIndicator==3]
#set default value
X3['frameId']=0
#X3.set_index('frameId')

#X3['frameId'] = np.where(((X3['binNum']>=2) & (X3['binNum']<=13)),1,2) # 930 1030
#X3['frameId'] = np.where(((X3['binNum']>=14) & (X3['binNum']<=25)),2,3) # 1030 1130
#X3['frameId'] = np.where(((X3['binNum']>=26) & (X3['binNum']<=37)),3,4) # 1130 0130
#X3['frameId'] = np.where(((X3['binNum']>=38) & (X3['binNum']<=49)),4,5) # 0130 0230
#X3['frameId'] = np.where(((X3['binNum']>=50) & (X3['binNum']<=69)),5,6) # 0230 0400
X3.loc[X3["binNum"] < 73, "frameId"] = 9 # 930 1038
X3.loc[X3["binNum"] < 65, "frameId"] = 8 # 930 1030
X3.loc[X3["binNum"] < 57, "frameId"] = 7
X3.loc[X3["binNum"] < 49, "frameId"] = 6
X3.loc[X3["binNum"] < 41, "frameId"] = 5
X3.loc[X3["binNum"] < 33, "frameId"] = 4
X3.loc[X3["binNum"] < 25, "frameId"] = 3
X3.loc[X3["binNum"] < 17, "frameId"] = 2
X3.loc[X3["binNum"] < 9, "frameId"] = 1
#X3['frameId'] = np.where(((X3['binNum']>=14) & (X3['binNum']<=25)),2,3) # 1030 1130
#X3['frameId'] = np.where(((X3['binNum']>=26) & (X3['binNum']<=37)),3,4) # 1130 0130
#X3['frameId'] = np.where(((X3['binNum']>=38) & (X3['binNum']<=49)),4,5) # 0130 0230
#X3['frameId'] = np.where(((X3['binNum']>=50) & (X3['binNum']<=69)),5,6) # 0230 0400


#X3['frameId'] = int(X3.binNum/10) #(X3.binNum/10) if (X3.binNum/10) > 0 else 1 
#X3.loc[(X3['binNum']>1 & X3['binNum']<14),'frameId'] = 1

#df=df[df.auctionIndicator != 'CloseAuction']

X1 = X[X.auctionIndicator==1]
X1 = X1.rename(columns={"volume":"OpenAuctionVolume"})
X1 = X1.rename(columns={"binPrice":"OpenAuctionBinPrice"})
del X1['binNum']
del X1['auctionIndicator']
#X2 = X[X.auctionIndicator==2]

Y = X3.groupby(['date','stock','frameId'],as_index=False).agg({'volume':'mean','binPrice':'mean'})

Y1 =pd.pivot_table(Y, values = 'volume', index=['date','stock'],columns = ['frameId']).reset_index()
Y1 =Y1.rename(columns={1.0:"1_volume"})
Y1 =Y1.rename(columns={2.0:"2_volume"})
Y1 =Y1.rename(columns={3.0:"3_volume"})
Y1 =Y1.rename(columns={4.0:"4_volume"})
Y1 =Y1.rename(columns={5.0:"5_volume"})
Y1 =Y1.rename(columns={6.0:"6_volume"})
Y1 =Y1.rename(columns={7.0:"7_volume"})
Y1 =Y1.rename(columns={8.0:"8_volume"})
Y1 =Y1.rename(columns={9.0:"9_volume"})
Y1.dtypes

Y2 =pd.pivot_table(Y, values = 'binPrice', index=['date','stock'],columns = ['frameId']).reset_index()
Y2 =Y2.rename(columns={1.0:"1_binPrice"})
Y2 =Y2.rename(columns={2.0:"2_binPrice"})
Y2 =Y2.rename(columns={3.0:"3_binPrice"})
Y2 =Y2.rename(columns={4.0:"4_binPrice"})
Y2 =Y2.rename(columns={5.0:"5_binPrice"})
Y2 =Y2.rename(columns={6.0:"6_binPrice"})
Y2 =Y2.rename(columns={7.0:"7_binPrice"})
Y2 =Y2.rename(columns={8.0:"8_binPrice"})
Y2 =Y2.rename(columns={9.0:"9_binPrice"})

result = pd.merge(Y1,Y2,on=['date','stock'])


#Y = Y.rename(columns={"volume":"trading_volume"})
#X1 = X1.rename(columns={"volume":"OpenAuctionVolume"})
#X2 = X2.rename(columns={"volume":"CloseAuctionVolume"})

result2 = pd.merge(X1,result,on=['date','stock'])

expected_output = train_data[['auctionIndicator','output_closeAuctionVolume','output_closePriceDirection']]
expected_output = expected_output[expected_output.auctionIndicator==1]
expected_output = expected_output.reset_index()
del expected_output['index']
del expected_output['auctionIndicator']


expected_train_data=expected_output[['output_closeAuctionVolume']]
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(result2, expected_train_data)

from sklearn import metrics


#print("Accuracy: {0:.4f}".format(np.sqrt(metrics.mean_squared_error(expected_train_data, lg_pred_train))))
#print("Accuracy: {0:.4f}".format(metrics.r2_score(y_train, lg_pred_train)))

#----------------------------------------------------------------------------------------------



#result = pd.merge(result,Y,on=['date','stock'])















#---------------------------------------------------------------------------

scaling_df = train_data[['date','stock','binStartPrice','binEndPrice']]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(scaling_df[['binStartPrice','binEndPrice']])
MinMaxScaler(copy=False, feature_range=(0, 1))
scaling_array = scaler.transform(scaling_df[['binStartPrice','binEndPrice']])
scaling_df = pd.DataFrame(scaling_array)



#day_stock = ''
#volume = 0
#for rec in X.rows:
#    if day_stock =='':
#        day_stock = rec['date']+rec['stock']
#    if day_stock == rec['date']+rec['stock']:
#        
        


