from sklearn import preprocessing
import torch
import numpy as np
import pandas as pd

def data_generating_bike(data_path, num_random=0, noise_level=0):
    df = pd.read_csv(data_path+"hour.csv", sep=",")
    df = df.dropna(axis=0)
    # remove the instant code and date
    df = df.drop(['instant', 'dteday', 'holiday', 'temp', 'weekday', 'season'], axis = 1)
    # drop casual and registered
    df = df.drop(['casual', 'registered'], axis = 1)
    np.random.seed(129)
    msk_1 = np.random.rand(len(df)) < 0.8
    traindf = df[msk_1]
    testdf = df[~msk_1]
    # training set
    trainx = traindf.drop('cnt', axis=1).values
    trainy = traindf['cnt'].values
    # test set
    testx = testdf.drop('cnt', axis=1).values
    testy = testdf['cnt'].values

    scalerx = preprocessing.StandardScaler().fit(trainx)
    trainx = scalerx.transform(trainx)
    testx = scalerx.transform(testx)

    scalery = preprocessing.StandardScaler().fit(trainy.reshape(-1,1))
    trainy = scalery.transform(trainy.reshape(-1,1)).reshape(1, -1)
    testy = scalery.transform(testy.reshape(-1,1)).reshape(1, -1)
    
    # transfer data into tensor
    x = torch.tensor(trainx, dtype = torch.float)
    y = torch.tensor(trainy[0], dtype = torch.float)
    x_test = torch.tensor(testx, dtype = torch.float)
    y_test = torch.tensor(testy[0], dtype = torch.float)
    
    if num_random != 0:
        rand_features = torch.randn(x.shape[0], num_random) * 2
        x = torch.cat((x, rand_features), 1)
        rand_features = torch.randn(x_test.shape[0], num_random) *2
        x_test = torch.cat((x_test, rand_features), 1)
    
    if noise_level != 0:
        noise = torch.randn_like(y) * np.sqrt(noise_level); noise_test = torch.randn_like(y_test) * np.sqrt(noise_level)
        y = y * np.sqrt(1-noise_level) + noise; y_test = y_test * np.sqrt(1-noise_level) + noise_test;         

    num_useful = 8.; num_feature = x.shape[1]
    
    return x, y, x_test, y_test

## California Housing Experiments
def data_generating_cal(data_path, num_random=0, noise_level=0):
    df = pd.read_csv(data_path + "housing.csv", sep=",")
    df = df.dropna(axis=0)
    # remove ocean_proximity
    df = df.drop('ocean_proximity', axis = 1)
    # split data into training and test set
    np.random.seed(129)
    msk_1 = np.random.rand(len(df)) < 0.8
    traindf = df[msk_1]
    testdf = df[~msk_1]
    # training set
    trainx = traindf.drop('median_house_value', axis=1).values
    trainy = traindf['median_house_value'].values
    # test set
    testx = testdf.drop('median_house_value', axis=1).values
    testy = testdf['median_house_value'].values 

    scalerx = preprocessing.StandardScaler().fit(trainx)
    trainx = scalerx.transform(trainx)
    testx = scalerx.transform(testx)

    scalery = preprocessing.StandardScaler().fit(trainy.reshape(-1,1))
    trainy = scalery.transform(trainy.reshape(-1,1)).reshape(1, -1)
    testy = scalery.transform(testy.reshape(-1,1)).reshape(1, -1)
    
    # transfer data into tensor
    x = torch.tensor(trainx, dtype = torch.float)
    y = torch.tensor(trainy[0], dtype = torch.float)
    x_test = torch.tensor(testx, dtype = torch.float)
    y_test = torch.tensor(testy[0], dtype = torch.float)
    
    if num_random != 0:
        rand_features = torch.randn(x.shape[0], num_random) * 2
        x = torch.cat((x, rand_features), 1)
        rand_features = torch.randn(x_test.shape[0], num_random) *2
        x_test = torch.cat((x_test, rand_features), 1)
    
    if noise_level != 0:
        noise = torch.randn_like(y) * np.sqrt(noise_level); noise_test = torch.randn_like(y_test) * np.sqrt(noise_level)
        y = y * np.sqrt(1-noise_level) + noise; y_test = y_test * np.sqrt(1-noise_level) + noise_test;    
        
    num_useful = 8.; num_feature = x.shape[1]

    return x, y, x_test, y_test

## Concrete datasets cross validation
def data_generating_concrete(data_path, num_random=0, noise_level=0):
    df = pd.read_csv(data_path + "concrete.csv", sep=",")
    np.random.seed(129)
    msk_1 = np.random.rand(len(df)) < 0.8
    traindf = df[msk_1]
    testdf = df[~msk_1]
    # training set
    trainx = traindf.drop('y', axis=1).values
    trainy = traindf['y'].values
    # test set
    testx = testdf.drop('y', axis=1).values
    testy = testdf['y'].values

    scalerx = preprocessing.StandardScaler().fit(trainx)
    trainx = scalerx.transform(trainx)
    testx = scalerx.transform(testx)

    scalery = preprocessing.StandardScaler().fit(trainy.reshape(-1,1))
    trainy = scalery.transform(trainy.reshape(-1,1)).reshape(1, -1)
    testy = scalery.transform(testy.reshape(-1,1)).reshape(1, -1)
    
    # transfer data into tensor
    x = torch.tensor(trainx, dtype = torch.float)
    y = torch.tensor(trainy[0], dtype = torch.float)
    x_test = torch.tensor(testx, dtype = torch.float)
    y_test = torch.tensor(testy[0], dtype = torch.float)
    
    if num_random != 0:
        rand_features = torch.randn(x.shape[0], num_random) * 2
        x = torch.cat((x, rand_features), 1)
        rand_features = torch.randn(x_test.shape[0], num_random) *2
        x_test = torch.cat((x_test, rand_features), 1)
    
    if noise_level != 0:
        noise = torch.randn_like(y) * np.sqrt(noise_level); noise_test = torch.randn_like(y_test) * np.sqrt(noise_level)
        y = y * np.sqrt(1-noise_level) + noise; y_test = y_test * np.sqrt(1-noise_level) + noise_test;   
        
    num_useful = 8.; num_feature = x.shape[1]

    return x, y, x_test, y_test

def data_generating_energy(data_path, num_random=0, noise_level=0):
    df = pd.read_csv(data_path + "energy.csv", sep=",")
    df = df.drop(['Unnamed: 10','Unnamed: 11','Y1'], axis = 1)
    df = df.dropna(axis=0)
    np.random.seed(129)
    msk_1 = np.random.rand(len(df)) < 0.8
    traindf = df[msk_1]
    testdf = df[~msk_1]
    # training set
    trainx = traindf.drop('Y2', axis=1).values
    trainy = traindf['Y2'].values
    # test set
    testx = testdf.drop('Y2', axis=1).values
    testy = testdf['Y2'].values

    scalerx = preprocessing.StandardScaler().fit(trainx)
    trainx = scalerx.transform(trainx)
    testx = scalerx.transform(testx)

    scalery = preprocessing.StandardScaler().fit(trainy.reshape(-1,1))
    trainy = scalery.transform(trainy.reshape(-1,1)).reshape(1, -1)
    testy = scalery.transform(testy.reshape(-1,1)).reshape(1, -1)
    
    # transfer data into tensor
    x = torch.tensor(trainx, dtype = torch.float)
    y = torch.tensor(trainy[0], dtype = torch.float)
    x_test = torch.tensor(testx, dtype = torch.float)
    y_test = torch.tensor(testy[0], dtype = torch.float)
    
    if num_random != 0:
        rand_features = torch.randn(x.shape[0], num_random) * 2
        x = torch.cat((x, rand_features), 1)
        rand_features = torch.randn(x_test.shape[0], num_random) *2
        x_test = torch.cat((x_test, rand_features), 1)
    
    if noise_level != 0:
        noise = torch.randn_like(y) * np.sqrt(noise_level); noise_test = torch.randn_like(y_test) * np.sqrt(noise_level)
        y = y * np.sqrt(1-noise_level) + noise; y_test = y_test * np.sqrt(1-noise_level) + noise_test;     
        
    num_useful = 8.; num_feature = x.shape[1]

    return x, y, x_test, y_test

## Concrete datasets
def data_generating_yacht(data_path, num_random=0, noise_level=0):
    df = pd.read_csv(data_path+"yacht", sep=" ")
    np.random.seed(129)
    msk_1 = np.random.rand(len(df)) < 0.8
    traindf = df[msk_1]
    testdf = df[~msk_1]
    # training set
    trainx = traindf.drop('y', axis=1).values
    trainy = traindf['y'].values
    # test set
    testx = testdf.drop('y', axis=1).values
    testy = testdf['y'].values

    scalerx = preprocessing.StandardScaler().fit(trainx)
    trainx = scalerx.transform(trainx)
    testx = scalerx.transform(testx)

    scalery = preprocessing.StandardScaler().fit(trainy.reshape(-1,1))
    trainy = scalery.transform(trainy.reshape(-1,1)).reshape(1, -1)
    testy = scalery.transform(testy.reshape(-1,1)).reshape(1, -1)
    
    # transfer data into tensor
    x = torch.tensor(trainx, dtype = torch.float)
    y = torch.tensor(trainy[0], dtype = torch.float)
    x_test = torch.tensor(testx, dtype = torch.float)
    y_test = torch.tensor(testy[0], dtype = torch.float)
    
    if num_random != 0:
        rand_features = torch.randn(x.shape[0], num_random) * 2
        x = torch.cat((x, rand_features), 1)
        rand_features = torch.randn(x_test.shape[0], num_random) *2
        x_test = torch.cat((x_test, rand_features), 1)
    
    if noise_level != 0:
        torch.manual_seed(129)
        noise = torch.randn_like(y) * np.sqrt(noise_level); noise_test = torch.randn_like(y_test) * np.sqrt(noise_level)
        y = y * np.sqrt(1-noise_level) + noise; y_test = y_test * np.sqrt(1-noise_level) + noise_test;   
        
    num_useful = 6.; num_feature = x.shape[1]

    return x, y, x_test, y_test

def data_generating_boston(data_path, num_random=0, noise_level=0):
    df = pd.read_csv(data_path + "boston.csv", sep=",")
    np.random.seed(129)
    msk_1 = np.random.rand(len(df)) < 0.8
    traindf = df[msk_1]
    testdf = df[~msk_1]
    # training set
    trainx = traindf.drop('MEDV', axis=1).values
    trainy = traindf['MEDV'].values
    # test set
    testx = testdf.drop('MEDV', axis=1).values
    testy = testdf['MEDV'].values

    scalerx = preprocessing.StandardScaler().fit(trainx)
    trainx = scalerx.transform(trainx)
    testx = scalerx.transform(testx)

    scalery = preprocessing.StandardScaler().fit(trainy.reshape(-1,1))
    trainy = scalery.transform(trainy.reshape(-1,1)).reshape(1, -1)
    testy = scalery.transform(testy.reshape(-1,1)).reshape(1, -1)
    
    # transfer data into tensor
    x = torch.tensor(trainx, dtype = torch.float)
    y = torch.tensor(trainy[0], dtype = torch.float)
    x_test = torch.tensor(testx, dtype = torch.float)
    y_test = torch.tensor(testy[0], dtype = torch.float)
    
    if num_random != 0:
        rand_features = torch.randn(x.shape[0], num_random) * 2
        x = torch.cat((x, rand_features), 1)
        rand_features = torch.randn(x_test.shape[0], num_random) *2
        x_test = torch.cat((x_test, rand_features), 1)
    
    if noise_level != 0:
        noise = torch.randn_like(y) * np.sqrt(noise_level); noise_test = torch.randn_like(y_test) * np.sqrt(noise_level)
        y = y * np.sqrt(1-noise_level) + noise; y_test = y_test * np.sqrt(1-noise_level) + noise_test;   
        
    num_useful = 3.; num_feature = x.shape[1]

    return x, y, x_test, y_test

def data_generating_kin8nm(data_path, num_random=0, noise_level=0):
    df = pd.read_csv(data_path + "kin8nm.csv", sep=",")
    np.random.seed(129)
    msk_1 = np.random.rand(len(df)) < 0.8
    traindf = df[msk_1]
    testdf = df[~msk_1]
    # training set
    trainx = traindf.drop('y', axis=1).values
    trainy = traindf['y'].values
    # test set
    testx = testdf.drop('y', axis=1).values
    testy = testdf['y'].values

    scalerx = preprocessing.StandardScaler().fit(trainx)
    trainx = scalerx.transform(trainx)
    testx = scalerx.transform(testx)

    scalery = preprocessing.StandardScaler().fit(trainy.reshape(-1,1))
    trainy = scalery.transform(trainy.reshape(-1,1)).reshape(1, -1)
    testy = scalery.transform(testy.reshape(-1,1)).reshape(1, -1)
    
    # transfer data into tensor
    x = torch.tensor(trainx, dtype = torch.float)
    y = torch.tensor(trainy[0], dtype = torch.float)
    x_test = torch.tensor(testx, dtype = torch.float)
    y_test = torch.tensor(testy[0], dtype = torch.float)
    
    if num_random != 0:
        rand_features = torch.randn(x.shape[0], num_random) * 2
        x = torch.cat((x, rand_features), 1)
        rand_features = torch.randn(x_test.shape[0], num_random) *2
        x_test = torch.cat((x_test, rand_features), 1)
    
    if noise_level != 0:
        noise = torch.randn_like(y) * np.sqrt(noise_level); noise_test = torch.randn_like(y_test) * np.sqrt(noise_level)
        y = y * np.sqrt(1-noise_level) + noise; y_test = y_test * np.sqrt(1-noise_level) + noise_test;  
        
    num_useful = 8.; num_feature = x.shape[1]

    return x, y, x_test, y_test
