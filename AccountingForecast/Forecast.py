import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
# import xgboost as xgb
import random
datelist = pd.date_range(pd.datetime.today(), periods=100).tolist()


def timestamp(dt,minYear):#date in %y-%m-%d format
    dt = dt.split("-")
    return (int(dt[0])-minYear)*365+int(dt[1])*30+int(dt[2])

def get_datetime(dt): #date in %y-%m-%d format
    dt = dt.split("-")
    return datetime.datetime(int(dt[0]),int(dt[1]),int(dt[2]))

def get_weekly_forecast(nWeeks):
    df = pd.read_csv("input.csv")
    minYear = int(df.DATE.values[0].split("-")[0])
    df["TIMESTAMP"] = df.apply(lambda row: timestamp(row.DATE,minYear),axis=1)
    df = get_weekly_data(df)
    lastWeek = df.WEEK.values[-1]
    lastTimestamp = df.TIMESTAMP.values[-1]
    xtrain = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    xtest = get_weekly_test_data(lastWeek,lastTimestamp,nWeeks)
    clf,dtest = get_model(xtrain,xtest,y)
    forecast = clf.predict(dtest)
    return forecast

def get_monthly_forecast(nMonths):
    df = pd.read_csv("input.csv")
    minYear = int(df.DATE.values[0].split("-")[0])
    df["TIMESTAMP"] = df.apply(lambda row: timestamp(row.DATE,minYear),axis=1)
    df = get_monthly_data(df)
    lastWeek = df.MONTH.values[-1]
    lastTimestamp = df.TIMESTAMP.values[-1]
    xtrain = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    xtest = get_monthly_test_data(lastWeek,lastTimestamp,nMonths)
    clf,dtest = get_model(xtrain,xtest,y)
    forecast = clf.predict(dtest)
    return forecast

def get_weekly_test_data(lastWeek,lastTimestamp,nWeeks):
    columns = ["WEEK","TIMESTAMP"]
    data = []
    for i in range(nWeeks):
        data.append([lastWeek+i+1, lastTimestamp+7*(i+1)])
    return pd.DataFrame(data,columns = columns)

def get_monthly_test_data(lastMonth,lastTimestamp,nMonths):
    columns = ["MONTH","TIMESTAMP"]
    data = []
    for i in range(nMonths):
        data.append([lastMonth+i+1, lastTimestamp+30*(i+1)])
    return pd.DataFrame(data,columns = columns)

def get_weekly_data(dailyData):
    week = 0
    prevDay = None
    amount = 0
    columns = ["WEEK","TIMESTAMP","AMOUNT"]
    data  = []
    for i,row in dailyData.iterrows():
        if(prevDay==None):
            prevDay = get_datetime(row.DATE)
        currDay = get_datetime(row.DATE)
        if((currDay-prevDay).days<7):
            amount+=row.AMOUNT
            timestamp = row.TIMESTAMP
        else:
            data.append([week,timestamp,amount])
            amount = row.AMOUNT
            prevDay = currDay
            week+=1
    return pd.DataFrame(data,columns = columns)

def get_monthly_data(dailyData):
    month = 0
    prevDay = None
    amount = 0
    columns = ["MONTH","TIMESTAMP","AMOUNT"]
    data  = []
    for i,row in dailyData.iterrows():
        if(prevDay==None):
            prevDay = get_datetime(row.DATE)
        currDay = get_datetime(row.DATE)
        if((currDay-prevDay).days<31):
            amount+=row.AMOUNT
            timestamp = row.TIMESTAMP
        else:
            data.append([month,timestamp,amount])
            amount = row.AMOUNT
            prevDay = currDay
            month+=1
    return pd.DataFrame(data,columns = columns)

def get_data(npoints):
    columns = ["DATE","CUSTOMER","ACCOUNT","AMOUNT","ACTION"]
    #generate data for only one customer
    data = []
    customer = 1
    account = 1
    action = "paid"
    timeseries = get_time_series(npoints)
    for i in range(npoints):
        data.append([timeseries[0][i],customer,account,timeseries[1][i],action])
    return pd.DataFrame(data,columns = columns)

def get_time_series(npoints):
    num_payments_in_a_day = 5
    base = datetime.datetime.today()
    datelist = [base - datetime.timedelta(days=x) for x in range(0, npoints//num_payments_in_a_day)]
    datelist = [x.strftime("%Y-%m-%d") for x in datelist]
    datelist = sorted(datelist)
    datelist_multiplied = []
    for d in datelist:
        for i in range(num_payments_in_a_day):
            datelist_multiplied.append(d)
    data = []
    for i in range(npoints):
        data.append(500+i*5)
#     print(len(datelist_multiplied)==len(data))
    return (datelist_multiplied,data)

def get_model(Xtrain,Xtest,ytrain):
    model = LinearRegression()
#     model = RandomForestRegressor(n_estimators=10)
#     model = KNeighborsRegressor(n_neighbors=5)
#     model = SVR()
    scaler = StandardScaler()
    scaler.fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    model.fit(Xtrain,ytrain)
    return (model,Xtest)

# def get_xgb_model(Xtrain,Xtest,ytrain):
#     dtrain = xgb.DMatrix(np.array(Xtrain), label=ytrain)
#     dtest = xgb.DMatrix(np.array(Xtest))
#     xgb_params = {
#         'objective':'reg:linear',
#         'booster': 'gbtree',
#         'eval_metric': 'auc',
#         'eta': 0.02,
#         'max_depth': 8,
#         'lambda': 4,
#         'alpha': 0.02,
#         'subsample': 0.8,
#         'colsample_bytree': 0.8,
#         'min_child_weight':4,
#         'silent': 1
#     }
#     num_round=100
#     gbdt = xgb.train(xgb_params, dtrain,num_round)
# #     get_xgb_feat_importances(gbdt)
#     return (gbdt,dtest)

def save_results(datalist,filename):
    datalist = list(datalist)
    datalist = list(map(str,datalist))
    with open(filename,"wb") as fp:
        fp.write("\n".join(datalist).encode())

if __name__ == "__main__":
    save_results(get_monthly_forecast(5),"monthly_forecast.txt")
