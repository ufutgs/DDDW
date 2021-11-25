import math
import os
import numpy as np
import pandas as pd

class Lregress():
  def __init__(self):
    self.y = None # record the result 
    self.x = None # record the input
    self.beta = None # beta ndarray
    self.train_x = None # x train ndarray
    self.train_y = None # y train ndarray
    self.cost = None # cost array , can plot to see the graph
    self.is_parameter_empty = True # status for parameter excluding slef.beta , if empty then 0 , else 1.
    self.is_trained = False # check if model trained or not
    self.pred = None
    
  def prepare_feature(self,df_feature):
    x=df_feature
    x = np.concatenate((np.full((x.shape[0],1),1),x),axis=1)
    return x

  def concen(self,a): # add column of 1 in the front of ndarray
    return np.concatenate((np.full((a.shape[0],1),1),a),axis=1)

  def concat_and_numpy(self,a):
        return pd.concat(a[0]).sample(frac=1,random_state=100).to_numpy(),pd.concat(a[1]).to_numpy(),pd.concat(a[2]).sample(frac=1,random_state=100).to_numpy(),pd.concat(a[3]).to_numpy()
        
  def init_MLR(self,x,y,array): # validate data, split data , store data
    if(isinstance(x,pd.DataFrame) and isinstance(y,pd.DataFrame)):
      if(x.shape[0]==y.shape[0]):
        if "location" in x.columns: # if we have location, we split to split_data each country data for equality
            self.y = y.drop("location",axis=1)
            self.x = x.drop("location",axis=1)
            self.train_x = self.x.to_numpy()
            self.train_y = self.y.to_numpy()
        else:
            self.train_x,self.test_x,self.train_y,self.test_y = self.split_data(x,y,0.3,300)
            self.y = y
            self.x = x
            
        self.beta = np.zeros((self.train_x.shape[1]+1,1))
        print("MLR Initialization completed .")
        self.is_parameter_empty = False
        return



  def split_data(self,X,Y,test_size,random_state=None,to_numpy = True):
      x_test = X.sample(n = math.floor(X.shape[0]*test_size) , random_state = random_state)
      y_test = Y.sample(n = math.floor(Y.shape[0]*test_size) , random_state = random_state)
      x_train = X.copy()
      y_train = Y.copy()
      for index in x_test.index:
          x_train=x_train.drop(index=index)
      for index in y_test.index:
          y_train=y_train.drop(index=index) 
      if to_numpy:
          return x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()
      return [x_train, x_test, y_train, y_test]
  
  def regress(self,alpha, num_iters): # regression part
    if(not self.is_parameter_empty and isinstance(alpha,float) and isinstance(num_iters,int)):
      self.beta, self.cost = self.gradient_descent(self.prepare_feature(self.normalize_z(self.train_x)),self.train_y,self.beta,alpha,num_iters)
      return
    return None
  
  def normalize_z(self,df): # min max scalling
    mini = df.min(axis=0)
    maxi = df.max(axis=0)
    shape = (df.shape[0],1)
    a = df.copy()
    for i in range(df.shape[1]):
        a[:,i]=((a[:,i].reshape(shape)-np.full(shape,mini[i]))/(maxi[i]-mini[i]))[:,0]
    np.nan_to_num(a,copy=False) # prevent 0/a situation
    return a

  def predict_norm(self,X, beta):
    return np.matmul(X,beta)
    pass

  def predict(self,df):
    if(not self.is_parameter_empty and self.is_trained):
      self.pred =  self.predict_norm(self.prepare_feature(df),self.beta) # normalize data, add column of 1 to data,
    return

  def gradient_descent(self,X, y, beta, alpha, num_iters): # the real regression part
      m = y.shape[0]
      J_storage=[]
      for i in range(num_iters):
          a = np.matmul(X,beta)-y
          J_storage.append(np.square(a).sum() /(2*m))
          beta[0] = beta[0] - (a).sum() *alpha/m
          for i in range(1,X.shape[1]):
              beta[i] = beta[i] - (X[:,i].reshape(m,1)*a).sum()*alpha/m
      self.is_trained = True
      return beta, J_storage



def take_data(data,parameter,countries):
    test_data = data.loc[data["location"].isin(countries)]
    for i in countries:
        test_data[test_data["location"]==i] = test_data[test_data["location"]==i].fillna(method="ffill")
    test_data = test_data.fillna(0)
    test_para = test_data[parameter]
    #test_para.loc[:,test_para.columns != "location"] = test_para.loc[:,test_para.columns != "location"].multiply(test_data["total_cases"], axis="index")
    test_death = test_data[["location","total_vaccinations"]]
    test_death["total_vaccinations"]=test_death["total_vaccinations"]/test_data["population"]
    return test_data,test_para,test_death    

def take_country_data(country,test_para,test_death,numpy = False):
    if numpy:
        return test_para.loc[test_para["location"]==country].drop("location",axis=1).to_numpy() ,test_death.loc[test_death["location"]==country].drop("location",axis=1).to_numpy()
    return test_para.loc[test_para["location"]==country].drop("location",axis=1),test_death.loc[test_death["location"]==country].drop("location",axis=1)

def start():
  data = pd.read_csv(os.path.join(os.path.abspath(os.path.dirname(__file__)),"covid.csv"))
  parameter = ["location","gdp_per_capita","human_development_index","total_deaths","aged_65_older"]
  countries =   data["location"].unique()
  test_data,test_para,test_death = take_data(data,parameter,countries)
  model = Lregress()
  model.init_MLR(test_para,test_death,countries)
  model.regress(0.02,10000)
  return list(model.beta.flatten()) , list(model.train_x.max(axis=0)),list(model.train_x.min(axis=0))