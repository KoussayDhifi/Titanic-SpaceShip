import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model,preprocessing
from sklearn.neighbors import KNeighborsClassifier
import pickle as pk
import matplotlib.pyplot as plt


data = pd.read_csv('spaceship-titanic/train.csv')

le = preprocessing.LabelEncoder()

hp = le.fit_transform(list(data['HomePlanet']))
name = le.fit_transform(list(data['Name']))
cs = le.fit_transform(list(data['CryoSleep']))
cab = le.fit_transform(list(data['Cabin']))
dest = le.fit_transform(list(data['Destination']))
age = le.fit_transform(list(data['Age']))
vip = le.fit_transform(list(data['VIP']))
Rs = le.fit_transform(list(data['RoomService']))
fc = le.fit_transform(list(data['FoodCourt']))
sm = le.fit_transform(list(data['ShoppingMall']))
vrdeck = le.fit_transform(list(data['VRDeck']))
spa = le.fit_transform(list(data['Spa']))
trans = le.fit_transform(list(data['Transported']))
accM = 0
for i in range(20):
    X = list(zip(hp,cs,cab,dest,age,vip,Rs,fc,sm,spa,vrdeck,name))
    Y = list(trans)
    for j in range(700):
        model = KNeighborsClassifier(n_neighbors=i+1)

        x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.1)
        model.fit(x_train,y_train)
        acc = model.score(x_test,y_test)
        if (acc>accM):
            print(acc)
            f = open("model.pk","wb")
            pk.dump(model,f)
            accM = acc
