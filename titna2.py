import pandas as pd
import sklearn
from sklearn import linear_model,preprocessing
from sklearn.neighbors import KNeighborsClassifier
import pickle as pk

data = pd.read_csv('spaceship-titanic/test.csv')
le = preprocessing.LabelEncoder()
n_ins = list(data['PassengerId'])
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

X = list(zip(hp,cs,cab,dest,age,vip,Rs,fc,sm,spa,vrdeck,name))
f = open("model.pk","rb")
model = pk.load(f)
answer = ["False","True"]
PREDICT = model.predict(X)
f2 = open("submit.csv","wt")
for i in range(len(PREDICT)):
    f2.write(n_ins[i]+","+answer[PREDICT[i]]+"\n")

f2.close()