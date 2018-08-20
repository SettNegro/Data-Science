from sklearn import tree
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#[height, weight, shoe size]
X = [[181,80,44], [177,70,43], [160,60,38], [154,54,37],[166,65,40], [175,64,39], [177,70,40],[190,90,47],[175,64,39]]

Y = ['male', 'female', 'female', 'female', 'female', 'male', 'male', 'male', 'male']

clf = tree.DecisionTreeClassifier()#initialization of the decision tree by calling the decision tree classifier method and clf is now a tree variable

clf = clf.fit(X,Y) #train the classifier with fit method 

prediction = clf.predict([[156,45,34], [170,42,38]])
print(prediction)



df = pd.DataFrame(dict(height=[item[0] for item in X],weight=[item[1] for item in X],shoes=[item[2] for item in X], label=Y))
fig = plt.figure()
colors={'male':'red', 'female':'orange'}
ax = fig.add_subplot(111, projection= '3d')
ax.scatter(df['height'],df['weight'],df['shoes'], edgecolors=(1, 1, 1), c=df['label'].apply(lambda x: colors[x]), s=2000)




plt.show()
ax.figure.savefig('data.png')
