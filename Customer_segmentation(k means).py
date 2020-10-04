import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("c:/Users/Tejas/Downloads/Mall_Customers.csv")
#print(df)

df.drop(["CustomerID"], axis = 1, inplace=True)
#print(df)

plt.figure()
plt.title("Age")
sns.violinplot(y=df["Age"])
#plt.show()


sns.displot(df["Age"],bins=5)
#plt.show()

plt.figure()
plt.title("Spending Score")
sns.boxplot(y=df["Spending Score (1-100)"])
#plt.show()

sns.displot(df["Spending Score (1-100)"],bins=5)

plt.figure()
plt.title("Annual Income")
sns.boxplot(y=df["Annual Income (k$)"])

sns.displot(df["Annual Income (k$)"],bins=5)

genders = df.Gender.value_counts()
plt.figure()
sns.barplot(x=genders.index, y=genders.values)
plt.show()

#print(df.iloc[:,1:])

wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(df.iloc[:,1:])
    wcss.append(kmeans.inertia_)
plt.figure()    
plt.grid()
plt.plot(range(1,11),wcss, marker ="8")
plt.xlabel("K Value")
plt.xticks(np.arange(1,11,1))
plt.ylabel("WCSS")
plt.show()

kmeans_5 = KMeans(n_clusters=5)
clusters = kmeans_5.fit_predict(df.iloc[:,1:])
df["label"] = clusters
#x=[[27,89,35]]
#print(kmeans_5.predict(x))
 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.Age[df.label == 0], df["Annual Income (k$)"][df.label == 0], df["Spending Score (1-100)"][df.label == 0], c='blue', s=60)
ax.scatter(df.Age[df.label == 1], df["Annual Income (k$)"][df.label == 1], df["Spending Score (1-100)"][df.label == 1], c='red', s=60)
ax.scatter(df.Age[df.label == 2], df["Annual Income (k$)"][df.label == 2], df["Spending Score (1-100)"][df.label == 2], c='green', s=60)
ax.scatter(df.Age[df.label == 3], df["Annual Income (k$)"][df.label == 3], df["Spending Score (1-100)"][df.label == 3], c='orange', s=60)
ax.scatter(df.Age[df.label == 4], df["Annual Income (k$)"][df.label == 4], df["Spending Score (1-100)"][df.label == 4], c='purple', s=60)
ax.view_init(30, 185)
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel('Spending Score (1-100)')
plt.show()