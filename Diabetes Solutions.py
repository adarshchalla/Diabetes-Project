#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r'C:\Users\adarsh\Downloads\diabetes.csv') 
import warnings 
warnings. simplefilter(action='ignore', category=Warning)


# In[2]:


import missingno as msno 
df.head()


# In[3]:


df.columns


# In[4]:


df.info()


# ->All the values are numerical quantities.It would be further used for predicting if a person has  Diabetes or Not.
# ->There seemed to be many zeros in the Parameters where there shouldn't be zeros. In all the parameters except outcome , there shouldn't be any zero. Converting all the cells of zeros to np.NaN or None values  other than outcome column.

# In[5]:


import numpy as np
c=[ 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']
for i in c :
    for j in range(len(df[i])):
         if df[i][j] ==0:
                df[i][j] = np.NaN
#conversion of all zeros in input columns to None


# In[6]:


msno.matrix(df)


# There seemed to be many Null values in skin thickness and insulin columns. followed by blood pressure. Let's check if we can check whether the Null values of skin thickness and insulin are common for same rows.

# In[7]:


sns.boxplot(df['Insulin'])


# Checking for outliers in the Insulin column.

# In[8]:


df.isnull().sum()


# Finding the number of NaN values in each column.

# In[9]:


(df.isnull().sum()/ len(df) ) * 100


# Finding the % percentage of Null values in each column . ( dividing the above cell values with total no of columns)

# In[10]:


df[(df['Insulin'].isna()) & (df['SkinThickness'].isna())].isnull().sum()


# In the above cell the filtering method is used to find if the commonnalities are present in columns of Skinthickness and Insulin. From the above filter and the msno cell, we found that if there a Null value in Insulin , there is also a Null value in Skinthickenss.

# In[11]:


df.describe()


# Five point summary for the dataframe before filling null values with mean or median.

# In[12]:


df.hist(figsize=(20,20))
plt.show()


# In[13]:


df['Glucose'].fillna(df['Glucose'].mean(), inplace = True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace = True)
df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace = True)
df['Insulin'].fillna(df['Insulin'].median(), inplace = True)
df['BMI'].fillna(df['BMI'].median(), inplace = True)
#imputing null values with median and mode.


# In[14]:


df.isna().sum()


# In[15]:


df.hist(figsize=(20,20))
plt.show()


# Histogram after imputing Null values with mean or median.

# In[16]:


# drop_A=df.index[df["Pregnancies"] == 0].tolist()
# drop_B = df.index[df["Glucose"] == 0].tolist()
# drop_C = df.index[df["BloodPressure"] == 0].tolist()
# drop_D = df.index[df['SkinThickness'] == 0].tolist()
# drop_E = df.index[df['Insulin'] == 0].tolist()
# drop_F = df.index[df["BMI"] == 0].tolist()
# drop_G = df.index[df["DiabetesPedigreeFunction"] == 0].tolist()
# drop_H = df.index[df["Age"] == 0].tolist()
# c = drop_A+drop_B + drop_C + drop_D + drop_E + drop_F + drop_G + drop_H 
# df = df.drop(df.index[c])


# the above code was used to find the number of rows remained after removing all the null values . The number was very low. So , the values were kept.

# In[17]:


df.describe()


# Five point statistics to understand the change after imputation.

# In[18]:


df.info()


# In[19]:


df['Outcome'].value_counts().plot(kind='bar')


# here most of the values are with outcome 0.

# In[20]:


import seaborn as sns
sns.set_theme(style="whitegrid")

for i in df.columns[0:-1]:
    plt.figure()
    sns.boxenplot(x=i, color="b",scale="linear", data=df)


# ->Most of the pregnancies are in the range 1 to 5 . However there are outliers which are more than 12.5
# ->The maximum glucose content is between 100 and 140.
# -> the outliers are found out with the help of box plot.
# 

# In[21]:


import seaborn as sns
sns.set_theme(style="ticks")

sns.pairplot(df, hue= 'Outcome')


# In[22]:


df.corr()['Outcome'].sort_values()


# Here the correlation was found out with all the other columns. The glucose column was highly correlated. Here we can observe that all the columns are positively correlated. 
# -> the glucose and bmi columns are highly correlated follwed by age ,pregnancies and skinthickness.

# In[23]:


sns.heatmap(df.corr())


# -> top three factors affecting the outcome are 1) Glucose 2) BMI 3) Age
# -> least significant factors are 1) Bloodpressure 2) Diabetes Pedigree Function 3) Insulin

# In[24]:


df['Pregnancies'].value_counts().plot(kind='bar')


# Most of the pregnancies are in the range 0 to 2.

# In[25]:


sns.displot(
    data=df, kind="hist",
    x="Age", y="BMI", hue="Outcome",
     palette="Set2", alpha=.6, height=6)

sns.set()


# -> people with bmi in range around 20 , tend to not have diabetes in all age groups.

# In[26]:


import seaborn as sns

sns.set_theme(style="darkgrid")
sns.set()
sns.displot(
    df, x="Age", hue="Outcome",
     facet_kws=dict(margin_titles=True),rug=True
)
sns.set()


# -> combining both bar graphs of outcomes 0 and 1 

# In[27]:


sns.displot(
    df, x="Pregnancies",
     facet_kws=dict(margin_titles=True),hue='Outcome',rug=True
)
sns.set()


# -> people with more pregnancies tend to have diabetes.

# In[28]:


df.groupby("Outcome").agg(['mean','median'])


# In the above cell we can observe that all the parameters for 1 outcome are more(mean). For example, people with more DiabetesPedigreeFunction column tend to have more value for 1 outcome. People with 0 outcome have less diabetes pedigree function. The same applies for all the columns.

# In[29]:


pd.pivot_table(df, index =['Outcome'], aggfunc={'Insulin': np.mean,'BMI': np.mean})


# The mean of BMI and Insulin are more for outcome 1 of diabetes.

# In[30]:


import seaborn as sns
sns.set_theme(style="whitegrid")

# Load the example tips dataset


# Draw a nested violinplot and split the violins for easier comparison
sns.violinplot(data=df, x="Outcome", y="Insulin",
               split=True, inner="quart", linewidth=1)
sns.despine(left=True)


# In[31]:


for i in df.columns[:-1]:
    sns.swarmplot(data=df, x="Outcome", y=i,
                   split=True, linewidth=1)
    sns.despine(left=True)
    plt.show()


# In[32]:


for i in df.columns[:-1]:
    sns.violinplot(data=df, x="Outcome", y=i,
                   split=True, inner="quart", linewidth=1)
    sns.despine(left=True)
    plt.show()


# In[33]:


df.groupby("Outcome").agg(['mean','median'])


# Most useful insights.
# -> People with more BMI , Glucose , Pregnancies , etc tend to have diabetes. 
# -> The top three factors affecting the outcome are 1) Glucose 2) BMI 3) Age
# -> people with more pregnancies are more susceptible to diabetes.
# -> from the above table we can say that all the above columns if they are in higher level, they are positively correlated to the presence of diabetes.

# In[34]:


#Prediction of Outcome


# In[35]:


df.Outcome


# In[36]:


#Splitting the data into dependent and independent variables
Y = df["Outcome"]
x = df.drop(["Outcome"], axis = 1)
columns = x.columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(x)
data_x = pd.DataFrame(X, columns = columns)


# In[37]:


#Splitting the data into training and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_x, Y, test_size = 0.15, random_state = 45)


# In[38]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))


# In[39]:


from sklearn.metrics import accuracy_score, precision_score, recall_score
print("The precision score is :" ,precision_score(y_test, y_pred, average="macro"))
print("The recall score is :",recall_score(y_test, y_pred, average="macro"))
print("The accuracy score is :",accuracy_score(y_test, y_pred))


# In[40]:


from sklearn.svm import SVC
classifier_rbf = SVC(kernel = 'rbf')
classifier_rbf.fit(x_train, y_train)
y_pred = classifier_rbf.predict(x_test)
print('Accuracy of SVC (RBF) classifier on test set: {:.2f}'.format(classifier_rbf.score(x_test, y_test)))

print("The precision score is :" ,precision_score(y_test, y_pred, average="macro"))
print("The recall score is :",recall_score(y_test, y_pred, average="macro"))
print("The accuracy score is :",accuracy_score(y_test, y_pred))


# In[41]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=300, bootstrap = True, max_features = 'sqrt')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('Accuracy of Random Forest on test set: {:.2f}'.format(model.score(x_test, y_test)))
print("The precision score is :" ,precision_score(y_test, y_pred, average="macro"))
print("The recall score is :",recall_score(y_test, y_pred, average="macro"))
print("The accuracy score is :",accuracy_score(y_test, y_pred))


# We thus select the Random Forest Classifier as the right model due to high accuracy, precision and recall score. One reason why Random Forest Classifier showed an improved performance was because of the presence of outliers. As mentioned before, since Random Forest is not a a distance based algorithm, it is not much affected by outliers, whereas distance based algorithm such as Logistic Regression and Support Vector showed a lower performance.
