import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report

# Get the directory where the current script is located
script_dir = os.path.dirname(__file__)

# Construct the full path to train.csv
file_path = os.path.join(script_dir, 'train.csv')

print(f"Attempting to load from: {file_path}") # This helps verify the path

df = pd.read_csv(file_path)

#getting an overview of the data
'''
print(df.head())
print(df.info())
print(df.describe())
'''
# Survived, Sex and pclass are our categorical values
# survived - 0,1
# Sex - Male, Female
# pclass - 1,2,3
'''
frequency_survived = df["Survived"].value_counts().idxmax()
print (frequency_survived)
'''
#Identifying missing values in the data
#printing missing values
'''
missing_values = df.isnull().sum()
print(missing_values[missing_values>0].sort_values())
'''

#heatmap allows to see correlation between missing values and other features(e.g., most missing values in cabin are for lower class people)
'''
plt.figure(figsize=(8,5))
sns.heatmap(df.isnull(),cbar = False, cmap = 'coolwarm', yticklabels= False)
plt.title('Missing data heatmap')
plt.show()
'''

#Creating bar plots to identify the most frequent values for categorical data

#Not survived has higher frequency
#Male has higher frequency
#lower class has higher frequency
'''
pclass_counts= df['Pclass'].value_counts().sort_index()
sex_counts= df['Sex'].value_counts().sort_index()
survived_counts= df['Survived'].value_counts().sort_index()
embarked_counts = df['Embarked'].value_counts().sort_index()

fig, axes = plt.subplots(nrows=2,ncols=2)
fig.tight_layout(pad=3.0)
axes[0,0].bar(pclass_counts.index,pclass_counts.values)
axes[0,0].set_xlabel('Pclass')
axes[0,0].set_ylabel('frequency')
axes[0,0].set_xticks(pclass_counts.index)
axes[0,0].set_title('Pclass')

axes[0,1].bar(sex_counts.index,sex_counts.values)
axes[0,1].set_xlabel('Sex')
axes[0,1].set_ylabel('frequency')
axes[0,1].set_title('Sex')

axes[1,0].bar(survived_counts.index,survived_counts.values)
axes[1,0].set_xlabel('Survived')
axes[1,0].set_ylabel('frequency')
axes[1,0].set_xticks(survived_counts.index)
axes[1,0].set_title('Survived')

axes[1,1].bar(embarked_counts.index,embarked_counts.values)
axes[1,1].set_xlabel('Embarked')
axes[1,1].set_ylabel('frequency')
axes[1,1].set_title('Embarked')
plt.show()
'''

#Creating box plots to identify outliers and also retrieving the five - number summary for numerical and continuous data

#data is skewed to the right for age and more so for fare. Fare contains numerous outliers
#meaning the older aged people were fewer and a lot of people paid lower fair price
'''
fig, axes = plt.subplots(1,2,figsize=(15,6))
sns.boxplot(df['Age'],ax=axes[0])
axes[0].set_ylabel('Age')
axes[0].set_ylim(-10,100)

sns.boxplot(df['Fare'],ax=axes[1])
axes[1].set_ylabel('Fare')
axes[1].set_ylim(-10,100)
plt.title('minimum,Q1,Q2,Q3,maximum,outliers for age & fare')
plt.show()
'''

#Kernel Density Estimate plot
'''
fig, axes = plt.subplots(1,2,figsize=(15,6))
sns.kdeplot(df['Age'],bw_adjust=0.75,ax=axes[0])
axes[0].set_title('Age')

sns.kdeplot(df['Fare'],ax=axes[1])
axes[1].set_title('Fare')
plt.show()
'''

#Bar plot for finding most frequent values for numerical and distinct data 
# For SibSp, 0 and 1 are most frequent and for Parch, 0 is most frequent
'''
sibsp_sort = df['SibSp'].value_counts().sort_index()
parch_sort = df['Parch'].value_counts().sort_index()

fig, axes = plt.subplots(1,2,figsize=(15,6))
axes[0].bar(sibsp_sort.index,sibsp_sort.values)
axes[0].set_xlabel('no of sibling/spouse')
axes[0].set_ylabel('frequency')
axes[0].set_title('SibSp')

axes[1].bar(parch_sort.index,parch_sort.values)
axes[1].set_xlabel('no of parent/child')
axes[1].set_ylabel('frequency')
axes[1].set_title('Parch')
plt.show()
'''
#Bivariate analysis of categorical data with survival

#survival count according to the gender
'''
survivors = df[df['Survived'].isin([1])]['Sex']
plt.figure()
plt.bar(survivors.unique(),survivors.value_counts())
plt.show()
'''
#Alternative
'''
plt.figure()
plt.title('Survival count / Gender')
sns.countplot(data=df,x=df['Sex'],hue=df['Survived'])
plt.show()
'''
#most of the men did not survive and a lot of the women did

#pclass correlation with survivors
'''
plt.figure()
plt.title('Survival count / Pclass')
sns.countplot(data=df,x=df['Pclass'],hue=df['Survived'])
plt.show()
'''
#lower class people had a lower survival chance

#finding the survival count according to age range
'''
df['AgeRange'] = pd.cut(df['Age'],bins=[0,12,19,100],labels=('0-12','12-19','19-100'))
plt.figure()
plt.title('Survival count / Age Range')
sns.countplot(data=df,x=df['AgeRange'],hue=df['Survived'])
plt.show()
df = df.drop(columns=['AgeRange'])
'''
#the adults were the ones who had a lower survival rate

#Survival count with regard to SibSp
'''
plt.figure()
plt.title('Survival count / SibSp')
sns.countplot(data=df,x=df['SibSp'],hue=df['Survived'])
plt.show()
'''
#considering the concentration of SibSp values at 0 and 1 there doesnt seem to be much correlation

#survival count with regard to Parch
'''
plt.figure()
plt.title('Survival count / Parch')
sns.countplot(data=df,x=df['Parch'],hue=df['Survived'])
plt.show()
'''
#There doesnt seem to be much correlation

#survival count according to the fare
'''
df['FareRange'] = pd.cut(df['Fare'],bins=[0,30,50,70,100,200,600],labels=['very low','low','medium-low','medium','high','very high'])
plt.figure()
plt.title('Survival count / Fare')
sns.countplot(data=df,x=df['FareRange'],hue=df['Survived'])
plt.show()
df = df.drop(columns=['FareRange'])
'''
#people who paid low and very low fairs have lower chances of surviving

#survival count according to port of embarkation
'''
plt.figure()
plt.title('Survival count / Embarked')
sns.countplot(data=df,x=df['Embarked'],hue=df['Survived'])
plt.show()
'''
#people who embarked at soutampton are less likely to survive

titles = df['Name'].str.extract(r'\b(\w+\.)',expand=False)
title_count = titles.value_counts()
rare_titles = title_count[title_count<3].index
simplified_titles = titles.replace(rare_titles,'Rare')

#finding survival count based on titles
'''
plt.figure()
plt.title('Survival count / titles')
sns.countplot(x=simplified_titles,hue=df['Survived'])
plt.show()
'''
#people with Mr. in their name are much less likely to survive

#Cleaning the data
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize']==1).astype(int)

df['Title'] = simplified_titles
df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median())) #group by saparated the groups into different data frames
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0]) #median and mode return a series
df = df.drop(columns=['Name','Cabin','Ticket'])

#Preparing data for model
encoded_df = pd.get_dummies(df,columns=['Sex','Embarked','Title','SibSp','Parch'],drop_first=True) #drop first is necassary to remove redundancy of data for logistic regression
Y = encoded_df['Survived']
X = encoded_df.drop('Survived',axis=1)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

#training the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test,y_pred)
print(f'model accuracy: {accuracy*100:.2f}')
print(classification_report(Y_test,y_pred))

#classification report
#precision - out of all the predictions made for class 0, 84% were correctly predicted
#recall - out of all the actual occurences for class 0, 83% were correctly identified
#f1-score - the harmonic mean for precision and recall of class 0 is 0.83
#support - there were 105 occurences of class 0 in the dataset