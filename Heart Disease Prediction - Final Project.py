#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore") 


# In[2]:


# Loading the dataset
data= pd.read_csv("heart_disease.csv")


# In[3]:


# displaying the top 5 rows of the data
data.head()


# In[4]:


# getting the shape of the data
data.shape


# In[5]:


# Checking for null values
data.isnull().sum()


# In[6]:


# dataset info
data.info()


# In[7]:


# descriptive summary of data
data.describe()


# In[8]:


# Getting the column names
data.columns


# # Exploratory Data Analysis

# In[9]:


# Let us figure out gender-wise distribution of heart disease data among participants 

sns.catplot(x="Sex", kind="count",hue='HeartDisease', data=data)


# ##### form the graph it can be inferred the count of heart diseases is more for males than females

# In[10]:


# Heart Disease in different age groups

plt.figure(figsize=(15,10))
sns.histplot(x='AgeCategory', data=data, hue='HeartDisease', multiple='stack',palette='YlGn')
plt.show()


# ##### Heart disease is higher in people above the age of 60 years

# In[11]:


plt.figure(figsize=(15,10))
sns.histplot(x='Race', data=data, hue='HeartDisease', multiple='stack', palette='RdPu')
plt.show()


# ##### heart disease is more in Whites

# In[12]:


# analysing categorical columns to know which features have impact on heartdisease
x_axis=['SkinCancer', 'KidneyDisease', 'PhysicalActivity', 'AlcoholDrinking', 'Smoking', 'Stroke', 'Asthma', 'Diabetic', 'DiffWalking']

fig=plt.figure(figsize=(20,10))

for i in range(0,len(x_axis)):
    sub_plot=fig.add_subplot(3,3,i+1)
    plot_map=sns.countplot(data=data,x=x_axis[i],hue='HeartDisease',palette='Accent')


# ##### From all the graphs presented, it can be concluded that alcohol consumption and smoking are not the main factors in heart disease, as scientists testify to this.

# # Data Pre-Processing

# In[13]:


# Machine Learning Readiness of data
# dataset does have many categorical variables. Therefore, we will rearrange the columns.

df=data[['Race', 'AgeCategory', 'GenHealth','Sex','Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Diabetic', 'PhysicalActivity','Asthma', 'KidneyDisease', 'SkinCancer','BMI', 'SleepTime', 'PhysicalHealth', 'MentalHealth', 'HeartDisease']]


# In[14]:


#Let us understand more about different columns using unique values

for i in df.columns:
    print(f'Unique Values of {i.title()}: {df[i].unique()}')


# In[15]:


# Replacing the value "No, borderline diabetes", as "Yes"
data=data.replace(to_replace ="No, borderline diabetes", value ="Yes")
# Replacing the value "Yes (during pregnancy)", as "No"
data=data.replace(to_replace ="Yes (during pregnancy)", value ="No")


# In[16]:


# converting the values of specific categorical variables in the df DataFrame into binary numerical representations (0 and 1)
categorical_var=['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Diabetic', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer', 'HeartDisease']

for i in categorical_var:
    df[i]=data[i].apply(lambda x: 0 if x=='No'else 1).astype('int64')


# In[17]:


# using lambda function to convert Female as 1 and male as 0 in sex column
df['Sex']=df['Sex'].apply(lambda x: 0 if x=='Female'else 1).astype('int64')


# In[18]:


# Extract features (X) by dropping the target column
X=df.iloc[:,:-1].values
# Extract features (Y) by dropping the rest except for the target column
Y=df.iloc[:,-1:].values


# In[19]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# using OneHotEncoder to convert categorical values into binary matirx of numerical values for 'Race', 'AgeCategory' and 'GenHealth' columns which are at index 0,1,2 in the df

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,1,2])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[20]:


# Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state=42) 


# In[21]:


# Scaling the data
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train[:,-4:]= sc.fit_transform(X_train[:, -4:])
X_test[:,-4:]= sc.transform(X_test[:, -4:])


# In[22]:


# Now we will implement model pipeline guidelines and replicate above models and learn about best model
# amongst these classifiers

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

model_pipeline=[]
model_pipeline.append(DecisionTreeClassifier(random_state=41))
model_pipeline.append(LogisticRegression(solver='saga',random_state=42))
model_pipeline.append(MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=100))  # MLP (Neural Network)


# In[27]:


model_list=['Decision Tree', 'Logistic Regression','MLP Classifier']
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve
from sklearn import metrics

acc=[]
auc=[]


for name, classifier in zip(model_list, model_pipeline):
    print(f"\nModel: {name}")
    
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    acc.append(accuracy_score(Y_test, Y_pred))
    fpr,tpr,_thresholds=roc_curve(Y_test,Y_pred)
    auc.append(round(metrics.auc(fpr,tpr),2))
    conf_matrix = confusion_matrix(Y_test, Y_pred)
   
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Classification Report
    print("Classification Report:")
    print(classification_report(Y_test, Y_pred))


# In[28]:


# Accuracy and AUC scores
result=pd.DataFrame({'Model': model_list, 'Accuracy': acc, 'AUC': auc})
result


# In[29]:


# Get the trained Logistic Regression model from the pipeline
log_reg_model = model_pipeline[1]  # Logistic Regression is at index 1

# Use the Logistic Regression model to make predictions on the test set
log_reg_pred = log_reg_model.predict(X_test)

# Print the predictions
print("Predictions using Logistic Regression:")
print(log_reg_pred)


# In[30]:


plt.figure(figsize=(10,6))

# Plotting the bar chart for Accuracy
plt.bar(result['Model'], result['Accuracy'], color='skyblue', )
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha="right")
plt.ylim(0.85, 1)  
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()


# # Best model is Logistic Regression with 91.39% accuracy

# In[ ]:




