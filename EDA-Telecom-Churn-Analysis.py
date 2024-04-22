import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
#%matplotlib inline

#filtering out the warnings
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("C://Users//Aishwarya Dhabadi//OneDrive//Desktop//Task//Telecom_Churn.csv")
data.head()

df = data.copy()
df.head()


df.isnull().sum()

df.duplicated().value_counts()

df.shape
df.info()
plt.figure(figsize=(15,10))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
#ns.heatmap(np.round(df.corr(),2),annot=True, cmap=cmap)
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
sns.heatmap(np.round(correlation_matrix, 2), annot=True, cmap=cmap)
plt.show()

#Observation from headmap
#From the above correlation heatmap, we can see total day charge & total day minute, total evening charge & total evening minute, total night charge & total night minute are positiveliy highly correlated with a value of 1.
#Customer service call is positively correlated only with area code and negative correlated with rest variables.
#Rest all correlation can be depicted from the above chart.

#Data Visualization
#Checking for how many customer leave the company (Churn).
df.head()
df['Churn'].value_counts()

textprops = {"fontsize":15} # Font size of text in pie chart
plt.figure(figsize = (9,9)) # fixing pie chart size
df['Churn'].value_counts().plot(kind = 'pie', autopct = '%1.1f%%', shadow = True, textprops =textprops)
plt.title("Overall Churn Rate", fontsize = 15)
plt.show()

df.head()
df['Area code'].unique()
area_churn = df.groupby(['Area code'])['Churn'].value_counts().unstack()
area_churn

# Visulaizing the area wise churn
area_churn.plot(kind = 'bar', figsize = (10,6), fontsize = 15)
plt.xticks(rotation = 360)
plt.xlabel("Area code", fontsize = 20)
plt.title("Area code wise churn", fontsize = 20)

df['State'].unique()
state_churn = df.groupby(['State'])['Churn'].value_counts().unstack()
state_churn.head()

state_churn.plot(kind = 'bar', figsize = (20,6))
plt.show()

df.head()
int_plan = df['International plan'].value_counts()
int_plan

int_plan.plot(kind = 'pie', figsize = (8,8), fontsize = 15, autopct = '%1.1f%%')
plt.title("International Plan", fontsize = 15)
plt.show()

international_churn = df.groupby(['International plan'])['Churn'].value_counts().unstack()
international_churn

plot2 = international_churn.plot(kind = 'bar', figsize = (10,6), fontsize = 15)
plt.title("Churn on the basis of international plan", fontsize = 15)
plt.xlabel("International plan", fontsize = 15)
plt.xticks(rotation = 360)
plt.show()

voice_churn = df.groupby(['Voice mail plan'])['Churn'].value_counts().unstack()
voice_churn

voice_churn.plot(kind = 'bar', figsize = (10,6), fontsize = 15)
plt.title("Churn on the basis of Voice mail plan", fontsize = 15)
plt.xlabel("Voice mail plan", fontsize = 15)
plt.xticks(rotation = 360)
plt.show()

# Getting those data who has international plan and voice mail plan
voice_and_int = df[df['International plan'] == 'Yes']
voice_and_int = voice_and_int[voice_and_int['Voice mail plan']  == 'Yes']

# Getting those data who has international plan but no voice mail plan
voice_and_int2 = df[df['International plan'] == 'Yes']
voice_and_int2 = voice_and_int2[voice_and_int2['Voice mail plan']  == 'No']

# Getting those data who dont have international plan but have voice mail plan
voice_and_int3 = df[df['International plan'] == 'No']
voice_and_int3 = voice_and_int3[voice_and_int3['Voice mail plan']  == 'Yes']

# Getting those data who dont have international plan and voice mail plan
voice_and_int4 = df[df['International plan'] == 'No']
voice_and_int4 = voice_and_int4[voice_and_int4['Voice mail plan']  == 'No']

plt.figure(figsize = (15,10))

# Checking the churn customers who are having both International and voice mail plan
plt.subplot(2,2,1)
sns.barplot(data = voice_and_int, x = voice_and_int['Churn'].value_counts().keys(), 
           y = voice_and_int['Churn'].value_counts())
plt.title("International plan = Yes and Voice mail plan = Yes", fontsize = 15)

# Checking the churn customers who are having International plan but not the voice mail plan
plt.subplot(2,2,2)
sns.barplot(data = voice_and_int2, x = voice_and_int2['Churn'].value_counts().keys(), 
           y = voice_and_int2['Churn'].value_counts())
plt.title("International plan = Yes and Voice mail plan = No", fontsize = 15)

# Checking the churn customers who are not having International plan but have voice mail plan
plt.subplot(2,2,3)
sns.barplot(data = voice_and_int3, x = voice_and_int3['Churn'].value_counts().keys(), 
           y = voice_and_int3['Churn'].value_counts())
plt.title("International plan = No and Voice mail plan = Yes", fontsize = 15)

# Checking the churn customers who are neither having International plan nor have voice mail plan
plt.subplot(2,2,4)
sns.barplot(data = voice_and_int4, x = voice_and_int4['Churn'].value_counts().keys(), 
           y = voice_and_int4['Churn'].value_counts())
plt.title("International plan = No and Voice mail plan = No", fontsize = 15)
plt.show()

#Column wise Histogram and Box Plot (Univariate)
for col in df.describe().columns:
    fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(18,6))
    sns.histplot(df[col], ax = axes[0],kde = True)
    sns.boxplot(df[col], ax = axes[1],orient='h',showmeans=True,color='orange')
    fig.suptitle("Distribution plot of "+ col, fontsize = 15)
    plt.show()

#State vs Average Top true churn percentage
df.head()
df['Churn'].value_counts()
to_state_churn = ((df.groupby(['State'])['Churn'].mean()*100).sort_values(ascending = False).reset_index().head(10))
to_state_churn

plt.figure(figsize = (10,6))
sns.barplot(data = to_state_churn, x = to_state_churn['State'], y = to_state_churn['Churn'])
plt.title(" State with most churn percentage", fontsize = 20)
plt.show()

#State vs Average bottom true churn percentage
bottom_state_churn = ((df.groupby(['State'])['Churn'].mean()*100).sort_values(ascending = True).reset_index().head(10))
bottom_state_churn

plt.figure(figsize = (10,6))
sns.barplot(data = bottom_state_churn, x = bottom_state_churn['State'], y = bottom_state_churn['Churn'])
plt.title(" State with least churn percentage", fontsize = 20)
plt.show()

df.head()
one_digit = df[df['Account length'] <= 9]
one_digit.head()

one_digit['Churn'].value_counts()
one_digit['Churn'].value_counts().plot(kind = 'pie', figsize = (8,8), autopct = '%1.1f%%', fontsize = 15)
plt.title('One Digit Account Length churn rate', fontsize=18)

two_digit = df[(df['Account length'] > 9) & (df['Account length'] <= 99)]
two_digit.head()
two_digit['Churn'].value_counts()

two_digit['Churn'].value_counts().plot(kind = 'pie', figsize = (8,8), autopct = '%1.1f%%', fontsize = 15)
plt.title('Two Digit Account Length churn rate', fontsize=18)

three_digit = df[(df['Account length'] > 99)]
three_digit.head()
three_digit['Churn'].value_counts()
three_digit['Churn'].value_counts().plot(kind = 'pie', figsize = (8,8), autopct = '%1.1f%%', fontsize = 15)
plt.title('Three Digit Account Length churn rate', fontsize=18)

plt.figure(figsize=(10,8))
sns.boxplot(data = df, x ='Churn', y = 'Account length')
plt.title('Account Length Boxplot with Churn', fontsize=18)

#voice mail
voice_mail_plan = df['Voice mail plan'].value_counts()
voice_mail_plan

voice_mail_plan.plot(kind = 'pie', autopct = '%1.1f%%', figsize = (8,8), fontsize = 15)
plt.title('Distribution of Voice mail plan', fontsize = 15)
plt.show()

voice_churn = ((df.groupby(['Voice mail plan'])['Churn'].mean())*100)
voice_churn
voice_churn.plot(kind = 'bar', figsize = (10,6), fontsize = 15, color = ['b','g'])
plt.xticks(rotation = 360)
plt.title("Percentage of churn on the basis of voice mail plan")
plt.show()

df['Area code'].value_counts()
area_churn = ((df.groupby(['Area code'])['Churn'].mean())*100).reset_index()
area_churn

plt.figure(figsize = (10,6))
sns.barplot(data = area_churn, x = 'Area code', y = 'Churn')
plt.title('Area Code wise Churn percentage', fontsize = 15)
plt.ylabel('Churn Percentage', fontsize = 15)
plt.xlabel("Area Code", fontsize = 15)
plt.show()


overall_calls_day = df.groupby('Churn')[['Total day minutes', 'Total day calls', 'Total day charge']].mean()
overall_calls_day

plt.figure(figsize = (10,6))
overall_calls_day.plot(kind = 'bar', figsize = (10,6))
plt.title('Mean of overall calls in a day', fontsize = 15)
plt.xticks(rotation = 360)
plt.ylabel('Mean', fontsize = 15)
plt.xlabel("Churn", fontsize = 15)
plt.show()

#Overall Calls in evening
df.head()
overall_calls_eve = df.groupby('Churn')[['Total eve minutes', 'Total eve calls', 'Total eve charge']].mean()
overall_calls_eve

plt.figure(figsize = (10,6))
overall_calls_eve.plot(kind = 'bar', figsize = (10,6))
plt.title('Mean of overall calls in the evening', fontsize = 15)
plt.xticks(rotation = 360)
plt.ylabel('Mean', fontsize = 15)
plt.xlabel("Churn", fontsize = 15)
plt.show()

#overall calls in night
overall_calls_night = df.groupby('Churn')[['Total night minutes', 'Total night calls', 'Total night charge']].mean()
overall_calls_night

plt.figure(figsize = (10,6))
overall_calls_night.plot(kind = 'bar', figsize = (10,6))
plt.title('Mean of overall calls in the night', fontsize = 15)
plt.xticks(rotation = 360)
plt.ylabel('Mean', fontsize = 15)
plt.xlabel("Churn", fontsize = 15)
plt.show()

#Average calls of total day calls, evening calls & night calls on basis of churn
df.head()
avg_calls = df.groupby('Churn')[['Total day calls', 'Total eve calls', 'Total night calls']].mean().T
avg_calls

plt.figure(figsize = (10,6))
avg_calls.plot(kind = 'bar', figsize = (10,6))
plt.title('Average calls of total day calls, evening calls & night calls on basis of churn', fontsize = 15)
plt.xticks(rotation = 360)
plt.ylabel('Mean', fontsize = 15)
plt.xlabel("Calls", fontsize = 15)
plt.show()

#Average calls of total day minutes, evening minutes & night minutes on basis of churn
df.head()
avg_minutes = df.groupby('Churn')[['Total day minutes', 'Total eve minutes', 'Total night minutes']].mean().T
avg_minutes

plt.figure(figsize = (10,6))
avg_minutes.plot(kind = 'bar', figsize = (10,6))
plt.title('Average calls of total day minutes, evening minutes & night minutes on basis of churn', fontsize = 15)
plt.xticks(rotation = 360)
plt.ylabel('Mean', fontsize = 15)
plt.xlabel("Minutes", fontsize = 15)
plt.show()

#Average calls of total day Charges, evening Charges & night Charges on basis of churn
df.head()
avg_charges = df.groupby('Churn')[['Total day charge', 'Total eve charge', 'Total night charge']].mean().T
avg_charges

plt.figure(figsize = (10,6))
avg_charges.plot(kind = 'bar', figsize = (10,6))
plt.title('Average calls of total day Charges, evening Charges & night Charges on basis of churn', fontsize = 15)
plt.xticks(rotation = 360)
plt.ylabel('Mean', fontsize = 15)
plt.xlabel("Charges", fontsize = 15)
plt.show()

#Customer Service calls
df.head()
customer_churn = ((df.groupby(['Customer service calls'])['Churn'].mean())*100).reset_index()
customer_churn

plt.figure(figsize = (10,6))
sns.barplot(data = customer_churn, x = customer_churn['Customer service calls'], y = customer_churn['Churn'])
plt.title("Churn rate per service call", fontsize = 20)
plt.xlabel('No of cust service call', fontsize = 15)
plt.ylabel('Percentage', fontsize = 15)
plt.show()