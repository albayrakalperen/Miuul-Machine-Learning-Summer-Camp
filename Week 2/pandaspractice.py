import seaborn as sns
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

# 1
titanic_df = sns.load_dataset("titanic")

# 2
male_count = titanic_df['sex'].value_counts()['male']
print(male_count)
female_count = titanic_df['sex'].value_counts()['female']
print(female_count)

# 3
print(titanic_df.nunique())

# 4
print(titanic_df['pclass'].nunique())

# 5
print(titanic_df[['pclass', 'parch']].nunique())

# 6
print(titanic_df['embarked'].dtype)
titanic_df['embarked'] = titanic_df['embarked'].astype('category')
print(titanic_df['embarked'].dtype)

# 7
print(titanic_df[titanic_df['embarked'] == 'C'])

# 8
print(titanic_df[titanic_df['embarked'] != 'S'])

# 9
print(titanic_df.query('(age < 30) & (sex == "female")'))

# 10
print(titanic_df.query('(fare > 500) | (age > 70)'))

# 11
print(titanic_df.isnull().sum())

# 12
titanic_df.drop(columns='who', inplace=True)

# 13
deck_mode = titanic_df['deck'].mode()[0]
titanic_df['deck'].fillna(deck_mode, inplace=True)

# 14
age_median = titanic_df['age'].median()
titanic_df['age'].fillna(age_median, inplace=True)

# 15
sub_df = titanic_df[['survived', 'pclass', 'sex']]
sub_df.groupby(['pclass', 'sex']).survived.sum()
sub_df.groupby(['pclass', 'sex']).survived.count()
sub_df.groupby(['pclass', 'sex']).survived.mean()

# 16
age_flag_series = titanic_df['age'].apply(lambda x: 1 if (x < 30) else 0)
titanic_df['age_flag'] = age_flag_series

# 17
tips_df = sns.load_dataset("tips")

# 18
tips_df.groupby('time').agg({'total_bill': ['sum', 'min', 'max', 'mean']})

# 19
tips_df.groupby(['day', 'time']).total_bill.sum()
tips_df.groupby(['day', 'time']).total_bill.min()
tips_df.groupby(['day', 'time']).total_bill.max()
tips_df.groupby(['day', 'time']).total_bill.mean()

# 20
lunch_female_df = tips_df.query('time == "Lunch" & sex == "Female"')
lunch_female_df[['day', 'total_bill', 'tip']].groupby('day').sum()
lunch_female_df[['day', 'total_bill', 'tip']].groupby('day').min()
lunch_female_df[['day', 'total_bill', 'tip']].groupby('day').max()
lunch_female_df[['day', 'total_bill', 'tip']].groupby('day').mean()

# 21
tips_df.query('size < 3 & total_bill > 10').loc[:, 'total_bill'].mean()

# 22
tips_df["total_bill_tip_sum"] = tips_df['total_bill'] + tips_df['tip']

# 23
mean_by_sex = tips_df.groupby('sex')['total_bill'].mean()
male_mean = mean_by_sex[0]
female_mean = mean_by_sex[1]
tips_df['total_bill_flag'] = np.where((((tips_df['sex'] == 'Male') & (tips_df['total_bill'] > male_mean)) | ((tips_df['sex'] == 'Female') & (tips_df['total_bill'] > female_mean))), 1, 0)

# 24
tips_df.groupby(['sex', 'total_bill_flag'])['total_bill'].count()

# 25
first_30_df = tips_df.sort_values('total_bill_tip_sum', ascending=False).head(30)

