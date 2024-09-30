#import basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load the dataset
df = pd.read_csv('C:/Users/Lenovo/Downloads/250k Medicines Usage, Side Effects and Substitutes.csv')

#display the first few rows
print(df.head())

# knowing all the columns
print(df.columns)
print(df.shape)

#display basic statistics
print(df.describe())

#display information about the dataset
print(df.info())

#missing value analysis
missing_vals = df.isnull().sum() / len(df)
print(missing_vals)


missing_more_15 = missing_vals[missing_vals > 0.15]
print(missing_more_15)


print(list(missing_more_15.index))

print(missing_vals[missing_vals <= 0.15])

#removing duplicate rows
df1 = df.drop(list(missing_more_15.index) , axis = 'columns')
print(df1.head())

df1 = df1.drop_duplicates()

#Data PreProcessing
#data cleaning and normalization
df1['name'] = df1['name'].str.lower()
df1['name'] = df1['name'].str.strip()

print(df1['name'].value_counts())

print(df1.isnull().sum())

df1['use0'] = df1['use0'].str.lower()
df1['use0'] = df1['use0'].str.strip()

print(df1['use0'].value_counts())

#Identifying the Top 15 Treatments
top_treatments = list(df1['use0'].value_counts().head(15).index)
print(top_treatments)
#Summing Occurrences of the Top 15 Treatments
print(df1['use0'].value_counts().head(15).sum())
#Calculating the Remaining Non-Top Treatment Count
print(df1.shape[0] - df1['use0'].value_counts().head(15).sum())

#Filtering Based on Top Treatments
def filter_high_uses(x):
    return x in top_treatments

#data filtering and cleaning
#Filtering Data Based on Top Treatments
df2 = df1[df1['use0'].apply(filter_high_uses)]
print(df2.head())
#Checking for Missing Values
print(df2.isnull().sum())
#Dropping Rows with Missing Values
df3 = df2.dropna()
print(df3.head())
#Calculating the Number of Rows Removed
print(df2.shape[0] - df3.shape[0])
#Checking the Shape and Validity of the Final Dataset
print(df3.shape)
print(df3.isnull().sum())
print(df3.head())

# data cleaning
df4 = df3.drop('id', axis='columns', errors='ignore')  # Safely drop 'id' if it exists

# Apply lowercase and strip to string columns only
df4 = df4.apply(lambda col: col.str.lower().str.strip() if col.dtype == 'object' else col)

# Check first few rows
print(df4.head())

# Print column names
print(list(df4.columns))

# Print unique value counts for each column
for col in df4.columns:
    print(f'{col}: {len(df4[col].unique())}')

# Plot value counts for 'Habit Forming'
df4['Habit Forming'].value_counts().plot.bar()
plt.title('Habit Forming Value Counts')
plt.show()

# Print value counts and plot bar graphs for side effects
print(df4['sideEffect0'].value_counts())

df4['sideEffect0'].value_counts().head(10).plot.bar()
plt.title('Top 10 Side Effects (sideEffect0)')
plt.show()

df4['sideEffect1'].value_counts().head(10).plot.bar()
plt.title('Top 10 Side Effects (sideEffect1)')
plt.show()

df4['sideEffect2'].value_counts().head(10).plot.bar()
plt.title('Top 10 Side Effects (sideEffect2)')
plt.show()

# Collect side effects for each substitute
substitute0_sideeffects = {}
for sub, sideeffect in df4.groupby('substitute0')['sideEffect0']:
    print('Substitute:', sub)
    print('Side effects:', list(sideeffect))
    substitute0_sideeffects[sub] = list(sideeffect)

# Convert the dictionary into a DataFrame
st0_sideeffects = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in substitute0_sideeffects.items()]))
print(st0_sideeffects)

# data cleaning and data inspection
st0_sideeffects = st0_sideeffects.drop_duplicates()
st0_sideeffects.shape
print(df4.head())
# Print the columns of the new DataFrame
print(st0_sideeffects.columns)
print(st0_sideeffects['zycin 500mg injection'].unique())
print(st0_sideeffects['lcfex tablet'].unique())

print(df4['Therapeutic Class'].unique())
print(df4['Therapeutic Class'].value_counts())

print(df4['Habit Forming'].value_counts())
print(df4.groupby('Therapeutic Class')['Habit Forming'].value_counts())
print(df4.groupby('Therapeutic Class')['Habit Forming'].value_counts().unstack().fillna(0))
#Grouping and Counting Data
plot_data = df4.groupby('Therapeutic Class')['Habit Forming'].value_counts().unstack().fillna(0)
#Plotting the Data
plot_data.plot(kind='bar')
#Adding Titles and Labels
plt.title('Habit Forming Counts by Therapeutic Class')
plt.xlabel('Therapeutic Class')
# Adjust layout for better fit
plt.tight_layout()  
plt.ylabel('Count')
plt.show()

print(df4.head())
#Data Preparation for Modeling
#Defining the Features (X)
X = df4.drop(['Therapeutic Class','Habit Forming'] , axis = 1)
#Defining the Target Variable (Y)
Y = df4['Therapeutic Class']

#Initializing an Empty List
all_sideeffects = []
#Iterating Over Rows
for i in range(0,len(df4)):
    sideeffects = list(df4.iloc[i , 6:9])  #Extracting Specific Columns
    sideeffects = sorted(sideeffects)      #Sorting the Side Effects
    all_sideeffects.append(sideeffects)    #Appending to the List


#data manipulation and preparation
df4.insert(9,'All Sideeffects' , all_sideeffects)
print(df4.head())

#Dropping Unused Columns
df5 = df4.drop(['sideEffect0','sideEffect1','sideEffect2'] , axis = 1)
print(df5.head())

#Changing Data Type of 'All Sideeffects'
df5['All Sideeffects'] = df5['All Sideeffects'].astype(str)
#Counting Unique Values in 'All Sideeffects'
print(df5['All Sideeffects'].nunique())

count_all_sideeffects = df5['All Sideeffects'].value_counts()
print(count_all_sideeffects)
 
print(count_all_sideeffects[count_all_sideeffects > 100])
all_subs = []
for i in range(len(df4)):
    subs = list(df4.iloc[i ,1: 6])
    subs = sorted(subs)
    all_subs.append(subs)

df5.insert(6 , 'All Substitutes' , all_subs)
print(df5.head())

df6 = df5.drop(['substitute0','substitute1','substitute2',
                'substitute3','substitute4'] , axis=1)

print(df6.head())
df6['All Substitutes'] = df6['All Substitutes'].astype(str)
print(df6['All Substitutes'].nunique())
print(df6.shape)
count_all_subs = df6['All Substitutes'].value_counts()
print(count_all_subs)
print(count_all_subs[count_all_subs > 100])
