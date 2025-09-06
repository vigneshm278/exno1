# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output
```
import pandas as pd  
df=pd.read_csv("/content/SAMPLEIDS.csv")
df
```
<img width="1076" height="876" alt="Screenshot 2025-09-05 221136" src="https://github.com/user-attachments/assets/82535a24-6f85-4afd-ac32-ec65c0fc8690" />

```
df.head()
```
<img width="1066" height="254" alt="Screenshot 2025-09-05 222015" src="https://github.com/user-attachments/assets/a020919d-b1fd-460c-9ee5-cea0db3a35ca" />

```
df.tail()
```
<img width="1090" height="247" alt="Screenshot 2025-09-05 222114" src="https://github.com/user-attachments/assets/ab25be12-a4f4-47bc-a022-daffd33c2c1b" />

```
df.info()
```
<img width="412" height="421" alt="Screenshot 2025-09-05 222304" src="https://github.com/user-attachments/assets/de9e3790-6a3a-484f-a0c7-a814914c72d7" />


```
df.describe()
```
<img width="983" height="360" alt="Screenshot 2025-09-05 222415 - Copy" src="https://github.com/user-attachments/assets/0e6362f2-71e2-4eaa-b373-0cd93ececdfc" />

```
df.isnull().sum()
```
<img width="874" height="571" alt="Screenshot 2025-09-05 222522" src="https://github.com/user-attachments/assets/3b7f4fad-fb59-4c9a-846c-0e62b32f6cae" />

```
df.isnull().any()
```
<img width="377" height="571" alt="Screenshot 2025-09-05 222613" src="https://github.com/user-attachments/assets/f7b3fdc9-0136-4b36-acad-a42deaf9d552" />

```
df.dropna()
```
<img width="1099" height="568" alt="Screenshot 2025-09-05 222701" src="https://github.com/user-attachments/assets/a2aff1e5-1bdf-4dd0-a3f2-0b5093ad1e46" />

```
df.fillna(0)
```
<img width="1160" height="872" alt="Screenshot 2025-09-05 222746" src="https://github.com/user-attachments/assets/3f465143-b1e4-4769-8eef-438f2a4c5ee2" />

```
df.fillna(method='ffill')
```
<img width="1700" height="823" alt="Screenshot 2025-09-05 223440" src="https://github.com/user-attachments/assets/1f42fdb9-4be6-4ff1-960d-72e9d1e1458b" />

```
df.fillna({'NAME':'VICKY','DOB': '2005-8-27', 'GENDER': 'Male', 'ADDRESS': 'Ambethkar st',
          'M1': 27.9, 'M2': 26.0, 'M3': 25.0, 'M4': 24.0, 'TOTAL': 100.0})
```
<img width="1241" height="768" alt="Screenshot 2025-09-06 102241" src="https://github.com/user-attachments/assets/92e70261-d93b-44d1-ba9d-6d684730cd06" />


```
ir=pd.read_csv("/content/iris.csv")
ir
```

<img width="734" height="513" alt="Screenshot 2025-09-05 223913" src="https://github.com/user-attachments/assets/ef8e50b0-bdb1-404c-96ed-365605526cdf" />

```
ir.describe()
```

<img width="672" height="365" alt="Screenshot 2025-09-05 224004" src="https://github.com/user-attachments/assets/a0827249-2a21-4a09-a37c-2516135101e7" />


```
import seaborn as sns

sns.boxplot(x='sepal_width',data=ir)

```
<img width="770" height="562" alt="Screenshot 2025-09-05 224057" src="https://github.com/user-attachments/assets/15d68e84-e698-4d0e-86b3-bc2f2810dde4" />

```
Q1=ir.sepal_width.quantile(0.25)
Q3=ir.sepal_width.quantile(0.75)
(IQR)=Q3-Q1
print(IQR)
```
<img width="225" height="38" alt="Screenshot 2025-09-05 224300" src="https://github.com/user-attachments/assets/5eeb55ac-884c-4817-ba2a-49d1512d7f75" />

```
ran=ir[((ir.sepal_width<(Q1-1.5*IQR))|(ir.sepal_width>(Q3+1.5*IQR)))]
ran['sepal_width']
```

<img width="540" height="253" alt="Screenshot 2025-09-05 224352" src="https://github.com/user-attachments/assets/00452a63-e59e-49d6-8137-2d53aeba6c46" />

```
ran=ir[~((ir.sepal_width<(Q1-1.5*IQR))|(ir.sepal_width>(Q3+1.5*IQR)))]
ran['sepal_width']
```

<img width="342" height="566" alt="Screenshot 2025-09-05 224444" src="https://github.com/user-attachments/assets/aef4cc78-1f97-4bd7-af0c-fca43e424615" />

```
sns.boxplot(x='sepal_width',data=ran)
```

<img width="777" height="569" alt="Screenshot 2025-09-05 224538" src="https://github.com/user-attachments/assets/c0c3bd9b-293b-49fe-99dd-d54887acee45" />

```
import numpy as np
import scipy.stats as stats
```
```
z=np.abs(stats.zscore(ir['petal_length']))
z
```


<img width="766" height="664" alt="Screenshot 2025-09-05 224710" src="https://github.com/user-attachments/assets/913d64f8-c95a-4b11-b3bb-98dcb935de0d" />

```
ir1=ir[z<3]
ir1
```


<img width="748" height="517" alt="Screenshot 2025-09-05 224852" src="https://github.com/user-attachments/assets/cc5d4078-90ba-48b4-8c3e-05d7b2f40da7" />


# Result
Thus the given data successfully performed data cleaning and saved the cleaned data to a file.
