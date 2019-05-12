import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
# Use a predefined style set
plt.style.use('ggplot')
from faker import Faker
fake = Faker()

# To ensure the results are reproducible
fake.seed(4321)

names_list = []

for _ in range(100):
  names_list.append(fake.name())

# Verify if 100 names were generated
len(names_list)

# To ensure the results are reproducible
np.random.seed(7)

salaries = []
for _ in range(100):
    salary = np.random.randint(1000,2500)
    salaries.append(salary)

    # Verify if 100 salariy values were generated
len(salaries)

# Create pandas DataFrame
salary_df = pd.DataFrame(
    {'Person': names_list,
     'Salary (in USD)': salaries
    })

# Print a subsection of the DataFrame"
salary_df.head()

#Let's now manually change the salary entries of two individuals.
salary_df.at[16, 'Salary (in USD)'] = 23
#or salary_df.iloc[16, 1]=24 could work
salary_df.at[65, 'Salary (in USD)'] = 17

salary_df['Salary (in USD)'].plot(kind='box')
plt.show()

ax = salary_df['Salary (in USD)'].plot(kind='hist')
ax.set_xlabel('Salary (in USD)')
plt.show()

# Convert the salary values to a numpy array
salary_raw = salary_df['Salary (in USD)'].values

# For compatibility with the SciPy implementation
salary_raw = salary_raw.reshape(-1, 1)
salary_raw = salary_raw.astype('float64')

from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans=kmeans.fit_predict(salary_raw)

plt.scatter(salary_raw, y_kmeans, c='blue')
plt.xlabel('Salaries in (USD)')
plt.ylabel('Groups')
plt.show()

#convertin the unsupervised problem to a supervised problem by adding class column
# First assign all the instances to 
salary_df['class'] = 0

# Manually edit the labels for the anomalies
salary_df.at[16, 'class'] = 1
salary_df.at[65, 'class'] = 1

# Veirfy 
salary_df.loc[16]

# Importing KNN module from PyOD
#Python library called PyOD which is specifically developed 
#for anomaly detection purposes.
from pyod.models.knn import KNN

# Segregate the salary values and the class labels 
X = salary_df['Salary (in USD)'].values.reshape(-1,1)
y = salary_df['class'].values

# Train kNN detector
clf = KNN(contamination=0.02, n_neighbors=5)
clf.fit(X)

# Get the prediction labels of the training data
y_train_pred = clf.labels_ 

# Outlier scores
y_train_scores = clf.decision_scores_


from pyod.utils import evaluate_print

# Evaluate on the training data
evaluate_print('KNN', y, y_train_scores)

# A salary of $37 (an anomaly right?)
X_test = np.array([[37.]])
# Check what the model predicts on the given test data point
clf.predict(X_test)

# A salary of $1256
X_test_abnormal = np.array([[1256.]])

# Predict
clf.predict(X_test_abnormal)
