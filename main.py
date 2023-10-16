import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt


# Load the dataset
df = pd.read_csv('ds_salaries.csv')

# Check for missing values
missing_values = df.isnull().sum()
# print("Missing values:\n", missing_values)

# Drop rows with missing values
df = df.dropna()

# Remote Ratio - Salary
sns.boxplot(x='remote_ratio', y='salary_in_usd', data=df)
plt.title(' Remote Ratio - Salary')
plt.xlabel('Remote Ratio')
plt.ylabel('Salary ')
plt.show()

print("\n")

# Company Size - Salary
company_size_order = ['S', 'M', 'L']
sns.boxplot(x='company_size', y='salary_in_usd', data=df, order=company_size_order)
plt.title(' Company Size - Salary')
plt.xlabel('Company Size')
plt.ylabel('Salary ')
plt.show()

print("\n")


# count of data points for each country
country_counts = df['company_location'].value_counts()

# Select the countries with more than 15 data points
selected_countries = country_counts[country_counts > 15].index

# Filter the dataframe based on the selected countries
filtered_df = df[df['company_location'].isin(selected_countries)]

# Calculate the average salary for each country
average_salary_by_country = filtered_df.groupby('company_location')['salary_in_usd'].mean().reset_index()
# Plot the average salary by country
sns.barplot(x='company_location', y='salary_in_usd', data=average_salary_by_country)
plt.title('Average Salary by Country (with >15 data points)')
plt.xlabel('Country')
plt.ylabel('Average Salary')
plt.xticks(rotation=45)
plt.show()

print("\n")

# Calculate the average salary for each experience level
average_salary_by_experience = df.groupby('experience_level')['salary_in_usd'].mean().reset_index()

# Define the order of experience levels
experience_order = ['EN', 'MI', 'SE', 'EX']

# Sort the dataframe based on the defined order
average_salary_by_experience['experience_level'] = pd.Categorical(average_salary_by_experience['experience_level'], categories=experience_order, ordered=True)
average_salary_by_experience = average_salary_by_experience.sort_values('experience_level')

# Plot the average salary by experience level
sns.barplot(x='experience_level', y='salary_in_usd', data=average_salary_by_experience)
plt.title('Average Salary by Experience Level')
plt.xlabel('Experience Level')
plt.ylabel('Average Salary')
plt.show()
print("\n")


# Classification
# Define salary classes
salary_classes = ['low', 'medium', 'high']

# Transform salary values into classes
salary_quantiles = df['salary_in_usd'].quantile([0.25, 0.75]).values
df['salary_class'] = pd.cut(df['salary_in_usd'], bins=[0, salary_quantiles[0], salary_quantiles[1], float('inf')], labels=salary_classes)

# Select features for classification
X_class = df[['work_year', 'experience_level','employment_type', 'job_title','remote_ratio','company_location', 'company_size']]
y_class = df['salary_class']

# Select features for regression
X_reg = df[['work_year', 'experience_level','employment_type', 'job_title','remote_ratio','company_location', 'company_size']]
y_reg = df['salary_in_usd']

# Get the top 25 most common job titles
top_job_titles = df['job_title'].value_counts().nlargest(25).index
X_class = X_class[X_class['job_title'].isin(top_job_titles)]
y_class = y_class[X_class.index]
X_reg = X_reg[X_reg['job_title'].isin(top_job_titles)]
y_reg = y_reg[X_reg.index]

# Preprocess categorical features using one-hot encoding
preprocessor = ColumnTransformer([('encoder', OneHotEncoder(), [1, 2, 3, 4, 5 ,6])], remainder='passthrough')
X_class_encoded = preprocessor.fit_transform(X_class)
X_reg_encoded = preprocessor.transform(X_reg)

# Split the dataset into training and test sets for classification
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(X_class_encoded, y_class, test_size=0.2, random_state=22)

# Split the dataset into training and test sets for regression
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg_encoded, y_reg, test_size=0.2, random_state=22)

# Create a classification model
classification_model = LogisticRegression(max_iter=1000)

# Train the classification model
classification_model.fit(X_class_train, y_class_train)

# Make predictions on the test set for classification
y_class_pred = classification_model.predict(X_class_test)

# Evaluate the classification model
classification_report = classification_report(y_class_test, y_class_pred)

print("Classification Report:\n", classification_report)

# Create a regression model
regression_model = LinearRegression()

# Train the regression model
regression_model.fit(X_reg_train, y_reg_train)

# Make predictions on the test set for regression
y_reg_pred = regression_model.predict(X_reg_test)

# Example instance
new_instance = [[2023,'MI', 'FT', 'ML Engineer',100,'IN', 'M']]

# Preprocess the new instance
new_instance_encoded = preprocessor.transform(pd.DataFrame(new_instance, columns=X_class.columns))

# Predict the salary class
predicted_class = classification_model.predict(new_instance_encoded)

# Get the range of each salary class
class_ranges = {
    'low': f"0-{salary_quantiles[0]:.2f}",
    'medium': f"{salary_quantiles[0]:.2f}-{salary_quantiles[1]:.2f}",
    'high': f"{salary_quantiles[1]:.2f}+"
}

# Print the predicted salary class and its range
predicted_range = class_ranges[predicted_class[0]]
print("Predicted Salary Class:", predicted_class, "Range:", predicted_range)

# Predict the salary using regression
predicted_salary = regression_model.predict(new_instance_encoded)

print("Predicted Salary (in number): {:.2f}".format(predicted_salary[0]))