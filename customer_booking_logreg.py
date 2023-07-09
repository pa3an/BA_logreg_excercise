
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# import csv file - I had an encoding issue, so I specified the character encoding
customer_booking = pd.read_csv("/customer_booking.csv", encoding='latin1') # adjust file path accordingly

customer_booking.head()

customer_booking.shape[0]

# encode categorical variables
encoder = LabelEncoder()
customer_booking['flight_day_encoded'] = encoder.fit_transform(customer_booking['flight_day'])
customer_booking['trip_type_encoded'] = encoder.fit_transform(customer_booking['trip_type'])
customer_booking['sales_channel_encoded'] = encoder.fit_transform(customer_booking['sales_channel'])

# check n. of rows with missing values...
customer_booking.isna().sum(axis=1).gt(0).sum()

# run logistic regression
# Prepare the data
features = ['num_passengers', 'sales_channel_encoded', 'trip_type_encoded', 'purchase_lead',
            'length_of_stay', 'flight_hour', 'flight_day_encoded', 'wants_extra_baggage',
            'wants_preferred_seat', 'wants_in_flight_meals', 'flight_duration']
X = customer_booking[features]
y = customer_booking['booking_complete']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the models
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Random Forest', RandomForestClassifier())
]

for name, model in models:
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)

    # Print evaluation metrics
    print(f'{name}:')
    print(f'  Accuracy: {accuracy}')
    print(f'  Precision: {precision}')
    print(f'  Recall: {recall}')
    print(f'  F1 Score: {f1}')
    print(f'  AUC-ROC: {auc_roc}')

# checking for multicollinearity
# Calculate the correlation matrix
correlation_matrix = X_train.corr()

# Set the figure size
plt.figure(figsize=(10, 8))

# Create a heatmap of the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# Display the heatmap
plt.show()


# the matrix above shows moderate correlation (0.31) between wants_preferred_seat 
# and want_in_flight_meals - to avoid multicollinearity, we'll drop the latter
# and run the models again
# Prepare the data
features = ['num_passengers', 'sales_channel_encoded', 'trip_type_encoded', 'purchase_lead',
            'length_of_stay', 'flight_hour', 'flight_day_encoded', 'wants_extra_baggage',
            'wants_preferred_seat', 'flight_duration']
X = customer_booking[features]
y = customer_booking['booking_complete']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the models
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Random Forest', RandomForestClassifier())
]

for name, model in models:
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)

    # Print evaluation metrics
    print(f'{name}:')
    print(f'  Accuracy: {accuracy}')
    print(f'  Precision: {precision}')
    print(f'  Recall: {recall}')
    print(f'  F1 Score: {f1}')
    print(f'  AUC-ROC: {auc_roc}')

# run regression to check feature coefficients
# Create and fit the logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Obtain the coefficient values
coefficients = logreg.coef_[0]
intercept = logreg.intercept_[0]

# Create a DataFrame to display the coefficients
coef_df = pd.DataFrame({'Variable': features, 'Coefficient': coefficients})
coef_df['Absolute Coefficient'] = np.abs(coefficients)
coef_df['Odds Ratio'] = np.exp(coefficients)

# Sort the DataFrame by absolute coefficient values
sorted_coef_df = coef_df.sort_values('Absolute Coefficient', ascending=False)

# Display the sorted coefficients
print(sorted_coef_df)

# Sort the DataFrame by absolute coefficient values
sorted_coef_df = coef_df.sort_values('Absolute Coefficient', ascending=False)

# Create a bar plot of the sorted coefficients
plt.figure(figsize=(10, 6))
plt.barh(sorted_coef_df['Variable'], sorted_coef_df['Coefficient'], color='steelblue')
plt.xlabel('Coefficient')
plt.ylabel('Variable')
plt.title('Logistic Regression Coefficients (Ordered by Magnitude)')
plt.grid(axis='x')
plt.savefig('logistic_regression_coefficients.png', bbox_inches='tight')
plt.show()

# we can also plot the absolute coefficients, to show the stength of the relationship
# (but not its direction)
# Take the absolute values of the coefficients
sorted_coef_df['Coefficient'] = abs(sorted_coef_df['Coefficient'])

# Create a bar plot of the sorted coefficients
plt.figure(figsize=(10, 6))
plt.barh(sorted_coef_df['Variable'], sorted_coef_df['Coefficient'], color='steelblue')
plt.xlabel('Coefficient')
plt.ylabel('Variable')
plt.title('Logistic Regression Coefficients (Ordered by Magnitude)')
plt.grid(axis='x')
plt.gca().invert_yaxis()  # Invert the y-axis to have bars in descending order
plt.savefig('logistic_regression_coefficients.png', bbox_inches='tight')
plt.show()

# create a bar plot of'booking_complete' and 'purchase_lead'

# Group the data by 'purchase_lead' and calculate the proportion of bookings made
booking_counts = customer_booking.groupby('purchase_lead')['booking_complete'].mean()

# Create a bar plot
plt.bar(booking_counts.index, booking_counts)
plt.xlabel('Purchase Lead')
plt.ylabel('Proportion of Bookings Made')
plt.title('Proportion of Bookings Made by Purchase Lead')

# Show the plot
plt.show()

# Encode 'booking_complete' into dummy variables so we can run linear rather than logistic regression

# Perform one-hot encoding on 'booking_complete'
booking_complete_encoded = pd.get_dummies(customer_booking['booking_complete'], prefix='booking')

# Concatenate the encoded variables with the original DataFrame
customer_booking_encoded = pd.concat([customer_booking, booking_complete_encoded], axis=1)

# Drop the original 'booking_complete' column
customer_booking_encoded.drop('booking_complete', axis=1, inplace=True)

# run linear regression

# Split the data into training and testing sets
X = customer_booking_encoded[['purchase_lead']]
y = customer_booking_encoded[['booking_0', 'booking_1']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = linear_reg.predict(X_test)

# Assess the model's performance
r2_score = linear_reg.score(X_test, y_test)
print('R-squared Score:', r2_score)



