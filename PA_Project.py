from google.colab import drive
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
%matplotlib inline
df = pd.read_csv(file)
df.head()
df.tail()
df.shape
df.columns
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()

# Show the number of orders for each category of pizza
categories = df.pizza_category.value_counts()

# Find the index of the highest value (highest number of orders)
highest_index = categories.idxmax()

# Set the color of the highest bar to green, and others to blue color
colors = ['#EE6A50' if i == highest_index else '#76EEC6' for i in categories.index]

# Plot the bar chart
plots = categories.plot.barh(figsize = (10,5), color = colors)
plt.ylabel('Pizza category')
plt.xlabel('Number of orders')
plt.title('Number of orders by pizza categories', fontsize = 15)

# Total number of orders
total = len(df)

# Iterating over the bars one-by-one
for bar in plots.patches:

  # Using Matplotlib's annotate function and
  # passing the coordinates where the annotation shall be done
  # y-coordinate: bar.get_y() + bar.get_width() / 2
  # x-coordinate: bar.get_width()
  # ha and va stand for the horizontal and vertical alignment
    plots.annotate(format(bar.get_width() / total, '.0%'),
                   (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    ha='center', va='center',
                   size=13, xytext=(-25, 0),
                   textcoords='offset points', color = 'white', fontweight = 600)

# Hide the right and top spines
plots.spines[['right', 'top']].set_visible(False);

total = len(df) # total number of orders
L = len(df[df.pizza_size == 'L']) / total # % of Large pizzas to total no. of orders
M = len(df[df.pizza_size == 'M']) / total # % of Medium pizzas to total no. of orders
S = len(df[df.pizza_size == 'S']) / total # % of Small pizzas to total no. of orders
XL = len(df[df.pizza_size == 'XL']) / total # % of Extra Large pizzas to total no. of orders
XXL = len(df[df.pizza_size == 'XXL']) / total # % of Extra Extra Large pizzas to total no. of orders

# Plot the pie chart
figure, ax = plt.subplots()
patches, texts, pcts = ax.pie([L,M,S,XL+XXL],
                              labels = ["L","M","S","XL+XXL"],
                              autopct = '%.1f%%',
                              wedgeprops={'linewidth':1.0, 'edgecolor': 'white'},
                              textprops = {'size': 'x-large'},
                              shadow = False,
                              colors= ['#7EC0EE','#EED2EE','#F4A460','#388E8E'])

# Title of pie chart
ax.set_title('Pizza sizes by orders (%)', fontsize = 15)

# Show the pie chart
plt.tight_layout();

print(f'The best selling pizza is: {df.pizza_name.value_counts().nlargest(1)}' '\n')
print(f'The worst selling pizza is : {df.pizza_name.value_counts().nsmallest(1)}')

print(f'Mean: USD {df.total_price.mean():.2f}')
print(f'Median: USD {df.total_price.median():.2f}')

df.total_price.sum()

len(df)

# Extract the month and month_name from order_date
month = []
month_name = []
df['date_column'] = pd.to_datetime(df['order_date'])
for i in df.date_column:
    # 1 = January, 2 = February, ..., 12 = December
    month = i.month
    # January, February, ..., December
    month_name.append(i.month_name())

df['month'] = month
df['month_name'] = month_name
total_revenue = df.total_price.sum() # total revenue = sum of prices of all orders

# Find the total revenue by each month using groupby
revenue_per_month_df = df.groupby(['month_name']).agg({'total_price': 'sum'})

# Rank the monthly total revenue from highest to lowest
revenue_per_month_df['rank'] = revenue_per_month_df['total_price'].rank(method="dense", ascending=False)

# Sort from the lowest revenue to the highest revenue
revenue_per_month_df = revenue_per_month_df.sort_values('rank', ascending=True).reset_index()
revenue_per_month_df

# Create a bar plot to visualize monthly total revenue
# Define a custom color palette for the bars
custom_colors = ['#8B475D', '#ff7f0e', '#2ca02c', '#d62728',
                 '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                 '#17becf','#B0C4DE','#FFC0CB']

# Create a bar plot with the custom color palette
plt.figure(figsize=(12, 6))  # Set the figure size
bars = plt.bar(revenue_per_month_df['month_name'],
               revenue_per_month_df['total_price'], color=custom_colors)
plt.xlabel('Month')
plt.ylabel('Total Revenue')
plt.title('Monthly Total Revenue')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Annotate the bars with revenue values
for bar, revenue in zip(bars, revenue_per_month_df['total_price']):
    plt.text(bar.get_x() + bar.get_width() / 2, revenue + 100,
             f'{revenue:.2f}', ha='center', va='bottom')

plt.tight_layout()  # Ensure the labels fit within the figure area

# Display the plot
plt.show()

top_pizza_analysis = df.groupby('pizza_name').agg(
    average_unit_price=('unit_price', 'mean'),
    revenue_per_pizza=('unit_price', lambda x: (x * df['quantity']).sum())
).nlargest(5, 'revenue_per_pizza')
print("Average Unit Price and Revenue of Top 3 Pizzas:\n", top_pizza_analysis)

# Create a bar plot to visualize revenue by top pizzas
plt.figure(figsize=(10, 6))  # Set the figure size
bars = plt.bar(top_pizza_analysis.index, top_pizza_analysis
 ['revenue_per_pizza'], color='#FFE1FF')
plt.xlabel('Pizza Name')
plt.ylabel('Revenue')
plt.title('Revenue by Top Pizzas')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Annotate the bars with revenue values
for bar, revenue in zip(bars, top_pizza_analysis['revenue_per_pizza']):
    plt.text(bar.get_x() + bar.get_width() / 2, revenue, f'{revenue:.2f}',
             ha='center', va='bottom')

plt.tight_layout()  # Ensure the labels fit within the figure area

# Display the plot
plt.show()

df.groupby('order_date').agg({'order_id': 'count'}).plot(kind = 'line', figsize=(15,7), legend=False, color="#71C671")
plt.ylabel('Number of orders')
plt.title('Number of orders per day');

df.groupby('order_date').agg({'total_price': 'sum'}).plot(kind = 'line', figsize=(15,7), legend=False, color="#D02090")
plt.ylabel('Revenue')
plt.title('Revenue per day');

plt.figure(figsize=(15,10))
sns.boxplot(y='unit_price', x='pizza_name', data=df)
plt.xticks(rotation=90)
plt.ylabel('Price')
plt.xlabel('Pizza Types')

sns.scatterplot(y='unit_price', x='pizza_name', data=df)
plt.xticks(rotation=90)

df.corr()

#Data preparation
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder();
df['pizza_id'] = le.fit_transform(df['pizza_id'])
df['pizza_category'] = le.fit_transform(df['pizza_category'])
df['pizza_ingredients'] = le.fit_transform(df['pizza_ingredients'])
df['pizza_name'] = le.fit_transform(df['pizza_name'])
df['month_name'] = le.fit_transform(df['month_name'])
df['order_date'] = le.fit_transform(df['order_date'])
df['order_time'] = le.fit_transform(df['order_time'])
df['date_column'] = le.fit_transform(df['date_column'])
df['pizza_size'] = le.fit_transform(df['pizza_size'])

#model building
# decision tree
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Define features and target
X = df[['unit_price','total_price','pizza_category','pizza_name','month_name','pizza_size','order_time']]
y = df['quantity']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train a Decision Tree model (you can customize hyperparameters as needed)
dct_model = DecisionTreeRegressor(random_state=42)
dct_model.fit(X_train, y_train)

# Make predictions
y_pred = dct_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = (np.sqrt(mse))
r2 = r2_score(y_test, y_pred)

print(f'The RMSE for the Decision Tree model is : {rmse}')
print(f'The R2 Score for the Decision Tree model is : {r2}')

# Graph for the above code
import matplotlib.pyplot as plt

# Create a scatter plot of actual vs. predicted values with different colors for actual and predicted points
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, c='blue', label='Actual', marker='o')  # Actual points in blue circles
plt.scatter(y_test, y_test, alpha=0.5, c='red', label='Predicted', marker='x')  # Predicted points in red X's
plt.title('Actual vs. Predicted Quantity of Pizza Orders')
plt.xlabel('Actual Quantity')
plt.ylabel('Predicted Quantity')
plt.legend()
plt.grid(True)
plt.show()

#To know feature importance
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Define features and target
X = df[['unit_price','total_price','pizza_category','pizza_name','month_name','pizza_size','order_time']]
y = df['quantity']

# Train a Decision Tree model
dct_model = DecisionTreeRegressor(random_state=42)
dct_model.fit(X, y)  # Train on the entire dataset

# Get feature importances
feature_importances = dct_model.feature_importances_

# Create a DataFrame to display feature names and their importances
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Create a bar plot to visualize feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.title('Feature Importances in Decision Tree Model for Quantity Prediction')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


