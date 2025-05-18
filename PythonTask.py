#!/usr/bin/env python
# coding: utf-8

# # Importing the Libraries

# In[1]:


import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine, Column, Integer, Float, String, Table, MetaData
from sqlalchemy.orm import sessionmaker
from bokeh.plotting import figure, show, output_notebook
from bokeh.layouts import gridplot
import unittest
import os

# Enable Bokeh plots to display in the notebook
output_notebook()


# # Load and Inspect the Data

# In[2]:


# Load the datasets from CSV files
train_df = pd.read_csv('train.csv')
ideal_df = pd.read_csv('ideal.csv')
test_df = pd.read_csv('test.csv')

# Display first few rows of each dataframe
print("Training Data")
display(train_df.head())

print("Ideal Functions Data")
display(ideal_df.head())

print("Test Data")
display(test_df.head())


# In[3]:


# Verify x values in test data are within the range of x values in ideal functions
min_x, max_x = ideal_df['x'].min(), ideal_df['x'].max()
out_of_range_test_data = test_df[(test_df['x'] < min_x) | (test_df['x'] > max_x)]
if not out_of_range_test_data.empty:
    print("Warning: Some test data x-values are out of range of the ideal functions dataset.")
    display(out_of_range_test_data)


# # Task 2: Creating SQLite Database and Tables

# In[4]:


from sqlalchemy import create_engine, Column, Float, Integer, Table, MetaData

# Set up the SQLite database connection
db_path = 'data_analysis.db'
engine = create_engine(f'sqlite:///{db_path}')
metadata = MetaData()

# Define the database tables
train_table = Table('training_data', metadata,
                    Column('x', Float, primary_key=True),
                    Column('y1', Float),
                    Column('y2', Float),
                    Column('y3', Float),
                    Column('y4', Float))

ideal_table = Table('ideal_functions', metadata,
                    Column('x', Float, primary_key=True),
                    *(Column(f'y{i}', Float) for i in range(1, 51)))

test_table = Table('test_data', metadata,
                   Column('x', Float, primary_key=True),
                   Column('y', Float),
                   Column('delta_y', Float),
                   Column('ideal_func_no', Integer))

# Create the tables in the database
metadata.create_all(engine)

# Insert the data into the database
with engine.connect() as conn:
    train_df.to_sql('training_data', conn, if_exists='replace', index=False)
    ideal_df.to_sql('ideal_functions', conn, if_exists='replace', index=False)
    test_df.to_sql('test_data', conn, if_exists='replace', index=False)

print("Data successfully loaded into the SQLite database.")


# # Task 3: Identifying the Best Fit Ideal Functions for the Training Data

# In[5]:


import numpy as np

# Function to find the best fit ideal functions based on least square deviation
def find_best_fit_functions(train_df, ideal_df):
    best_functions = []
    for train_col in train_df.columns[1:]:
        y_train = train_df[train_col].values
        min_error = float('inf')
        best_func = None
        for ideal_col in ideal_df.columns[1:]:
            y_ideal = ideal_df[ideal_col].values
            error = np.sum((y_train - y_ideal) ** 2)
            if error < min_error:
                min_error = error
                best_func = ideal_col
        best_functions.append(best_func)
    return best_functions

# Find the best functions
best_functions = find_best_fit_functions(train_df, ideal_df)
print("Best fitting functions for the training data:", best_functions)


# # Task 4: Mapping the Test Data to the Ideal Functions

# In[6]:


# Function to map test data to the best fit ideal functions
def map_test_data(test_df, ideal_df, best_functions, threshold_factor=np.sqrt(2)):
    mapped_data = []
    for _, row in test_df.iterrows():
        x = row['x']
        y_test = row['y']
        # Ensure x exists in the ideal dataset
        if not (ideal_df['x'] == x).any():
            print(f"Warning: x={x} not found in the ideal dataset, skipping this test data point.")
            continue
        for func_no, func_name in enumerate(best_functions, start=1):
            y_ideal = ideal_df.loc[ideal_df['x'] == x, func_name].values[0]
            deviation = abs(y_test - y_ideal)
            max_allowed_dev = ideal_df[func_name].max() * threshold_factor
            if deviation <= max_allowed_dev:
                mapped_data.append({'x': x, 'y': y_test, 'delta_y': deviation, 'ideal_func_no': func_no})
                break
    return pd.DataFrame(mapped_data)

# Map the test data
mapped_test_df = map_test_data(test_df, ideal_df, best_functions)

# Insert the mapped data into the database
with engine.connect() as conn:
    mapped_test_df.to_sql('mapped_test_data', conn, if_exists='replace', index=False)

print("Test data mapped to the ideal functions and saved to the database.")
display(mapped_test_df.head())


# # Task 5: Visualizing the Data

# In[7]:


from bokeh.plotting import figure, show, output_notebook
from bokeh.layouts import gridplot

output_notebook()

# Function to create Bokeh plots
def plot_data(train_df, test_df, ideal_df, best_functions):
    plots = []
    for i, func in enumerate(best_functions):
        p = figure(title=f"Training Data vs Ideal Function {func}", x_axis_label='x', y_axis_label='y')
        p.line(train_df['x'], train_df.iloc[:, i+1], legend_label='Training Data', color='blue', line_width=2)
        p.line(ideal_df['x'], ideal_df[func], legend_label='Ideal Function', color='green', line_width=2)
        p.circle(test_df['x'], test_df['y'], legend_label='Test Data', color='red', size=8)
        plots.append(p)
    
    grid = gridplot(plots, ncols=2)
    show(grid)

# Visualize the data
plot_data(train_df, test_df, ideal_df, best_functions)


# # Task 6: Writing Unit Tests

# In[8]:


import unittest

class TestFunctionMatcher(unittest.TestCase):
    def setUp(self):
        self.train_df = pd.DataFrame({
            'x': [1, 2, 3],
            'y1': [1, 4, 9],
            'y2': [1, 2, 3],
            'y3': [2, 3, 4],
            'y4': [3, 6, 9]
        })
        self.ideal_df = pd.DataFrame({
            'x': [1, 2, 3],
            'y1': [1, 4, 8],
            'y2': [1, 2, 3],
            'y3': [2, 3, 5],
            'y4': [3, 6, 8],
            'y5': [1, 4, 9]
        })

    def test_find_best_fit_functions(self):
        best_funcs = find_best_fit_functions(self.train_df, self.ideal_df)
        self.assertEqual(best_funcs, ['y5', 'y2', 'y3', 'y4'])

unittest.main(argv=[''], exit=False)


# In[ ]:


# --- Visualization of Training, Ideal, and Test Data ---
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import gridplot

# Use previously identified best functions or load from result
best_functions = ['y13', 'y24', 'y36', 'y40']

# Create Bokeh plots
plots = []
for i, func in enumerate(best_functions):
    p = figure(title=f"Training vs Ideal vs Test (Function: {func})", width=600, height=400)
    p.line(train_df['x'], train_df.iloc[:, i+1], legend_label='Training', line_color='blue', line_width=2)
    p.line(ideal_df['x'], ideal_df[func], legend_label='Ideal', line_color='green', line_width=2)
    p.scatter(test_df['x'], test_df['y'], legend_label='Test', fill_color='red', size=4)
    p.legend.location = "top_left"
    plots.append(p)

# Output to HTML file
output_file("Training_Ideal_Test_Comparison.html")
show(gridplot(plots, ncols=2))
