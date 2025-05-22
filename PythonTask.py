import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from bokeh.plotting import figure, output_file, save
import unittest

class DataProcessingError(Exception):
    """Custom exception raised for errors in loading and saving data."""
    pass

class DataHandler:
    def __init__(self, db_name='data_analysis.db'):
        self.engine = create_engine(f'sqlite:///{db_name}')

    def load_csv(self, filepath):
        """Load a CSV file into a pandas DataFrame."""
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            raise DataProcessingError(f"Failed to load file {filepath}: {e}")

    def save_to_db(self, df, table_name):
        """Save a pandas DataFrame to a SQLite database."""
        try:
            df.to_sql(table_name, self.engine, index=False, if_exists='replace')
        except Exception as e:
            raise DataProcessingError(f"Failed to write to table {table_name}: {e}")

class Analyzer(DataHandler):
    def __init__(self):
        super().__init__()
        self.best_matches = []

    def select_best_ideal_functions(self, train_df, ideal_df):
        """
        Compare each training function with all ideal functions using least squares error.
        Store the best matching ideal function for each training column.
        """
        for col in train_df.columns[1:]:
            min_error = float('inf')
            best_func = None
            for ideal_col in ideal_df.columns[1:]:
                error = ((train_df[col] - ideal_df[ideal_col]) ** 2).sum()
                if error < min_error:
                    min_error = error
                    best_func = ideal_col
            self.best_matches.append(best_func)
        print("\nChosen Ideal Functions for each training column:")
        for i, func in enumerate(self.best_matches):
            print(f"Training column y{i+1} --> Ideal Function: {func}")

    def map_test_data(self, test_df, ideal_df):
        """
        Map each test point to the closest ideal function using a √2 deviation threshold.
        Returns a DataFrame of matched results including deviation values.
        """
        mapped = []
        for _, row in test_df.iterrows():
            x, y = row['x'], row['y']
            ideal_row = ideal_df[ideal_df['x'] == x]
            if ideal_row.empty:
                continue
            for func in self.best_matches:
                max_dev = abs(ideal_df[func] - y).max()
                deviation = abs(ideal_row[func].values[0] - y)
                if deviation <= max_dev * np.sqrt(2):
                    mapped.append({
                        'x': x,
                        'y': y,
                        'delta_y': round(deviation, 4),
                        'ideal_func': func
                    })
                    break
        mapped_df = pd.DataFrame(mapped)
        print("\nSample of Mapped Test Data:")
        print(mapped_df.head(10))
        return mapped_df

    def visualize_training_data(self, train_df):
        """
        Visualize the training dataset using Bokeh.
        """
        output_file("training_data_plot.html")
        p = figure(title="Training Data Visualization", x_axis_label="X", y_axis_label="Y", width=900, height=500)
        for col in train_df.columns[1:]:
            p.line(train_df['x'], train_df[col], line_width=2, legend_label=col)
        p.legend.location = "top_left"
        save(p)

    def visualize_results(self, train_df, ideal_df, test_df, mapped_df):
        """
        Generate a Bokeh plot showing:
        - Ideal functions
        - Test data
        - Deviation lines connecting test points and closest ideal function values
        """
        output_file("deviation_lines_plot.html")
        p = figure(title="Mapped Test Data and Ideal Functions", x_axis_label="X", y_axis_label="Y", width=900, height=500)
        colors = ["blue", "orange", "green", "red"]
        for i, func in enumerate(self.best_matches[:4]):
            p.line(ideal_df['x'], ideal_df[func], line_width=2, color=colors[i], legend_label=f"Ideal {func}")
        p.scatter(test_df['x'], test_df['y'], size=6, color="black", legend_label="Test Data")
        for _, row in mapped_df.iterrows():
            x = row['x']
            y_test = row['y']
            func = row['ideal_func']
            y_ideal = ideal_df.loc[ideal_df['x'] == x, func].values[0]
            p.line([x, x], [y_test, y_ideal], line_color="gray", line_dash="dashed", line_width=1)
        p.legend.location = "top_right"
        save(p)

class TestAnalyzer(unittest.TestCase):
    def setUp(self):
        self.anal = Analyzer()
        self.train = pd.DataFrame({'x': [0,1,2], 'y1': [1,2,3], 'y2': [2,3,4]})
        self.ideal = pd.DataFrame({'x': [0,1,2], 'i1': [1.1,2.1,3.1], 'i2': [5,6,7]})
        self.test = pd.DataFrame({'x': [0,1,2], 'y': [1.05,2.05,3.05]})

    def test_best_func_selection(self):
        self.anal.select_best_ideal_functions(self.train, self.ideal)
        self.assertEqual(self.anal.best_matches, ['i1', 'i1'])

    def test_test_mapping_logic(self):
        self.anal.best_matches = ['i1', 'i1']
        out = self.anal.map_test_data(self.test, self.ideal)
        self.assertEqual(len(out), 3)
        self.assertIn('delta_y', out.columns)

if __name__ == "__main__":
    try:
        analyzer = Analyzer()
        train = analyzer.load_csv("train.csv")
        ideal = analyzer.load_csv("ideal.csv")
        test = analyzer.load_csv("test.csv")

        analyzer.save_to_db(train, "training_data")
        analyzer.save_to_db(ideal, "ideal_functions")
        analyzer.save_to_db(test, "test_data")

        analyzer.select_best_ideal_functions(train, ideal)
        mapped = analyzer.map_test_data(test, ideal)

        analyzer.save_to_db(mapped, "mapped_test_data")
        mapped.to_csv("mapped_results.csv", index=False)

        analyzer.visualize_training_data(train)
        analyzer.visualize_results(train, ideal, test, mapped)

        print("\nExecution completed successfully. Output files generated.")

    except DataProcessingError as e:
        print(f"[Data Error] {e}")
    except Exception as ex:
        print(f"[Unexpected Error] {ex}")

    unittest.main(argv=[''], verbosity=2, exit=False)
