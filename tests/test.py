"""Tests for 'zeitpy' package."""

# Imports
import zeitpy as zp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unittest
from unittest.mock import patch


class TestZeit(unittest.TestCase):    
    # Creating sample data for testing    
    def setUp(self):    
        data = np.random.randint(22500, 1e6, 100)
        index = pd.date_range('2024-09-07', periods = len(data), freq = 'D')
        self.data = pd.Series(data = data, index = index, name = 'sales')
        self.df = pd.DataFrame(data = {'date': index, 'sales': data})
        
    # Class initialization test for Pandas series as input
    def test_init_series(self):
        zo = zp.Zeit(self.data)
        self.assertIsInstance(zo.data, pd.Series)
        self.assertIsInstance(zo.data.index, pd.DatetimeIndex)
        self.assertEqual(len(zo.data), 100)
        
    # Class initialization test for DataFrame inputs
    def test_init_dataframe(self):    
        zo = zp.Zeit(self.df, date_format = '%Y-%m-%d', date_col = 'date', data_col = 'sales')
        self.assertIsInstance(zo.data, pd.Series)
        self.assertIsInstance(zo.data.index, pd.DatetimeIndex)
        self.assertEqual(len(zo.data), 100)
        
    # Mocking the 'inject' method with a csv file as input
    @patch('zeitpy.Zeit.inject')
    def test_inject_from_csv_file(self, mock_inject):
        mock_inject.return_value = self.data
        zo = zp.Zeit(
            'fake_dataset.csv', date_format = '%Y-%m-%d', 
            date_col = 'date', data_col = 'sales'
            )
        self.assertIsInstance(zo.data, pd.Series)
        self.assertIsInstance(zo.data.index, pd.DatetimeIndex)
        self.assertEqual(len(zo.data), 100)
        mock_inject.assert_called_once()
    
    # Testing 'seasonal_decomposition' method  
    def test_seasonal_decomposition(self):
        zo = zp.Zeit(self.data)
        with patch.object(plt, 'show') as mock_show:
            zo.seasonal_decomposition()
            mock_show.assert_called_once()
     
    # Testing the method for trend analysis
    def test_trend(self):
        zo = zp.Zeit(self.data)
        zo.data.index = zo.data.index.astype(str)
        with patch.object(plt, 'show') as mock_show:
            zo.trend(30)
            mock_show.assert_called_once()
    
    # Test for 'periodogram' method
    def test_periodogram(self):
        zo = zp.Zeit(self.data)
        with patch.object(plt, 'show') as mock_show:
            zo.periodogram()
            mock_show.assert_called_once()
            
    # Testing the method for displaying seasonal plots        
    def test_seasonal_plots(self):
        zo = zp.Zeit(self.data)
        with patch.object(plt, 'show') as mock_show:
            zo.seasonal_plots(period = 'year', freq = 'month', ylabel = 'sales')
            mock_show.assert_called_once()       
    
    # Test for Augmented Dickey-Fuller test
    def test_adfuller_test(self):
        zo = zp.Zeit(self.data)
        with patch('builtins.print') as mock_print:
            zo.adfuller_test()
            self.assertEqual(mock_print.call_count, 3)

            
    # Testing the method for splitting data into train and test sets        
    def test_split_data(self):
        zo = zp.Zeit(self.data)
        train, test = zo.split_data()
        self.assertIsInstance(test, zp.Zeit)
        self.assertEqual(len(train.data), 80)
        self.assertEqual(len(test.data), 20)
        
    # Test for plotting correlograms
    def test_correlogram(self):
        zo = zp.Zeit(self.data)
        train, test = zo.split_data()
        with patch.object(plt, 'show') as mock_show:
            train.correlogram()
            mock_show.assert_called_once() 
            
    # Test for assessing forecasting performance      
    def test_evaluate(self):
        zo = zp.Zeit(self.data)
        _, test = zo.split_data()
        sarima_forecast, prophet_forecast = test.data + 1, test.data + .5
        forecast_results = [
            ('SARIMA', sarima_forecast), 
            ('Prophet', prophet_forecast)
            ]
        with patch('zeitpy.core.display') as mock_display:
            zo.evaluate(forecast_results, test.data)
            mock_display.assert_called_once()
        with patch('zeitpy.core.display') as mock_display:
            zo.evaluate(forecast_results, test.data, 'metrics')
            mock_display.assert_called_once()