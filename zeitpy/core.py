'''
        IMPORTING MODULES
        ------------
'''

import pandas as pd
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from IPython.display import display
from typing import Self
from scipy.signal import periodogram
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error, root_mean_squared_error, \
    mean_absolute_error, mean_absolute_percentage_error


class Zeit:  
    '''
    A class for computing main time series operations, including visualizations, statistical tests, 
    and forecasting performance evaluation.
    ''' 

    '''
    Implementation
    --------------
    '''  

    def __init__(
            self, dataset: pd.DataFrame | pd.Series | str, date_format: str = None, 
            date_col: str = None, data_col: str = None
            ):
        '''
        Initializes a Zeit object given a Pandas Series/DataFrame or csv file.

        Parameters and:
        ---------- 
        dataset : str
            The DataFrame, Series or csv file path containing the time series data.
        date_format : str
            The format of the "date_col" instances to be converted into datetime — Ex.: '20' 
            should be passed as '%y' for annual observations; '10/27/2019' as '%m/%d/%Y', etc.
        date_col : str
            The column containing the time observations.
        data_col : str
            The column containing the values of the series.
        
        Example with 'data' attribute:
        ------------------------------
        In [1]: import zeitpy as zp
        ...: file_path = 'ibovespa_stock.csv'
        ...: zo = zp.Zeit(
        ...:     file_path, 
        ...:     date_format = '%Y-%m-%d', 
        ...:     date_col = 'Date', 
        ...:     data_col = 'Close'
        ...:     )
        ...: zo.data.head()
        Out [1]:
        2018-01-02    77891.0
        2018-01-03    77995.0
        2018-01-04    78647.0
        2018-01-05    79071.0
        2018-01-08    79379.0
        Name: Close, dtype: float64
        '''  
        # Class attributes
        self.dataset = dataset
        self.format = date_format
        self.ts_attr = date_col, data_col
        self.data = self.inject()

    # Class methods
    def inject(self) -> pd.Series:
        '''
        Transforms any time series data whether it is a Pandas Series/DataFrame or csv file into a Pandas
        time series.

        Returns:
        -------
        pd.Series
            The Pandas time series.
        
        Example:
        --------
        In [2]: zo.inject()
        '''
        date_col, data_col = self.ts_attr
        if self.format is None:
            if isinstance(self.dataset.index,  pd.DatetimeIndex) and isinstance(self.dataset,  pd.Series):
                ts = self.dataset.copy()
            else:
                raise TypeError(
                    'For series with indexes not in datetime format, "date_format" field must ' \
                    'not be None'
                                )        
        else:        
            if isinstance(self.dataset, pd.Series):         
                    ts = self.dataset.copy()
                    ts.index = pd.to_datetime(ts.index, format = self.format)
                    ts.index.name = None
            elif isinstance(self.dataset, pd.DataFrame):            
                if (data_col is None) or (date_col is None):
                    raise TypeError('For DataFrames, either "data_col" or "date_col" must not be "None"')
                else:            
                    df = self.dataset.copy()
                    ts = pd.Series(
                        data = df[data_col].values, 
                        index = pd.to_datetime(df[date_col], format = self.format),
                        name = data_col
                        )
            elif isinstance(self.dataset, str):
                if self.dataset.endswith('.csv'):
                    df = pd.read_csv(self.dataset)
                    ts = pd.Series(
                        data = df[data_col].values, 
                        index = pd.to_datetime(df[date_col], format = self.format),
                        name = data_col
                        )
                    ts.index.name = None
                else:
                    raise ValueError('"data" requires a Pandas DataFrame/Series or csv file path.')                        
        return ts
    
    
    def seasonal_decomposition(self, model: str = 'additive', period: int = 12, **plot_args) -> None:
        '''
        Presents the seasonal decomposition of the time series using moving averages.

        Parameters:
        ----------        
        model : str, optional
            The type of seasonal decomposition, whether it is Additive ('additive') or Multiplicative 
            ('multiplicative'). Default is 'additive'. 
        period : int, optional
            Period of the series (e.g., 1 for annual, 4 for quarterly, etc). Default is 12.

        Returns:
        -------
        None
        
        Example:
        --------
        In [3]: zo.seasonal_decomposition()
        '''
        # Decomposing the time series
        ts = self.data.copy()
        ts.name = None # Removing the series name to not be set as the plot title
        decomposition_result = seasonal_decompose(ts, model, period = period)
        
        # Setting up the decomposition plot 
        title = plot_args['title'] if 'title' in plot_args else model.title() + ' Decomposition'
        rotation = plot_args['rotation'] if 'rotation' in plot_args else 45
        fontsize = plot_args['fontsize'] if 'fontsize' in plot_args else 12
        
        # Displaying the plot
        decomposition_result.plot().suptitle(title, fontsize = fontsize)
        plt.xticks(rotation = rotation)
        plt.tight_layout()
        plt.show()
        
        
    def trend(self, window: int = 52, **plot_args) -> None:
        '''
        Plots the original time series and its trend using moving average.

        Parameters:
        ----------
        window : int, optional
            Number of periods to calculate the moving average. Default is 52.

        Returns:
        -------
        None
        
        Example:
        --------
        In [4]: zo.trend()
        '''
        # Series and main parameters of plot        
        ts = self.data
        ts_rolling = ts.rolling(window = window).mean().dropna()
        rotation = plot_args['rotation'] if 'rotation' in plot_args else 45
        ylabel = plot_args['ylabel'] if 'ylabel' in plot_args else ts.name
        title = plot_args['title'] if 'title' in plot_args else 'Trend Analysis'            
        
        # Plotting the series and trend      
        plt.plot(ts, label = 'Original Data')
        plt.plot(ts_rolling, label = 'Moving Average')
        plt.title(title)
        plt.xticks(rotation = rotation)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()        
    
    
    def periodogram(
            self, detrend: str = 'linear', ax: mpl.axes._axes.Axes = None,
            fs: int = 365, color: str = 'brown'
            ) -> None:     
        '''
        Displays the periodogram of the given time series data.

        The periodogram helps identify the dominant frequencies in the time series, providing insights
        into its periodic components.

        Parameters:
        ----------
        detrend : str, optional
            The detrending method to use. Options include 'linear' (default) or 'constant'. Detrending
            removes trends from the data before calculating the periodogram.

        ax : matplotlib.axes._axes.Axes, optional
            The Axes object on which to plot the periodogram. If None, a new figure and axes are created.

        fs : int, optional
            The sampling frequency of the data. Default is 365, which is suitable for daily data over a
            year.

        color : str, optional
            The color of the plot line. Default is 'brown'.

        Returns:
        -------
        None
            It displays the periodogram plot and does not return any value.
        
        Example:
        --------
        In [5]: zo.periodogram()
        '''           
        # Time series and Number of observations
        ts, nobs = self.data, len(self.data)
     
        # Computing frequencies and spectrum
        frequencies, spectrum = periodogram(
            ts, fs, window = 'boxcar', detrend = detrend, scaling = 'spectrum'
            )
        
        # Frequency adjustment
        freqs = [1, 2, 3, 4, 6, 12, 26, 52, 104]
        freqs_labels = [
            'Annual (1)', 'Semiannual (2)', 'Triannual (3)',
           'Quarterly (4)', 'Bimonthly (6)', 'Monthly (12)',
           'Biweekly (26)', 'Weekly (52)', 'Semiweekly (104)'
           ]

        # Plotting the periodogram
        if ax is None:
            _, ax = plt.subplots()        
        ax.step(frequencies, spectrum, color = color)
        ax.set_xscale('log')
        ax.set_xticks(freqs)
        ax.set_xticklabels(freqs_labels, rotation = 55, fontsize = 11)
        ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0))
        ax.set_ylabel('Variance', fontsize = 12)
        ax.set_title(f'Periodogram ({nobs} Observations)')
        plt.show()


    def seasonal_plots(
            self, period: str, freq: str, ax: mpl.axes._axes.Axes = None, 
            title: str = None, xlabel: str = None, ylabel: str = None
            ) -> None:   
        '''
        Displays seasonal plots for visualizing seasonal patterns in time series data.

        This method generates line plots that illustrate the variations in the time series across 
        specified seasonal periods, helping to identify trends and seasonal effects.

        Parameters:
        ----------
        period : str
            The seasonal period. This is used to group the data for the seasonal plot. Options 
            include "year", "month", "week of year", "day of month", "day of week", and "day 
            of year".

        freq : str
            The frequency of the seasonality. This typically denotes the time interval at which the
            seasonal effect is observed (shares the same options with "period" parameter).

        ax : matplotlib.axes._axes.Axes, optional
            The Axes object on which to plot the seasonal plot. If None, a new figure and axes are
            created.

        title : str, optional
            Title of the plot. If None, a default title will be generated.

        xlabel : str, optional
            Label for the x-axis. If None, the frequency name will be used. If 'hide', the label 
            will not be shown.

        ylabel : str, optional
            Label for the y-axis. If None, the name of the series will be used.

        Returns:
        -------
        None

        Example:
        --------
        In [6]: zo.seasonal_plots('year', 'month', xlabel = 'hide')
        '''
        # Extracting date elements from the index
        ts = self.data
        data = ts.to_frame()
        data['year'] = ts.index.year
        data['month'] = ts.index.strftime('%b')
        data['week of year'] = ts.index.isocalendar().week.astype(int)  
        data['day of month'] = ts.index.day    
        data['day of week'] = ts.index.strftime('%a')
        data['day of year'] = ts.index.dayofyear
    
        # Plotting seasonal effects
        if ax is None:
            _, ax = plt.subplots()
        palette = sns.color_palette('rocket', n_colors = data[period].nunique())
        ax = sns.lineplot(
            data = data, x = freq, y = data.columns[0], hue = period, ax = ax, palette = palette
            )
    
        # Setting options for the plot title
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Seasonal Plot ({period}/{freq})')
    
        # Setting label options for x axis    
        if xlabel is None:
            plt.xlabel(freq)    
        elif xlabel == 'hide':
            plt.xlabel('')    
        else:
            plt.xlabel(xlabel)
    
        # Setting label options for y axis     
        if ylabel is None:
            plt.ylabel(data.columns[0])    
        else:
            plt.ylabel(ylabel)
    
        plt.legend(fontsize = 8, loc = 'best')
        plt.show()            


    def lags(self, n_lags: int = 10, title: str = 'Lag Plots') -> None:
        '''
        Generates lag plots to visualize the autocorrelation of a time series.

        This method creates scatter plots of the time series against its lagged versions, helping 
        to assess the degree of autocorrelation at different lags. Each plot includes a regression
        line to indicate the strength of the relationship between the original series and its 
        lagged version.

        Parameters:
        ----------        
        n_lags : int, optional
            Number of lag plots to generate. Default is 10, creating lag plots from 1 to 10 lags.

        title : str, optional
            Title of the overall figure. Default is 'Lag Plots'.

        Returns:
        -------
        None
            This method displays the lag plots and does not return any value.

        Example:
        --------
        In [7]: zo.lags()
        '''
        ts = self.data    
        fig, axes = plt.subplots(
            2, n_lags // 2, figsize = (10, 6), sharex = False, sharey = True, dpi = 240
            )
        for i, ax in enumerate(axes.flatten()[:n_lags]):
            lag_data = pd.DataFrame({'x': ts, 'y': ts.shift(i + 1)}).dropna()

            x, y = lag_data['x'], lag_data['y']
            slope, intercept, r, p_value, std_err = linregress(x, y)
            reg_line = [(slope * k) + intercept for k in x]   
            ax.scatter(x, y, c = 'k', alpha = .6)
            ax.plot(x, reg_line, color = 'm', label = f'{r**2:.2f}')
            ax.set_title(f'Lag {i + 1}')
            ax.legend()
            ax.grid(True) 
        plt.tight_layout()
        plt.suptitle(title, y = 1.05)
        plt.show()


    def adfuller_test(self) -> None:
        '''
        Checks the stationarity of a time series using the Augmented Dickey-Fuller Test.

        This method performs the Augmented Dickey-Fuller (ADF) test to assess whether a given time series 
        is stationary or contains a unit root. It prints the test results and displays key statistics for 
        evaluation.

        Returns:
        -------
        None
            Prints the results of the ADF test and displays a DataFrame containing the test statistics.

        Example:
        --------
        In [8]: zo.adfuller_test()
        '''
        # Performing the Augmented Dickey-Fuller Test    
        adft = adfuller(self.data, autolag = 'AIC')
         
        # DataFrame to store the results of the test    
        scores_df = pd.DataFrame({'Scores': [adft[0], adft[1], 
                                             adft[2], adft[3], adft[4]['1%'],
                                             adft[4]['5%'], adft[4]['10%']]},
                                 index = ['Test Statistic',
                                          'p-value', 'Lags Used',
                                          'Observations', 
                                          'Critical Value (1%)',
                                        'Critical Value (5%)',
                                        'Critical Value (10%)'])
         
        # Printing the result of the test    
        if adft[1] > 0.05 and abs(adft[0]) > adft[4]['5%']:
            print('\033[1mThis series is not stationary!\033[0m')
        else:
            print('\033[1mThis series is stationary!\033[0m')             
        print('\nResults of Dickey-Fuller Test\n' + '=' * 29)           
        display(scores_df)
        
        
    def split_data(self, train_proportion: float = .8) -> tuple[Self, Self]:
        '''
        Splits a Series into trainining and testing sets.

        This method divides the provided Series into two subsets: a training set and a testing set,
        based on a specified proportion. The training set contains the first portion of the data, while
        the testing set contains the remaining data.

        Parameters:
        ----------
        train_proportion : float, optional
            The proportion of the data to include in the training set. Must be between 0 and 1.
            Default is 0.8, meaning 80% of the data will be used for training.

        Returns:
        -------
        tuple
            A tuple containing two Zeit objects:
            - train_set (Zeit): The Zeit object of training set.
            - test_set (Zeit): The Zeit object of testing set.

        Example:
        --------
        In [9]: train, test = zo.split_data()
        '''
        train_size = round(len(self.data) * train_proportion)
        train_set = self.data.iloc[:train_size]
        test_set = self.data.iloc[train_size:]   
        splitted_data = Zeit(train_set), Zeit(test_set)
        return splitted_data


    def correlogram(self, lags: int = 6, ACF: bool = True, PACF: bool = True) -> None:   
        '''
        Plots the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) of a time
        series.

        This method generates correlograms to help assess the autocorrelation and partial autocorrelation 
        of the time series, which are crucial for identifying the orders of Autoregressive (AR) and Moving
        Average (MA) components in time series models.

        Parameters:
        ----------
        lags : int, optional
            The number of lags to include in the ACF and PACF plots. Default is 6.

        ACF : bool, optional
            Whether to plot the Autocorrelation Function. Default is True.

        PACF : bool, optional
            Whether to plot the Partial Autocorrelation Function. Default is True.

        Returns:
        -------
        None
            Displays the ACF and PACF plots and does not return any value.

        Example:
        --------
        In [10]: zo.correlogram(lags = 8)
        '''     
        # Setting the plots for correlograms
        fig, axes = plt.subplots(1, 2, figsize = (15, 6))
        
        # ACF Plot
        if ACF:       
            plot_acf(self.data, lags = lags, ax = axes[0])
            axes[0].set_title('ACF Plot')
    
        # PACF Plot    
        if PACF:
            plot_pacf(self.data, lags = lags, ax = axes[1])  
            axes[1].set_title('PACF Plot')  
        plt.tight_layout()
        plt.show()


    def evaluate(
            self, forecast_models: list[tuple[str, pd.Series]], test_data: pd.Series,
            view: str = 'results'
            ) -> None:
        '''
        Assesses the performance of forecast models using various metrics.

        This method compares the forecasts from multiple models against a test dataset by calculating 
        common performance metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), 
        Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE). It can display either 
        the performance metrics or a comparison of the actual vs. predicted values.

        Parameters:
        ----------
        forecast_models : list of tuples
            A list containing tuples where each tuple consists of a model name (str) and the
            corresponding forecasted data (pandas.Series).

        test_data : pandas.Series
            The actual test dataset containing the true values for comparison.

        view : str, optional
            Specifies the output view. Options are 'metrics' to display model performance metrics or 
            'results' to display a comparison of actual vs. predicted values. Default is 'results'.

        Returns:
        -------
        None
            Displays the performance metrics or results comparison and does not return any value.

        Example:
        --------
        In [11]: forecast_models = [
        ...:    ('SARIMA', sarima_forecast),
        ...:    ('Prophet', prophet_forecast)
        ...:    ]
        ...: test_data = test.data
        ...: zo.evaluate(forecast_models, test_data)
        '''    
        # Inserting the test set to form the first column of the comparison DataFrame
        comparison_df = pd.DataFrame(
            data = test_data.values, 
            index = test_data.index, 
            columns = ['Actual']
            )
        
        for model, forecast_data in forecast_models:
            # Computing the performance according to different parameters
            mse = mean_squared_error(test_data, forecast_data)
            rmse = root_mean_squared_error(test_data, forecast_data)
            mae = mean_absolute_error(test_data, forecast_data)
            mape = mean_absolute_percentage_error(test_data, forecast_data)
            metrics_scores = [mse, rmse, mae, mape]
            metrics = ['MSE', 'RMSE', 'MAE', 'MAPE']
            temp_df = pd.DataFrame({model: metrics_scores}, index = metrics)
            metrics_df = pd.concat([temp_df], axis = 1)
            
            # Adding the forecast to the comparison DataFrame
            comparison_df[model] = forecast_data 
            
        # Option to display the transposed performance metrics DataFrame 
        if view == 'metrics':
            display(metrics_df.T)
            
        # Option to display the transposed comparison DataFrame 
        elif view == 'results':
            display(comparison_df.head(10).T)
        else:
            raise ValueError(
                "This option is not available! :(\nPlease choose whether you want to display the"
                " performance assessment inserting \033[1m'metrics'\033[0m or \033[1m'results'\033[0m "
                "in case of forecast comparison."
                )