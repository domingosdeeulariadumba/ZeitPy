# ZeitPy

ZeitPy is a package for time series analysis. Its main purpose is to abstract some vital operations to analyse time series data. At its core is the `Zeit` class, which provides attributes and methods for initializing a (Pandas) time series, performing Exploratory Data Analysis (Augmented Dickey-Fuller test, visualize periodograms, seasonal, and lag plots, etc.), forecasting and performance assessment.

---

## Class Overview ℹ️

### `Zeit`

#### **Class Initialization**
```python
In [1]: import zeitpy as zp
...: zo = zp.Zeit('sales_luanda.csv', date_format = '%Y-%m-%d', date_col = 'date', data_col = 'sales')
```

- **dataset**: the DataFrame, Series or csv file path containing the time series data.
- **date_format**: the format of the "date_col" instances to be converted into datetime.
- **date_col**: the column containing the time observations.
- **data_col**: the column containing the values (in case of csv files or DataFrames).

#### **Attributes and Methods**

- **`data`**
   - this attribute retrieves the time series data wrapped by the Zeit object.

- **`seasonal_decomposition(model: str = 'additive', period: int = 12, **plot_args) -> None`**
   - A method for plotting the seasonal decomposition of the time series using moving averages.
   - Parameters:
     - `model`: the type of seasonal decomposition
     - `period`: period of the series (12 for monthly data, 1 for annual, etc.)

---

## Usage ⚙️📦

### How to get the package?

#### Installation via PyPI
**`pip install zeitpy`**

#### Installation of source distribution (.tar.gz)
**`pip install path/to/zeitpy-0.1.1.tar.gz`**

#### Installation of pre-built distribution (.whl)
**`pip install path/to/zeitpy-0.1.1-py3-none-any.whl`**

#### Cloning the package repository
**`git clone https://github.com/domingosdeeulariadumba/ZeitPy.git`**


### Importing the package

```python
In [1]: import zeitpy as zp
```

### Example: Injecting a dataset and accessing the first five records of the time series
```python
In [2]: zo = zp.Zeit('sales_luanda.csv', date_format = '%Y-%m-%d', date_col = 'date', data_col = 'sales')
...: zo.data.head()
Out[2]: 
2024-09-07     86662
2024-09-08    449329
2024-09-09     64041
2024-09-10    420328
2024-09-11    351528
Freq: D, Name: sales, dtype: int32
```
📑 You can view all operations provided by the `Zeit` class implemented [here](https://github.com/domingosdeeulariadumba/ZeitPy/examples.ipynb).

---

## License ⚖️

This project is licensed under the MIT `LICENSE`.

---

## Issues and Contributions 🧱

Feel free to [submit any issue](https://github.com/domingosdeeulariadumba/ZeitPy/issues) you may find in this package or recommend additional features not already listed [here](https://github.com/domingosdeeulariadumba/ZeitPy/TODO.md). If you find this useful, please fork the repository, create a feature branch, and submit a pull request.

---
## Connect with me 🌐

<img src = 'https://i.postimg.cc/t4vNmLB0/linktree-icon.png' width = '25' height = '25'/>  **[/domingosdeeulariadumba](https://linktr.ee/domingosdeeulariadumba)**

<img src = 'https://i.postimg.cc/wj3w1mjG/kofi-icon.png' width = '25' height = '25'/>  **[/domingosdeeulariadumba](https://ko-fi.com/domingosdeeulariadumba)**

<img src = 'https://i.postimg.cc/W1178266/linkedin-icon.png' width = '25' height = '25'/>  **[/domingosdeeulariadumba](https://linkedin.com/in/domingosdeeulariadumba/)**