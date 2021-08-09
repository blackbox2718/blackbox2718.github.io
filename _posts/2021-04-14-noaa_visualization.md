---
title: Data Visualization with NOAA Climate Data Set
date: 2021-04-14 19:57:50 -0700
math: true
categories: [PIC16B-blog]
tags: [sql, data-visualization, climate-change]
---
In this assignment, we will be working with the [NOAA](https://www.ncdc.noaa.gov/data-access/land-based-station-data/land-based-datasets/global-historical-climatology-network-monthly-version-4) climate data set to create some interesting visualizations.

## $$\S 1.$$ Database
There are three tables that we need to create within a database which are *stations*, *countries*, and *temperatures*. In order to generate a database, we need the help from ``sqlite3`` which is a module that allows us to conveniently create and query databases.


```python
import sqlite3

# make a connection
conn = sqlite3.connect("noaa.db")
```

First, let's access the stations data set with ``pandas``.


```python
import pandas as pd

url = "https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/noaa-ghcn/station-metadata.csv"
stations = pd.read_csv(url)
stations.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: center;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th style="text-align: center">ID</th>
      <th style="text-align: center">LATITUDE</th>
      <th style="text-align: center">LONGITUDE</th>
      <th style="text-align: center">STNELEV</th>
      <th style="text-align: center">NAME</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACW00011604</td>
      <td>57.7667</td>
      <td>11.8667</td>
      <td>18.0</td>
      <td>SAVE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AE000041196</td>
      <td>25.3330</td>
      <td>55.5170</td>
      <td>34.0</td>
      <td>SHARJAH_INTER_AIRP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AEM00041184</td>
      <td>25.6170</td>
      <td>55.9330</td>
      <td>31.0</td>
      <td>RAS_AL_KHAIMAH_INTE</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AEM00041194</td>
      <td>25.2550</td>
      <td>55.3640</td>
      <td>10.4</td>
      <td>DUBAI_INTL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AEM00041216</td>
      <td>24.4300</td>
      <td>54.4700</td>
      <td>3.0</td>
      <td>ABU_DHABI_BATEEN_AIR</td>
    </tr>
  </tbody>
</table>
</div>



Now, we can add this table to our database through the connection ``conn`` that we defined earlier.


```python
stations.to_sql("stations", conn, if_exists = "replace", index = False)
```

We proceed similarly with the country data set.


```python
url = "https://raw.githubusercontent.com/mysociety/gaze/master/data/fips-10-4-to-iso-country-codes.csv"
countries = pd.read_csv(url)
# change the column "Name" for easier reference later on
countries = countries.rename(columns = {"Name": "Country"})
countries.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: center;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: center;">
      <th></th>
      <th style="text-align: center">FIPS 10-4</th>
      <th style="text-align: center">ISO 3166</th>
      <th style="text-align: center">Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AF</td>
      <td>AF</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AX</td>
      <td>-</td>
      <td>Akrotiri</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AL</td>
      <td>AL</td>
      <td>Albania</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AG</td>
      <td>DZ</td>
      <td>Algeria</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AQ</td>
      <td>AS</td>
      <td>American Samoa</td>
    </tr>
  </tbody>
</table>
</div>



To avoid potential issues with the spacing in the column names, let's rename them in a more friendly and convenient format to work with.


```python
# we don't need ISO column, so we will drop it
countries = countries.drop(["ISO 3166"], axis = 1)
# rename the column with name with space in it
countries = countries.rename(columns = {"FIPS 10-4": "FIPS"})
countries.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th style="text-align: center">FIPS</th>
      <th style="text-align: center">Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AF</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AX</td>
      <td>Akrotiri</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AL</td>
      <td>Albania</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AG</td>
      <td>Algeria</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AQ</td>
      <td>American Samoa</td>
    </tr>
  </tbody>
</table>
</div>



We add the table to the database as usual.


```python
countries.to_sql("countries", conn, if_exists = "replace", index = False)
```

For the temperature data set, we will load it from our local directory directly instead of accessing it through an url as the data from the [source](https://github.com/PhilChodrow/PIC16B/tree/master/datasets/noaa-ghcn/decades) is already splitted/categorized into decades.


```python
temperatures = pd.read_csv("temps.csv")
temperatures.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th style="text-align: center">ID</th>
      <th>Year</th>
      <th>VALUE1</th>
      <th>VALUE2</th>
      <th>VALUE3</th>
      <th>VALUE4</th>
      <th>VALUE5</th>
      <th>VALUE6</th>
      <th>VALUE7</th>
      <th>VALUE8</th>
      <th>VALUE9</th>
      <th>VALUE10</th>
      <th>VALUE11</th>
      <th>VALUE12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>-89.0</td>
      <td>236.0</td>
      <td>472.0</td>
      <td>773.0</td>
      <td>1128.0</td>
      <td>1599.0</td>
      <td>1570.0</td>
      <td>1481.0</td>
      <td>1413.0</td>
      <td>1174.0</td>
      <td>510.0</td>
      <td>-39.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACW00011604</td>
      <td>1962</td>
      <td>113.0</td>
      <td>85.0</td>
      <td>-154.0</td>
      <td>635.0</td>
      <td>908.0</td>
      <td>1381.0</td>
      <td>1510.0</td>
      <td>1393.0</td>
      <td>1163.0</td>
      <td>994.0</td>
      <td>323.0</td>
      <td>-126.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACW00011604</td>
      <td>1963</td>
      <td>-713.0</td>
      <td>-553.0</td>
      <td>-99.0</td>
      <td>541.0</td>
      <td>1224.0</td>
      <td>1627.0</td>
      <td>1620.0</td>
      <td>1596.0</td>
      <td>1332.0</td>
      <td>940.0</td>
      <td>566.0</td>
      <td>-108.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACW00011604</td>
      <td>1964</td>
      <td>62.0</td>
      <td>-85.0</td>
      <td>55.0</td>
      <td>738.0</td>
      <td>1219.0</td>
      <td>1442.0</td>
      <td>1506.0</td>
      <td>1557.0</td>
      <td>1221.0</td>
      <td>788.0</td>
      <td>546.0</td>
      <td>112.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACW00011604</td>
      <td>1965</td>
      <td>44.0</td>
      <td>-105.0</td>
      <td>38.0</td>
      <td>590.0</td>
      <td>987.0</td>
      <td>1500.0</td>
      <td>1487.0</td>
      <td>1477.0</td>
      <td>1377.0</td>
      <td>974.0</td>
      <td>31.0</td>
      <td>-178.0</td>
    </tr>
  </tbody>
</table>
</div>



At this point, there are some data manipulation that we have to do so that the querying process is easier to deal with later on.


```python
# set index
temperatures = temperatures.set_index(keys = ["ID", "Year"])
# stack data based on months
temperatures = temperatures.stack()
# reset index
temperatures = temperatures.reset_index()
# let's take a look
temperatures.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th style="text-align: center">ID</th>
      <th style="text-align: center">Year</th>
      <th style="text-align: center">level_2</th>
      <th style="text-align: center">0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>VALUE1</td>
      <td>-89.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>VALUE2</td>
      <td>236.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>VALUE3</td>
      <td>472.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>VALUE4</td>
      <td>773.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>VALUE5</td>
      <td>1128.0</td>
    </tr>
  </tbody>
</table>
</div>



Now, looking at the table above, we can immediately observe that the name of the "month" and "temperature" columns do not make much sense, so let's change them and also the value within the month column as well. In addition, since the temperature is 100x of its unit degree Celsius, we divide the column accordingly.


```python
# rename some of the columns
temperatures = temperatures.rename(columns = {"level_2": "Month", 0: "Temp"})
# change VALUE (of month) to integer represent month in a year
temperatures["Month"] = temperatures["Month"].str[5:].astype(int)
# add a FIPS column for querying database later on
temperatures["FIPS"] = temperatures["ID"].str[:2]
# convert temperature to (C)
temperatures["Temp"] = temperatures["Temp"] / 100
temperatures.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th style="text-align: center">ID</th>
      <th style="text-align: center">Year</th>
      <th style="text-align: center">Month</th>
      <th style="text-align: center">Temp</th>
      <th style="text-align: center">FIPS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>1</td>
      <td>-0.89</td>
      <td>AC</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>2</td>
      <td>2.36</td>
      <td>AC</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>3</td>
      <td>4.72</td>
      <td>AC</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>4</td>
      <td>7.73</td>
      <td>AC</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>5</td>
      <td>11.28</td>
      <td>AC</td>
    </tr>
  </tbody>
</table>
</div>



Things look pretty good here! We can now add this data set to our database as before.


```python
temperatures.to_sql("temperatures", conn, if_exists = "replace", index = False)
```

To verify whether our database does indeed contain the three data sets, we use a sql cursor, ``.cursor``, to interact with it.


```python
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print(cursor.fetchall())
```

    [('stations',), ('countries',), ('temperatures',)]


And that's what we exactly want to be outputted. Finally, as a safeguard, we close the connection to the database to avoid any undesirable/unexpected situations such as leak, etc.


```python
conn.close()
```

Let's now move on to the next part of this post...
## $$\S 2.$$ Query the Database
In this section, we will define a function in which it accepts some certain information that we need to query on the data base and return the corresponding Pandas dataframe with data from the three tables we have in the database.


```python
def query_climate_database(country, year_begin, year_end, month):
    """
    FUNCTION
    --------
    Generate a dataframe which provides us with information of stations,
    latitude, longitude, and average temperatures from a set of conditions
    set by the user
    
    PARAMETERS
    ----------
    country   : name of a country (string)
    year_begin: starting year (int)
    year_end  : ending year (int)
    month     : the only month included in the dataframe
    
    RETURN
    ------
    A dataframe filled with info specified by the user along with the
    corresponding data/columns (station, latitude, longitude, country,
    year, month, temperature)
    """
    
    cmd = \
    f"""
    SELECT S.name, S.latitude, S.longitude, C.country, T.year, T.month, T.temp
    FROM temperatures T
    LEFT JOIN stations S ON T.id = S.id
    LEFT JOIN countries C ON T.fips = C.fips
    WHERE C.country = ?
        AND T.year BETWEEN ? AND ?
        AND month = ?
    """
    # remember to open the connection before accessing the database
    conn = sqlite3.connect("noaa.db")
    # query the database
    df = pd.read_sql_query(cmd, conn, params=(country, year_begin, year_end, month))
    # close the connection
    conn.close()
    return df
```


For example, let's say we want to take a look at India from 1980 to 2020 in the month of January.


```python
query_climate_database(country = 'India', 
                       year_begin = 1980, 
                       year_end = 2020,
                       month = 1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th style="text-align: center">NAME</th>
      <th style="text-align: center">LATITUDE</th>
      <th style="text-align: center">LONGITUDE</th>
      <th style="text-align: center">Country</th>
      <th style="text-align: center">Year</th>
      <th style="text-align: center">Month</th>
      <th style="text-align: center">Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1980</td>
      <td>1</td>
      <td>23.48</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1981</td>
      <td>1</td>
      <td>24.57</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1982</td>
      <td>1</td>
      <td>24.19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>23.51</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1984</td>
      <td>1</td>
      <td>24.81</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3147</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>5.10</td>
    </tr>
    <tr>
      <th>3148</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1986</td>
      <td>1</td>
      <td>6.90</td>
    </tr>
    <tr>
      <th>3149</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1994</td>
      <td>1</td>
      <td>8.10</td>
    </tr>
    <tr>
      <th>3150</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1995</td>
      <td>1</td>
      <td>5.60</td>
    </tr>
    <tr>
      <th>3151</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1997</td>
      <td>1</td>
      <td>5.70</td>
    </tr>
  </tbody>
</table>
<p>3152 rows × 7 columns</p>
</div>



Fantastic! Our function does work as we expected.
## $$\S 3.$$ Geographic Scatter Plot

For this section, we will define a function which answers the following question:
> How does the average yearly change in temperature vary within a given country?

{::options parse_block_html="true" /}
<div class="gave-help">
First, we have to get a sense of how to compute the average yearly change in temperature of each station. Recall from PIC16A that there is a tool oftentimes used for such a task which is *linear regression*. Mathematically speaking, the coefficient of the line of best fit is in fact an estimate of the yearly temperature change.
</div>

```python
from sklearn.linear_model import LinearRegression
import numpy as np

def coef(df):
    """
    Estimate the coefficient, which is also the estimate of the average
    yearly change in temperature.
    -------------------------------
    Input: a pandas dataframe
    Ouput: the linear coefficient
    """
    
    # X needs to be a pandas dataframe
    X = df[["Year"]]
    # y needs to be a pandas series
    y = df["Temp"]
    LR = LinearRegression()
    LR.fit(X, y)
    return np.round(LR.coef_[0], 3)
```

To visually answer the main question, we will need the help from ``plotly`` to produce an interactive scatter plot which showcases the the yearly change of temperature at a specific time for each station within a given country.


{::options parse_block_html="true" /}
<div class="gave-help">
A demonstration of coding documentation and the use of ``<br />`` in long title that I suggested to my classmate.
```python
from plotly import express as px

def temperature_coefficient_plot(country, year_begin, year_end, month, min_obs, **kwargs):
    """
    FUNCTION
    --------
    Generate an interactive geographic scatter plot that shows the
    yearly change in temperature at a given time and contry specified
    by the user
    
    PARAMETERS
    ----------
    country   : name of the country (string)
    year_begin: starting year (int)
    year_end  : ending year(int)
    month     : month (int)
    min_obs   : minimum required number of years for a station (int)
    
    RETURN
    ------
    the produced interactive scatter plot
    """
    
    # collect the data
    df = query_climate_database(country, year_begin, year_end, month)
    # filter data that satisfies the minimum observation requirements
    df = df.groupby(["NAME"]).filter(lambda x: len(x) >= min_obs)
    # compute the average yearly change in temperature
    coefs = df.groupby(["NAME", "LATITUDE", "LONGITUDE", "Month"]).apply(coef)
    coefs = coefs.reset_index()
    # rename the column with info about the yearly rate
    coefs = coefs.rename(columns = {0: "Estimated Yearly<br>Change (C)"})
    # time to plot
    fig = px.scatter_mapbox(coefs,
                            lat = "LATITUDE",
                            lon = "LONGITUDE",
                            hover_name = "NAME",
                            title = "Estimates of Yearly Average Change of Temperature<br>in "
                                    + country + " stations from " + str(year_begin)
                                    + "-" + str(year_end) + " in " + month
                            color = "Estimated Yearly<br>Change (C)",
                            **kwargs)
    fig.update_layout(margin={"r":0,"t":55,"l":0,"b":0})
    return fig
```
</div>
{::options parse_block_html="false" /}

```python
# choose a colormap
color_map_1 = px.colors.sequential.Turbo

fig = temperature_coefficient_plot("India", 1980, 2020, 1, 
                                   min_obs = 10,
                                   zoom = 2.4,
                                   mapbox_style = "carto-positron",
                                   color_continuous_scale = color_map_1)
fig.show()
```
{% include hw1-fig1.html %}
<br />
Such a cool map we have produced! Observe that a majority of stations in India over this time period have a slight increase/decrease in yearly temperature. Nevertheless, there are still a small number of cases where the yearly change is very alarming, i.e. the temperature changes of these stations are significantly higher than their neighborhood. Now, let's do some even cooler visualization while exploring these data sets.
## $$\S 4.$$ Geographic Coordinates with 3D Scatter Plots
For this section of the post, I will try to construct a 3d scatter plot in an attempt to address the following question:
> How does temperature vary with respect to latitude and longitude (geographic coordinates) in a given year?

As this is a rather straightforward question to answer, we can certainly define a simple query function to get the desired data.


```python
def query_climate_database_3d(year, lat_min, lat_max, lon_min, lon_max):
    """
    Generate a pandas dataframe by querying the database with sql
    restricted to a certain set of conditions set by the user.
    ------------
    Input
    year   : year (int)
    lat_min: minimum latitude (float)
    lat_max: maximum latitude (float)
    lon_min: minimum longitude (float)
    lon_max: maximum longitude (float)
    
    Output
    A pandas dataframe that satisfies those conditions
    """
    cmd = \
    f"""
    SELECT S.latitude, S.longitude, T.temp, T.month
    FROM temperatures T
    LEFT JOIN stations S ON T.id = S.id
    WHERE T.year = ?
        AND S.latitude BETWEEN ? AND ?
        AND S.longitude BETWEEN ? AND ?
    """
    # open the connection
    conn = sqlite3.connect("noaa.db")
    # query the database
    df = pd.read_sql_query(cmd, conn, 
                           params=(year, lat_min, lat_max, lon_min, lon_max))
    # close the connection
    conn.close()
    return df
```

We're ready now to get started with implementing the plotting function.

{::options parse_block_html="true" /}
<div class="got-help">
I want to say thank to an anonymous friend for suggesting to add docstring to this function!
```python
def geo_temp_3d_plot(year, lat_min, lat_max, lon_min, lon_max, **kwargs):
    """
    FUNCTION
    --------
    Generate a 3D interactive plot demonstrating the relationship
    between geographic coordinates and temperatures with respect to
    each month in a year
    
    PARAMETERS
    ----------
    year    : desired year (int)
    lat_min : minimum latitude (float)
    lat_max : maximum latitude (float)
    lon_min : minimum longitude (float)
    lon_max : maximum longitude (float)
    **kwargs: additional arguments in px.scatter_3d()
    
    RETURN
    ------
    the interactive 3d figure generated by px.scatter_3d()
    """
    geo = query_climate_database_3d(year, lat_min, lat_max, lon_min, lon_max)
    fig = px.scatter_3d(geo,
                        x = "LATITUDE",
                        y = "LONGITUDE",
                        z = "Month",
                        labels = {"Temp": "Temperature (C)"},
                        color = "Temp",
                        **kwargs,)
    fig.update_layout(margin={"r":0,"t":25,"l":0,"b":15},
                      title = {
                          "text": "Temperatures with respect to Geographic Coordinates in "
                          + str(year),
                          "y": .95,
                          "x": .4,
                          "xanchor": "center",
                          "yanchor": "top",
                      },
                      title_font_size = 20,
                      title_font_color = "DarkBlue")
    return fig
```
</div>
{::options parse_block_html="false" /}

Let's test our function.


```python
color_map_2 = px.colors.diverging.Portland
fig = geo_temp_3d_plot(2000, -8, 8, -8, 8,
                 opacity = .7,
                 height = 530,
                 color_continuous_scale = color_map_2)
fig.show()
```
{% include hw1-fig2.html %}
<br />
From the plot, it's intuitive that the temperature is high from Jan to July and has a tendency to decrease rapidly onward from July to Dec which is certainly aligned with our common sense. With respect to the geographic coordinates, the lower the latitude the lower the temperature appears to be. In addition, it seems that in this specific plot, longitude does not have an observable effect with regard to temperature, though if we look closely enough, it seems like the temperature is cooler toward the two ends of the spectrum. For the last visualization of this blog post, let's change gear a bit and meet our old friend ``seaborn``.
## $$\S 5.$$ Time-Series Plot of a Station

For this section, we will utilize the power of relation plot and line plot from ``seaborn`` to tackle the following question
> Given any station in the world, how does the average temperature there range in a certain time period?

Let's now construct a function that would get us the data that we want.


```python
def query_station(country, station, year_begin, year_end):
    """
    Generate a panadas dataframe of a specified station for a specified
    period of time
    --------------
    Input
    country   : name of a country (string)
    station   : name of a station within that country (string)
    year_begin: beginning year (int)
    year_end  : ending year (int)
    
    Output
    A pandas dataframe that satisfies those conditions
    """
    b = pd.DataFrame()
    # open the connection
    conn = sqlite3.connect("noaa.db")
    # 12 months in a year
    for i in range(1, 13):
        a = query_climate_database(country, year_begin, year_end, i)
        b = pd.concat([b, a])
    b = b[b["NAME"] == station].sort_values(by=["Year", "Month"], ignore_index=True)
    # close the connection
    conn.close()
    return b
```

**Note**: Because of some technical errors with accessing the database using a newly defined function (for the third time), I cannot implement a new query in this section. Instead, I will be recycling ``query_climate_database()`` from the first plot and add some nuances to the function to serve the purpose.

At this point, we're fully prepared to construct a plotting function which would address the stated question.


```python
import seaborn as sns
sns.set_theme(style="dark")
def mult_line_plot_station(country, station, year_begin, year_end):
    """
    FUNCTION
    --------
    Construct a group of plots of a station's temperatures within
    a given period of time
    
    PARAMETERS
    ----------
    country   : name of a country to inspect (string)
    station   : name of a station in that country (string)
    year_begin: beginning year (int)
    year_end  : ending year (int)
    
    RETURN
    ------
    A facet grids that showcases temperatures of the station
    within the given time period
    """
    
    # get the data
    s = query_station(country, station, year_begin, year_end)
    # plot the relational plot for each year
    g = sns.relplot(data=s, x="Month", y="Temp", col="Year", hue="Year",
                    kind="line", palette="husl", linewidth=2.5, zorder=5,
                    col_wrap=3, height=2.2, aspect=1.6, legend=False)
    
    # plot the the other year in the background for comparison
    for year, ax in g.axes_dict.items():
        # denote year (bold) at the top right of the plot
        ax.text(.81, .9, year, transform=ax.transAxes, fontweight="bold")
        # Plot every year's time series in the background
        sns.lineplot(data=s, x="Month", y="Temp", units="Year",
                     estimator=None, color=".75", linewidth=1, ax=ax)

    # set x-tick labels
    ax.set_xticks(ax.get_xticks()[::2])

    # add title, axis label and annotation
    g.fig.suptitle("Temperatures of " + station + " station"
                   + " from " + str(year_begin) + " to " + str(year_end))
    g.set_titles("")
    g.set_axis_labels("Month", "Temperature (C)")
    g.tight_layout()
```

Now, let's check a station in Vietnam named TAN_SON_HOA from 1999 to 2010.


```python
mult_line_plot_station("Vietnam", "TAN_SON_HOA", 1999, 2010)
```


    
![png](/images/2021-04-14-homework1_files/2021-04-14-homework1_48_0.png)
    

Thank you Philip for helping me realize that I need to add more explanations on the meaning of this plot.
{::options parse_block_html="true" /}
<div class="got-help">
The series of graphs appears to fluctuate quite erratically, but a more thorough inspection reveals that the temperature, in general, has increased over time. Especially, if we observe more closely at the winter season, the temperature looks like it follows an indicative trend in which it increases quite rapidly. This is a rather solid indication of the realness and severity of climate change, which should be taken seriously by every single one of us.
</div>
{::options parse_block_html="false" /}