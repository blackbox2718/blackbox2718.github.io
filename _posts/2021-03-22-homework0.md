---
title: Feature Selection Through Visualization
date: 2021-03-22 23:50:59 -0700
categories: [PIC16B-blog]
tags: [seaborn, heatmap, palmer-penguins]
---
In this very first post of PIC16B, I will demonstrate how to visualize the [Palmer Penguins](https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/palmer_penguins.csv) data set with the help of ``panda`` and ``seaborn`` which is a data visualization library based on ``matplotlib``.

## Data Preparation
First, we need to load the data set using ``read_csv()`` function from ``panda``.


```python
import pandas as pd
url = "https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/palmer_penguins.csv"
penguins = pd.read_csv(url)
```

As one may spend some time to scrutinize the data set, it can be easily seen that there are several variables that has a very low correlation or even not all correlated to other features presented. Experiences of working with data and specifically with this data set for the mini-project in PIC16A can be very useful here as it can help us with which features to choose. Notice that we can immediately disregard quite many "irrelevant" columns like ``Comments``, ``Individual ID``, ``Date Egg``, etc in this context.


```python
cols = ["Species", "Island", "Culmen Length (mm)", "Culmen Depth (mm)",
        "Flipper Length (mm)", "Body Mass (g)", "Sex", "Delta 15 N (o/oo)",
        "Delta 13 C (o/oo)"]
penguins = penguins[cols]
```

Then, let's take a look at the first few rows of our data set.


```python
penguins.head()
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
      <th>Species</th>
      <th>Island</th>
      <th>Culmen Length (mm)</th>
      <th>Culmen Depth (mm)</th>
      <th>Flipper Length (mm)</th>
      <th>Body Mass (g)</th>
      <th>Sex</th>
      <th>Delta 15 N (o/oo)</th>
      <th>Delta 13 C (o/oo)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelie Penguin (Pygoscelis adeliae)</td>
      <td>Torgersen</td>
      <td>39.1</td>
      <td>18.7</td>
      <td>181.0</td>
      <td>3750.0</td>
      <td>MALE</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adelie Penguin (Pygoscelis adeliae)</td>
      <td>Torgersen</td>
      <td>39.5</td>
      <td>17.4</td>
      <td>186.0</td>
      <td>3800.0</td>
      <td>FEMALE</td>
      <td>8.94956</td>
      <td>-24.69454</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adelie Penguin (Pygoscelis adeliae)</td>
      <td>Torgersen</td>
      <td>40.3</td>
      <td>18.0</td>
      <td>195.0</td>
      <td>3250.0</td>
      <td>FEMALE</td>
      <td>8.36821</td>
      <td>-25.33302</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Adelie Penguin (Pygoscelis adeliae)</td>
      <td>Torgersen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adelie Penguin (Pygoscelis adeliae)</td>
      <td>Torgersen</td>
      <td>36.7</td>
      <td>19.3</td>
      <td>193.0</td>
      <td>3450.0</td>
      <td>FEMALE</td>
      <td>8.76651</td>
      <td>-25.32426</td>
    </tr>
  </tbody>
</table>
</div>



Now, observe that there are Nan values floating around in our data, so we want to eliminate them so that the computation can be dealt with a lot more ease.


```python
penguins = penguins.dropna()
```

Through some sets of trial and error, we can also figure out that there is an invalid value of ``Sex`` (.) at index 336 which can affect how the program performs.


```python
penguins = penguins.drop([336])
```

Now, in order to generate a plot, we have to borrow some help from our best companion in PIC16A, which is ``sklearn``, to encode all of the qualitative features in our data set. Otherwise, nothing can be logically deduced from data filled with strings.


```python
from sklearn import preprocessing
 
le = preprocessing.LabelEncoder()
penguins['Sex'] = le.fit_transform(penguins['Sex'])
penguins['Species'] = le.fit_transform(penguins['Species'])
penguins['Island'] = le.fit_transform(penguins['Island'])
```

## Visualizing with **Seaborn**

At this point, there are different paths that we can take to visualize the relationship between various variables in this data set. As we may recall from the mini-project, we have to perform *feature selection* to get the best combination of variables that maximizes a machine learning model's ability to correctly predict a penguin's species. With that being said, in this blog post, I want to construct a plot that showcases the correlation between each variables in our data set using the heatmap method from ``seaborn``. This would certainly give us a very comprehensive picture of how each variable is connected to the others.


{::options parse_block_html="true" /}
<div class="got-help">
In fact, when working on the project, my groupmate came up with this ingenious idea to systematically select the best features for the model. Thus, the code that I am about to show here is credited to Emily Nguyen; thank you for all of your hard work throughout the quarter.


```python
import seaborn as sns
from matplotlib import pyplot as plt

plt.figure(figsize=(12, 10))
# create a panda dataframe consisting of the pairwise correlation between all variables
cor = penguins.corr()
# plot the correlation coefficient on a heatmap
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
```
</div>
{::options parse_block_html="false" /}

    
![png](/images/2021-03-22-homework0_files/2021-03-22-homework0_15_0.png)
    


{::options parse_block_html="true" /}
<div class="gave-help">
Observe that we mapped the correlation levels on a heat map in order to visualize the relationship between any two features, in which the darker red implies a more positive correlation. On the opposite spectrum, the white region signifies a more negative correlation. However, those negative values are not necessarily labeled as the undesirable features for the machine learning model. In fact, they hold the same "credentials" as those positive ones because what may happen here is that two variables may have an inverse relationship in which an increase in one variable leads to the decrease of the other. What we really want to ignore from the heat map are those sets of variables with coefficient close to 0 with respect to species. These features do not contribute helpful information or an indicative trend in determining species of a penguin.
</div>
{::options parse_block_html="false" /}
