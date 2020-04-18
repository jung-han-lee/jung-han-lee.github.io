---
title: "[Text Mining with Python] Extracting and Normalizing dates"
date: 2020-04-17
tags: [machine learning, data science, text mining, regex, regular expression]

excerpt: "Machine Learning, Text Mining, Data Science"
mathjax: "true"
---

# Working with Text in Python(Regular Expression)

Each line of the dates.txt file corresponds to a medical note. Each note has a date that needs to be extracted, but each date is encoded in one of many formats.

The goal of this assignment is to correctly identify all of the different date variants encoded in this dataset and to properly normalize and sort the dates.

Here is a list of some of the variants you might encounter in this dataset:


>-04/20/2009; 04/20/09; 4/20/09; 4/3/09

>-Mar-20-2009; Mar 20, 2009; March 20, 2009; Mar. 20, 2009; Mar 20 2009;

>-20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009

>-Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009

>-Feb 2009; Sep 2009; Oct 2010

>-6/2008; 12/2009

>-2009; 2010


Based on the list, I made 4 classified cases:


case1 : 04/20/2009; 04/20/09; 4/20/09; 4/3/09; Mar-20-2009

case2 : Mar 20, 2009; March 20, 2009; Mar. 20, 2009; Mar 20 2009; 20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009; Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009; Feb 2009; Sep 2009; Oct 2010

case3 : 6/2008; 12/2009

case4 : 2009; 2010


Once you have extracted these date patterns from the text, the next step is to sort them in ascending chronological order according to the following rules:

```
-Assume all dates in xx/xx/xx format are mm/dd/yy

-Assume all dates where year is encoded in only two digits are years from the 1900's (e.g. 1/5/89 is January 5th, 1989)

-If the day is missing (e.g. 9/2009), assume it is the first day of the month (e.g. September 1, 2009).

-If the month is missing (e.g. 2010), assume it is the first of January of that year (e.g. January 1, 2010).

-Watch out for potential typos as this is a raw, real-life derived dataset.
```

With these rules in mind, find the correct date in each note and return a pandas Series in chronological order of the original Series' indices.



```python
import pandas as pd

doc = []
with open('dates.txt') as file:
    for line in file:
        doc.append(line)

df = pd.Series(doc)
```

```python
df.head(10)
```
<img src="{{ site.url }}{{ site.baseurl }}/images/tm/s1.png" alt="">


```python
import re
```

```python
#case 1
df.str.findall(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}').head()
```

<img src="{{ site.url }}{{ site.baseurl }}/images/tm/s2.png" alt="">

```python
#case2
df.str.findall(r'(?:\d{1,2} )?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*.? (?:\d{1,2}[a-z]*, |\d{1,2}[a-z]* )?\d{4}')[300:305]
```

<img src="{{ site.url }}{{ site.baseurl }}/images/tm/s3.png" alt="">

```python
#case3
df.str.findall(r'\d{1,2}[-/][1|2]\d{3}')[400:405]
```

<img src="{{ site.url }}{{ site.baseurl }}/images/tm/s4.png" alt="">

```python
#case4
df.str.findall(r'[1|2]\d{3}')[495:500]
```

<img src="{{ site.url }}{{ site.baseurl }}/images/tm/s6.png" alt="">


```python
#case 4 to compare the results to case3
df.str.findall(r'[1|2]\d{3}')[400:405]
```

<img src="{{ site.url }}{{ site.baseurl }}/images/tm/s5.png" alt="">

However as we can find from above, in some texts, the results from case3 and case4 are different. To avoid this situation, I applied case1 to all texts, and then did case2, case3, case4 sequentially.

Therefore, my answer is as follows:
```python
def date_sorter():

    case1 = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
    case2 = r'(?:\d{1,2} )?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*.? (?:\d{1,2}[a-z]*, |\d{1,2}[a-z]* )?\d{4}'
    case3 = r'\d{1,2}[-/][1|2]\d{3}'
    case4 = r'[1|2]\d{3}'

    regex = '(%s|%s|%s|%s)' %(case1, case2, case3, case4)

    date = df.str.extract(regex)
    date = date.str.replace('Janaury', 'January').str.replace('Decemeber', 'December')
    date = pd.Series(pd.to_datetime(date))
    date = date.sort_values().index

    return pd.Series(date)
```
