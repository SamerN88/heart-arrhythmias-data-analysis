# Analysis of ECG Data to Diagnose Heart Arrhythmias

**Samer Najjar<br>
30 November 2021**

#### About my dataset
My dataset is about heart arrhythmias (irregular heartbeat). Each observation represents a patient, containing ECG values, the name of the patient's arrhythmia, and the heart condition(s) the patient has, if any.<br>
Dataset source: https://figshare.com/collections/ChapmanECG/4560497/2

#### Research question and learning model
I am trying to use the numeric ECG values of a patient to predict certain arrhythmias and/or heart conditions. Thus, I will use classification models. I will use _Logistic Regression_, _Support Vector Machine (SVM)_, _K-Nearest Neighbors (KNN)_, and _Decision Tree_ to predict certain selected heart conditions.

#### Evaluating the model
I will use a train/test split to train and test the model, then I will look at the accuracy, precision, and recall to evaluate the performance of the model. In medicine we really don't like false negatives, as an undiagnosed heart condition could be life-threatening, so the recall will be especially important. I relate to this personally; despite having multiple ECGs done on me, doctors failed to diagnose me until I had two cardiac arrests in one hour. I was extremely lucky to survive, but if I had been diagnosed earlier we could have avoided that which kills 475,000 Americans yearly.

#### My prediction
Based on what I've learned about cardiology, the ECG is fairly effective in diagnosing or partially diagnosing many heart conditions. Thus, I expect my models to have at least 80% accuracy at worst, and hopefully very few false negatives, i.e. high recall (although I'm not sure if I will be able to achieve a high enough recall for clinical use).

<hr>

## 0) Basic anatomy
In this project we are mainly concerned with diagnosing arrhythmogenic conditions related to the atriums (upper chambers) and the ventricles (lower chambers):

![heart_diagram.png](attachment:heart_diagram.png)
[ Source: https://yourheartvalve.com/heart-basics/heart-anatomy/ ]

<hr>

## 1) Prepare data


```python
# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, classification_report

from itertools import combinations

# Set random state for project (8)
RANDOM_STATE = 8

# Enable inline mode for matplotlib so that Jupyter displays graphs
%matplotlib inline
```


```python
# Import dataset of ECG values and diagnostic info for various heart arrhythmias

arrhythmias = pd.read_excel('Diagnostics.xlsx')
arrhythmias
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
      <th>FileName</th>
      <th>Rhythm</th>
      <th>Beat</th>
      <th>PatientAge</th>
      <th>Gender</th>
      <th>VentricularRate</th>
      <th>AtrialRate</th>
      <th>QRSDuration</th>
      <th>QTInterval</th>
      <th>QTCorrected</th>
      <th>RAxis</th>
      <th>TAxis</th>
      <th>QRSCount</th>
      <th>QOnset</th>
      <th>QOffset</th>
      <th>TOffset</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MUSE_20180113_171327_27000</td>
      <td>AFIB</td>
      <td>RBBB TWC</td>
      <td>85</td>
      <td>MALE</td>
      <td>117</td>
      <td>234</td>
      <td>114</td>
      <td>356</td>
      <td>496</td>
      <td>81</td>
      <td>-27</td>
      <td>19</td>
      <td>208</td>
      <td>265</td>
      <td>386</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MUSE_20180112_073319_29000</td>
      <td>SB</td>
      <td>TWC</td>
      <td>59</td>
      <td>FEMALE</td>
      <td>52</td>
      <td>52</td>
      <td>92</td>
      <td>432</td>
      <td>401</td>
      <td>76</td>
      <td>42</td>
      <td>8</td>
      <td>215</td>
      <td>261</td>
      <td>431</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MUSE_20180111_165520_97000</td>
      <td>SA</td>
      <td>NONE</td>
      <td>20</td>
      <td>FEMALE</td>
      <td>67</td>
      <td>67</td>
      <td>82</td>
      <td>382</td>
      <td>403</td>
      <td>88</td>
      <td>20</td>
      <td>11</td>
      <td>224</td>
      <td>265</td>
      <td>415</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MUSE_20180113_121940_44000</td>
      <td>SB</td>
      <td>NONE</td>
      <td>66</td>
      <td>MALE</td>
      <td>53</td>
      <td>53</td>
      <td>96</td>
      <td>456</td>
      <td>427</td>
      <td>34</td>
      <td>3</td>
      <td>9</td>
      <td>219</td>
      <td>267</td>
      <td>447</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MUSE_20180112_122850_57000</td>
      <td>AF</td>
      <td>STDD STTC</td>
      <td>73</td>
      <td>FEMALE</td>
      <td>162</td>
      <td>162</td>
      <td>114</td>
      <td>252</td>
      <td>413</td>
      <td>68</td>
      <td>-40</td>
      <td>26</td>
      <td>228</td>
      <td>285</td>
      <td>354</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10641</th>
      <td>MUSE_20181222_204306_99000</td>
      <td>SVT</td>
      <td>NONE</td>
      <td>80</td>
      <td>FEMALE</td>
      <td>196</td>
      <td>73</td>
      <td>168</td>
      <td>284</td>
      <td>513</td>
      <td>258</td>
      <td>244</td>
      <td>32</td>
      <td>177</td>
      <td>261</td>
      <td>319</td>
    </tr>
    <tr>
      <th>10642</th>
      <td>MUSE_20181222_204309_22000</td>
      <td>SVT</td>
      <td>NONE</td>
      <td>81</td>
      <td>FEMALE</td>
      <td>162</td>
      <td>81</td>
      <td>162</td>
      <td>294</td>
      <td>482</td>
      <td>110</td>
      <td>-75</td>
      <td>27</td>
      <td>173</td>
      <td>254</td>
      <td>320</td>
    </tr>
    <tr>
      <th>10643</th>
      <td>MUSE_20181222_204310_31000</td>
      <td>SVT</td>
      <td>NONE</td>
      <td>39</td>
      <td>MALE</td>
      <td>152</td>
      <td>92</td>
      <td>152</td>
      <td>340</td>
      <td>540</td>
      <td>250</td>
      <td>38</td>
      <td>25</td>
      <td>208</td>
      <td>284</td>
      <td>378</td>
    </tr>
    <tr>
      <th>10644</th>
      <td>MUSE_20181222_204312_58000</td>
      <td>SVT</td>
      <td>NONE</td>
      <td>76</td>
      <td>MALE</td>
      <td>175</td>
      <td>178</td>
      <td>128</td>
      <td>310</td>
      <td>529</td>
      <td>98</td>
      <td>-83</td>
      <td>29</td>
      <td>205</td>
      <td>269</td>
      <td>360</td>
    </tr>
    <tr>
      <th>10645</th>
      <td>MUSE_20181222_204314_78000</td>
      <td>SVT</td>
      <td>NONE</td>
      <td>75</td>
      <td>MALE</td>
      <td>117</td>
      <td>104</td>
      <td>140</td>
      <td>312</td>
      <td>435</td>
      <td>263</td>
      <td>144</td>
      <td>19</td>
      <td>208</td>
      <td>278</td>
      <td>364</td>
    </tr>
  </tbody>
</table>
<p>10646 rows × 16 columns</p>
</div>




```python
# Show attribute info

attribute_info = pd.read_excel('AttributesDictionary.xlsx')
attribute_info
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
      <th>Attributes</th>
      <th>Type</th>
      <th>ValueRange</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FileName</td>
      <td>String</td>
      <td>NaN</td>
      <td>ECG data file name(unique ID)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Rhythm</td>
      <td>String</td>
      <td>NaN</td>
      <td>Rhythm Label</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Beat</td>
      <td>String</td>
      <td>NaN</td>
      <td>Other conditions Label</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PatientAge</td>
      <td>Numeric</td>
      <td>0-999</td>
      <td>Age</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Gender</td>
      <td>String</td>
      <td>MALE/FEMAL</td>
      <td>Gender</td>
    </tr>
    <tr>
      <th>5</th>
      <td>VentricularRate</td>
      <td>Numeric</td>
      <td>0-999</td>
      <td>Ventricular rate in BPM</td>
    </tr>
    <tr>
      <th>6</th>
      <td>AtrialRate</td>
      <td>Numeric</td>
      <td>0-999</td>
      <td>Atrial rate in BPM</td>
    </tr>
    <tr>
      <th>7</th>
      <td>QRSDuration</td>
      <td>Numeric -</td>
      <td>0-999</td>
      <td>QRS duration in msec</td>
    </tr>
    <tr>
      <th>8</th>
      <td>QTInterval</td>
      <td>Numeric</td>
      <td>0-999</td>
      <td>QT interval in msec</td>
    </tr>
    <tr>
      <th>9</th>
      <td>QTCorrected</td>
      <td>Numeric</td>
      <td>0-999</td>
      <td>Corrected QT interval in msec</td>
    </tr>
    <tr>
      <th>10</th>
      <td>RAxis</td>
      <td>Numeric</td>
      <td>-179~180</td>
      <td>R axis</td>
    </tr>
    <tr>
      <th>11</th>
      <td>TAxis</td>
      <td>Numeric</td>
      <td>-179~181</td>
      <td>T axis</td>
    </tr>
    <tr>
      <th>12</th>
      <td>QRSCount</td>
      <td>Numeric</td>
      <td>0-254</td>
      <td>QRS count</td>
    </tr>
    <tr>
      <th>13</th>
      <td>QOnset</td>
      <td>Numeric</td>
      <td>16 Bit Unsigned</td>
      <td>Q onset(In samples)</td>
    </tr>
    <tr>
      <th>14</th>
      <td>QOffset</td>
      <td>Numeric</td>
      <td>17 Bit Unsigned</td>
      <td>Q offset(In samples)</td>
    </tr>
    <tr>
      <th>15</th>
      <td>TOffset</td>
      <td>Numeric</td>
      <td>18 Bit Unsigned</td>
      <td>T offset(In samples)</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check if columns have NaN values

print('Number of NaN values by column:\n')
for col in arrhythmias.columns:
    print(f'{col}: {arrhythmias[col].isnull().sum()}')
```

    Number of NaN values by column:
    
    FileName: 0
    Rhythm: 0
    Beat: 0
    PatientAge: 0
    Gender: 0
    VentricularRate: 0
    AtrialRate: 0
    QRSDuration: 0
    QTInterval: 0
    QTCorrected: 0
    RAxis: 0
    TAxis: 0
    QRSCount: 0
    QOnset: 0
    QOffset: 0
    TOffset: 0



```python
# No NaN values, so data is clean
# Drop FileName column

arrhythmias = arrhythmias.drop('FileName', axis=1).reset_index(drop=True)
arrhythmias
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
      <th>Rhythm</th>
      <th>Beat</th>
      <th>PatientAge</th>
      <th>Gender</th>
      <th>VentricularRate</th>
      <th>AtrialRate</th>
      <th>QRSDuration</th>
      <th>QTInterval</th>
      <th>QTCorrected</th>
      <th>RAxis</th>
      <th>TAxis</th>
      <th>QRSCount</th>
      <th>QOnset</th>
      <th>QOffset</th>
      <th>TOffset</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AFIB</td>
      <td>RBBB TWC</td>
      <td>85</td>
      <td>MALE</td>
      <td>117</td>
      <td>234</td>
      <td>114</td>
      <td>356</td>
      <td>496</td>
      <td>81</td>
      <td>-27</td>
      <td>19</td>
      <td>208</td>
      <td>265</td>
      <td>386</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SB</td>
      <td>TWC</td>
      <td>59</td>
      <td>FEMALE</td>
      <td>52</td>
      <td>52</td>
      <td>92</td>
      <td>432</td>
      <td>401</td>
      <td>76</td>
      <td>42</td>
      <td>8</td>
      <td>215</td>
      <td>261</td>
      <td>431</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SA</td>
      <td>NONE</td>
      <td>20</td>
      <td>FEMALE</td>
      <td>67</td>
      <td>67</td>
      <td>82</td>
      <td>382</td>
      <td>403</td>
      <td>88</td>
      <td>20</td>
      <td>11</td>
      <td>224</td>
      <td>265</td>
      <td>415</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SB</td>
      <td>NONE</td>
      <td>66</td>
      <td>MALE</td>
      <td>53</td>
      <td>53</td>
      <td>96</td>
      <td>456</td>
      <td>427</td>
      <td>34</td>
      <td>3</td>
      <td>9</td>
      <td>219</td>
      <td>267</td>
      <td>447</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AF</td>
      <td>STDD STTC</td>
      <td>73</td>
      <td>FEMALE</td>
      <td>162</td>
      <td>162</td>
      <td>114</td>
      <td>252</td>
      <td>413</td>
      <td>68</td>
      <td>-40</td>
      <td>26</td>
      <td>228</td>
      <td>285</td>
      <td>354</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10641</th>
      <td>SVT</td>
      <td>NONE</td>
      <td>80</td>
      <td>FEMALE</td>
      <td>196</td>
      <td>73</td>
      <td>168</td>
      <td>284</td>
      <td>513</td>
      <td>258</td>
      <td>244</td>
      <td>32</td>
      <td>177</td>
      <td>261</td>
      <td>319</td>
    </tr>
    <tr>
      <th>10642</th>
      <td>SVT</td>
      <td>NONE</td>
      <td>81</td>
      <td>FEMALE</td>
      <td>162</td>
      <td>81</td>
      <td>162</td>
      <td>294</td>
      <td>482</td>
      <td>110</td>
      <td>-75</td>
      <td>27</td>
      <td>173</td>
      <td>254</td>
      <td>320</td>
    </tr>
    <tr>
      <th>10643</th>
      <td>SVT</td>
      <td>NONE</td>
      <td>39</td>
      <td>MALE</td>
      <td>152</td>
      <td>92</td>
      <td>152</td>
      <td>340</td>
      <td>540</td>
      <td>250</td>
      <td>38</td>
      <td>25</td>
      <td>208</td>
      <td>284</td>
      <td>378</td>
    </tr>
    <tr>
      <th>10644</th>
      <td>SVT</td>
      <td>NONE</td>
      <td>76</td>
      <td>MALE</td>
      <td>175</td>
      <td>178</td>
      <td>128</td>
      <td>310</td>
      <td>529</td>
      <td>98</td>
      <td>-83</td>
      <td>29</td>
      <td>205</td>
      <td>269</td>
      <td>360</td>
    </tr>
    <tr>
      <th>10645</th>
      <td>SVT</td>
      <td>NONE</td>
      <td>75</td>
      <td>MALE</td>
      <td>117</td>
      <td>104</td>
      <td>140</td>
      <td>312</td>
      <td>435</td>
      <td>263</td>
      <td>144</td>
      <td>19</td>
      <td>208</td>
      <td>278</td>
      <td>364</td>
    </tr>
  </tbody>
</table>
<p>10646 rows × 15 columns</p>
</div>




```python
# Import dictionary for rhythm names

rhythm_names_df = pd.read_excel('RhythmNames.xlsx')
rhythm_names_df
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
      <th>Acronym Name</th>
      <th>Full Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SB</td>
      <td>Sinus Bradycardia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SR</td>
      <td>Sinus Rhythm</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AFIB</td>
      <td>Atrial Fibrillation</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ST</td>
      <td>Sinus Tachycardia</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AF</td>
      <td>Atrial Flutter</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SI</td>
      <td>Sinus Irregularity</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SVT</td>
      <td>Supraventricular Tachycardia</td>
    </tr>
    <tr>
      <th>7</th>
      <td>AT</td>
      <td>Atrial Tachycardia</td>
    </tr>
    <tr>
      <th>8</th>
      <td>AVNRT</td>
      <td>Atrioventricular  Node Reentrant Tachycardia</td>
    </tr>
    <tr>
      <th>9</th>
      <td>AVRT</td>
      <td>Atrioventricular Reentrant Tachycardia</td>
    </tr>
    <tr>
      <th>10</th>
      <td>SAAWR</td>
      <td>Sinus Atrium to Atrial Wandering Rhythm</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Import dictionary for condition names

condition_names_df = pd.read_excel('ConditionNames.xlsx')
condition_names_df
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
      <th>Acronym Name</th>
      <th>Full Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1AVB</td>
      <td>1 degree atrioventricular block</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2AVB</td>
      <td>2 degree atrioventricular block</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2AVB1</td>
      <td>2 degree atrioventricular block(Type one)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2AVB2</td>
      <td>2 degree atrioventricular block(Type two)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3AVB</td>
      <td>3 degree atrioventricular block</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ABI</td>
      <td>atrial bigeminy</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ALS</td>
      <td>Axis left shift</td>
    </tr>
    <tr>
      <th>7</th>
      <td>APB</td>
      <td>atrial premature beats</td>
    </tr>
    <tr>
      <th>8</th>
      <td>AQW</td>
      <td>abnormal Q wave</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ARS</td>
      <td>Axis right shift</td>
    </tr>
    <tr>
      <th>10</th>
      <td>AVB</td>
      <td>atrioventricular block</td>
    </tr>
    <tr>
      <th>11</th>
      <td>CCR</td>
      <td>countercolockwise rotation</td>
    </tr>
    <tr>
      <th>12</th>
      <td>CR</td>
      <td>colockwise rotation</td>
    </tr>
    <tr>
      <th>13</th>
      <td>ERV</td>
      <td>Early repolarization of the ventricles</td>
    </tr>
    <tr>
      <th>14</th>
      <td>FQRS</td>
      <td>fQRS Wave</td>
    </tr>
    <tr>
      <th>15</th>
      <td>IDC</td>
      <td>Interior differences conduction</td>
    </tr>
    <tr>
      <th>16</th>
      <td>IVB</td>
      <td>Intraventricular block</td>
    </tr>
    <tr>
      <th>17</th>
      <td>JEB</td>
      <td>junctional escape beat</td>
    </tr>
    <tr>
      <th>18</th>
      <td>JPS</td>
      <td>J point shift</td>
    </tr>
    <tr>
      <th>19</th>
      <td>JPT</td>
      <td>junctional premature beat</td>
    </tr>
    <tr>
      <th>20</th>
      <td>LBBB</td>
      <td>left bundle branch block</td>
    </tr>
    <tr>
      <th>21</th>
      <td>LBBBB</td>
      <td>left back bundle branch block</td>
    </tr>
    <tr>
      <th>22</th>
      <td>LFBBB</td>
      <td>left front bundle branch block</td>
    </tr>
    <tr>
      <th>23</th>
      <td>LRRI</td>
      <td>Long RR interval</td>
    </tr>
    <tr>
      <th>24</th>
      <td>LVH</td>
      <td>left ventricle hypertrophy</td>
    </tr>
    <tr>
      <th>25</th>
      <td>LVHV</td>
      <td>left ventricle high voltage</td>
    </tr>
    <tr>
      <th>26</th>
      <td>LVQRSAL</td>
      <td>lower voltage QRS in all lead</td>
    </tr>
    <tr>
      <th>27</th>
      <td>LVQRSCL</td>
      <td>lower voltage QRS in chest lead</td>
    </tr>
    <tr>
      <th>28</th>
      <td>LVQRSLL</td>
      <td>lower voltage QRS in limb lead</td>
    </tr>
    <tr>
      <th>29</th>
      <td>MI</td>
      <td>myocardial infarction</td>
    </tr>
    <tr>
      <th>30</th>
      <td>MIBW</td>
      <td>myocardial infraction in back wall</td>
    </tr>
    <tr>
      <th>31</th>
      <td>MIFW</td>
      <td>Myocardial infgraction in the front wall</td>
    </tr>
    <tr>
      <th>32</th>
      <td>MILW</td>
      <td>Myocardial infraction in the lower wall</td>
    </tr>
    <tr>
      <th>33</th>
      <td>MISW</td>
      <td>Myocardial infraction in the side wall</td>
    </tr>
    <tr>
      <th>34</th>
      <td>PRIE</td>
      <td>PR interval extension</td>
    </tr>
    <tr>
      <th>35</th>
      <td>PWC</td>
      <td>P wave Change</td>
    </tr>
    <tr>
      <th>36</th>
      <td>QTIE</td>
      <td>QT interval extension</td>
    </tr>
    <tr>
      <th>37</th>
      <td>RAH</td>
      <td>right atrial hypertrophy</td>
    </tr>
    <tr>
      <th>38</th>
      <td>RAHV</td>
      <td>right atrial high voltage</td>
    </tr>
    <tr>
      <th>39</th>
      <td>RBBB</td>
      <td>right bundle branch block</td>
    </tr>
    <tr>
      <th>40</th>
      <td>RVH</td>
      <td>right ventricle hypertrophy</td>
    </tr>
    <tr>
      <th>41</th>
      <td>STDD</td>
      <td>ST drop down</td>
    </tr>
    <tr>
      <th>42</th>
      <td>STE</td>
      <td>ST extension</td>
    </tr>
    <tr>
      <th>43</th>
      <td>STTC</td>
      <td>ST-T Change</td>
    </tr>
    <tr>
      <th>44</th>
      <td>STTU</td>
      <td>ST tilt up</td>
    </tr>
    <tr>
      <th>45</th>
      <td>TWC</td>
      <td>T wave Change</td>
    </tr>
    <tr>
      <th>46</th>
      <td>TWO</td>
      <td>T wave opposite</td>
    </tr>
    <tr>
      <th>47</th>
      <td>UW</td>
      <td>U wave</td>
    </tr>
    <tr>
      <th>48</th>
      <td>VB</td>
      <td>ventricular bigeminy</td>
    </tr>
    <tr>
      <th>49</th>
      <td>VEB</td>
      <td>ventricular escape beat</td>
    </tr>
    <tr>
      <th>50</th>
      <td>VFW</td>
      <td>ventricular fusion wave</td>
    </tr>
    <tr>
      <th>51</th>
      <td>VPB</td>
      <td>ventricular premature beat</td>
    </tr>
    <tr>
      <th>52</th>
      <td>VPE</td>
      <td>ventricular preexcitation</td>
    </tr>
    <tr>
      <th>53</th>
      <td>VET</td>
      <td>ventricular escape trigeminy</td>
    </tr>
    <tr>
      <th>54</th>
      <td>WAVN</td>
      <td>Wandering in the atrioventricalualr node</td>
    </tr>
    <tr>
      <th>55</th>
      <td>WPW</td>
      <td>WPW</td>
    </tr>
  </tbody>
</table>
</div>



<hr>

## 2) Atrial Fibrilation (AFib)

AFib is an arrhythmia in which the atriums beat irregularly and rapidly. Usually AFib on its own is not deadly, but it increases the risk of stroke, heart failure, and other complications. It is the most commonly diagnosed arrhythmia and affects millions of Americans.
<br><br>
We will create a new dataframe aimed at predicting AFib from ECG values.


```python
# See how many patients have AFib

arrhythmias['Rhythm'].value_counts()
```




    SB       3889
    SR       1826
    AFIB     1780
    ST       1568
    SVT       587
    AF        445
    SA        399
    AT        121
    AVNRT      16
    AVRT        8
    SAAWR       7
    Name: Rhythm, dtype: int64




```python
# Create new dataframe aimed at predicting AFib

afib_df = arrhythmias.copy()
afib_df = afib_df.rename(columns={'Rhythm': 'AFib'})
afib_df['AFib'] = (afib_df['AFib'] == 'AFIB')

afib_df
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
      <th>AFib</th>
      <th>Beat</th>
      <th>PatientAge</th>
      <th>Gender</th>
      <th>VentricularRate</th>
      <th>AtrialRate</th>
      <th>QRSDuration</th>
      <th>QTInterval</th>
      <th>QTCorrected</th>
      <th>RAxis</th>
      <th>TAxis</th>
      <th>QRSCount</th>
      <th>QOnset</th>
      <th>QOffset</th>
      <th>TOffset</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>True</td>
      <td>RBBB TWC</td>
      <td>85</td>
      <td>MALE</td>
      <td>117</td>
      <td>234</td>
      <td>114</td>
      <td>356</td>
      <td>496</td>
      <td>81</td>
      <td>-27</td>
      <td>19</td>
      <td>208</td>
      <td>265</td>
      <td>386</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>TWC</td>
      <td>59</td>
      <td>FEMALE</td>
      <td>52</td>
      <td>52</td>
      <td>92</td>
      <td>432</td>
      <td>401</td>
      <td>76</td>
      <td>42</td>
      <td>8</td>
      <td>215</td>
      <td>261</td>
      <td>431</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>NONE</td>
      <td>20</td>
      <td>FEMALE</td>
      <td>67</td>
      <td>67</td>
      <td>82</td>
      <td>382</td>
      <td>403</td>
      <td>88</td>
      <td>20</td>
      <td>11</td>
      <td>224</td>
      <td>265</td>
      <td>415</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>NONE</td>
      <td>66</td>
      <td>MALE</td>
      <td>53</td>
      <td>53</td>
      <td>96</td>
      <td>456</td>
      <td>427</td>
      <td>34</td>
      <td>3</td>
      <td>9</td>
      <td>219</td>
      <td>267</td>
      <td>447</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>STDD STTC</td>
      <td>73</td>
      <td>FEMALE</td>
      <td>162</td>
      <td>162</td>
      <td>114</td>
      <td>252</td>
      <td>413</td>
      <td>68</td>
      <td>-40</td>
      <td>26</td>
      <td>228</td>
      <td>285</td>
      <td>354</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10641</th>
      <td>False</td>
      <td>NONE</td>
      <td>80</td>
      <td>FEMALE</td>
      <td>196</td>
      <td>73</td>
      <td>168</td>
      <td>284</td>
      <td>513</td>
      <td>258</td>
      <td>244</td>
      <td>32</td>
      <td>177</td>
      <td>261</td>
      <td>319</td>
    </tr>
    <tr>
      <th>10642</th>
      <td>False</td>
      <td>NONE</td>
      <td>81</td>
      <td>FEMALE</td>
      <td>162</td>
      <td>81</td>
      <td>162</td>
      <td>294</td>
      <td>482</td>
      <td>110</td>
      <td>-75</td>
      <td>27</td>
      <td>173</td>
      <td>254</td>
      <td>320</td>
    </tr>
    <tr>
      <th>10643</th>
      <td>False</td>
      <td>NONE</td>
      <td>39</td>
      <td>MALE</td>
      <td>152</td>
      <td>92</td>
      <td>152</td>
      <td>340</td>
      <td>540</td>
      <td>250</td>
      <td>38</td>
      <td>25</td>
      <td>208</td>
      <td>284</td>
      <td>378</td>
    </tr>
    <tr>
      <th>10644</th>
      <td>False</td>
      <td>NONE</td>
      <td>76</td>
      <td>MALE</td>
      <td>175</td>
      <td>178</td>
      <td>128</td>
      <td>310</td>
      <td>529</td>
      <td>98</td>
      <td>-83</td>
      <td>29</td>
      <td>205</td>
      <td>269</td>
      <td>360</td>
    </tr>
    <tr>
      <th>10645</th>
      <td>False</td>
      <td>NONE</td>
      <td>75</td>
      <td>MALE</td>
      <td>117</td>
      <td>104</td>
      <td>140</td>
      <td>312</td>
      <td>435</td>
      <td>263</td>
      <td>144</td>
      <td>19</td>
      <td>208</td>
      <td>278</td>
      <td>364</td>
    </tr>
  </tbody>
</table>
<p>10646 rows × 15 columns</p>
</div>



### Logistic Regression


```python
# Build Logistic Regression model

# Select all numeric ECG values as features
all_features = [
    'VentricularRate',
    'AtrialRate',
    'QRSDuration',
    'QTInterval',
    'QTCorrected',
    'RAxis',
    'TAxis',
    'QRSCount',
    'QOnset',
    'QOffset',
    'TOffset'
]

X = afib_df[all_features]
y = afib_df['AFib']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```




    LogisticRegression(max_iter=1000)




```python
# Prints classification report in a convenient format (strictly for binary target)
def print_report(y_test, y_pred):
    # Accuracy
    print(f"accuracy = {sum(y_pred == y_test) / len(y_test)}")
    
    # precision = (True Positives) /  (True Positives + False Positives)
    # recall = (True Positives) / (True Positives + False Negatives)
    # F1-score = 2*((precision*recall)/(precision+recall))
    
    true_pos = false_pos = false_neg = 0

    for pred, true in zip(y_pred, y_test):
        if true == True and pred == True:
            true_pos += 1
        elif true == False and pred == True:
            false_pos += 1
        elif true == True and pred == False:
            false_neg += 1

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    
    print(f'precision = {precision}')
    print(f'recall = {recall}')
    print(f'F1-score = {2 * ((precision*recall) / (precision+recall))}')
```


```python
# Test model
    
y_pred = model.predict(X_test)
print_report(y_test, y_pred)
```

    accuracy = 0.8474830954169797
    precision = 0.6147540983606558
    recall = 0.1728110599078341
    F1-score = 0.2697841726618705


#### Interpretation:

The logistic regression model performed very poorly. Even though the model had 84% accuracy, a recall of 0.173 is far too low for any context, let alone a medical one. This means that the model was unable to diagnose most of the patients that actually had AFib. This may partially be due to the fact that, proportionally speaking, there are not many AFib-positive patients in the dataset.
<br><br>
We will try some other classification models to improve our performance.

### SVM


```python
# Build and test SVM model

model = svm.SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print_report(y_test, y_pred)
```

    accuracy = 0.8534936138241923
    precision = 0.7037037037037037
    recall = 0.17511520737327188
    F1-score = 0.28044280442804426


#### Interpretation:

This model also performs poorly.

### KNN


```python
# Build and test KNN model

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print_report(y_test, y_pred)
```

    accuracy = 0.8741547708489857
    precision = 0.6701030927835051
    recall = 0.44930875576036866
    F1-score = 0.5379310344827586


#### Interpretation:

This model performs a bit better but still not that well. However, instead of using all the features in the dataset, we can improve our performance by optimizing which features we use, and since KNN performed the best out of the three classifiers, we will try some optimization techniques on it. 

### Optimize KNN model
Test different combinations of numeric features and see which set of features produces the best model.


```python
# Given a certain number of features n_features, this function finds the best set of features of size
# n_features that produces the highest accuracy model
def optimize_features(n_features, X, y, model, random_state=RANDOM_STATE):
    all_features = X.columns
    feature_combos = combinations(all_features, n_features)
    
    best_acc = 0
    best_features = None
    for features in feature_combos:
        features = list(features)
        
        X_train, X_test, y_train, y_test = train_test_split(X[features], y, test_size=0.25, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        if acc > best_acc:
            best_features = features
            best_acc = acc
            
    return best_features, best_acc
```


```python
# Instantiate new KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
    
# Run the optimize_features function using various numbers of features (ranging from 1 to 11)
# NOTE: There are 11 total features, so this takes some time to run
for n in range(1, len(all_features) + 1):
    best_features, best_acc = optimize_features(n, X, y, knn_model)
    print(f'Optimal set of {n} feature{"" if n == 1 else "s"} (accuracy={best_acc}):')
    print(best_features)
    print()
```

    Optimal set of 1 feature (accuracy=0.8377160030052592):
    ['QOffset']
    
    Optimal set of 2 features (accuracy=0.9380165289256198):
    ['VentricularRate', 'AtrialRate']
    
    Optimal set of 3 features (accuracy=0.9383921863260706):
    ['VentricularRate', 'AtrialRate', 'QRSCount']
    
    Optimal set of 4 features (accuracy=0.9256198347107438):
    ['VentricularRate', 'AtrialRate', 'QRSCount', 'QOnset']
    
    Optimal set of 5 features (accuracy=0.9158527422990232):
    ['VentricularRate', 'AtrialRate', 'QRSCount', 'QOnset', 'QOffset']
    
    Optimal set of 6 features (accuracy=0.9064613072877535):
    ['VentricularRate', 'AtrialRate', 'QRSCount', 'QOnset', 'QOffset', 'TOffset']
    
    Optimal set of 7 features (accuracy=0.8996994740796393):
    ['VentricularRate', 'AtrialRate', 'QTInterval', 'QTCorrected', 'QRSCount', 'QOffset', 'TOffset']
    
    Optimal set of 8 features (accuracy=0.8974455296769346):
    ['VentricularRate', 'AtrialRate', 'QTInterval', 'QTCorrected', 'QRSCount', 'QOnset', 'QOffset', 'TOffset']
    
    Optimal set of 9 features (accuracy=0.886175807663411):
    ['VentricularRate', 'AtrialRate', 'QRSDuration', 'QTInterval', 'QTCorrected', 'QRSCount', 'QOnset', 'QOffset', 'TOffset']
    
    Optimal set of 10 features (accuracy=0.8835462058602555):
    ['VentricularRate', 'AtrialRate', 'QRSDuration', 'QTInterval', 'QTCorrected', 'TAxis', 'QRSCount', 'QOnset', 'QOffset', 'TOffset']
    
    Optimal set of 11 features (accuracy=0.8741547708489857):
    ['VentricularRate', 'AtrialRate', 'QRSDuration', 'QTInterval', 'QTCorrected', 'RAxis', 'TAxis', 'QRSCount', 'QOnset', 'QOffset', 'TOffset']
    


Based on the results, the optimal model (accuracy=0.938) uses the following 3 features:
* VentricularRate
* AtrialRate
* QRSCount

Now we see how the optimized model performs:


```python
best_features = ['VentricularRate', 'AtrialRate', 'QRSCount']

X = afib_df[best_features]
y = afib_df['AFib']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)
        
print_report(y_test, y_pred)
```

    accuracy = 0.9383921863260706
    precision = 0.8214285714285714
    recall = 0.7949308755760369
    F1-score = 0.8079625292740047


#### Interpretation:

This optimized KNN model is much better than the first three models. The accuracy was improved from 0.874 to 0.938, and more significantly, the recall was improved from 0.449 to 0.795 (the precision is also pretty good). Despite these big improvements in model performance, I don't think this model is quite ready for clinical implementation because it still misses quite a few AFib diagnoses; about 20% of patients with AFib were not diagnosed by this model. It may be a good model when someone's life isn't at risk, but in medicine, the model should be held to a higher standard. In conclusion, it's a fairly strong model but would require more improvement to be deployed in the real world.

<hr>

## 3) Predicting multiple heart rhythms

So far, I've used classification models to predict a binary target value (has AFib / doesn't have AFib). AFib is not the most concerning arrhythmia. In fact, generally speaking, arrhythmias of the ventricles are more deadly than atrial arrhythmias and can often cause cardiac arrest. For example, while atrial fibrillation often goes unnoticed, ventricular fibrilliation is highly fatal with a mortality rate of 90-95% if not treated immediately (ventricular fibrillation is the type of cardiac arrest I had). Some arrhythmias in this data set, such as supraventricular tachycardia (SVT) can lead to ventricular fibrillation. Thus, it would be useful to build a model that can predict multiple arrhythmias, both atrial and ventricular. For this, I will use a Decision Tree. I will use some optimization techniques such as optimizing the feature set (like before) and testing various max depths for the tree.

### Decision Tree


```python
# The previous optimize_features function found the optimal set of features given a fixed number of
# features (dimension); this function finds the optimal number of features AND the optimal set of
# features of that size
def optimize_features_and_dimension(X, y, model, random_state=RANDOM_STATE):   
    best_acc = 0
    best_features = None
    
    for n in range(1, len(X.columns) + 1):
        features, acc = optimize_features(n, X, y, model)
        if acc > best_acc:
            best_features = features
            best_acc = acc
    
    return best_features
```


```python
# Determine optimal feature set for Decision Tree model
# (takes some time to run)

X = arrhythmias[all_features]
y = arrhythmias['Rhythm']

tree_model = DecisionTreeClassifier(random_state=RANDOM_STATE)

best_features = optimize_features_and_dimension(X, y, tree_model)
best_features
```




    ['VentricularRate', 'AtrialRate']




```python
# Build and test Decision Tree model

X_train, X_test, y_train, y_test = train_test_split(X[best_features], y, test_size=0.25, random_state=RANDOM_STATE)

# Test various max depths from 2 to 10 and select model with best accuracy
best_acc = 0
best_depth = 2
for depth in range(2, 11):
    tree_model.max_depth = depth
    tree_model.fit(X_train, y_train)
    
    y_pred = tree_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    if acc > best_acc:
        best_depth = depth
        best_acc = acc

# Build model with optimal max depth
tree_model.max_depth = best_depth
tree_model.fit(X_train, y_train)

y_pred = tree_model.predict(X_test)

print(f'max_depth = {tree_model.max_depth}\n')
print(classification_report(y_test, y_pred, digits=3, zero_division=0))
```

    max_depth = 9
    
                  precision    recall  f1-score   support
    
              AF      0.149     0.069     0.095       101
            AFIB      0.785     0.730     0.757       434
              AT      0.000     0.000     0.000        24
           AVNRT      0.000     0.000     0.000         6
            AVRT      0.000     0.000     0.000         1
              SA      0.000     0.000     0.000        89
           SAAWR      0.000     0.000     0.000         2
              SB      0.987     0.990     0.989      1006
              SR      0.750     0.996     0.856       459
              ST      0.905     0.963     0.933       404
             SVT      0.639     0.743     0.687       136
    
        accuracy                          0.852      2662
       macro avg      0.383     0.408     0.392      2662
    weighted avg      0.806     0.852     0.825      2662
    


#### Interpretation:

Based on the classification report, it is clear that in many cases the Decision Tree did not do so well, with an accuracy of 0.852, a weighted average recall of 0.832, and bad F1-scores for most rhythms. I'm sure this is partly due to the fact that we have so many different arrhythmias but a relatively small dataset (in some cases we only have a few observations of a particular arrhythmia); notice how the rhythms with higher support (more observations) have better performance. For the last four heart rhythms (SB, SR, ST, SVT) the model was not so bad.
<br><br>
Instead of just looking at how accurate the model was in predicting each unique heart rhythm, I want to see how often the Decision Tree misclassifies an abnormal heart rhythm as normal (false negative). With previous classifiers, I did this by computing the recall score. However, since the Decision Tree is a multiclass classifier, I first need to map the heart rhythms to a binary value (True/False) and then fit the model again. This way, although the Decision Tree won't predict specific heart rhythms, it would at least predict when the heart rhythm is abnormal.
<br><br>
If you look at the rhythm names below, you will see one called "SR" which stands for "Sinus Rhythm". SR is a normal heart rhythm, so we will consider this as "doesn't have arrhythmia", i.e. `False`, and we will consider the other rhythms as "has arrhythmia", i.e `True`. The code below maps "SR" heart rhythms to `False` and all other rhythms to `True`.


```python
# See all heart rhythms
rhythm_names_df
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
      <th>Acronym Name</th>
      <th>Full Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SB</td>
      <td>Sinus Bradycardia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SR</td>
      <td>Sinus Rhythm</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AFIB</td>
      <td>Atrial Fibrillation</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ST</td>
      <td>Sinus Tachycardia</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AF</td>
      <td>Atrial Flutter</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SI</td>
      <td>Sinus Irregularity</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SVT</td>
      <td>Supraventricular Tachycardia</td>
    </tr>
    <tr>
      <th>7</th>
      <td>AT</td>
      <td>Atrial Tachycardia</td>
    </tr>
    <tr>
      <th>8</th>
      <td>AVNRT</td>
      <td>Atrioventricular  Node Reentrant Tachycardia</td>
    </tr>
    <tr>
      <th>9</th>
      <td>AVRT</td>
      <td>Atrioventricular Reentrant Tachycardia</td>
    </tr>
    <tr>
      <th>10</th>
      <td>SAAWR</td>
      <td>Sinus Atrium to Atrial Wandering Rhythm</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Map rhythms to binary outcome
y_binary = (y != 'SR')
```


```python
# Build and test Decision Tree model

binary_tree_model = DecisionTreeClassifier(random_state=RANDOM_STATE)

# Use best_features from before (VentricularRate and AtrialRate)
X_train, X_test, y_train, y_test = train_test_split(X[best_features], y_binary, test_size=0.25, random_state=RANDOM_STATE)

# Test various max depths from 2 to 10 and select model with best accuracy
best_acc = 0
best_depth = 2
for depth in range(2, 11):
    binary_tree_model.max_depth = depth
    binary_tree_model.fit(X_train, y_train)
    
    y_pred = binary_tree_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    if acc > best_acc:
        best_depth = depth
        best_acc = acc

# Build model with optimal max depth
binary_tree_model.max_depth = best_depth
binary_tree_model.fit(X_train, y_train)

y_pred = binary_tree_model.predict(X_test)

print(f'max_depth = {binary_tree_model.max_depth}\n')
print_report(y_test, y_pred)
```

    max_depth = 10
    
    accuracy = 0.941397445529677
    precision = 0.9990248659190639
    recall = 0.9300953245574217
    F1-score = 0.9633286318758816


#### Interpretation:

Now we have a pretty strong tree model with an accuracy of 0.941 and a recall score of 0.930, compared with those of the multiclass Decision Tree (0.852 for both accuracy and weighted average recall). The recall score of the binary Decision Tree is also much better than the recall score of the KNN model (0.795); so, while the KNN model failed to diagnose about 20% of AFib patients, the binary Decision Tree only failed to recognize about 7% of patients as having abnormal heart rhythms, even if it couldn't always tell exactly which arrhythmia it was. It may be unfair to compare the two models since the KNN model was focused on AFib while the binary Decision Tree clumped all arrhythmias into one group, but in any case the binary Decision Tree performed pretty well (much better than the multiclass Decision Tree).
<br><br>
Now, ideally, I'd want the accuracy to be at least 95% and the recall to be closer to 100% before I can comfortably say it is ready for clinical implementation; however, if any of our models are to be clinically deployed, it is definitely the binary Decision Tree. It performs very well and its numbers would be considered great in many contexts, but I still think it needs a little more improvement before it can be used medically.
<br><br>
I think it's worth noting that usually the benefit of the Decision Tree is that it can predict many different classes, as opposed to just one. We did not take advantage of this in the binary Decision Tree, and instead we made it easier for the model to correctly predict if some abnormality exists. The model is still worth something, but a truly sophisticated model would be able to do both: predict multiple heart rhythms AND perform with very high accuracy and recall.

<hr>

## 4) Conclusion

To summarize, we developed three promising models:

1. **KNN** - predicts AFib
    * accuracy = 0.938
    * recall = 0.795
    * Performs with high accuracy, but needs a significantly higher recall score to be considered for clinical use.
   
   
2. **Multiclass Decision Tree** - predicts multiple heart rhythms
    * accuracy = 0.852
    * weighted average recall = 0.852
    * Overall accuracy is not horrible but should be improved. Performs very poorly for some rhythms, and pretty well for other rhythms; a larger dataset would improve performance for more heart rhythms. Not ready for clinical use, unless we cut out the rhythms for which the model performs poorly.
    
    
3. **Binary Decision Tree** - predicts abnormal heart rhythm
    * accuracy = 0.941
    * recall = 0.930
    * Performs with high accuracy and high recall; best candidate for clinical use. Still needs a little improvement in recall score (ideally >95%) to truly be ready for clinical use.


```python
knn_model  # predicts AFib
tree_model  # predicts multiple heart rhythms
binary_tree_model  # predicts abnormal heart rhythm

pass
```
