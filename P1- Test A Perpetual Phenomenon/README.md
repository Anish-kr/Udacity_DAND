
# UDACITY DATA ANALYST NANODEGREE

## P1- TEST A PERCEPTUAL PHENOMENON

**Author-[ANISH KUMAR](https://github.com/Anish-kr)  
Date    - April 25, 2017 **


## Background Information

In a Stroop task, participants are presented with a list of words, with each word displayed in a color of ink. The participant’s task is to say out loud the color of the ink in which the word is printed. The task has two conditions: a congruent words condition, and an incongruent words condition. In the congruent words condition, the words being displayed are color words whose names match the colors in which they are printed: for example <span style="color:blue">BLUE</span>,<span style="color:red">RED</span>. In the incongruent words condition, the words displayed are color words whose names do not match the colors in which they are printed: for example <span style="color:green">PURPLE</span>,<span style="color:blue">ORANGE</span>. In each case, we measure the time it takes to name the ink colors in equally-sized lists. Each participant will go through and record a time from each condition.

## Questions For Investigation

1.. What is our independent variable? What is our dependent variable?

* Independent variable: the words and their color(congruent words or incongruent words)

* Dependent variable: Time taken to name the ink colors in equally-sized lists

2..  What is an appropriate set of hypotheses for this task? What kind of statistical test do you expect to perform? Justify your choices.



+ The null hypothesis in this case should be that the mean time taken to name the ink colors in the congruent condition  should be greater than equal to the mean time taken to name the ink colors in the incongruent condition. 
+ And the alternative hypothesis for this task is that the mean time taken to name the ink colors in congruent condition should be less than the mean time taken to name the ink colors in incongruent condition.
+ From the above itself, we can say that we need to perform one-tailed test where we are interested only in the fact that time taken to read the ink color in congruent condition is less or not. We are not interested in the fact that time taken to read the ink color in congruent condition is more than in the case of incongruent condition because not only it is intuitive but and also clearly stated [here](https://en.wikipedia.org/wiki/Stroop_effect)

Therefore if we assume-

+ $\mu$<sub>C</sub> to be mean time taken to name the ink colors in the congruent condition
+ $\mu$<sub>I </sub> to be mean time taken to name the ink colors in the incongruent condition
+ H<sub>0</sub> to denote the NULL Hypothesis
+ H<sub>A</sub> to denote the Alternative Hypothesis

Then- 

+ H<sub>0</sub> : $\mu$<sub>C</sub> $\geqslant$ $\mu$<sub>I </sub>
+ H<sub>A</sub> : $\mu$<sub>C</sub> < $\mu$<sub>I </sub>

Now we Since the sample is small( size(n) < 30 ) and the population parameters are not known, We would go for the ** t-test. **And since from what we have already discussed before, We can say that we are going for ** One-tailed t-test**

##### The assumptions for the t-test for dependent samples are the following-
 * Your dependent variable should be measured on a continuous scale (i.e., it is measured at the interval or ratio level). Examples of variables that meet this criterion include revision time (measured in hours), intelligence (measured using IQ score), exam performance (measured from 0 to 100), weight (measured in kg), and so forth.
 
 * Your independent variable should consist of two categorical, "related groups" or "matched pairs". "Related groups" indicates that the same subjects are present in both groups. The reason that it is possible to have the same subjects in each group is because each subject has been measured on two occasions on the same dependent variable. 
 
 * There should be no significant outliers in the differences between the two related groups. Outliers are simply single data points within your data that do not follow the usual pattern.he problem with outliers is that they can have a negative effect on the dependent t-test, reducing the validity of your results. In addition, they can affect the statistical significance of the test.
 
 * The distribution of the differences in the dependent variable between the two related groups should be approximately normally distributed. We talk about the dependent t-test only requiring approximately normal data because it is quite "robust" to violations of normality, meaning that the assumption can be a little violated and still provide valid results.

The t-test should be of the dependent samples variety because the same subject is exposed to two conditions and tested for each, which are the defining criteria for "within-subjects" or "repeated-measures" statistical tests.[Read More](https://statistics.laerd.com/statistical-guides/dependent-t-test-statistical-guide.php)

Now to carry out the test we are provided with a dataset "stroopdata.csv".


```python
import seaborn as sns#improves the look of the graphs
%pylab inline        
#prevents graph from flowing out of the active window

import matplotlib.pyplot as plt
import numpy as np   #for handling arrays
import pandas as pd  #for handling dataframes
```

    Populating the interactive namespace from numpy and matplotlib
    


```python
## Reading the data in a suitable variable
data = pd.read_csv("stroopdata.csv")
data.head() # displaying the dataset
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Congruent</th>
      <th>Incongruent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12.079</td>
      <td>19.278</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16.791</td>
      <td>18.741</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.564</td>
      <td>21.214</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.630</td>
      <td>15.687</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14.669</td>
      <td>22.803</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Since we are very much interested in the difference of the two scores we'll create another column denoting the same.
diff=data
diff['Difference'] = data['Congruent']-data['Incongruent'] ##Using vectorized operations on dataframe

diff

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Congruent</th>
      <th>Incongruent</th>
      <th>Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12.079</td>
      <td>19.278</td>
      <td>-7.199</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16.791</td>
      <td>18.741</td>
      <td>-1.950</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.564</td>
      <td>21.214</td>
      <td>-11.650</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.630</td>
      <td>15.687</td>
      <td>-7.057</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14.669</td>
      <td>22.803</td>
      <td>-8.134</td>
    </tr>
    <tr>
      <th>5</th>
      <td>12.238</td>
      <td>20.878</td>
      <td>-8.640</td>
    </tr>
    <tr>
      <th>6</th>
      <td>14.692</td>
      <td>24.572</td>
      <td>-9.880</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.987</td>
      <td>17.394</td>
      <td>-8.407</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9.401</td>
      <td>20.762</td>
      <td>-11.361</td>
    </tr>
    <tr>
      <th>9</th>
      <td>14.480</td>
      <td>26.282</td>
      <td>-11.802</td>
    </tr>
    <tr>
      <th>10</th>
      <td>22.328</td>
      <td>24.524</td>
      <td>-2.196</td>
    </tr>
    <tr>
      <th>11</th>
      <td>15.298</td>
      <td>18.644</td>
      <td>-3.346</td>
    </tr>
    <tr>
      <th>12</th>
      <td>15.073</td>
      <td>17.510</td>
      <td>-2.437</td>
    </tr>
    <tr>
      <th>13</th>
      <td>16.929</td>
      <td>20.330</td>
      <td>-3.401</td>
    </tr>
    <tr>
      <th>14</th>
      <td>18.200</td>
      <td>35.255</td>
      <td>-17.055</td>
    </tr>
    <tr>
      <th>15</th>
      <td>12.130</td>
      <td>22.158</td>
      <td>-10.028</td>
    </tr>
    <tr>
      <th>16</th>
      <td>18.495</td>
      <td>25.139</td>
      <td>-6.644</td>
    </tr>
    <tr>
      <th>17</th>
      <td>10.639</td>
      <td>20.429</td>
      <td>-9.790</td>
    </tr>
    <tr>
      <th>18</th>
      <td>11.344</td>
      <td>17.425</td>
      <td>-6.081</td>
    </tr>
    <tr>
      <th>19</th>
      <td>12.369</td>
      <td>34.288</td>
      <td>-21.919</td>
    </tr>
    <tr>
      <th>20</th>
      <td>12.944</td>
      <td>23.894</td>
      <td>-10.950</td>
    </tr>
    <tr>
      <th>21</th>
      <td>14.233</td>
      <td>17.960</td>
      <td>-3.727</td>
    </tr>
    <tr>
      <th>22</th>
      <td>19.710</td>
      <td>22.058</td>
      <td>-2.348</td>
    </tr>
    <tr>
      <th>23</th>
      <td>16.004</td>
      <td>21.157</td>
      <td>-5.153</td>
    </tr>
  </tbody>
</table>
</div>



3.. Report some descriptive statistics regarding this dataset. Include at least one measure of central tendency and at least one measure of variability.


```python
#displaying the descriptive statistics measures
diff.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Congruent</th>
      <th>Incongruent</th>
      <th>Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>24.000000</td>
      <td>24.000000</td>
      <td>24.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>14.051125</td>
      <td>22.015917</td>
      <td>-7.964792</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.559358</td>
      <td>4.797057</td>
      <td>4.864827</td>
    </tr>
    <tr>
      <th>min</th>
      <td>8.630000</td>
      <td>15.687000</td>
      <td>-21.919000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>11.895250</td>
      <td>18.716750</td>
      <td>-10.258500</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>14.356500</td>
      <td>21.017500</td>
      <td>-7.666500</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>16.200750</td>
      <td>24.051500</td>
      <td>-3.645500</td>
    </tr>
    <tr>
      <th>max</th>
      <td>22.328000</td>
      <td>35.255000</td>
      <td>-1.950000</td>
    </tr>
  </tbody>
</table>
</div>



**Table 1.1**

4.. Provide one or two visualizations that show the distribution of the sample data. Write one or two sentences noting what you observe about the plot or plots.


```python
viz=sns.regplot(x='Congruent',y='Incongruent',scatter=True,data=diff);
viz.set_title('fig.1 Incongruent Condition versus Congruent Condition', fontsize=16);
```


![png](output_18_0.png)


From the above graph we notice that as the time for reading the ink-name for Incongruent condition increases, the time for congruent condition also increases. It is true in general sense as reading ability, and sharpness of mind depends on an individual. And that is why if one time increases for an individual so does the other.

Also we can notice two outliers, two individuals who take much larger time to perform in incongruent condition as compared to that in congruent condition.


```python
##Now Let us plot a histogram
viz2=data.plot(kind='hist',stacked=True,bins=15);
viz2.set_title("fig.2 Plotting a histogram of the given data", fontsize=16);
```


![png](output_20_0.png)


Above obtained is a histogram plotting of the given data. 

As we notice from the blue bars that represent the congruent condition, we find that that range of time taken in case of congruent condition is somewhere between 8 and 23. Mean time being at 14.05 units and mode being at 15.

While green bars that represent the incongruent tells us that time in this case vary from 15 to 35 units. Its is bidal data whith modes being somewhere at 18 and 23 units. 
Also there lies an outlier for this case at t equal 35.

5.. Now, perform the statistical test and report your results. What is your confidence level and your critical statistic value? Do you reject the null hypothesis or fail to reject it? Come to a conclusion in terms of the experiment task. Did the results match up with your expectations?


```python
from scipy.stats import ttest_ind
##Performing the test using pandas itself

ttest_ind(data['Congruent'],data['Incongruent'])
```




    Ttest_indResult(statistic=-6.5322505539032285, pvalue=4.5949489622951854e-08)




```python
##confidence interval for population mean of congruent data
import numpy as np, scipy.stats as st

st.t.interval(0.99, len(data['Congruent'])-1, loc=np.mean(data['Congruent']), scale=st.sem(data['Congruent']))
```




    (12.011452730450978, 16.090797269549029)




```python
# confidence interval for range of difference in time takes for the two cases
import numpy as np, scipy.stats as st

st.t.interval(0.99, len(diff['Difference'])-1, loc=np.mean(diff['Difference']), scale=st.sem(diff['Difference']))
```




    (-10.75255638808285, -5.1770269452504838)



+ Taking significance level($\alpha$) equal to 0.01 that corresponds to 99% significance level   
+ Degree of freedom(df)               equal to 23   
+ Therefore t<sub>crit</sub> from t-table equals -2.50   
+ Also mean time in congruent condition for the entire population will lie in range of 12.011 and 16.090

**Obtained value of t is -6.532250   
And p-value for corresponding statistic is << 0.01**

**Hence we can satisfactorily reject the Null **

### Hence we can say that mean time taken to name the ink colors in congruent condition is less than the mean time taken to name the ink colors in incongruent condition.
(There is sufficient evidence at the α = .01 level of significance to support the claim)

6.. Optional: What do you think is responsible for the effects observed?   
Can you think of an alternative or similar task that would result in a similar effect? Some research about the problem will be helpful for thinking about these two questions!

My hypothesis for the above results are that these are due to difference in the ability of the brain to recognize color of the word as compared to reading it. From the birth itself we spend a huge time reading texts. The action becomes so spontaneous that as soon a word appears, involuntarily end up reading it. Whereas the time devoted to recognizing color of the texts negligible when compared to previous case. So extra attention needs to be devoted to carry out the same. As a result a lag in time occurs.

Numerical/Physical size Stroop tasks, where numerical values and physical size are the factors that contribute to congruency/incongruency, results in a similar effect. It takes longer to recognize the number and physical size (two separate tasks) of small numbers that have a large physical size and large numbers that have a small physical size.[Read More..](https://en.wikipedia.org/wiki/Numerical_Stroop_effect)



## References

1. [Wikipedia: Numerical Stroop effect](https://en.wikipedia.org/wiki/Numerical_Stroop_effect)
2. [Wikipedia: More on StroopEffect](https://en.wikipedia.org/wiki/Stroop_effect)
3. [Applying color in Markdown](http://stackoverflow.com/questions/35465557/how-to-apply-color-in-markdown)
4. [Markdown help](http://www.rob-mcculloch.org/2017_bayes/webpage/R/rmarkdown1.pdf)
5. [More on dependent variables](https://statistics.laerd.com/spss-tutorials/dependent-t-test-using-spss-statistics.php)
6. [Drawing special symbols](https://math.meta.stackexchange.com/questions/21841/how-to-type-greater-than-or-equal-to-symbols)
7. [Drawing scatter plot using seaborn](http://seaborn.pydata.org/generated/seaborn.regplot.html)
8. [Drawing stacked bar chart](http://stackoverflow.com/questions/22226375/histogram-with-stacked-components)
9. [Carrying out t-test using pandas](http://stackoverflow.com/questions/13404468/t-test-in-pandas-python)

