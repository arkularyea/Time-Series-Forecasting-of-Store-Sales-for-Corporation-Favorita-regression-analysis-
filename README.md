# Time-Series-Forecasting-of-Store-Sales-for-Corporation-Favorita-regression-analysis-


## INTRODUCTION ##

Using data to make informed decisions is the best decision. Every day the amount of data available grows exponentially .as a result, effective interpretation is more important than ever. Data analytics is quickly becoming one of the world’s most exciting and rewarding career paths. Business analytics skills will most likely be in higher demand over the next decade than any other career (10.9% vs 5.2%). career of labor statistics).

Companies all over the world require qualified data analysts to solve problems and assist them in making the best business decisions possible. Currently ,59% of companies intend to add even more positions requiring data analysis skills (source: SHRM).

In this article, I will describe the process in achieving my outcome. This is a time series forecasting problem. this project, will predict store sales on data from Corporation Favorita, a large Ecuadorian-based grocery retailer. Specifically, how to build a model that accurately predicts the unit sales for thousands of items sold at different Favorita stores. The training data includes dates, store, and product information, whether that item was being promoted, as well as the sales numbers. 


Additional files include supplementary information that may be useful in building your models.


This will involve formulating and evaluating a hypothesis, preparing research questions, conducting analysis, and presenting insights through appropriate visualizations.


The goal of this regression analysis is to optimize stock management at Corporation Favorita by accurately predicting demand for products in order to ensure that the right quantity of each product is always in stock. I was also prepared to use the opportunity to develop and harness data analytic skills. 

By the time I am done with my investigations into corporation FAVORITA, I hope to have made smart and strategic decisions which is data driven. I hope to have asked the right questions, summarized data, connected business objectives to data analysis, identified and cleaned the data, created visualizations, and above all I hope to have told a data driven story.


Every data science or data analytics project follows a certain kind of data science process. Scrum, Kanban, and Agile are all methods that data science teams adopt to complete their projects but, in this project, I will be working with CRISP-DM.


 I used the cross industry standard process for data mining (CRISP-DM) model as the base for my data science process.it has six sequential phases:
1.	Business understanding - what does the business need?
2.	Data understanding – what data do we have/ need? Is it clean?
3.	Data preparation – how do I organize the data for modelling?
4.	Modelling- what modelling techniques should I apply?
5.	Evaluation – which model best meets the business objectives?
6.	Deployment- how do stakeholders access the results?

## THE DATA
The data for corporation favorita came in 7 seven different csv files, namely; stores, oil, train, sample_submission, test, holidays_event and transactions. 

File Descriptions and Data Field Information
train.csv

•	The training data, comprising time series of features store_nbr, family, and on promotion as well as the target sales.

•	store_nbr identifies the store at which the products are sold.

•	family identifies the type of product sold.

•	sales give the total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units (1.5 kg of cheese, for instance, as opposed to 1 bag of chips).

•	on promotion gives the total number of items in a product family that were being promoted at a store at a given date.

## test.csv

•	The test data, having the same features as the training data. You will predict the target sales for the dates in this file. 


•	The dates in the test data are for the 15 days after the last date in the training data.
transaction.csv


•	Contains date, store_nbr and transaction made on that specific date.


## sample_submission.csv
•	A sample submission file in the correct format.

## stores.csv

•	Store metadata, including city, state, type, and cluster.

•	cluster is a grouping of similar stores.


## oil.csv

•	Daily oil price which includes values during both the train and test data timeframes. (Ecuador is an oil-dependent country and its economic health is highly vulnerable to shocks in oil prices.)

## holidays_events.csv

•	Holidays and Events, with metadata
NOTE: Pay special attention to the transferred column. A holiday that is transferred officially falls on that calendar day but was moved to another date by the government.

 A transferred day is more like a normal day than a holiday. To find the day that it was celebrated, look for the corresponding row where type is Transfer.


For example, the holiday Independencies de Guayaquil was transferred from 2012-10-09 to 2012-10-12, which means it was celebrated on 2012-10-12. Days that are type Bridge are extra days that are added to a holiday (e.g., to extend the break across a long weekend). 

These are frequently made up by the type Work Day which is a day not normally scheduled for work (e.g., Saturday) that is meant to payback the Bridge.
•	Additional holidays are days added a regular calendar holiday, for example, as typically happens around Christmas (making Christmas Eve a holiday).

## Additional Notes
•	Wages in the public sector are paid every two weeks on the 15th and on the last day of the month. Supermarket sales could be affected by this.

•	A magnitude 7.8 earthquake struck Ecuador on April 16, 2016. People rallied in relief efforts donating water and other first need products which greatly affected supermarket sales for several weeks after the earthquake.


## ASK STAGE 
At this stage, we bring the objective into view and put down the questions that we intend to answer at the end of the analysis process. The first phrase of the data analysis process is asking the right questions.


Here, with the overarching goal of making a recommendation to the team to assist their goal of entering the Indian startup ecosystem, the whole picture was considered to make sure we get the situation right. The following hypothesis was stated and questions were asked to guide the analyses.

it's important to use new data when evaluating our model to prevent the likelihood of overfitting to the training set. However, sometimes it's useful to evaluate our model as we're building it to find that best parameters of a model - but we can't use the test set for this evaluation or else we'll end up selecting the parameters that perform best on the test data but maybe not the parameters that generalize best.

## HYPOTHESIS

## Number 1 ## 

H0 - Store sales are affected by a number of factors, including holidays, promotions, and the overall economy.

H1 - Store sales are not affected by the following factors; holidays, promotions, and the overall economy.

## Number 2 ##

H0 - The type of day does not play a significant role in determining the demand for oil

H1 - the type of day plays a significant role in determining the demand for oil

## Number 3 ##

H0 - The location does not have an impact for the for the demand for oil

H1 - The location has an impact for the demand for oil

## Number 4 ##

H0 - There is no significant correlation between oil price and increase sales

H1 - There is significant correlation between oil price and increase sales

## RESEARCH QUESTIONS 
1. Is the train dataset complete (has all the required dates)?

2. Which dates have the lowest and highest sales for each year?

3. Are certain groups of stores selling more products? (Cluster, city, state, type)

4. Are sales affected by promotions, oil prices and holidays?

5. What analysis can we get from the date and its extractable features?

6. What is the difference between RMSLE, RMSE, MSE (or why is the MAE greater than all of them?)

7.  What is the relationship between oil prices and sales?

8. What is the relationship between product and sales?

9. What is the trend of sales overtime?

10. What is the relationship between oil prices and promotion?

11. what is the best method for forecasting sales?


DATA PREPARATION AND PROCESSING

Here I organize the data to make it fit for analysis. Cleanliness and consistency of data are the objectives promotion? to make sure that all datatypes are correct.
Here are a few steps that you can use to validate your time series machine learning models:

•	Compare the results of your model with those of a baseline method, such as a simple moving average.

•	Compare the predictions of your model against actual data.

•	Use rolling windows to test how well the model performs on data that is one step or several steps ahead of the current time point.

•	Compare the predictions of your model against those made by a human expert.

•	Use machine learning techniques, such as k-fold cross-validation, to test the generalization accuracy of your model.


## LOADING PACKAGES

To start with, the basic packages for analysis were loaded into my jupyter notebook. These packages were:

Pandas: for data cleaning and manipulation

NumPy: for data cleaning and manipulation 

Glob: a module that has several functions, that can help in listing files under a specified folder.

Matplotlib: visualization tool

Seaborn

# Library for EDA
import pandas as pd

import NumPy as np 

import seaborn as sns


%Matplotlib inline

import matplotlib. pyplot as plt

import matplotlib. dates as mdates


from sklearn. impute import SimpleImputer

from pandas_profiling import ProfileReport

import warnings

warnings. filterwarnings('ignore')

## GENERAL NOTES FROM PREVIEWING THE DATA FRAMES
•	All the columns with amounts have to be set to float.

•	Upon examining the data frame, I discovered that some of the columns contain values that should be numerical but are currently strings (objects). I will need to convert the datatypes of the values in these columns to numerical (float and/or integer).

•	Also, there were a considerable number of null values in the datasets

•	Data inspected for null values

## ASSUMPTIONS
1.	Imputations will not be made for undisclosed and/or unavailable (missing) amounts due to the uncertainties, risks of misstatements and possible misleading effects on the analyses.

2.	All other things been equal (ceteris paribus)
DATA CLEANING
the major activities performed on the Data Frames with respect to data cleaning are explain below.
 The detailed functions will be found in the jupyter notebook, a link to which will be attached at the end of the article.
		

## ANSWERING RESEARCH QUESTIONS

At this point, I combine the analyses and share stages of the data analysis process the coding and visualization of the merged data.



## CONCLUSION 
In conclusion, we accept the null hypothesis for number 1and number 4, whereas rejecting the null hypothesis for number 2 and 3. Thank you 


## FINAL NOTES 

Thank you so much for reading.
I will be grateful for your comments, advice, suggestions, and recommendations. Too long or too short? Too detailed or missing some details? Please let me know. You can leave a comment here or find me on Twitter (@arku_laryea).

I’m also open to collaborating on projects.


Link to the project repository on GitHub: https://github.com/arkularyea/Time-Series-Forecasting-of-Store-Sales-for-Corporation-Favorita-regression-analysis-.git

LINKEDIN : https://www.linkedin.com/pulse/time-series-forecasting-store-sales-corporation-analysis-nii-laryea


MEDIUM: 