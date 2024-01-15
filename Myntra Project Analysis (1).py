#!/usr/bin/env python
# coding: utf-8

# #### PROJECT - Myntra Fasion Clothing
# 

# #### --------------------------------------------------------------------------------------

# ### ->FIRST WE IMPORT DIFFERENT PYTHON LIBRARIES WHICH WE WILL USE IN OUR PROJECT

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import math

warnings.filterwarnings("ignore")


# ### ->IMPORTING THE EXCEL FILE DATA

# In[4]:


data=pd.read_csv("Myntra Fasion Clothing.csv")


# In[3]:


#pd.set_option("display.max_rows", None, "display.max_columns", None)


# In[4]:


data.head()


# ## ->First we change the column order as we need to check the product_id first then at the end we need to see URL of the product which makes the dataset easy to read
# 

# In[5]:


new_order = [1,2,3,4,5,6,7,8,9,10,11,12,0]
data=data[data.columns[new_order]]


# In[6]:


data.head()


# ### *------------------*---------------------------*

# ## Getting dimension or shape of dataset ->

# In[7]:


data.shape


# ## Using pandas info() to find the non-null values and data types of different columns ->

# In[8]:


data.info()


# 
# ### This dataset contains float, int, object values

# ### ----------------------------------------

# ## Using pandas .isnull().sum()  to get the count of sum of all null values present ->

# In[9]:


data.isnull().sum() 


# ### -> There are 4 columns with  null values i.e.
# 1. DiscountPrice (in Rs)   - 193158
# 2. DiscountOffer           -  74306
# 3. Ratings                 - 336152
# 4. Reviews                 - 336152

# # DATA CLEANING

# ## As we know that the first column with null values is "DiscountPrice (in Rs)" 

# ### Getting the counts of each value present in the "DiscountPrice (in Rs)" column ->

# In[10]:


data["DiscountPrice (in Rs)"].value_counts()


# In[10]:


## Getting the statistical values of column 'DiscountPrice (in Rs)' using describe() funtion
data['DiscountPrice (in Rs)'].describe()


# #### As we can see there is lot of variations min max value so we can"t use any statistical value directly related to this columns

# # we can use  price values related to different categories ->

# In[11]:


data['Category'].value_counts()


# # Using groupby to get the mean of different discount price according to different category ->

# In[12]:


data.groupby('Category').mean()['DiscountPrice (in Rs)']


# # In the following method, we have used groupby() and transform() functions to replace the mean of different  Null values->
# 

# In[13]:


data['DiscountPrice (in Rs)'] = data['DiscountPrice (in Rs)'].fillna(data.groupby('Category')['DiscountPrice (in Rs)'].transform('mean'))


# In[14]:


data['DiscountPrice (in Rs)'].describe() 


# ### Conclusion
# #### As we can see there is not that much change in statistical values because we add mean values according to different categories 

# In[15]:


data.isnull().sum() # now we succefully remove the null values of this column


# ### conclusion 
# #### the null values of the column 'DiscountPrice (in Rs)' is seccessfully removed
# 

# ###  --------                     ----------------------                                --------------------

# # CLEANING NULL VALUES OF COLUMN 'Ratings'

# In[ ]:


data['Ratings'].value_counts()


# In[16]:


data['Ratings'].describe() # we can't use mean value in null values here because mean is close to max value which is not a good odea to use in this 


# #### By checking the statistical values we see we can't use mean value in null values here because mean is close to max value which is not a good 

# # 

# # Plotting boxplot and checking if there is ouliers ->

# In[17]:


sns.boxplot(data.Ratings)


# ## Let's see if these are outliers or not

# In[20]:


data.Ratings.quantile([0.5,0.7,0.9])


# #### No outliers is present 

# # 

# In[21]:


data.groupby('Category').mean()['Ratings'] 


# ####  we can't use mean according to each category so we have to drop this idea too

# ## Lets find the mean of rating with respect to the 'Individual_category' ->

# In[22]:


data.groupby('Individual_category').mean()['Ratings']


# ### ->Conclusion
# #### As we can see the variation of mean using the individual category is a good idea 
# #### But there are also some null values whose mean is not shown because of less value counts so we ahve to remove those numbers

# In[23]:


data['Individual_category'].value_counts()


# ##  Removing the Individual category data with less value counts

# In[24]:


counts = data['Individual_category'].value_counts()

data = data[~data['Individual_category'].isin(counts[counts < 11].index)]


# In[25]:


data['Individual_category'].value_counts()


# In[26]:


data.groupby('Individual_category').mean()['Ratings'] 


# In[27]:


data["Ratings"].describe()


# ### -> As we can see the statistical values of the ratings does not changed after removing some data of individual category

# ## Filling the Average values in null values in 'Ratings' columns with respect to the 'Individual_category' columns

# In[28]:


data['Ratings'] = data['Ratings'].fillna(data.groupby('Individual_category')['Ratings'].transform('mean'))


# In[29]:


data["Ratings"].describe() 


# ## Colnclusion
# ### So after filling the null values there is not much change in statistcial data

# ## -------------------------------------------------------------------------

# # 

# # ->Cleaning null values from the column 'DiscountPrice (in Rs)'

# In[30]:


data["DiscountPrice (in Rs)"].isnull().sum()


# ## Statistical values of column->

# In[31]:


data["DiscountPrice (in Rs)"].describe()


# # Checking Outliers using boxplot ->

# In[32]:


sns.boxplot(data["DiscountPrice (in Rs)"])


# 
# #### As we can see here is continuous increase upto 20000 but after that some outliers can be found

# # 

# In[38]:


data["DiscountPrice (in Rs)"].quantile([0.5,0.7,0.9,0.95,0.99,1])


# In[39]:


## As we can see after 99% there is outliers  but they are not affecting the data so we can ignore them


# # 

# ## Filling the Average values in null values in 'DiscountPrice (in Rs)' columns with respect to the 'Individual_category' columns  ->

# In[40]:


data['DiscountPrice (in Rs)'].value_counts().isnull()


# In[41]:


data['DiscountPrice (in Rs)'] = data['DiscountPrice (in Rs)'].fillna(data.groupby('Individual_category')['DiscountPrice (in Rs)'].transform('mean'))


# In[42]:


data['DiscountPrice (in Rs)'].isnull().sum()


# ## Removing the remaining null values  ->

# In[43]:


data.dropna(subset=['DiscountPrice (in Rs)'], inplace=True)


# In[44]:


data['DiscountPrice (in Rs)'].isnull().sum()


# In[45]:


sns.boxplot(data["DiscountPrice (in Rs)"])


# ### ---------------------------------------------------

# ## ->Cleaning null values from the column 'DiscountOffer'

# In[46]:


data.isnull().sum() 


# ### First we have to convert column datatype into Integer
# ### we have to change the discount price missing values according to discount offer 

# In[47]:


#data["DiscountOffer)"]=data["DiscountOffer"].str.replace(" OFF","% OFF")
data["DiscountOffer"]=data["DiscountOffer"].str.replace('[OFF,"OFF",Hurry*,' ',"%","Rs. "]', '')


# In[48]:


data["DiscountOffer"].value_counts()


# In[49]:


data['DiscountOffer'] = data['DiscountOffer'].fillna(0)


# ### Conclusion
# #### We fill the null values as 0 because there is no discount on those products

# In[50]:


data["DiscountOffer"].isnull().sum()


# ### ------------------------------------------------------------

# ## Cleaning null values from the column "Reviews" ->

# In[51]:


data["Reviews"].value_counts().head()


# ## Checking outliers using boxplot ->

# In[52]:


sns.boxplot(data.Reviews)


# #### we cannot treat them as outliers as there are feww highers values as continuous

# # 

# In[53]:


data.groupby('Individual_category').mean()['Reviews'] 


# In[54]:


data['Reviews'].describe()


# ## We can use forward and backword fill in reviews as in many cases the forwrd and backword values are same ->

# In[55]:


data['Reviews'] = data['Reviews'].fillna(method="ffill")
#data['Reviews'] = data['Reviews'].interpolate()


# ##### we are getting the same result as using the .interpolate() method which is use to guess the missing values in data
# 

# In[56]:


data['Reviews'].describe()


# In[57]:


data.isnull().sum()


# ### ----------------------------------------------------------

# # REMOVING outliers from column "Orignal price"

# In[58]:


data["OriginalPrice (in Rs)"].describe()


# In[59]:


sns.boxplot(data["OriginalPrice (in Rs)"])


# In[60]:


data["OriginalPrice (in Rs)"].quantile([0.5,0.7,0.9,0.95,0.96,0.97,0.99])


# #### As we can see there is outliers after 97% so we have to remove them

# In[62]:


min_thresold=data["OriginalPrice (in Rs)"].quantile(0.97)


# In[63]:


data.shape


# ## Removing the ouliers which is higher then 0.97 ->

# In[64]:


data=data[data["OriginalPrice (in Rs)"]<min_thresold]


# In[65]:


data.shape


# In[66]:


sns.boxplot(data["OriginalPrice (in Rs)"])


# # --------------------------------------------------------------------

# ## UNIVARIATE ANALYSIS

# In[ ]:


data.head()


# In[67]:


ax=plt.figure(figsize=[8,6])
explode = (0.1, 0)
data["category_by_Gender"].value_counts().plot.pie(explode=explode,autopct = "%.2f%%", shadow=True,colors = ['blue', 'yellow'],startangle=90)
plt.legend()


# ## Conclusion
# ### This graph shows the difference in male and female products using bar graph

# In[68]:


ax=plt.figure(figsize=[8,6])
data["Category"].value_counts().plot.barh(color='blue',edgecolor='red')
plt.legend()


# ## Conclusion
# ### This graph shows the difference in Category of different products using bar graph

# In[69]:


ax=plt.figure(figsize=[8,6])
data["BrandName"].value_counts().head(10).plot.barh(50,200,color='yellow',edgecolor='blue')
plt.legend()


# ## Conclusion
# ### This graph shows the TOP -10 product brand using bar graph
# 

# In[70]:


ax=plt.figure(figsize=[18,16])
data["Individual_category"].value_counts().plot.barh(color='green',edgecolor='red')
plt.legend()


# ## Conclusion
# ### This graph shows the TOP-20 product brand using bar graph
# 

# # 

# In[89]:


ax=plt.figure(figsize=[18,6])
data.Ratings.plot.hist(bins=40)
plt.show()


# ## Conclusion
# ### This graph shows the where the most ratings area of products using histogram4
# ### lie in 3.5 - 4.5

# # 

# In[88]:


ax=plt.figure(figsize=[18,6])
data.Reviews.plot.hist(bins=10)
plt.show()


# ## Conclusion
# ### This graph shows the where the most Reviews area of products using histogram4
# ### lie in 0-100

# # Bivariate Analysis

# In[ ]:


data.head()


# In[73]:


ax=plt.figure(figsize=[8,6])
data.groupby('Category').mean()['DiscountPrice (in Rs)'].head(15).plot.barh(color='green',edgecolor='red')
plt.legend()


# ## Conclusion
# ### It shows the mean discount on each category
# ### it is found that western wear has high mean descount price

# In[74]:


ax=plt.figure(figsize=[16,20])
data.groupby('Individual_category').mean()['DiscountPrice (in Rs)'].plot.barh(color='green',edgecolor='red')
plt.legend()


# ## Conclusion
# ### It shows the mean discount of all individual clothes
# ### it is found that suits ahev high discount price

# In[75]:


ax=plt.figure(figsize=[10,6])
data.groupby('Category').mean()['OriginalPrice (in Rs)'].plot.barh(color='blue',edgecolor='red')
plt.legend()


# ## Conclusion
# ### It shows the mean price of all individual category of clothes
# ### it is found that the indian wear has high mean price on myntra

# In[76]:


ax=plt.figure(figsize=[16,20])
data.groupby('Individual_category').mean()['OriginalPrice (in Rs)'].plot.barh(color='blue',edgecolor='red')
plt.legend()


# ## Conclusion
# ### It shows the mean price of all individual clothes
# ### it is found that suits are most expnesive average price on myntra platform

# # Multivariate

# In[77]:


data.corr()


# In[78]:


# displaying the plotted heatmap
ax=plt.figure(figsize=[9,7])
sns.heatmap(data = data.corr(),cmap="Blues",annot=True) 
plt.show()


# ## Conclusion
# ### this heatmap shows the realtionship of data as using correaltion 

# In[74]:


sns.pairplot(data,size=2.5)


# ## Conclusion-
# ### Using Pair plot we found the most relation of where the most of data lies when compare with other columns
# #### for example we c an see higher the reviews for high ratings 

# In[ ]:


data.head()


# In[93]:


plt.plot(data_Roadster["Reviews"],np.zeros_like(data_Roadster["Reviews"]),'o')
plt.plot(data_Pothys["Reviews"],np.zeros_like(data_Pothys["Reviews"]),'o')
plt.plot(data_KALINI["Reviews"],np.zeros_like(data_KALINI["Reviews"]),'o')
plt.show()


# ## Hypothesis Testing

# In[78]:


# importing the library for t-test 
#let alpha value be 0.05 or 5%

rating_mean=np.mean(data["Ratings"])
rating_mean


# # NULL hypothesis will say there is no difference in mean of ratings while alternate will say opposite

# In[80]:


from scipy.stats import ttest_1samp
# creating random sample from the column rating
sample_size=10000
rating_sample=np.random.choice(data["Ratings"],sample_size) 


# In[91]:


# finding p-value for sample
ttest,p_value=ttest_1samp(rating_sample,4)


# In[92]:


print(p_value)


# In[93]:


if(p_value < 0.01):
    print("We are rejecting the null hypothesis")
else:
    print("we are accepting the null hypothesis")


# # AS WE FOUND THAT IF WE TAKE LARGE DATA THE P-VALUE DECREASE TO 0.00 SO WE CANT USE P-VALUE TEST FOR THIS DATA

# In[81]:


## checking if ratings are normal distribur=tion or not
from matplotlib import pyplot
pyplot.figure(figsize=(14,6))
pyplot.hist(data["Ratings"])
pyplot.show()


# In[ ]:


#Report

Inferences –
As we can analyze that Roadster is mostly available brandfor the men in the Myntra
get_ipython().run_line_magic('pinfo', 'MYNTRA')
Inference – As we can see Western clothes is mostly available on Myntra .
Suggestions-
We should analyze what kind of categoryis mainly available for men.
get_ipython().run_line_magic('pinfo', 'WOMEN')
Inferences – As we can see Top Wear is mainly available for men while Western is for women which shows that western is not famous in men .
WHICH BRAND CLOTHES IS MOSTLY AVAILABLE ON MYNTRA ?
Inferences – Here we can see that Pothys is the brand which is mostly available and then Roadster. These two brand are quite famous that’s why present in large quantity on Myntra platform
CONCLUSION -
•Those products which have less Ratings and reviews we can recheck them and there quality why they have negative ratings take some action on selling them.

•We can advertise those products who has high and reviews to make it more profitable.

•We can make good partnership with those brands which have positive reviews or ratings by customers.

•We can increase discount offer on those products which have low ratings to increase there demand in market

