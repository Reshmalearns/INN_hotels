Problem statement
download.png

INN Hotels

A significant number of hotel bookings are called off due to cancellations or no-shows. The typical reasons for cancellations include change of plans, scheduling conflicts, etc.
This is often made easier by the option to do so free of charge or preferably at a low cost which is beneficial to hotel guests but it is a less desirable and possibly revenue-diminishing factor for hotels to deal with.
Such losses are particularly high on last-minute cancellations.
The new technologies involving online booking channels have dramatically changed customers’ booking possibilities and behavior. This adds a further dimension to the challenge of how hotels handle cancellations, which are no longer limited to traditional booking and guest characteristics.
The cancellation of bookings impacts a hotel on various fronts:
Loss of resources (revenue) when the hotel cannot resell the room.
Additional costs of distribution channels by increasing commissions or paying for publicity to help sell these rooms.
Lowering prices last minute, so the hotel can resell a room, resulting in reducing the profit margin.
Human resources to make arrangements for the guests.
Objective

The increasing number of cancellations calls for a Machine Learning based solution that can help in predicting which booking is likely to be canceled.
INN Hotels Group has a chain of hotels in Portugal, they are facing problems with the high number of booking cancellations and have reached out to your firm for data-driven solutions.
Aim of INN Hotels

Analyze the data provided to find which factors have a high influence on booking cancellations, build a predictive model that can predict which booking is going to be canceled in advance, and help in formulating profitable policies for cancellations and refunds
Data Description
Data Dictionary

The data contains the different attributes of customers' booking details. The detailed data dictionary is given below.

Booking_ID: the unique identifier of each booking

no_of_adults: Number of adults
no_of_children: Number of Children
no_of_weekend_nights: Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel
no_of_week_nights: Number of weeknights (Monday to Friday) the guest stayed or booked to stay at the hotel
type_of_meal_plan: Type of meal plan booked by the customer:
Not Selected – No meal plan selected
Meal Plan 1 – Breakfast
Meal Plan 2 – Half board (breakfast and one other meal)
Meal Plan 3 – Full board (breakfast, lunch, and dinner)
required_car_parking_space: Does the customer require a car parking space? (0 - No, 1- Yes)
room_type_reserved: Type of room reserved by the customer. The values are ciphered (encoded) by INN Hotels Group
lead_time: Number of days between the date of booking and the arrival date
arrival_year: Year of arrival date
arrival_month: Month of arrival date
arrival_date: Date of the month
market_segment_type: Market segment designation.
repeated_guest: Is the customer a repeated guest? (0 - No, 1- Yes)
no_of_previous_cancellations: Number of previous bookings that were canceled by the customer before the current booking
no_of_previous_bookings_not_canceled: Number of previous bookings not canceled by the customer before the current booking
avg_price_per_room: Average price per day of the reservation; prices of the rooms are dynamic. (in euros)
no_of_special_requests: Total number of special requests made by the customer (e.g. high floor, view from the room, etc)
booking_status: Flag indicating if the booking was canceled or not.
Importing Libraries
# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import scipy.stats as stats
from scipy.stats import ttest_ind
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chi2_contingency
from scipy.stats import shapiro
from statsmodels.stats.weightstats import ztest as ztest
%matplotlib inline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
Importing Data
# mounting to drive
from google.colab import drive
drive.mount('/content/drive')
Mounted at /content/drive
# reading the datafile
data = pd.read_csv('/content/drive/MyDrive/INNHotelsGroup.csv')
# making a copy of the dataset
df = data.copy()
Understanding the dataset
⚛ Head of the data set

# check first five rows
df.head()
Booking_ID	no_of_adults	no_of_children	no_of_weekend_nights	no_of_week_nights	type_of_meal_plan	required_car_parking_space	room_type_reserved	lead_time	arrival_year	arrival_month	arrival_date	market_segment_type	repeated_guest	no_of_previous_cancellations	no_of_previous_bookings_not_canceled	avg_price_per_room	no_of_special_requests	booking_status
0	INN00001	2	0	1	2	Meal Plan 1	0	Room_Type 1	224	2017	10	2	Offline	0	0	0	65.00	0	Not_Canceled
1	INN00002	2	0	2	3	Not Selected	0	Room_Type 1	5	2018	11	6	Online	0	0	0	106.68	1	Not_Canceled
2	INN00003	1	0	2	1	Meal Plan 1	0	Room_Type 1	1	2018	2	28	Online	0	0	0	60.00	0	Canceled
3	INN00004	2	0	0	2	Meal Plan 1	0	Room_Type 1	211	2018	5	20	Online	0	0	0	100.00	0	Canceled
4	INN00005	2	0	1	1	Not Selected	0	Room_Type 1	48	2018	4	11	Online	0	0	0	94.50	0	Canceled
⚛ Tail of the dataset

# check last five rows of the dataset
df.tail()
Booking_ID	no_of_adults	no_of_children	no_of_weekend_nights	no_of_week_nights	type_of_meal_plan	required_car_parking_space	room_type_reserved	lead_time	arrival_year	arrival_month	arrival_date	market_segment_type	repeated_guest	no_of_previous_cancellations	no_of_previous_bookings_not_canceled	avg_price_per_room	no_of_special_requests	booking_status
36270	INN36271	3	0	2	6	Meal Plan 1	0	Room_Type 4	85	2018	8	3	Online	0	0	0	167.80	1	Not_Canceled
36271	INN36272	2	0	1	3	Meal Plan 1	0	Room_Type 1	228	2018	10	17	Online	0	0	0	90.95	2	Canceled
36272	INN36273	2	0	2	6	Meal Plan 1	0	Room_Type 1	148	2018	7	1	Online	0	0	0	98.39	2	Not_Canceled
36273	INN36274	2	0	0	3	Not Selected	0	Room_Type 1	63	2018	4	21	Online	0	0	0	94.50	0	Canceled
36274	INN36275	2	0	1	2	Meal Plan 1	0	Room_Type 1	207	2018	12	30	Offline	0	0	0	161.67	0	Not_Canceled
⚛ Shape of the dataset

# check no. of rows and columns
df.shape
(36275, 19)
Number of rows in the dataset = 36275
Number of columns in the datset = 19
⚛ Data Types present in dataset

# Analyzing the data types present in dataset
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 36275 entries, 0 to 36274
Data columns (total 19 columns):
 #   Column                                Non-Null Count  Dtype  
---  ------                                --------------  -----  
 0   Booking_ID                            36275 non-null  object 
 1   no_of_adults                          36275 non-null  int64  
 2   no_of_children                        36275 non-null  int64  
 3   no_of_weekend_nights                  36275 non-null  int64  
 4   no_of_week_nights                     36275 non-null  int64  
 5   type_of_meal_plan                     36275 non-null  object 
 6   required_car_parking_space            36275 non-null  int64  
 7   room_type_reserved                    36275 non-null  object 
 8   lead_time                             36275 non-null  int64  
 9   arrival_year                          36275 non-null  int64  
 10  arrival_month                         36275 non-null  int64  
 11  arrival_date                          36275 non-null  int64  
 12  market_segment_type                   36275 non-null  object 
 13  repeated_guest                        36275 non-null  int64  
 14  no_of_previous_cancellations          36275 non-null  int64  
 15  no_of_previous_bookings_not_canceled  36275 non-null  int64  
 16  avg_price_per_room                    36275 non-null  float64
 17  no_of_special_requests                36275 non-null  int64  
 18  booking_status                        36275 non-null  object 
dtypes: float64(1), int64(13), object(5)
memory usage: 5.3+ MB
The data set has
float64 - 1
int64 - 13
object - 5
⚛ Duplicates in dataset

# checking if there are duplicate entries in the dataset
df.duplicated().sum()
0
⚛ Missing values

# checking if there are missing values
df.isnull().sum()
0
Booking_ID	0
no_of_adults	0
no_of_children	0
no_of_weekend_nights	0
no_of_week_nights	0
type_of_meal_plan	0
required_car_parking_space	0
room_type_reserved	0
lead_time	0
arrival_year	0
arrival_month	0
arrival_date	0
market_segment_type	0
repeated_guest	0
no_of_previous_cancellations	0
no_of_previous_bookings_not_canceled	0
avg_price_per_room	0
no_of_special_requests	0
booking_status	0

dtype: int64
The dataset has no missing values.
⚛ Shape of the dataset

#  shape of the data set
df.shape
(36275, 19)
There are 36275 rows and 19 columns
⚛ Descriptive statistical summary

# statistical summary
df.describe().T # Transpose for readability
count	mean	std	min	25%	50%	75%	max
no_of_adults	36275.0	1.844962	0.518715	0.0	2.0	2.00	2.0	4.0
no_of_children	36275.0	0.105279	0.402648	0.0	0.0	0.00	0.0	10.0
no_of_weekend_nights	36275.0	0.810724	0.870644	0.0	0.0	1.00	2.0	7.0
no_of_week_nights	36275.0	2.204300	1.410905	0.0	1.0	2.00	3.0	17.0
required_car_parking_space	36275.0	0.030986	0.173281	0.0	0.0	0.00	0.0	1.0
lead_time	36275.0	85.232557	85.930817	0.0	17.0	57.00	126.0	443.0
arrival_year	36275.0	2017.820427	0.383836	2017.0	2018.0	2018.00	2018.0	2018.0
arrival_month	36275.0	7.423653	3.069894	1.0	5.0	8.00	10.0	12.0
arrival_date	36275.0	15.596995	8.740447	1.0	8.0	16.00	23.0	31.0
repeated_guest	36275.0	0.025637	0.158053	0.0	0.0	0.00	0.0	1.0
no_of_previous_cancellations	36275.0	0.023349	0.368331	0.0	0.0	0.00	0.0	13.0
no_of_previous_bookings_not_canceled	36275.0	0.153411	1.754171	0.0	0.0	0.00	0.0	58.0
avg_price_per_room	36275.0	103.423539	35.089424	0.0	80.3	99.45	120.0	540.0
no_of_special_requests	36275.0	0.619655	0.786236	0.0	0.0	0.00	1.0	5.0
Most bookings are for 2 adults
The number of children per booking is very low.
Very few customers require a car parking space, as indicated by the low average of 0.03.
A small percentage of guests are repeated customers as average of repeated customers is less (0.0256).
⚛ Unique values

# checking unique values for each column
for column in df.columns:
  print("{} : {}".format(column,df[column].unique()))
  print("-"*90)
Booking_ID : ['INN00001' 'INN00002' 'INN00003' ... 'INN36273' 'INN36274' 'INN36275']
------------------------------------------------------------------------------------------
no_of_adults : [2 1 3 0 4]
------------------------------------------------------------------------------------------
no_of_children : [ 0  2  1  3 10  9]
------------------------------------------------------------------------------------------
no_of_weekend_nights : [1 2 0 4 3 6 5 7]
------------------------------------------------------------------------------------------
no_of_week_nights : [ 2  3  1  4  5  0 10  6 11  7 15  9 13  8 14 12 17 16]
------------------------------------------------------------------------------------------
type_of_meal_plan : ['Meal Plan 1' 'Not Selected' 'Meal Plan 2' 'Meal Plan 3']
------------------------------------------------------------------------------------------
required_car_parking_space : [0 1]
------------------------------------------------------------------------------------------
room_type_reserved : ['Room_Type 1' 'Room_Type 4' 'Room_Type 2' 'Room_Type 6' 'Room_Type 5'
 'Room_Type 7' 'Room_Type 3']
------------------------------------------------------------------------------------------
lead_time : [224   5   1 211  48 346  34  83 121  44   0  35  30  95  47 256  99  12
 122   2  37 130  60  56   3 107  72  23 289 247 186  64  96  41  55 146
  32  57   7 124 169   6  51  13 100 139 117  39  86  19 192 179  26  74
 143 177  18 267 155  46 128  20  40 196 188  17 110  68  73  92 171 134
 320 118 189  16  24   8  10 182 116 123 105 443 317 286 148  14  85  25
  28  80  11 162  82  27 245 266 112  88  69 273   4  97  31  62 197 280
 185 160 104  22 292 109 126 303  81  54  15 161 147  87 127 418 156  58
 433 111 195 119  59  78 335 103  70  76 144  49  77  36  79  21  33 164
 152  43 102  71 209  93  53 302 239  45 167 113  84   9 166 174  61 151
  52  67 282  38 175  89 133  65  66  50 159 386 115 237 125  91  29 221
 213 198  75 180 236 120 230  63 136 309 157 268 217  94 305  98  42 154
 330 137 184 232 304 114 257 265 191 101 259 149 170 271 207 108 210 222
 296 194 145 153 275 158 301 349 200 315 181 263 176 141 270 150 359 244
 219 142 138 276 178 163 377 290 216 226 258 254 193 131 208 215 190 381
 231 248 106 308 140 173 168 172  90 249 205 129 212 135 220 277 253 132
 183 255 223 336 288 229 319 199 203 228 246 235 294 281 202 361 287 291
 313 206 269 279 261 214 274 250 187 240 241 323 322 227 225 233 338 283
 327 204 352 165 251 299 314 285 238 328 278 332 243 201 307 272 252 242
 284 297 324 260 262 326 295 218 234 353 300 355 306 298 331 341 318 333
 372 311 310 345 264 325 293 348 350 351]
------------------------------------------------------------------------------------------
arrival_year : [2017 2018]
------------------------------------------------------------------------------------------
arrival_month : [10 11  2  5  4  9 12  7  6  8  3  1]
------------------------------------------------------------------------------------------
arrival_date : [ 2  6 28 20 11 13 15 26 18 30  5 10  4 25 22 21 19 17  7  9 27  1 29 16
  3 24 14 31 23  8 12]
------------------------------------------------------------------------------------------
market_segment_type : ['Offline' 'Online' 'Corporate' 'Aviation' 'Complementary']
------------------------------------------------------------------------------------------
repeated_guest : [0 1]
------------------------------------------------------------------------------------------
no_of_previous_cancellations : [ 0  3  1  2 11  4  5 13  6]
------------------------------------------------------------------------------------------
no_of_previous_bookings_not_canceled : [ 0  5  1  3  4 12 19  2 15 17  7 20 16 50 13  6 14 34 18  8 10 23 11 49
 47 53  9 33 22 24 52 21 48 28 39 25 31 38 26 51 42 37 35 56 44 27 32 55
 45 30 57 46 54 43 58 41 29 40 36]
------------------------------------------------------------------------------------------
avg_price_per_room : [ 65.   106.68  60.   ... 118.43 137.25 167.8 ]
------------------------------------------------------------------------------------------
no_of_special_requests : [0 1 3 2 4 5]
------------------------------------------------------------------------------------------
booking_status : ['Not_Canceled' 'Canceled']
------------------------------------------------------------------------------------------
Unique entries in each column as in above the column BOOKING ID is a unique identification number for recording entry of booking and has no impact on other columns
Exploratory Data Analysis
( Points : 12 )

➡ Univariate Analysis
Booking_id
Booking id is an identification number alloted to each booking made , further analysis of this column results in nothing .
2. no_of_adults
# visualizing the number of adults using bar plot
plt.figure(figsize=(7,4)) # size of the figure
sns.countplot(x='no_of_adults', data=df, palette="pastel")
plt.title('Number of adults') # title of the plot
plt.xlabel('Number of adults') # label of x-axis
plt.ylabel('Count') # label of y-axis
plt.show() # show the plot

Insight

The above plot shows that most bookings made are for a group of two people , rest being comparatively very less.
3. no_of_children
# visualizing no of children using a bar plot
plt.figure(figsize=(7,5)) # size of the figure
sns.countplot(x='no_of_children', data=df, palette="pastel")
plt.title('Number of children') # title of the plot
plt.xlabel('Number of children') # label of x-axis
plt.ylabel('Count') # label of y-axis
plt.show() # show the plot

Insight

The above plot shows most of the bookings have no children included .
The children being included is also very less with either 1 child or 2 children.
4.no_of_weekend_nights
# visualizing using a histogram for no of weekend nights
plt.figure(figsize=(7,4))
sns.histplot(df['no_of_weekend_nights'],kde=True , bins = 10, color="#adf7b6")
plt.title('Distribution of weekend nights (kernal density estimate)') # title of the plot
plt.xlabel('No of weekend nights [Saturday or Sunday]') # label of x-axis
plt.ylabel('count') # label of y-axis
plt.show() # show the plot

plt.figure(figsize=(7,4))
sns.histplot(df['no_of_weekend_nights'], bins = 10, color="#a0ced9")
plt.title('Distribution of weekend nights') # title of the plot
plt.xlabel('No of weekend nights [Saturday or Sunday]') # label of x-axis
plt.ylabel('count') # label of y-axis
plt.show() # show the plot


Insight

Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel.
Large number of bookings include 1 weekend night, with a noticeable proportion having none.
The plot shows right skewness.
5.no_of_week_nights
#visualizing no of week nights stay by customer using histogram
plt.figure(figsize=(7,4)) # size of the figure
sns.histplot(df['no_of_week_nights'], bins=10,kde=True, color="#adf7b6")
plt.title('Distribution of week nights (kernal density estimate)') # title of the plot
plt.xlabel('No of week nights [Monday to Friday]') # label of x-axis
plt.ylabel('count') # label of y-axis
plt.show() # show the plot


plt.figure(figsize=(7,4)) # size of the figure
sns.histplot(df['no_of_week_nights'], bins=10, color="#a0ced9")
plt.title('Distribution of week nights') # title of the plot
plt.xlabel('No of week nights [Monday to Friday]') # label of x-axis
plt.ylabel('count') # label of y-axis
plt.show() # show the plot


Insight

Number of weeknights (Monday to Friday) the guest stayed or booked to stay at the hotel
There plot shows more stays for weeknights is around 1 to 3.
The plot shows right skewness
6.type_of_meal_plan
# visualizing the type of meal plan selected by the customers using countplot
plt.figure(figsize=(7,4)) # size of the figure
sns.countplot(x='type_of_meal_plan', data=df, palette="pastel")
plt.title('Type of meal plan') # title of the plot
plt.xlabel('Type of meal plan')  # label of x-axis
plt.ylabel('Count') # label of y-axis
plt.show() # show the plot

Insight

Type of meal plan booked by the customer:

Not Selected – No meal plan selected
Meal Plan 1 – Breakfast
Meal Plan 2 – Half board (breakfast and one other meal)
Meal Plan 3 – Full board (breakfast, lunch, and dinner)
The above plot shows most customers choose the Meal plan 1 which includes only breakfast.

Significant number of customers choose not to select any Meal plan and few opting Meal plan 2 which includes breakfast and one other meal.
7.required_car_parking_space
# visualizing the required car parking space using a pie chart
plt.figure(figsize=(6,6))
df.required_car_parking_space.value_counts().plot(kind='pie', autopct='%1.0f%%', colors = ['#a0ced9', '#adf7b6'])
plt.title('Required car parking space') # title of the plot
plt.show()

Insight

Customer requirement for a car parking space (0 - No, 1- Yes)
The above pie chart shows significant percentage of customers do not need a car parking space.
Tiny portion of 3 % of customers need a car parking space.
8.room_type_reserved
# visualizing the room type reserved by the customers using countplot
plt.figure(figsize=(8,5)) # size of the figure
sns.countplot(x='room_type_reserved', data=df, palette="pastel")
plt.title('Room type reserved') # title of the plot
plt.xticks(rotation=90) # rotate the x-axis labels
plt.xlabel('Room type reserved') # label of x-axis
plt.ylabel('Count') # label of y-axis
plt.show() # show the plot

Insight

Type of room reserved by the customer. The values are ciphered (encoded) by INN Hotels Group
The plot shows Room type 1 is most preffered by customers in the INN Hotels followed by Room type 4.
9.lead_time
# visualizing the lead time using a box plot
plt.figure(figsize=(7,4))
sns.boxplot(y='lead_time', data=df, color = '#a0ced9')
plt.title('Lead time') # title of the plot
plt.xlabel('Lead time') # label of x-axis
plt.ylabel('Count') # label of y-axis
plt.show()

Insight

Number of days between the date of booking and the arrival date
The above plot shows the spread and the outliers of the data.
The length of the upper whisker is longer than the lower whisker, and there are several outliers, suggesting that the data is positively skewed.
The above plot indicates that while bookings are made within a shorter lead time, a few bookings are made with a much longer lead time.
10.arrival_year
# visualization of arrival year using bar plot
plt.figure(figsize=(7,4)) # size of the figure
sns.countplot(x='arrival_year', data=df, palette="pastel")
plt.title('Arrival year') # title of the plot
plt.xlabel('Year') # label of x-axis
plt.ylabel('Count')  # label of y-axis
plt.show() # show the plot

Insight

Year of arrival date
The above plot shows more coustmers arrival being 2018
11.arrival_month
# visualizing the arrival month using a line plot
plt.figure(figsize=(7,5))
sns.lineplot(x='arrival_month', y=df.arrival_month.value_counts(), data=df, color="#ff7477")
plt.title('Arrival month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.show()

Insight

Month of arrival date
The line plot shows most of the customershave an arrival date from months 7 - 10.
12.arrival_date
# visualizing the date in which customers arrive using line pot
plt.figure(figsize=(15,4)) # size of the figure
sns.lineplot(x='arrival_date', y=df.arrival_date.value_counts(), data=df, color="#ff7477")
plt.title('Arrival date') # title of the plot
plt.xlabel('Date') # label of x-axis
plt.ylabel('Count') # label of y-axis
plt.show() # show the plot

Insight

Date of the month
The above plot show various dates of arrival of the customer yet show a less circumstance of arrival in dates from 10 - 15 , in a month.
13.market_segment_type
# visualizating market segment type using barplot

plt.figure(figsize=(7,4))
sns.countplot(x='market_segment_type', data=df, palette="pastel")
plt.title('Market segment type')
plt.xlabel('Market segment type')
plt.ylabel('Count')
plt.show()

Insight

Market segment designation.
The plot shows that most of the market is designated at online followed by offline and corcporate.
14.repeated_guest
# visualizing repeated guests using pie chrt

plt.figure(figsize=(6,6)) # size of the figure
df.repeated_guest.value_counts().plot(kind='pie', colors = ['#a0ced9', '#adf7b6'])
plt.title('Repeated guest') # title of the plot
plt.show()

Insight

Is the customer a repeated guest? (0 - No, 1- Yes)

The above chart shows there is significantly less numbers of repeated guests for INN Hotels.

15.no_of_previous_cancellations
# Visualizing the cancellations made previously using a histogram
plt.figure(figsize=(7,4)) # size of the figure
sns.histplot(df['no_of_previous_cancellations'], bins=10 , color="#a0ced9")
plt.title('No of previous cancellations') # title of the plot
plt.xlabel('No of previous cancellations')  # label of x-axis
plt.ylabel('count')  # label of y-axis
plt.show()

Insight

Number of previous bookings that were canceled by the customer before the current booking
The histogram shows very less number of previous cancelations.
16.no_of_previous_bookings_not_canceled
# visualization of previous bookings not cancelled
plt.figure(figsize=(7,4)) # size of the figure
sns.histplot(df['no_of_previous_bookings_not_canceled'], bins=10 , color="#a0ced9")
plt.title('No of previous bookings not cancellled') # title of the plot
plt.xlabel('No of previous bookings not cancelled') # label of x-axis
plt.ylabel('count') # label of y-axis
plt.show()

Insight

Number of previous bookings that were not canceled by the customer before the current booking
Most customers have no previous bookings cancelled, but a small number have many.
17.avg_price_per_room
# visualizing the average price per room using a histogram
plt.figure(figsize=(7,4)) # size of the figure
sns.histplot(df['avg_price_per_room'], bins=10 , color="#a0ced9")
plt.title('Average price per room') # title of the plot
plt.xlabel('Average price per room') # label of x-axis
plt.ylabel('count') # label of y-axis
plt.show()

# visualizing avg price using boxplot
plt.figure(figsize=(7,4)) # size of the figure
sns.boxplot(y='avg_price_per_room', data=df, color = '#a0ced9')
plt.title('Average price per room') # title of the plot
plt.xlabel('Average price per room') # label of x-axis
plt.ylabel('Count') # label of y-axis
plt.show()

Insight

Average price per day of the reservation; prices of the rooms are dynamic. (in euros)
The above histogram shows the distribution of the prices highly from 50 euros to 150 euros.
The boxplot shows outliers where there are very high or low prices charged per day of reservation.
18.no_of_special_requests
# visualizing the no of special requests made by customers using bar plot
plt.figure(figsize=(7,4)) # size of the figure
sns.countplot(x='no_of_special_requests', data=df, palette="pastel")
plt.title('No of special requests') # title of the plot
plt.xlabel('No of special requests') # label of x-axis
plt.ylabel('Count') # label of y-axis
plt.show()

Insight

Total number of special requests made by the customer (e.g. high floor, view from the room, etc)
The plot shows that most customers make no special requests .
There are one or two special requests made by few customers .
19.booking_status
# visualizing the booking status using a countplot
plt.figure(figsize=(7,4)) # size of the figure
sns.countplot(x='booking_status', data=df, palette="pastel")
plt.title('Booking status') # title of the plot
plt.xlabel('Booking status') # label of x-axis
plt.ylabel('Count') # label of y-axis
plt.show() # show the plot

Insight

Flag indicating if the booking was canceled or not.
The above plot shows the number of bookings that have been canceled and not canceled where the above shows the cancelled bookings are less compared to not canceled bookings .
There is also a significant number of bookings being canceled which need to be looked into.
➡ Bivariate Analysis
Number of Adults and Average Room Price
Target :

Check how the number of adults affects the average room price.
# boxplot to visualize the distribution of average room prices for different numbers of adults.
plt.figure(figsize=(8, 6)) # size of the figure
sns.boxplot(x='no_of_adults', y='avg_price_per_room', data=df, palette='pastel')
plt.title('Average Room Price vs Number of Adults') # title of the plot
plt.xlabel('Number of Adults') # label of x-axis
plt.ylabel('Average Room Price') # label of y-axis
plt.show() # show the plot

Insight

The above box plot shows how the distribution of room prices varies with different numbers of adults.
The average room price tends to increase with increase in number of adults.
There are more outliers with more number of adults
Number of Children and Average Room Price
Target :

check if the presence of number of children influences room price.
#boxplot to visualize the distribution of average room prices for different numbers of children.
plt.figure(figsize=(8, 6)) # size of the figure
sns.boxplot(x='no_of_children', y='avg_price_per_room', data=df, palette='pastel')
plt.title('Average Room Price vs Number of Children') # title of the plot
plt.xlabel('Number of Children') # label of x-axis
plt.ylabel('Average Room Price') # label of y-axis
plt.show()

Insight

There are outliers present in the room prices for bookings with a number of children.
Slight increase in the average room price is observed as the number of children increases.
Number of weekend nights and Room price
Target :

Check if staying over the weekend affects room pricing.
# Analyzing the relation between stay on weekend night and room price with boxplot
plt.figure(figsize=(10, 6)) # size of the figure
sns.boxplot(x='no_of_weekend_nights', y='avg_price_per_room', data=df, palette='pastel')
plt.title('Average Room Price vs Number of Weekend Nights') # title of the plot
plt.xlabel('Number of Weekend Nights') # label of x-axis
plt.ylabel('Average Room Price') # label of y-axis
plt.show()

Insight

The above plot shows almost stable pricing with a very slight increase with increase in no. of weekend night stay.
There are outliers in the room prices for bookings with both low and high numbers of weekend nights.
Lead time and Room price
Target :

Check how lead time affects the price paid per room.
# Analyzing the effect of lead time over the average price per room

plt.figure(figsize=(8, 6)) # size of the figure
sns.scatterplot(x='lead_time', y='avg_price_per_room', data=df, color = '#a0ced9')
plt.title('Lead Time vs Average Room Price') # title of the plot
plt.xlabel('Lead Time (Days)') # label of x-axis
plt.ylabel('Average Room Price') # label of y-axis
plt.grid(True) # show the grid
plt.show()

Insight

The above plot appears to be negative correlated, where longer lead times have lower room prices.
Room prices show a wider distribution, being clustered more at shorter lead times compared to longer lead times.
Number of week nights and Room prices
Target :

Check how the number of weekdays affects the room price.
# Analyzing the relation between stay on week night and room price with boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='no_of_week_nights', y='avg_price_per_room', data=df, palette='pastel')
plt.title('Average Room Price vs Number of Week Nights')
plt.xlabel('Number of Week Nights')
plt.ylabel('Average Room Price')
plt.show()

Insight

The above plot shows the meddian remains almost constant for different number of week nights .
The interquartile range (IQR) for room prices widens slightly with an increasing number of weeknights, indicating some variability in pricing.
Lead time and number of special requests
Target:

Analyze if bookings made in advance tend to have more special requests.
# Analyze if bookings made in advance tend to have more special requests
plt.figure(figsize=(8, 6)) # size of the figure
sns.boxplot(x='no_of_special_requests', y='lead_time', data=df, palette='pastel')
plt.title('Lead Time vs Number of Special Requests') # title of the plot
plt.xlabel('Number of Special Requests') # label of x-axis
plt.ylabel('Lead Time (Days)') # label of y-axis
plt.show() # show the plot

Insight

There is higher variability in the number of special requests for shorter lead times.
Outliers are more common in the number of special requests for shorter lead times.
The number of special requests tends to decrease as the lead time increases.
Number of special requests and room prices
Target :

Check if a higher number of special requests increase the room price.
# box plot analysis between the number of special requests and room prices
plt.figure(figsize=(8, 6)) # size of the figure
sns.boxplot(x='no_of_special_requests', y='avg_price_per_room', data=df, palette='pastel')
plt.title('Average Room Price vs Number of Special Requests') # title of the plot
plt.xlabel('Number of Special Requests') # label of x-axis
plt.ylabel('Average Room Price') # label of y-axis
plt.show()

Insight

The median room price shows slight increase with a higher number of special requests.
The interquartile range (IQR) for room prices widens as the number of special requests increases, indicating greater price variability.
There are notable outliers in room prices, especially for bookings with a higher number of special requests.
Room price and arrival month
Target :

Check if there are any effect of specific month on the room price
# Analyzing if there is any sesonality
plt.figure(figsize=(12, 6)) # size of the figure
sns.boxplot(x='arrival_month', y='avg_price_per_room', data=df, palette='pastel')
plt.title('Average Room Price vs Arrival Month') # title of the plot
plt.xlabel('Arrival Month') # label of x-axis
plt.ylabel('Average Room Price') # label of y-axis
plt.show() # show the plot

Insight

The median room price varies across different months, this might indicate seasonal pricing trends.
Certain months may show higher median room prices, indicating higher demand periods.
There are outliers in room prices for most months, with more extreme prices possibly occurring during high-demand or low-demand periods.
Room price and arrival date
Target :

Determine if the specific date within a month affects room price

# scattrplot Room price and arrival date

plt.figure(figsize=(15, 6)) # size of the figure
sns.scatterplot(x='arrival_date', y='avg_price_per_room', data=df, color='#a0ced9')
plt.title('Average Room Price vs Arrival Date') # title of the plot
plt.xlabel('Arrival Date') # label of x-axis
plt.ylabel('Average Room Price') # label of y-axis
plt.grid(True) # show the grid
plt.show()

Insight

Room prices do not follow a clear trend with arrival dates, showing a scattered distribution. Room prices fluctuate across different arrival dates, suggesting daily adjustments
Room price and repeated guests
Target :

Check if repeated guests tend to pay more or less per room
# Analyzing if repeated guests tend to pay more or less per room
plt.figure(figsize=(8, 5)) # size of the figure
sns.boxplot(x='repeated_guest', y='avg_price_per_room', data=df, palette='pastel')
plt.title('Average Room Price vs Repeated Guest') # title of the plot
plt.xlabel('Repeated Guest (0 = No, 1 = Yes)') # label of x-axis
plt.ylabel('Average Room Price') # label of y-axis
plt.show()

Insight

Both the guests show outliers.
The average price of not repeated customer is more than that of repeated customers indicating there is some difference in price .
Number of previous cancellations and lead time
Target :

Check if guests with a history of cancellations tend to book closer to their stay date.
# Analyze the relation betwen lead time and previous cancellations
plt.figure(figsize=(8, 6)) # size of the figure
sns.scatterplot(x='no_of_previous_cancellations', y='lead_time', data=df, color='#a0ced9')
plt.title('Number of Previous Cancellations vs Lead Time') # title of the plot
plt.xlabel('Number of Previous Cancellations')  # label of x-axis
plt.ylabel('Lead Time (Days)') # label of y-axis
plt.grid(True) # show the grid
plt.show()

Insight

The distribution of lead times for guests with no previous cancellations is different from those with multiple cancellations.
Clusters indicating specific ranges of lead times associated with higher or lower cancellation frequencies.
Number of previous bookings not canceled and room price
Target :

Check the influence of the room price over the number of previous bookings not cancelled
# Analyzing the relation between number of previous bookings not cancelled and room price with boxplot
plt.figure(figsize=(20,9)) # size of the figure
sns.boxplot(x='no_of_previous_bookings_not_canceled', y='avg_price_per_room', data=df, palette='pastel')
plt.title('Average Room Price vs Number of Previous Bookings Not Cancelled') # title of the plot
plt.xlabel('Number of Previous Bookings Not Cancelled') # label of x-axis
plt.xticks(rotation=90) # rotate the x-axis labels for better visibility
plt.ylabel('Average Room Price') # label of y-axis
plt.show() # show the plot

Insight

customrs with more previous no cancellation have more similar price or consistent pricing
The outliers indicate difference in price with fewer or no previous non cancelations.
Market segmentation and Average price per room
Target :

Check price variations across different market segments.
# Analyzing the relation between market segmentation and price of the room
plt.figure(figsize=(10, 6)) # size of the figure
sns.boxplot(x='market_segment_type', y='avg_price_per_room', data=df, palette='pastel')
plt.title('Average Room Price vs Market Segment Type') # title of the plot
plt.xlabel('Market Segment Type') # label of x-axis
plt.ylabel('Average Room Price') # label of y-axis
plt.show() # show the plot

Insight

In the above plot different market segmentations show difference in their median room prices
Some segments may have a wider interquartile range indicating greater price variability compared to others
Outliers impling higher or lower room prices can be special cases or high-demand periods.
Booking status and Market segmentation
Target :

Compare the frequency of booking statuses within each market segment.
# Analyzing the frequency of booking with market segment
plt.figure(figsize=(10,5 )) # size of the figure
sns.countplot(x='market_segment_type', hue='booking_status', data=df, palette='pastel')
plt.title('Booking Status by Market Segment') # title of the plot
plt.xlabel('Market Segment Type') # label of x-axis
plt.ylabel('Count') # label of y-axis
plt.show() # show the plot

Insight

Market segment [online] shows higher cancellation compared to others.
Online segment has more no. of non cancelled customers.
Other segments need more improve ment to increase the non cancelled customers .
Special requirements and Booking status
Target :

Compare the number of special requests across different booking statuses (e.g., canceled or not canceled)
# Analyzing the booking status across no. of special requests made.
plt.figure(figsize=(8, 5)) # size of the figure
sns.boxplot(x='booking_status', y='no_of_special_requests', data=df, palette='pastel')
plt.title('Number of Special Requests vs Booking Status') # title of the plot
plt.xlabel('Booking Status')  # label of x-axis
plt.ylabel('Number of Special Requests') # label of y-axis
plt.show() # show the plot

Insight

The above plot shows no much difference with in the plot of not canceled and canceled
The ouliers in the above plot indicate certain customers make unusually high no. of special requests.
Special requirement and Average price per room
Target :

Check how the average room price varies with the number of special requests made by guests
# Analyzing how special requests influence the pric charged per room
plt.figure(figsize=(8, 6)) # size of the figure
sns.boxplot(x='no_of_special_requests', y='avg_price_per_room', data=df, palette='pastel')
plt.title('Average Room Price vs Number of Special Requests') # title of the plot
plt.xlabel('Number of Special Requests') # label of x-axis
plt.ylabel('Average Room Price') # label of y-axis
plt.show() # show the plot

Insight

The median price of rooms increases as the number of special requests increase.
There are outliers particularly in cases with no or few special requests where room prices are unusually high
With increase in the no. of requests the price seem to be consistant with no much difference.
Average price per room and Booking status
Target :

Compare how the average price per room varies based on whether the booking was Canceled or Not_Canceled
# Analyzing the effect of booking status on price charged
plt.figure(figsize=(8, 5)) # size of the figure
sns.boxplot(x='booking_status', y='avg_price_per_room', data=df, palette='pastel')
plt.title('Average Room Price vs Booking Status') # title of the plot
plt.xlabel('Booking Status')  # label of x-axis
plt.ylabel('Average Room Price') # label of y-axis
plt.show() # show the plot

Insight

Outliers in the plot indicate unusually high or low room prices that can influence booking behavior
The average price per room for canceled is more than that of not canceled
Lead time and Booking status
Target :

Compare the distribution of lead times between canceled and non-canceled bookings, providing insight into whether bookings with longer or shorter lead times are more likely to be canceled.
# Analying the booking staus and lead time using boxplot
plt.figure(figsize=(8,4 )) # size of the figure
sns.boxplot(x='booking_status', y='lead_time', data=df, palette='pastel')
plt.title('Lead Time vs Booking Status') # title of the plot
plt.xlabel('Booking Status') # label of x-axis
plt.ylabel('Lead Time (Days)') # label of y-axis
plt.show() # show the plot

Insight

The above plot shows cancellations are more frequent with long lead times
Outliers indiacte extremely long or short lead times are associated with cancellations or non-cancellations
Arrival month and Booking status
Target :

Check the influnce of the month on booking status
# Analyzing using  countplot for Arrival month and Booking status
plt.figure(figsize=(12, 6)) # size of the figure
sns.countplot(x='arrival_month', hue='booking_status', data=df, palette='pastel')
plt.title('Booking Status by Arrival Month') # title of the plot
plt.xlabel('Arrival Month') # label of x-axis
plt.ylabel('Count') # label of y-axis
plt.show() # show the plot

Insight

The above plot show the cancelation is least in Month no.1 [ January ] and highest in Month no. 10 [ October ].
The activity of both booking and canceling is high from the month no.6 to month no.10 [ June - October ].
Repeated guest and Booking status
Target :

The target is to understand whether repeat guests (repeated_guest = 1) are less likely to cancel their bookings compared to first-time guests (repeated_guest = 0).
# Analyzing booking status and repeated guest
plt.figure(figsize=(8, 5)) # size of the figure
sns.countplot(x='repeated_guest', hue='booking_status', data=df, palette='pastel')
plt.title('Booking Status by Repeated Guest') # title of the plot
plt.xlabel('Repeated Guest (0 = No, 1 = Yes)') # label of x-axis
plt.ylabel('Count') # label of y-axis
plt.show() # show the plot

Insight

The plot shows repeat guests have a lower likelihood of canceling bookings compared to non-repeat guests
Arrival month and Average price
Target :

Check how room prices vary for each month.
# Analyzing the the affect of month on the price charged
plt.figure(figsize=(12, 6)) # size of the figure
sns.boxplot(x='arrival_month', y='avg_price_per_room', data=df, palette='pastel')
plt.title('Average Room Price vs Arrival Month') # title of the plot
plt.xlabel('Arrival Month') # label of x-axis
plt.ylabel('Average Room Price') # label of y-axis
plt.show()

# Analyzing the the affect of month on the price charged
plt.figure(figsize=(12, 6)) # size of the figure
sns.barplot(x='arrival_month', y='avg_price_per_room', data=df, palette='pastel')
plt.title('Average Room Price vs Arrival Month') # title of the plot
plt.xlabel('Arrival Month') # label of x-axis
plt.ylabel('Average Room Price') # label of y-axis
plt.show() # show the plot

Insight

The median and spread of prices in each month shows peak and off-peak seasons where average room prices are higher and lower respectively.
Outliers show months where there are exceptionally high or low prices, which might correlate with special events or discounts.
➡ Multivariate Analysis
Avg Price Per Room vs Market Segment Type vs Booking Status
Target :

Check how market segments (e.g., online, offline) affect room prices and booking cancellations.
# clustered bar plot of Avg Price Per Room vs Market Segment Type vs Booking Status
plt.figure(figsize=(12, 6)) # size of the figure
sns.barplot(x='market_segment_type', y='avg_price_per_room', hue='booking_status', data=df, palette='pastel')
plt.title('Average Room Price vs Market Segment Type vs Booking Status') # title of the plot
plt.xlabel('Market Segment Type') # label of x-axis
plt.ylabel('Average Room Price') # label of y-axis
plt.xticks(rotation=90) # rotate the x-axis labels for better visibility
plt.show() # show the plot

Insights

Some segments like corporate , offline show higher average prices for canceled bookings .
Segments with lower average prices and a higher rate of cancellations like offline are be more price-sensitive
Average Price Per Room vs Arrival Month vs Booking Status
Target :

Examine how arrival month affect room prices and booking cancellations.
# Analyzing using line plot for Average Price Per Room vs Arrival Month vs Booking Status
plt.figure(figsize=(12, 6))  # size of the figure
sns.lineplot(x='arrival_month', y='avg_price_per_room', hue='booking_status', data=df, marker='o')
plt.title('Average Room Price vs Arrival Month vs Booking Status') # title of the plot
plt.xlabel('Arrival Month') # label of x-axis
plt.ylabel('Average Room Price') # label of y-axis
plt.xticks(rotation=90) # rotate the x-axis labels for better visibility
plt.show() # show the plot

Insights

Some months like march show higher cancellation rates at both high and low price points, maybe caused by market conditions leading to cancellations.

Certain months like June - September show stable average prices with low cancellation rates, while other months show high variability in pricing with more cancellation

Average Price Per Room vs Lead Time vs Booking Status
Target :

Analyze how room prices and lead times impact the likelihood of cancellations or successful bookings.
# visualizing the relation with  scatterplot of  Average Price Per Room vs Lead Time vs Booking Status
plt.figure(figsize=(12, 6))  # size of the figure
sns.scatterplot(x='lead_time', y='avg_price_per_room', hue='booking_status', data=df)
plt.title('Average Room Price vs Lead Time vs Booking Status')  # title of the plot
plt.xlabel('Lead Time')  # label of x-axis
plt.ylabel('Average Room Price')  # label of y-axis
plt.show()  # show the plot

Insight

The above plot shows with increase in leadd time the rate of cancellation increases.
Higher average room prices is associated with shorter lead time , last minute bookings having high price and low cancelations.
Canceled bookings show higher prices at certain lead times indicating that price-sensitive customers cancel when prices spike
No. of Adults vs Avg Price Per Room vs Booking Status
Target :

Check how the number of adults in a booking correlates with room pricing and booking outcomes (canceled or not).
# visualizing with clustered bar plot of No. of Adults vs Avg Price Per Room vs Booking Status
plt.figure(figsize=(12, 6)) # size of the figure
sns.barplot(x='no_of_adults', y='avg_price_per_room', hue='booking_status', data=df, palette='pastel')
plt.title('Average Room Price vs Number of Adults vs Booking Status') # title of the plot
plt.xlabel('Number of Adults') # label of x-axis
plt.ylabel('Average Room Price') # label of y-axis
plt.show() # show the plot

Insight

The price of the room increases with increase in no. of adults. specifically for not canceled
The cancelation with more number of adult is pretty high
The price is more alike withing few no. of adults (0-2)
Avg Price Per Room vs Room Type Reserved vs Booking Status
Target :

Check if certain room types (e.g., deluxe, suite) have higher prices and more cancellations or lower rates of booking failures.
#  boxplot of Avg Price Per Room vs Room Type Reserved vs Booking Status
plt.figure(figsize=(12,7 ))  # size of the figure
sns.boxplot(y='room_type_reserved', x='avg_price_per_room', hue='booking_status', data=df, palette='pastel')
plt.title('Average Room Price vs Room Type Reserved vs Booking Status')  # title of the plot
plt.xlabel('Room Type Reserved')  # label of x-axis
plt.ylabel('Average Room Price')  # label of y-axis
plt.xticks(rotation=90)  # rotate the x-axis labels for better visibility
plt.show()  # show the plot

Insight

The average price per room is significantly different across each room types
Certain room types may have higher cancellation rates, particularly for more expensive or premium room types
Outliers represent rooms that were booked at significantly higher or lower prices than the majority for a given room type and booking status.
No. of Special Requests vs Avg Price Per Room vs Booking Status
Target :

Analyze how special requests influence room pricing and cancellations.
# visualizing with grouped bar plot of No. of Special Requests vs Avg Price Per Room vs Booking Status
plt.figure(figsize=(12, 6))  # size of the figure
sns.barplot(x='no_of_special_requests', y='avg_price_per_room', hue='booking_status', data=df, palette='pastel')
plt.title('Average Room Price vs Number of Special Requests vs Booking Status')  # title of the plot
plt.xlabel('Number of Special Requests')  # label of x-axis
plt.ylabel('Average Room Price')  # label of y-axis
plt.show()  # show the plot

Insights

Bookings with more special requests have higher average room prices
With increase in speacial requests there is less likely for the customer to cancel the booking
Arrival Year vs Arrival Month vs Avg Price Per Room
Target :

Check how average room prices have trended over time across different months and years.
# Analyzing relationship with heatmap of Arrival Year vs Arrival Month vs Avg Price Per Room
heatmap_data = df.pivot_table(values='avg_price_per_room', index='arrival_year', columns='arrival_month', aggfunc='mean')
plt.figure(figsize=(12, 8))  # size of the figure
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis")
plt.title('Average Room Price by Arrival Year and Month')  # title of the plot
plt.xlabel('Arrival Month')  # label of x-axis
plt.ylabel('Arrival Year')  # label of y-axis
plt.show()  # show the plot

Insights

The above heatmap shows there are specific months of the year that show high prices.
The heatmaps shows there is increase in price of the room in 2018 when compared to 2017
Lead Time vs No. of Weekend Nights vs Avg Price Per Room
Target :

Explore the relationship between how early people book, the number of weekend nights, and room pricing.
# prompt:  visualizing with scatterplot for Lead Time vs No. of Weekend Nights vs Avg Price Per Room

plt.figure(figsize=(12,6))  # size of the figure
sns.scatterplot(x='lead_time', y='no_of_weekend_nights', hue='avg_price_per_room', data=df)
plt.title('Lead Time vs No. of Weekend Nights vs Avg Price Per Room')  # title of the plot
plt.xlabel('Lead Time')  # label of x-axis
plt.ylabel('No. of Weekend Nights')  # label of y-axis
plt.show()  # show the plot

Insights

Rooms booked with shorter lead time show higher prices for weekend stay.
Higher prices may be clustered for shorter lead times and higher numbers of weekend nights
Lead Time vs Market Segment Type vs Booking Status
Target :

Check how booking lead times differ across market segments and their influence on cancellations.
# Visualizing with box plot of Lead Time vs Market Segment Type vs Booking Status
plt.figure(figsize=(12, 8))  # size of the figure
sns.boxplot(x='market_segment_type', y='lead_time', hue='booking_status', data=df, palette='pastel')
plt.title('Lead Time vs Market Segment Type vs Booking Status')  # title of the plot
plt.xlabel('Market Segment Type')  # label of x-axis
plt.ylabel('Lead Time')  # label of y-axis
plt.xticks(rotation=90)  # rotate the x-axis labels for better visibility
plt.show()  # show the plot

Insights

There is significant variability in the data, with many extreme values that fall far from the normal range.
Certain market segments, such as offline, exhibit longer lead times compared to others, indicating earlier booking.
No. of Children vs No. of Special Requests vs Booking Status
Target :

Check how families with children make requests and how that affects booking success or cancellations.
# Visualizing with  bar plot of No. of Children vs No. of Special Requests vs Booking Status
plt.figure(figsize=(10, 6))  # size of the figure
sns.countplot(x='no_of_children', hue='booking_status', data=df, palette='pastel')
plt.title('No. of Children vs No. of Special Requests vs Booking Status')  # title of the plot
plt.xlabel('No. of Children')  # label of x-axis
plt.ylabel('count of customers with special requests')  # label of y-axis
plt.show()  # show the plot

Insights

The no of children is not more than three as in the plot .
For booking with lower children and more special requests has high cancellation rate.
No. of Weekend Nights vs No. of Week Nights vs Avg Price Per Room
Target :

Check how different lengths of stay (week vs. weekend) influence room pricing.
# Analyzing with scatterplot of No. of Weekend Nights vs No. of Week Nights vs Avg Price Per Room
plt.figure(figsize=(12, 6))  # size of the figure
sns.scatterplot(x='no_of_weekend_nights', y='no_of_week_nights', hue='avg_price_per_room', data=df)
plt.title('No. of Weekend Nights vs No. of Week Nights vs Avg Price Per Room')  # title of the plot
plt.xlabel('No. of Weekend Nights')  # label of x-axis
plt.ylabel('No. of Week Nights')  # label of y-axis
plt.show()  # show the plot

# Analyzing relation with heatmap of No. of Weekend Nights vs No. of Week Nights vs Avg Price Per Room
heatmap_data = df.pivot_table(values='avg_price_per_room', index='no_of_weekend_nights', columns='no_of_week_nights', aggfunc='mean')
plt.figure(figsize=(12, 8))  # size of the figure
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis")
plt.title('Average Room Price by Weekend Nights and Week Nights')  # title of the plot
plt.xlabel('Number of Week Nights')  # label of x-axis
plt.ylabel('Number of Weekend Nights')  # label of y-axis
plt.show()  # show the plot

Insights

As the number of both weekend and weeknights increases, the average price per room rises.
The plot reveals that rooms booked for more weekend nights are priced higher than those with more weeknights.
No. of Previous Cancellations vs Lead Time vs Booking Status
Target :

Examine how customers with prior cancellations behave in terms of booking lead time and how it affects future bookings.
# Visualizing with scatterplot of No. of Previous Cancellations vs Lead Time vs Booking Status
plt.figure(figsize=(12, 6))  # size of the figure
sns.scatterplot(x='no_of_previous_cancellations', y='lead_time', hue='booking_status', data=df)
plt.title('No. of Previous Cancellations vs Lead Time vs Booking Status')  # title of the plot
plt.xlabel('No. of Previous Cancellations')  # label of x-axis
plt.ylabel('Lead Time')  # label of y-axis
plt.show()  # show the plot

# Visualizing with barplot of No. of Previous Cancellations vs Lead Time vs Booking Status
plt.figure(figsize=(12, 6))  # size of the figure
sns.barplot(x='no_of_previous_cancellations', y='lead_time', hue='booking_status', data=df, palette='pastel')
plt.title('No. of Previous Cancellations vs Lead Time vs Booking Status')  # title of the plot
plt.xlabel('No. of Previous Cancellations')  # label of x-axis
plt.ylabel('Lead Time')  # label of y-axis
plt.show()  # show the plot

Insights

Bookings with a higher number of previous cancellations have longer lead times, which indicates that customers who book far in advance tend to cancel more.
Market Segment Type vs Room Type Reserved vs Avg Price Per Room
Target :

Check how market segments prefer different room types and how this affects average pricing.
# Analyzing relationship with heatmap of Market Segment Type vs Room Type Reserved vs Avg Price Per Room
heatmap_data = df.pivot_table(values='avg_price_per_room', index='market_segment_type', columns='room_type_reserved', aggfunc='mean')
plt.figure(figsize=(12, 8))  # size of the figure
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis")
plt.title('Average Room Price by Market Segment Type and Room Type Reserved')  # title of the plot
plt.xlabel('Room Type Reserved')  # label of x-axis
plt.xticks(rotation=90)  # rotate the x-axis labels for better visibility
plt.ylabel('Market Segment Type')  # label of y-axis
plt.show()  # show the plot

Insights

Different room types within the same market segment may have significantly different average prices.
Room type 7 tends to be the costliest one compared to other type of rooms
Online booking show almost all category of room types being booked and complimentary being the least prized
Arrival Month vs No. of Special Requests vs Booking Status
Target :

Check if special requests vary across different months and how that relates to cancellations.
# Visualize using a barplot of Arrival Month vs No. of Special Requests vs Booking Status
plt.figure(figsize=(12, 6))  # size of the figure
sns.barplot(x='arrival_month', y='no_of_special_requests', hue='booking_status', data=df, palette='pastel')
plt.title('Arrival Month vs No. of Special Requests vs Booking Status')  # title of the plot
plt.xlabel('Arrival Month')  # label of x-axis
plt.ylabel('No. of Special Requests')  # label of y-axis
plt.xticks(rotation=90)  # rotate the x-axis labels for better visibility
plt.show()  # show the plot

Insights

Certain months show a higher average number of special requests, possibly due to holiday periods, events, or seasonal preferences.
Bookings with a higher number of special requests have a higher cancellation rate
During high-demand months [7-10] there is an increase in special requests
Arrival Month vs Lead Time vs Booking Status
Target :

Check how the time before arrival affects cancellations across different months.
# Analyzing relationship with barplot of Arrival Month vs Lead Time vs Booking Status
plt.figure(figsize=(12, 6))  # size of the figure
sns.barplot(x='arrival_month', y='lead_time', hue='booking_status', data=df, palette='pastel')
plt.title('Arrival Month vs Lead Time vs Booking Status')  # title of the plot
plt.xlabel('Arrival Month')  # label of x-axis
plt.ylabel('Lead Time')  # label of y-axis
plt.xticks(rotation=90)  # rotate the x-axis labels for better visibility
plt.show()  # show the plot

Insights

There is an increase in the cancellation rate from 4 th month and is at highest in 9th and 10th month.
Market Segment Type vs Arrival Month vs Booking Status
Target :

Check if different market segments behave differently across months in terms of booking status.
# analyzing with barplot of Market Segment Type vs Arrival Month vs Booking Status
plt.figure(figsize=(12, 6))  # size of the figure
sns.countplot(x='arrival_month', hue='market_segment_type', data=df, palette='pastel')
plt.title('Market Segment Type vs Arrival Month vs Booking Status')  # title of the plot
plt.xlabel('Arrival Month')  # label of x-axis
plt.ylabel('Count')  # label of y-axis
plt.xticks(rotation=90)  # rotate the x-axis labels for better visibility
plt.show()  # show the plot

Insights

Certain months have longer lead time , indicating people book in advace during these months.
Bookings with shorter lead times may have a higher rate of cancellations, particularly in certain months, suggesting last-minute bookings are more likely to be canceled.
Certain months may show higher cancellation rates regardless of lead time, suggesting that external factors affect booking status.
No. of Adults vs Lead Time vs Avg Price Per Room
Target :

Check the impact of no. of adults on booking lead time and room pricing.
# analyzing with scatterplot of No. of Adults vs Lead Time vs Avg Price Per Room
plt.figure(figsize=(12, 6))  # size of the figure
sns.scatterplot(x='no_of_adults', y='lead_time', hue='avg_price_per_room', data=df)
plt.title('No. of Adults vs Lead Time vs Avg Price Per Room')  # title of the plot
plt.xlabel('No. of Adults')  # label of x-axis
plt.ylabel('Lead Time')  # label of y-axis
plt.show()  # show the plot

Insights

As the number of adults increse there is an increase in the price and also the lead time increase which has effect on both.
➡ EDA Questions
Question no :01
-- What are the busiest months in the hotel?

# To find the busiest months, let's calculate the number of bookings per month.
busiest_months = df.groupby('arrival_month')['Booking_ID'].count().reset_index()
busiest_months.columns = ['Month', 'No_of_Bookings']
# Sort by number of bookings in descending order to identify the busiest months
busiest_months = busiest_months.sort_values(by='No_of_Bookings', ascending=False)

busiest_months
Month	No_of_Bookings
9	10	5317
8	9	4611
7	8	3813
5	6	3203
11	12	3021
10	11	2980
6	7	2920
3	4	2736
4	5	2598
2	3	2358
1	2	1704
0	1	1014
ANSWER

The busiest months of INN Hotels is 10th month i.e October ,followed by 9th month - September and 8th month - August.
This can be seen even in the Lineplot of the Arrival month in univariate analysis.
Question no :02
-- Which market segment do most of the guests come from?

# To determine which market segment most guests come from, let's calculate the count of bookings per market segment.
market_segment_counts = data['market_segment_type'].value_counts()
market_segment_counts
count
market_segment_type	
Online	23214
Offline	10528
Corporate	2017
Complementary	391
Aviation	125

dtype: int64
ANSWER

The market segment from which most bookings come from is Online , followed by offline and corporate.
The above result can be observed in univariate analysis through bar plot of market segment type.
Question no :03
-- Hotel rates are dynamic and change according to demand and customer demographics. What are the differences in room prices in different market segments?

# Group by market segment and calculate the mean room price
room_price_by_segment = data.groupby('market_segment_type')['avg_price_per_room'].mean().reset_index()
print(room_price_by_segment)
  market_segment_type  avg_price_per_room
0            Aviation          100.704000
1       Complementary            3.141765
2           Corporate           82.911740
3             Offline           91.632679
4              Online          112.256855
ANSWER

The Hotel room prices vary with the market segment . Different prices in different segments of market are as follows
Online - 112.256855 euros
Aviation - 100.704000 euros
Offline - 91.632679 euros
Corporate - 28.911740 euros
Complimentary - 3.141795 euros
Question no :04
-- What percentage of bookings are canceled?

# Total number of bookings
total_bookings = len(df)

# Number of canceled bookings
canceled_bookings = len(df[df['booking_status'] == 'Canceled'])

# Calculate the percentage of canceled bookings
percentage_canceled = (canceled_bookings / total_bookings) * 100
print("Percentage of canceled bookings:", percentage_canceled)
Percentage of canceled bookings: 32.76361130254997
Answer

The percentage of bookings canceled is 32.7636.
This can be observed in univariate analysis of booking status using a pie chart or a bar chat as in EDA done above.
Question no :05
-- Repeating guests are the guests who stay in the hotel often and are important to brand equity. What percentage of repeating guests cancel?

# Filter for repeating guests
repeating_guests = df[df['repeated_guest'] == 1]

# Total number of repeating guests
total_repeating_guests = len(repeating_guests)

# Number of repeating guests who canceled
repeating_guests_canceled = len(repeating_guests[repeating_guests['booking_status'] == 'Canceled'])

# Calculate the percentage of repeating guests who canceled
percentage_repeating_guests_canceled = (repeating_guests_canceled / total_repeating_guests) * 100
print(f"Percentage of repeating guests who canceled: {percentage_repeating_guests_canceled:.2f}%")
Percentage of repeating guests who canceled: 1.72%
Answer

The percentage of repeating guest who canceled is 1.72 %
Question no :06
-- Many guests have special requirements when booking a hotel room. Do these requirements affect booking cancellation?

# Guests with special requests
guests_with_requests = df[df['no_of_special_requests'] > 0]

# Guests without special requests
guests_without_requests = df[df['no_of_special_requests'] == 0]

# Calculate the cancellation rate for both groups
cancellation_rate_with_requests = (len(guests_with_requests[guests_with_requests['booking_status'] == 'Canceled'])
                                   / len(guests_with_requests)) * 100

cancellation_rate_without_requests = (len(guests_without_requests[guests_without_requests['booking_status'] == 'Canceled'])
                                      / len(guests_without_requests)) * 100
print(f"Cancellation rate for guests with special requests: {cancellation_rate_with_requests:.2f}%")
print(f"Cancellation rate for guests without special requests: {cancellation_rate_without_requests:.2f}%")
Cancellation rate for guests with special requests: 20.24%
Cancellation rate for guests without special requests: 43.21%
Answer

Cancellation rate for guests with special requests: 20.24%
Cancellation rate for guests without special requests: 43.21%
Key Observations of EDA
Observations
⚛ Booking Cancellations

Certain market segments (e.g., online bookings) exhibit a significantly higher cancellation rate.
Last-minute bookings are more prone to cancellations.
High-demand months like summer or holiday periods have a higher rate of cancellations, especially for segments with higher prices.
⚛ Special Requests

Bookings with multiple special requests are more likely to be canceled, potentially due to customer expectations and the complexity of fulfilling those requests.
Guests with multiple previous non-canceled bookings tend to make fewer special requests
⚛ Pricing

Room prices fluctuate significantly based on the arrival month and lead time.
Customers with longer lead times often pay lower average prices and tend to cancel less frequently.
⚛ Lead Time and Booking Status

Bookings made with short lead times are often associated with lower room prices, but they also have a higher cancellation rate.
Peak seasons have longer lead times, where rooms are booked well in advance, and these bookings are less likely to be canceled.
⚛ Market Segmentation

Certain market segments are more likely to have higher room prices and are less prone to cancellations.
Online booking segments tend to have more variability in room prices and higher cancellation rates.
Recommendations
⚛ Cancellation Policies

Offer more policies of cancellation options for higher-risk market segments, such as online bookings like refunds etc
Add cancellation penalties based on lead time, such that closer cancellations incur higher penalties to reduce last-minute cancellations.
⚛ Improve customer Experience with Special Requests

The issue of higher cancellation rates for bookings with many special requests by ensuring those requests are met.
Communicate the hotel’s ability to meet special requests during the booking process, especially for peak seasons, to prevent cancellations related to unmet expectations.
⚛ Pricing Strategies

Charge higher prices for shorter lead times and lower prices for bookings made well in advance.
Room pricing based on market segments like corporate and family stays can be priced to increase rebooking.
⚛ Targeted Marketing

Enhance loyalty programs to retain customers with previous non-canceled bookings by offering special discounts or perks.
promotions for segments that tend to cancel frequently, such as online customers, offering incentives to encourage commitment by giving rewards or perks for early booking
⚛ Peak Seasons Management

Increase room prices for peak seasons but offer added-value services by giving breakfast, free parking etc to justify the higher cost, reducing the risk of cancellations due to high prices.
Create targeted marketing and loyalty programs to retain repeat customers and engage new segments.
Utilize lead time data to offer tailored promotions and pricing, especially for peak seasons and last-minute bookings
Data Processing
(Points : 4 )

data= df.copy()
data.loc[data['booking_status']=='Not_Canceled','booking_status'] = False
data.loc[data['booking_status']=='Canceled','booking_status'] = True
numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
# drop column because they were either time, or not helpful
numeric_columns.remove("arrival_year")


plt.figure(figsize=(15, 12))

for i, variable in enumerate(numeric_columns):
    data.boxplot()

plt.xticks(rotation=45)
plt.show()

There are two heavy outlier columns, lead_time & avg_room_price. I will only treat avg_room_price as a log because I am going to bin lead time and that should handle those outliers.
#Solving the IQR fro avg price room
quartiles = np.quantile(data['avg_price_per_room'][data['avg_price_per_room'].notnull()], [.25, .75])
power_4iqr = 4 * (quartiles[1] - quartiles[0])
print(f'Q1 = {quartiles[0]}, Q3 = {quartiles[1]}, 4*IQR = {power_4iqr}')
outlier_powers = data.loc[np.abs(data['avg_price_per_room'] - data['avg_price_per_room'].median()) > power_4iqr, 'avg_price_per_room']
outlier_powers.shape
Q1 = 80.3, Q3 = 120.0, 4*IQR = 158.8
(49,)
# creating a list of columns
dist_cols = [
    item for item in data.select_dtypes(include=np.number).columns
]

plt.figure(figsize=(15, 45))
#looping the list and ploting histograms
for i in range(len(dist_cols)):
    plt.subplot(12, 3, i + 1)
    plt.hist(data[dist_cols[i]], bins=50)
    plt.tight_layout()
    plt.title(dist_cols[i], fontsize=15)

plt.show()

data2 = data.copy()

# Print dist_cols to see its contents
print("Contents of dist_cols:", dist_cols)

# List of columns to be removed
columns_to_remove = [
    'no_of_week_nights', 'no_of_adults', 'length_stay', 'avg_price_per_room',
    'arrival_year', 'required_car_parking_space', 'arrival_date',
    'arrival_month', 'repeated_guest', 'lead_time'
]

# Safely remove columns if they exist in dist_cols
for col in columns_to_remove:
    if col in dist_cols:
        dist_cols.remove(col)

# Print updated dist_cols to confirm removal
print("Updated dist_cols:", dist_cols)
Contents of dist_cols: ['no_of_children', 'no_of_weekend_nights', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 'no_of_special_requests']
Updated dist_cols: ['no_of_children', 'no_of_weekend_nights', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 'no_of_special_requests']
# using log transforms on some columns
for col in dist_cols:
    data2[col + "_log"] = np.log(data2[col] + 1)

# dropping the original columns
data2.drop(dist_cols, axis=1, inplace=True)
data2.head()
Booking_ID	no_of_adults	no_of_week_nights	type_of_meal_plan	required_car_parking_space	room_type_reserved	lead_time	arrival_year	arrival_month	arrival_date	market_segment_type	repeated_guest	avg_price_per_room	booking_status	no_of_children_log	no_of_weekend_nights_log	no_of_previous_cancellations_log	no_of_previous_bookings_not_canceled_log	no_of_special_requests_log
0	INN00001	2	2	Meal Plan 1	0	Room_Type 1	224	2017	10	2	Offline	0	65.00	False	0.0	0.693147	0.0	0.0	0.000000
1	INN00002	2	3	Not Selected	0	Room_Type 1	5	2018	11	6	Online	0	106.68	False	0.0	1.098612	0.0	0.0	0.693147
2	INN00003	1	1	Meal Plan 1	0	Room_Type 1	1	2018	2	28	Online	0	60.00	True	0.0	1.098612	0.0	0.0	0.000000
3	INN00004	2	2	Meal Plan 1	0	Room_Type 1	211	2018	5	20	Online	0	100.00	True	0.0	0.000000	0.0	0.0	0.000000
4	INN00005	2	1	Not Selected	0	Room_Type 1	48	2018	4	11	Online	0	94.50	True	0.0	0.693147	0.0	0.0	0.000000
# Drop booking id
data2 = data2.drop('Booking_ID', axis=1)
data2.head()
no_of_adults	no_of_week_nights	type_of_meal_plan	required_car_parking_space	room_type_reserved	lead_time	arrival_year	arrival_month	arrival_date	market_segment_type	repeated_guest	avg_price_per_room	booking_status	no_of_children_log	no_of_weekend_nights_log	no_of_previous_cancellations_log	no_of_previous_bookings_not_canceled_log	no_of_special_requests_log
0	2	2	Meal Plan 1	0	Room_Type 1	224	2017	10	2	Offline	0	65.00	False	0.0	0.693147	0.0	0.0	0.000000
1	2	3	Not Selected	0	Room_Type 1	5	2018	11	6	Online	0	106.68	False	0.0	1.098612	0.0	0.0	0.693147
2	1	1	Meal Plan 1	0	Room_Type 1	1	2018	2	28	Online	0	60.00	True	0.0	1.098612	0.0	0.0	0.000000
3	2	2	Meal Plan 1	0	Room_Type 1	211	2018	5	20	Online	0	100.00	True	0.0	0.000000	0.0	0.0	0.000000
4	2	1	Not Selected	0	Room_Type 1	48	2018	4	11	Online	0	94.50	True	0.0	0.693147	0.0	0.0	0.000000
# viewing the distributions after the log transformation.
dist_cols = [
    item for item in data2.select_dtypes(include=np.number).columns
]

#  plot histogram of all numeric columns

plt.figure(figsize=(15, 45))

for i in range(len(dist_cols)):
    plt.subplot(12, 3, i + 1)
    plt.hist(data2[dist_cols[i]], bins=50)  # Adjust the number of bins as needed
    sns.histplot(data=data2, x=dist_cols[i], kde=True)
    plt.tight_layout()
    plt.title(dist_cols[i], fontsize=25)

plt.show()

# OneHotEncoding catergorical variables
dummy_data = pd.get_dummies (
    data2,
    columns = [
        'type_of_meal_plan',
        'room_type_reserved',
        'market_segment_type',
    ],
    drop_first=True,
)
dummy_data.head()
no_of_adults	no_of_week_nights	required_car_parking_space	lead_time	arrival_year	arrival_month	arrival_date	repeated_guest	avg_price_per_room	booking_status	...	room_type_reserved_Room_Type 2	room_type_reserved_Room_Type 3	room_type_reserved_Room_Type 4	room_type_reserved_Room_Type 5	room_type_reserved_Room_Type 6	room_type_reserved_Room_Type 7	market_segment_type_Complementary	market_segment_type_Corporate	market_segment_type_Offline	market_segment_type_Online
0	2	2	0	224	2017	10	2	0	65.00	False	...	False	False	False	False	False	False	False	False	True	False
1	2	3	0	5	2018	11	6	0	106.68	False	...	False	False	False	False	False	False	False	False	False	True
2	1	1	0	1	2018	2	28	0	60.00	True	...	False	False	False	False	False	False	False	False	False	True
3	2	2	0	211	2018	5	20	0	100.00	True	...	False	False	False	False	False	False	False	False	False	True
4	2	1	0	48	2018	4	11	0	94.50	True	...	False	False	False	False	False	False	False	False	False	True
5 rows × 28 columns

dummy_data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 36275 entries, 0 to 36274
Data columns (total 28 columns):
 #   Column                                    Non-Null Count  Dtype  
---  ------                                    --------------  -----  
 0   no_of_adults                              36275 non-null  int64  
 1   no_of_week_nights                         36275 non-null  int64  
 2   required_car_parking_space                36275 non-null  int64  
 3   lead_time                                 36275 non-null  int64  
 4   arrival_year                              36275 non-null  int64  
 5   arrival_month                             36275 non-null  int64  
 6   arrival_date                              36275 non-null  int64  
 7   repeated_guest                            36275 non-null  int64  
 8   avg_price_per_room                        36275 non-null  float64
 9   booking_status                            36275 non-null  object 
 10  no_of_children_log                        36275 non-null  float64
 11  no_of_weekend_nights_log                  36275 non-null  float64
 12  no_of_previous_cancellations_log          36275 non-null  float64
 13  no_of_previous_bookings_not_canceled_log  36275 non-null  float64
 14  no_of_special_requests_log                36275 non-null  float64
 15  type_of_meal_plan_Meal Plan 2             36275 non-null  bool   
 16  type_of_meal_plan_Meal Plan 3             36275 non-null  bool   
 17  type_of_meal_plan_Not Selected            36275 non-null  bool   
 18  room_type_reserved_Room_Type 2            36275 non-null  bool   
 19  room_type_reserved_Room_Type 3            36275 non-null  bool   
 20  room_type_reserved_Room_Type 4            36275 non-null  bool   
 21  room_type_reserved_Room_Type 5            36275 non-null  bool   
 22  room_type_reserved_Room_Type 6            36275 non-null  bool   
 23  room_type_reserved_Room_Type 7            36275 non-null  bool   
 24  market_segment_type_Complementary         36275 non-null  bool   
 25  market_segment_type_Corporate             36275 non-null  bool   
 26  market_segment_type_Offline               36275 non-null  bool   
 27  market_segment_type_Online                36275 non-null  bool   
dtypes: bool(13), float64(6), int64(8), object(1)
memory usage: 4.6+ MB
dummied_cut = pd.cut(dummy_data['lead_time'], 5, labels=['lat_min','short','med','long','advanced'])
dummied_cut.head(10)
lead_time
0	med
1	lat_min
2	lat_min
3	med
4	lat_min
5	long
6	lat_min
7	lat_min
8	short
9	lat_min

dtype: category
data3 = pd.merge(dummy_data, dummied_cut, left_index=True, right_index=True)

data3.head().T
0	1	2	3	4
no_of_adults	2	2	1	2	2
no_of_week_nights	2	3	1	2	1
required_car_parking_space	0	0	0	0	0
lead_time_x	224	5	1	211	48
arrival_year	2017	2018	2018	2018	2018
arrival_month	10	11	2	5	4
arrival_date	2	6	28	20	11
repeated_guest	0	0	0	0	0
avg_price_per_room	65.0	106.68	60.0	100.0	94.5
booking_status	False	False	True	True	True
no_of_children_log	0.0	0.0	0.0	0.0	0.0
no_of_weekend_nights_log	0.693147	1.098612	1.098612	0.0	0.693147
no_of_previous_cancellations_log	0.0	0.0	0.0	0.0	0.0
no_of_previous_bookings_not_canceled_log	0.0	0.0	0.0	0.0	0.0
no_of_special_requests_log	0.0	0.693147	0.0	0.0	0.0
type_of_meal_plan_Meal Plan 2	False	False	False	False	False
type_of_meal_plan_Meal Plan 3	False	False	False	False	False
type_of_meal_plan_Not Selected	False	True	False	False	True
room_type_reserved_Room_Type 2	False	False	False	False	False
room_type_reserved_Room_Type 3	False	False	False	False	False
room_type_reserved_Room_Type 4	False	False	False	False	False
room_type_reserved_Room_Type 5	False	False	False	False	False
room_type_reserved_Room_Type 6	False	False	False	False	False
room_type_reserved_Room_Type 7	False	False	False	False	False
market_segment_type_Complementary	False	False	False	False	False
market_segment_type_Corporate	False	False	False	False	False
market_segment_type_Offline	True	False	False	False	False
market_segment_type_Online	False	True	True	True	True
lead_time_y	med	lat_min	lat_min	med	lat_min
# dropping time variables and lead_time_x since it has been binned into 5 columns.
data3_5 = data3.drop(['lead_time_x','arrival_date', 'arrival_year'], axis=1)
data4 = pd.get_dummies (
    data3_5,
    columns = [
        'lead_time_y',
    ],
    drop_first=True,
)
data4.head().T
0	1	2	3	4
no_of_adults	2	2	1	2	2
no_of_week_nights	2	3	1	2	1
required_car_parking_space	0	0	0	0	0
arrival_month	10	11	2	5	4
repeated_guest	0	0	0	0	0
avg_price_per_room	65.0	106.68	60.0	100.0	94.5
booking_status	False	False	True	True	True
no_of_children_log	0.0	0.0	0.0	0.0	0.0
no_of_weekend_nights_log	0.693147	1.098612	1.098612	0.0	0.693147
no_of_previous_cancellations_log	0.0	0.0	0.0	0.0	0.0
no_of_previous_bookings_not_canceled_log	0.0	0.0	0.0	0.0	0.0
no_of_special_requests_log	0.0	0.693147	0.0	0.0	0.0
type_of_meal_plan_Meal Plan 2	False	False	False	False	False
type_of_meal_plan_Meal Plan 3	False	False	False	False	False
type_of_meal_plan_Not Selected	False	True	False	False	True
room_type_reserved_Room_Type 2	False	False	False	False	False
room_type_reserved_Room_Type 3	False	False	False	False	False
room_type_reserved_Room_Type 4	False	False	False	False	False
room_type_reserved_Room_Type 5	False	False	False	False	False
room_type_reserved_Room_Type 6	False	False	False	False	False
room_type_reserved_Room_Type 7	False	False	False	False	False
market_segment_type_Complementary	False	False	False	False	False
market_segment_type_Corporate	False	False	False	False	False
market_segment_type_Offline	True	False	False	False	False
market_segment_type_Online	False	True	True	True	True
lead_time_y_short	False	False	False	False	False
lead_time_y_med	True	False	False	True	False
lead_time_y_long	False	False	False	False	False
lead_time_y_advanced	False	False	False	False	False
data4 = data4.astype(float)
data4.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 36275 entries, 0 to 36274
Data columns (total 29 columns):
 #   Column                                    Non-Null Count  Dtype  
---  ------                                    --------------  -----  
 0   no_of_adults                              36275 non-null  float64
 1   no_of_week_nights                         36275 non-null  float64
 2   required_car_parking_space                36275 non-null  float64
 3   arrival_month                             36275 non-null  float64
 4   repeated_guest                            36275 non-null  float64
 5   avg_price_per_room                        36275 non-null  float64
 6   booking_status                            36275 non-null  float64
 7   no_of_children_log                        36275 non-null  float64
 8   no_of_weekend_nights_log                  36275 non-null  float64
 9   no_of_previous_cancellations_log          36275 non-null  float64
 10  no_of_previous_bookings_not_canceled_log  36275 non-null  float64
 11  no_of_special_requests_log                36275 non-null  float64
 12  type_of_meal_plan_Meal Plan 2             36275 non-null  float64
 13  type_of_meal_plan_Meal Plan 3             36275 non-null  float64
 14  type_of_meal_plan_Not Selected            36275 non-null  float64
 15  room_type_reserved_Room_Type 2            36275 non-null  float64
 16  room_type_reserved_Room_Type 3            36275 non-null  float64
 17  room_type_reserved_Room_Type 4            36275 non-null  float64
 18  room_type_reserved_Room_Type 5            36275 non-null  float64
 19  room_type_reserved_Room_Type 6            36275 non-null  float64
 20  room_type_reserved_Room_Type 7            36275 non-null  float64
 21  market_segment_type_Complementary         36275 non-null  float64
 22  market_segment_type_Corporate             36275 non-null  float64
 23  market_segment_type_Offline               36275 non-null  float64
 24  market_segment_type_Online                36275 non-null  float64
 25  lead_time_y_short                         36275 non-null  float64
 26  lead_time_y_med                           36275 non-null  float64
 27  lead_time_y_long                          36275 non-null  float64
 28  lead_time_y_advanced                      36275 non-null  float64
dtypes: float64(29)
memory usage: 8.0 MB
# Assuming data has the original 'booking_status' column
if 'booking_status' in data.columns:
    data4['booking_status'] = data['booking_status']
else:
    print("'booking_status' not found in the original data")
print(data4.columns)
Index(['no_of_adults', 'no_of_week_nights', 'required_car_parking_space',
       'arrival_month', 'repeated_guest', 'avg_price_per_room',
       'no_of_children_log', 'no_of_weekend_nights_log',
       'no_of_previous_cancellations_log',
       'no_of_previous_bookings_not_canceled_log',
       'no_of_special_requests_log', 'type_of_meal_plan_Meal Plan 2',
       'type_of_meal_plan_Meal Plan 3', 'type_of_meal_plan_Not Selected',
       'room_type_reserved_Room_Type 2', 'room_type_reserved_Room_Type 3',
       'room_type_reserved_Room_Type 4', 'room_type_reserved_Room_Type 5',
       'room_type_reserved_Room_Type 6', 'room_type_reserved_Room_Type 7',
       'market_segment_type_Complementary', 'market_segment_type_Corporate',
       'market_segment_type_Offline', 'market_segment_type_Online',
       'lead_time_y_short', 'lead_time_y_med', 'lead_time_y_long',
       'lead_time_y_advanced', 'booking_status'],
      dtype='object')
# Using the SCIEM method I will split the train test data first.
X = data4.drop("booking_status" , axis=1)
y = data4.pop("booking_status")
# Import add_constant from statsmodels
from statsmodels.api import add_constant

# Adding a constant (intercept) column to X
X = add_constant(X)

# Now, X has a constant column added
print(X.head())
   const  no_of_adults  no_of_week_nights  required_car_parking_space  \
0    1.0           2.0                2.0                         0.0   
1    1.0           2.0                3.0                         0.0   
2    1.0           1.0                1.0                         0.0   
3    1.0           2.0                2.0                         0.0   
4    1.0           2.0                1.0                         0.0   

   arrival_month  repeated_guest  avg_price_per_room  no_of_children_log  \
0           10.0             0.0               65.00                 0.0   
1           11.0             0.0              106.68                 0.0   
2            2.0             0.0               60.00                 0.0   
3            5.0             0.0              100.00                 0.0   
4            4.0             0.0               94.50                 0.0   

   no_of_weekend_nights_log  no_of_previous_cancellations_log  ...  \
0                  0.693147                               0.0  ...   
1                  1.098612                               0.0  ...   
2                  1.098612                               0.0  ...   
3                  0.000000                               0.0  ...   
4                  0.693147                               0.0  ...   

   room_type_reserved_Room_Type 6  room_type_reserved_Room_Type 7  \
0                             0.0                             0.0   
1                             0.0                             0.0   
2                             0.0                             0.0   
3                             0.0                             0.0   
4                             0.0                             0.0   

   market_segment_type_Complementary  market_segment_type_Corporate  \
0                                0.0                            0.0   
1                                0.0                            0.0   
2                                0.0                            0.0   
3                                0.0                            0.0   
4                                0.0                            0.0   

   market_segment_type_Offline  market_segment_type_Online  lead_time_y_short  \
0                          1.0                         0.0                0.0   
1                          0.0                         1.0                0.0   
2                          0.0                         1.0                0.0   
3                          0.0                         1.0                0.0   
4                          0.0                         1.0                0.0   

   lead_time_y_med  lead_time_y_long  lead_time_y_advanced  
0              1.0               0.0                   0.0  
1              0.0               0.0                   0.0  
2              0.0               0.0                   0.0  
3              1.0               0.0                   0.0  
4              0.0               0.0                   0.0  

[5 rows x 29 columns]
# Train/Test Split 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=1)
print("Number of rows in train data =", X_train.shape[0])
print("Number of rows in test data =", X_test.shape[0])
Number of rows in train data = 25392
Number of rows in test data = 10883
print("Percentage of classes in training set:")
print(y_train.value_counts(normalize=True))
print("Percentage of classes in test set:")
print(y_test.value_counts(normalize=True))
Percentage of classes in training set:
booking_status
0.0    0.670644
1.0    0.329356
Name: proportion, dtype: float64
Percentage of classes in test set:
booking_status
0.0    0.676376
1.0    0.323624
Name: proportion, dtype: float64
X_train.info()
<class 'pandas.core.frame.DataFrame'>
Index: 25392 entries, 13662 to 33003
Data columns (total 28 columns):
 #   Column                                    Non-Null Count  Dtype  
---  ------                                    --------------  -----  
 0   no_of_adults                              25392 non-null  float64
 1   no_of_week_nights                         25392 non-null  float64
 2   required_car_parking_space                25392 non-null  float64
 3   arrival_month                             25392 non-null  float64
 4   repeated_guest                            25392 non-null  float64
 5   avg_price_per_room                        25392 non-null  float64
 6   no_of_children_log                        25392 non-null  float64
 7   no_of_weekend_nights_log                  25392 non-null  float64
 8   no_of_previous_cancellations_log          25392 non-null  float64
 9   no_of_previous_bookings_not_canceled_log  25392 non-null  float64
 10  no_of_special_requests_log                25392 non-null  float64
 11  type_of_meal_plan_Meal Plan 2             25392 non-null  float64
 12  type_of_meal_plan_Meal Plan 3             25392 non-null  float64
 13  type_of_meal_plan_Not Selected            25392 non-null  float64
 14  room_type_reserved_Room_Type 2            25392 non-null  float64
 15  room_type_reserved_Room_Type 3            25392 non-null  float64
 16  room_type_reserved_Room_Type 4            25392 non-null  float64
 17  room_type_reserved_Room_Type 5            25392 non-null  float64
 18  room_type_reserved_Room_Type 6            25392 non-null  float64
 19  room_type_reserved_Room_Type 7            25392 non-null  float64
 20  market_segment_type_Complementary         25392 non-null  float64
 21  market_segment_type_Corporate             25392 non-null  float64
 22  market_segment_type_Offline               25392 non-null  float64
 23  market_segment_type_Online                25392 non-null  float64
 24  lead_time_y_short                         25392 non-null  float64
 25  lead_time_y_med                           25392 non-null  float64
 26  lead_time_y_long                          25392 non-null  float64
 27  lead_time_y_advanced                      25392 non-null  float64
dtypes: float64(28)
memory usage: 5.6 MB
# It is a good idea to explore the data once again after manipulating it.
plt.figure(figsize=(20,10))
sns.heatmap(
data4.corr(), annot=True, vmin=-1, vmax=1, fmt='.2f')
<Axes: >

In order to make statistical inferences from a logistic regression model, it is important to ensure that there is no multicollinearity present in the data.

import statsmodels.api as sm
from statsmodels.api import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
# let's check the VIF of the predictors
vif_series = pd.Series(
    [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])],
    index=X_train.columns,
    dtype=float,
)
print("VIF values: \n\n{}\n".format(vif_series))
VIF values: 

no_of_adults                                18.197098
no_of_week_nights                            3.680150
required_car_parking_space                   1.075494
arrival_month                                7.097404
repeated_guest                               3.417968
avg_price_per_room                          17.989022
no_of_children_log                           2.006153
no_of_weekend_nights_log                     2.133807
no_of_previous_cancellations_log             1.608889
no_of_previous_bookings_not_canceled_log     3.570483
no_of_special_requests_log                   2.202569
type_of_meal_plan_Meal Plan 2                1.334174
type_of_meal_plan_Meal Plan 3                1.025507
type_of_meal_plan_Not Selected               1.432984
room_type_reserved_Room_Type 2               1.111269
room_type_reserved_Room_Type 3               1.003573
room_type_reserved_Room_Type 4               1.638586
room_type_reserved_Room_Type 5               1.034873
room_type_reserved_Room_Type 6               1.903476
room_type_reserved_Room_Type 7               1.114383
market_segment_type_Complementary            1.296787
market_segment_type_Corporate                2.418865
market_segment_type_Offline                  9.385691
market_segment_type_Online                  23.630149
lead_time_y_short                            1.424498
lead_time_y_med                              1.218744
lead_time_y_long                             1.207261
lead_time_y_advanced                         1.054674
dtype: float64

#dropping the number of weekend & week nights because I have combined them into one & market segements because they all have large multi values
X_train1 = X_train.drop(['no_of_weekend_nights_log',
                         'no_of_week_nights',
                         'market_segment_type_Online',
                         'market_segment_type_Offline',
                         'market_segment_type_Corporate',
                        'market_segment_type_Complementary'],
                       axis=1)
Building a Logistic Regression model

# Import necessary libraries
import statsmodels.api as sm
from statsmodels.api import add_constant
logit = sm.Logit(y_train, X_train1.astype(float))
lg = logit.fit()
Optimization terminated successfully.
         Current function value: 0.495874
         Iterations 10
# print the logistic regression summary
print(lg.summary())
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         booking_status   No. Observations:                25392
Model:                          Logit   Df Residuals:                    25370
Method:                           MLE   Df Model:                           21
Date:                Sun, 08 Sep 2024   Pseudo R-squ.:                  0.2175
Time:                        16:03:56   Log-Likelihood:                -12591.
converged:                       True   LL-Null:                       -16091.
Covariance Type:            nonrobust   LLR p-value:                     0.000
============================================================================================================
                                               coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------------------------------------
no_of_adults                                -0.4712      0.028    -17.051      0.000      -0.525      -0.417
required_car_parking_space                  -1.3018      0.130     -9.990      0.000      -1.557      -1.046
arrival_month                               -0.1376      0.005    -27.464      0.000      -0.147      -0.128
repeated_guest                              -3.2515      0.644     -5.050      0.000      -4.513      -1.990
avg_price_per_room                           0.0086      0.001     16.833      0.000       0.008       0.010
no_of_children_log                           0.5743      0.091      6.341      0.000       0.397       0.752
no_of_previous_cancellations_log             1.2858      0.459      2.799      0.005       0.385       2.186
no_of_previous_bookings_not_canceled_log    -1.0502      0.527     -1.993      0.046      -2.083      -0.017
no_of_special_requests_log                  -1.6744      0.041    -40.640      0.000      -1.755      -1.594
type_of_meal_plan_Meal Plan 2               -0.2106      0.055     -3.841      0.000      -0.318      -0.103
type_of_meal_plan_Meal Plan 3                0.5421      1.286      0.422      0.673      -1.978       3.062
type_of_meal_plan_Not Selected               0.5950      0.046     12.887      0.000       0.504       0.685
room_type_reserved_Room_Type 2              -0.3656      0.119     -3.074      0.002      -0.599      -0.132
room_type_reserved_Room_Type 3              -0.3770      0.987     -0.382      0.703      -2.312       1.558
room_type_reserved_Room_Type 4               0.5761      0.046     12.576      0.000       0.486       0.666
room_type_reserved_Room_Type 5              -0.6209      0.192     -3.228      0.001      -0.998      -0.244
room_type_reserved_Room_Type 6               0.0997      0.127      0.783      0.434      -0.150       0.349
room_type_reserved_Room_Type 7              -0.0272      0.264     -0.103      0.918      -0.545       0.490
lead_time_y_short                            1.1559      0.037     31.123      0.000       1.083       1.229
lead_time_y_med                              2.7193      0.057     47.914      0.000       2.608       2.831
lead_time_y_long                             2.8778      0.076     37.914      0.000       2.729       3.027
lead_time_y_advanced                         4.3622      0.248     17.618      0.000       3.877       4.847
============================================================================================================
# let's check the VIF of the predictors
vif_series = pd.Series(
    [variance_inflation_factor(X_train1.values, i) for i in range(X_train1.shape[1])],
    index=X_train1.columns,
    dtype=float,
)
print("VIF values: \n\n{}\n".format(vif_series))
VIF values: 

no_of_adults                                11.519816
required_car_parking_space                   1.069077
arrival_month                                5.913945
repeated_guest                               3.260798
avg_price_per_room                          11.718683
no_of_children_log                           1.991388
no_of_previous_cancellations_log             1.587147
no_of_previous_bookings_not_canceled_log     3.488538
no_of_special_requests_log                   1.961361
type_of_meal_plan_Meal Plan 2                1.243078
type_of_meal_plan_Meal Plan 3                1.017756
type_of_meal_plan_Not Selected               1.276209
room_type_reserved_Room_Type 2               1.089233
room_type_reserved_Room_Type 3               1.000940
room_type_reserved_Room_Type 4               1.494463
room_type_reserved_Room_Type 5               1.020678
room_type_reserved_Room_Type 6               1.823130
room_type_reserved_Room_Type 7               1.065028
lead_time_y_short                            1.372970
lead_time_y_med                              1.192447
lead_time_y_long                             1.179291
lead_time_y_advanced                         1.051144
dtype: float64

# test performance
pred_train = lg.predict(X_train1) > 0.5
pred_train = np.round(pred_train)
X_train2 = X_train1.drop(['room_type_reserved_Room_Type 3'], axis=1)
X_train2.info()
<class 'pandas.core.frame.DataFrame'>
Index: 25392 entries, 13662 to 33003
Data columns (total 21 columns):
 #   Column                                    Non-Null Count  Dtype  
---  ------                                    --------------  -----  
 0   no_of_adults                              25392 non-null  float64
 1   required_car_parking_space                25392 non-null  float64
 2   arrival_month                             25392 non-null  float64
 3   repeated_guest                            25392 non-null  float64
 4   avg_price_per_room                        25392 non-null  float64
 5   no_of_children_log                        25392 non-null  float64
 6   no_of_previous_cancellations_log          25392 non-null  float64
 7   no_of_previous_bookings_not_canceled_log  25392 non-null  float64
 8   no_of_special_requests_log                25392 non-null  float64
 9   type_of_meal_plan_Meal Plan 2             25392 non-null  float64
 10  type_of_meal_plan_Meal Plan 3             25392 non-null  float64
 11  type_of_meal_plan_Not Selected            25392 non-null  float64
 12  room_type_reserved_Room_Type 2            25392 non-null  float64
 13  room_type_reserved_Room_Type 4            25392 non-null  float64
 14  room_type_reserved_Room_Type 5            25392 non-null  float64
 15  room_type_reserved_Room_Type 6            25392 non-null  float64
 16  room_type_reserved_Room_Type 7            25392 non-null  float64
 17  lead_time_y_short                         25392 non-null  float64
 18  lead_time_y_med                           25392 non-null  float64
 19  lead_time_y_long                          25392 non-null  float64
 20  lead_time_y_advanced                      25392 non-null  float64
dtypes: float64(21)
memory usage: 4.3 MB
logit = sm.Logit(y_train, X_train2.astype(float))
lg2 = logit.fit()
Optimization terminated successfully.
         Current function value: 0.495877
         Iterations 10
print(lg2.summary())
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         booking_status   No. Observations:                25392
Model:                          Logit   Df Residuals:                    25371
Method:                           MLE   Df Model:                           20
Date:                Sun, 08 Sep 2024   Pseudo R-squ.:                  0.2175
Time:                        16:07:42   Log-Likelihood:                -12591.
converged:                       True   LL-Null:                       -16091.
Covariance Type:            nonrobust   LLR p-value:                     0.000
============================================================================================================
                                               coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------------------------------------
no_of_adults                                -0.4712      0.028    -17.051      0.000      -0.525      -0.417
required_car_parking_space                  -1.3018      0.130     -9.990      0.000      -1.557      -1.046
arrival_month                               -0.1376      0.005    -27.477      0.000      -0.147      -0.128
repeated_guest                              -3.2514      0.644     -5.050      0.000      -4.513      -1.989
avg_price_per_room                           0.0086      0.001     16.836      0.000       0.008       0.010
no_of_children_log                           0.5743      0.091      6.341      0.000       0.397       0.752
no_of_previous_cancellations_log             1.2857      0.459      2.799      0.005       0.385       2.186
no_of_previous_bookings_not_canceled_log    -1.0503      0.527     -1.993      0.046      -2.083      -0.017
no_of_special_requests_log                  -1.6742      0.041    -40.638      0.000      -1.755      -1.593
type_of_meal_plan_Meal Plan 2               -0.2105      0.055     -3.839      0.000      -0.318      -0.103
type_of_meal_plan_Meal Plan 3                0.5422      1.286      0.422      0.673      -1.978       3.062
type_of_meal_plan_Not Selected               0.5949      0.046     12.887      0.000       0.504       0.685
room_type_reserved_Room_Type 2              -0.3655      0.119     -3.073      0.002      -0.599      -0.132
room_type_reserved_Room_Type 4               0.5762      0.046     12.577      0.000       0.486       0.666
room_type_reserved_Room_Type 5              -0.6208      0.192     -3.228      0.001      -0.998      -0.244
room_type_reserved_Room_Type 6               0.0996      0.127      0.782      0.434      -0.150       0.349
room_type_reserved_Room_Type 7              -0.0273      0.264     -0.103      0.918      -0.545       0.490
lead_time_y_short                            1.1558      0.037     31.121      0.000       1.083       1.229
lead_time_y_med                              2.7193      0.057     47.913      0.000       2.608       2.831
lead_time_y_long                             2.8780      0.076     37.917      0.000       2.729       3.027
lead_time_y_advanced                         4.3622      0.248     17.618      0.000       3.877       4.848
============================================================================================================
X_train3 = X_train2.drop(['no_of_previous_bookings_not_canceled_log'], axis=1)
logit = sm.Logit(y_train, X_train3.astype(float))
lg3 = logit.fit()
Optimization terminated successfully.
         Current function value: 0.495990
         Iterations 9
print(lg3.summary())
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         booking_status   No. Observations:                25392
Model:                          Logit   Df Residuals:                    25372
Method:                           MLE   Df Model:                           19
Date:                Sun, 08 Sep 2024   Pseudo R-squ.:                  0.2173
Time:                        16:08:29   Log-Likelihood:                -12594.
converged:                       True   LL-Null:                       -16091.
Covariance Type:            nonrobust   LLR p-value:                     0.000
====================================================================================================
                                       coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------------------
no_of_adults                        -0.4715      0.028    -17.060      0.000      -0.526      -0.417
required_car_parking_space          -1.3007      0.130     -9.983      0.000      -1.556      -1.045
arrival_month                       -0.1376      0.005    -27.482      0.000      -0.147      -0.128
repeated_guest                      -3.9896      0.551     -7.239      0.000      -5.070      -2.909
avg_price_per_room                   0.0086      0.001     16.848      0.000       0.008       0.010
no_of_children_log                   0.5749      0.091      6.348      0.000       0.397       0.752
no_of_previous_cancellations_log     1.0029      0.376      2.666      0.008       0.265       1.740
no_of_special_requests_log          -1.6758      0.041    -40.677      0.000      -1.757      -1.595
type_of_meal_plan_Meal Plan 2       -0.2122      0.055     -3.869      0.000      -0.320      -0.105
type_of_meal_plan_Meal Plan 3        0.5422      1.285      0.422      0.673      -1.977       3.062
type_of_meal_plan_Not Selected       0.5952      0.046     12.892      0.000       0.505       0.686
room_type_reserved_Room_Type 2      -0.3661      0.119     -3.077      0.002      -0.599      -0.133
room_type_reserved_Room_Type 4       0.5764      0.046     12.581      0.000       0.487       0.666
room_type_reserved_Room_Type 5      -0.6225      0.192     -3.238      0.001      -0.999      -0.246
room_type_reserved_Room_Type 6       0.0989      0.127      0.777      0.437      -0.151       0.348
room_type_reserved_Room_Type 7      -0.0296      0.264     -0.112      0.911      -0.547       0.488
lead_time_y_short                    1.1561      0.037     31.129      0.000       1.083       1.229
lead_time_y_med                      2.7213      0.057     47.943      0.000       2.610       2.833
lead_time_y_long                     2.8798      0.076     37.924      0.000       2.731       3.029
lead_time_y_advanced                 4.4037      0.250     17.620      0.000       3.914       4.894
====================================================================================================
# let's check the VIF of the predictors again to see if any Multicollinearity persist
vif_series = pd.Series(
    [variance_inflation_factor(X_train3.values, i) for i in range(X_train3.shape[1])],
    index=X_train3.columns,
    dtype=float,
)
print("VIF values: \n\n{}\n".format(vif_series))
VIF values: 

no_of_adults                        11.519270
required_car_parking_space           1.068324
arrival_month                        5.911260
repeated_guest                       1.508209
avg_price_per_room                  11.712940
no_of_children_log                   1.991288
no_of_previous_cancellations_log     1.436736
no_of_special_requests_log           1.952941
type_of_meal_plan_Meal Plan 2        1.242879
type_of_meal_plan_Meal Plan 3        1.017726
type_of_meal_plan_Not Selected       1.276122
room_type_reserved_Room_Type 2       1.089226
room_type_reserved_Room_Type 4       1.494420
room_type_reserved_Room_Type 5       1.019830
room_type_reserved_Room_Type 6       1.822871
room_type_reserved_Room_Type 7       1.064638
lead_time_y_short                    1.372770
lead_time_y_med                      1.192427
lead_time_y_long                     1.179198
lead_time_y_advanced                 1.050944
dtype: float64

X_train4 = X_train3.drop(['room_type_reserved_Room_Type 2'], axis=1)

logit = sm.Logit(y_train, X_train4.astype(float))
lg4 = logit.fit()
Optimization terminated successfully.
         Current function value: 0.496180
         Iterations 9
print(lg4.summary())
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         booking_status   No. Observations:                25392
Model:                          Logit   Df Residuals:                    25373
Method:                           MLE   Df Model:                           18
Date:                Sun, 08 Sep 2024   Pseudo R-squ.:                  0.2170
Time:                        16:09:27   Log-Likelihood:                -12599.
converged:                       True   LL-Null:                       -16091.
Covariance Type:            nonrobust   LLR p-value:                     0.000
====================================================================================================
                                       coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------------------
no_of_adults                        -0.4727      0.028    -17.118      0.000      -0.527      -0.419
required_car_parking_space          -1.3070      0.130    -10.021      0.000      -1.563      -1.051
arrival_month                       -0.1376      0.005    -27.504      0.000      -0.147      -0.128
repeated_guest                      -3.9887      0.551     -7.239      0.000      -5.069      -2.909
avg_price_per_room                   0.0086      0.001     16.834      0.000       0.008       0.010
no_of_children_log                   0.5073      0.088      5.771      0.000       0.335       0.680
no_of_previous_cancellations_log     1.0050      0.376      2.674      0.008       0.268       1.742
no_of_special_requests_log          -1.6788      0.041    -40.761      0.000      -1.760      -1.598
type_of_meal_plan_Meal Plan 2       -0.2055      0.055     -3.750      0.000      -0.313      -0.098
type_of_meal_plan_Meal Plan 3        0.5230      1.282      0.408      0.683      -1.991       3.037
type_of_meal_plan_Not Selected       0.6001      0.046     13.005      0.000       0.510       0.691
room_type_reserved_Room_Type 4       0.5847      0.046     12.782      0.000       0.495       0.674
room_type_reserved_Room_Type 5      -0.6109      0.192     -3.180      0.001      -0.987      -0.234
room_type_reserved_Room_Type 6       0.1712      0.125      1.369      0.171      -0.074       0.416
room_type_reserved_Room_Type 7       0.0189      0.263      0.072      0.943      -0.496       0.534
lead_time_y_short                    1.1533      0.037     31.065      0.000       1.080       1.226
lead_time_y_med                      2.7090      0.057     47.907      0.000       2.598       2.820
lead_time_y_long                     2.8751      0.076     37.901      0.000       2.726       3.024
lead_time_y_advanced                 4.3985      0.250     17.604      0.000       3.909       4.888
====================================================================================================
X_train5 = X_train4.drop(['room_type_reserved_Room_Type 4'], axis=1)

logit = sm.Logit(y_train, X_train5.astype(float))
lg5 = logit.fit()
Optimization terminated successfully.
         Current function value: 0.499373
         Iterations 9
print(lg5.summary())
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         booking_status   No. Observations:                25392
Model:                          Logit   Df Residuals:                    25374
Method:                           MLE   Df Model:                           17
Date:                Sun, 08 Sep 2024   Pseudo R-squ.:                  0.2120
Time:                        16:10:07   Log-Likelihood:                -12680.
converged:                       True   LL-Null:                       -16091.
Covariance Type:            nonrobust   LLR p-value:                     0.000
====================================================================================================
                                       coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------------------
no_of_adults                        -0.4324      0.027    -15.812      0.000      -0.486      -0.379
required_car_parking_space          -1.2989      0.130    -10.009      0.000      -1.553      -1.045
arrival_month                       -0.1445      0.005    -29.008      0.000      -0.154      -0.135
repeated_guest                      -4.0168      0.550     -7.300      0.000      -5.095      -2.938
avg_price_per_room                   0.0099      0.001     19.711      0.000       0.009       0.011
no_of_children_log                   0.4024      0.087      4.632      0.000       0.232       0.573
no_of_previous_cancellations_log     0.9906      0.377      2.629      0.009       0.252       1.729
no_of_special_requests_log          -1.6350      0.041    -40.101      0.000      -1.715      -1.555
type_of_meal_plan_Meal Plan 2       -0.3010      0.054     -5.548      0.000      -0.407      -0.195
type_of_meal_plan_Meal Plan 3        0.5454      1.349      0.404      0.686      -2.098       3.189
type_of_meal_plan_Not Selected       0.4520      0.044     10.200      0.000       0.365       0.539
room_type_reserved_Room_Type 5      -0.7607      0.191     -3.988      0.000      -1.135      -0.387
room_type_reserved_Room_Type 6      -0.0001      0.123     -0.001      0.999      -0.242       0.242
room_type_reserved_Room_Type 7      -0.2404      0.261     -0.922      0.356      -0.751       0.270
lead_time_y_short                    1.1233      0.037     30.394      0.000       1.051       1.196
lead_time_y_med                      2.6371      0.056     47.049      0.000       2.527       2.747
lead_time_y_long                     2.7832      0.075     36.876      0.000       2.635       2.931
lead_time_y_advanced                 4.3071      0.250     17.242      0.000       3.817       4.797
====================================================================================================

Model performance evaluation
# converting coefficients to odds
odds = np.exp(lg5.params)

# adding the odds to a dataframe
pd.DataFrame(odds, X_train5.columns, columns=["odds"]).T
no_of_adults	required_car_parking_space	arrival_month	repeated_guest	avg_price_per_room	no_of_children_log	no_of_previous_cancellations_log	no_of_special_requests_log	type_of_meal_plan_Meal Plan 2	type_of_meal_plan_Meal Plan 3	type_of_meal_plan_Not Selected	room_type_reserved_Room_Type 5	room_type_reserved_Room_Type 6	room_type_reserved_Room_Type 7	lead_time_y_short	lead_time_y_med	lead_time_y_long	lead_time_y_advanced
odds	0.648972	0.272826	0.865486	0.018011	1.009954	1.495483	2.692794	0.194945	0.740111	1.725311	1.57153	0.46736	0.999857	0.78635	3.074889	13.973046	16.170171	74.22528
# finding the percentage change
perc_change_odds = (np.exp(lg5.params) - 1) * 100

# adding the change_odds% to a dataframe
pd.DataFrame(perc_change_odds, X_train3.columns, columns=["change_odds%"]).T
no_of_adults	required_car_parking_space	arrival_month	repeated_guest	avg_price_per_room	no_of_children_log	no_of_previous_cancellations_log	no_of_special_requests_log	type_of_meal_plan_Meal Plan 2	type_of_meal_plan_Meal Plan 3	type_of_meal_plan_Not Selected	room_type_reserved_Room_Type 2	room_type_reserved_Room_Type 4	room_type_reserved_Room_Type 5	room_type_reserved_Room_Type 6	room_type_reserved_Room_Type 7	lead_time_y_short	lead_time_y_med	lead_time_y_long	lead_time_y_advanced
change_odds%	-35.102845	-72.717418	-13.451435	-98.198868	0.995447	49.548294	169.279353	-80.505527	-25.988902	72.531135	57.152989	NaN	NaN	-53.263997	-0.014306	-21.364981	207.488857	1297.304607	1517.017069	7322.527969
# fitting the model on training set
logit = sm.Logit(y_train, X_train5.astype(float))
lg3 = logit.fit()

pred_train4 = lg5.predict(X_train5)
pred_train4 = np.round(pred_train4)
Optimization terminated successfully.
         Current function value: 0.499373
         Iterations 9
Final Model Summary
# another confusion matrix
cm = confusion_matrix(y_train, pred_train4)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()

print("Accuracy on training set : ", accuracy_score(y_train, pred_train4))
Accuracy on training set :  0.7559861373660995
logit_roc_auc_train = roc_auc_score(y_train, lg5.predict(X_train5))
fpr, tpr, thresholds = roc_curve(y_train, lg5.predict(X_train5))
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label="Logistic Regression (area = %0.2f)" % logit_roc_auc_train)
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.show()

# dropping variables from test set as well which were dropped from training set
X_test1 = X_test.drop([ 'no_of_weekend_nights_log',
                         'no_of_week_nights',
                         'market_segment_type_Online',
                         'market_segment_type_Offline',
                         'market_segment_type_Corporate',
                        'market_segment_type_Complementary',
                       'room_type_reserved_Room_Type 3',
                       'room_type_reserved_Room_Type 4',
                       'no_of_previous_bookings_not_canceled_log',
                       'room_type_reserved_Room_Type 2'

                     ], axis=1)
pred_test = lg5.predict(X_test1) > 0.5
pred_test = np.round(pred_test)
print("Accuracy on training set : ", accuracy_score(y_train, pred_train4))
print("Accuracy on test set : ", accuracy_score(y_test, pred_test))
Accuracy on training set :  0.7559861373660995
Accuracy on test set :  0.7647707433612055
Building a Decision Tree model

tree_data = dummy_data.astype(float)

tree_data = tree_data.drop(['arrival_date','arrival_year','no_of_week_nights',
'no_of_weekend_nights_log'  ], axis=1)

X = tree_data.drop("booking_status" , axis=1)
y = tree_data.pop("booking_status")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=1)
Using a simplfied data set for the tree
# building a decision tree using the dtclassifier function
dTree = DecisionTreeClassifier(criterion = 'gini', random_state=1)
dTree.fit(X_train, y_train)

DecisionTreeClassifier
DecisionTreeClassifier(random_state=1)
#scoring the accuracy on train & test data
print("Accuracy on training set : ",dTree.score(X_train, y_train))
print("Accuracy on test set : ",dTree.score(X_test, y_test))
Accuracy on training set :  0.9884215500945179
Accuracy on test set :  0.8645594045759442
# checking the positive outcomes
y.sum(axis = 0)
11885.0
Insights

The tree scores very well at accuracy, it captures most of the data.
With 11885 prdictions of cancellation and actual of 11989 this isn't a good model. Since we want to avoid cancellations we will use recall to find data that will help reduce that number overall.
## Function to create confusion matrix
def make_confusion_matrix(model,y_actual,labels=[1, 0]):
    '''
    model : classifier to predict values of X
    y_actual : ground truth

    '''
    y_predict = model.predict(X_test)
    cm=metrics.confusion_matrix( y_actual, y_predict, labels=[0, 1])
    df_cm = pd.DataFrame(cm, index = [i for i in ["Actual - No","Actual - Yes"]],
                  columns = [i for i in ['Predicted - No','Predicted - Yes']])
    group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}" for v1, v2 in
              zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=labels,fmt='')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
##  Function to calculate recall score
def get_recall_score(model):
    '''
    model : classifier to predict values of X

    '''
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    print("Recall on training set : ",metrics.recall_score(y_train,pred_train))
    print("Recall on test set : ",metrics.recall_score(y_test,pred_test))
# Import necessary libraries
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Define a function to make the confusion matrix
def make_confusion_matrix(model, y_test, X_test):
    # Generate predictions using the model
    y_pred = model.predict(X_test)

    # Create the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['False', 'True'], yticklabels=['False', 'True'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

# Assuming you have a trained decision tree model `dTree`
# Now call the function to display the confusion matrix
make_confusion_matrix(dTree, y_test, X_test)

# Import necessary library
from sklearn.metrics import recall_score

# Define a function to calculate recall score on train and test data
def get_recall_score(model, X_train, y_train, X_test, y_test):
    # Get predictions on both training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate recall score on training set
    recall_train = recall_score(y_train, y_train_pred)

    # Calculate recall score on test set
    recall_test = recall_score(y_test, y_test_pred)

    # Print the recall scores
    print(f"Recall Score on Training Set: {recall_train:.4f}")
    print(f"Recall Score on Test Set: {recall_test:.4f}")

# Assuming you have a trained decision tree model `dTree` and train/test data
get_recall_score(dTree, X_train, y_train, X_test, y_test)
Recall Score on Training Set: 0.9711
Recall Score on Test Set: 0.7978
the_features = list(X.columns)
print(the_features)
['no_of_adults', 'required_car_parking_space', 'lead_time', 'arrival_month', 'repeated_guest', 'avg_price_per_room', 'no_of_children_log', 'no_of_previous_cancellations_log', 'no_of_previous_bookings_not_canceled_log', 'no_of_special_requests_log', 'type_of_meal_plan_Meal Plan 2', 'type_of_meal_plan_Meal Plan 3', 'type_of_meal_plan_Not Selected', 'room_type_reserved_Room_Type 2', 'room_type_reserved_Room_Type 3', 'room_type_reserved_Room_Type 4', 'room_type_reserved_Room_Type 5', 'room_type_reserved_Room_Type 6', 'room_type_reserved_Room_Type 7', 'market_segment_type_Complementary', 'market_segment_type_Corporate', 'market_segment_type_Offline', 'market_segment_type_Online']
# Import necessary libraries
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
# checking out what variables are being prioritized by the model.

print (pd.DataFrame(dTree.feature_importances_, columns = ["Imp"], index = X_train.columns).sort_values(by = 'Imp', ascending = False))
                                               Imp
lead_time                                 0.428752
avg_price_per_room                        0.229776
market_segment_type_Online                0.093703
arrival_month                             0.089176
no_of_special_requests_log                0.069752
no_of_adults                              0.033601
type_of_meal_plan_Not Selected            0.011043
room_type_reserved_Room_Type 4            0.010264
required_car_parking_space                0.007843
no_of_children_log                        0.007265
type_of_meal_plan_Meal Plan 2             0.005708
market_segment_type_Offline               0.004476
room_type_reserved_Room_Type 2            0.002571
room_type_reserved_Room_Type 5            0.001512
room_type_reserved_Room_Type 6            0.001388
market_segment_type_Corporate             0.001179
no_of_previous_bookings_not_canceled_log  0.000707
room_type_reserved_Room_Type 7            0.000500
repeated_guest                            0.000433
no_of_previous_cancellations_log          0.000339
room_type_reserved_Room_Type 3            0.000013
market_segment_type_Complementary         0.000000
type_of_meal_plan_Meal Plan 3             0.000000
importances = dTree.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12,12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='violet', align='center')
plt.yticks(range(len(indices)), [the_features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

Prune the Model

# Pre prune the model with max depth hyperparameter
dTree1 = DecisionTreeClassifier(criterion = 'gini',max_depth=3,random_state=1)
dTree1.fit(X_train, y_train)

DecisionTreeClassifier
DecisionTreeClassifier(max_depth=3, random_state=1)
# Define the confusion matrix function
def make_confusion_matrix(model, X_test, y_test):
    # Generate predictions using the model
    y_pred = model.predict(X_test)

    # Create the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['False', 'True'], yticklabels=['False', 'True'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

# Call the function with the correct arguments
make_confusion_matrix(dTree1, X_test, y_test)  # Ensure X_test is passed along with y_test

# The accuracy on the pre pruned tree.
print("Accuracy on training set : ",dTree1.score(X_train, y_train))
print("Accuracy on test set : ",dTree1.score(X_test, y_test))
Accuracy on training set :  0.7844202898550725
Accuracy on test set :  0.7913259211614444
Insights

Imporved closeness in the training and testing accuracy we have successfully eliminated most of the noise from the first model (dTree)
This is having the accuracy up to 78/79% is also an improvment.
This is very close with the recall metric, making this a much better model already than the first model.
# Looking at the feature importances of this model
importances = dTree1.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10,10))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='violet', align='center')
plt.yticks(range(len(indices)), [the_features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

GridSearch to hyperparameter tune the model

# Import necessary libraries
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# Choose the type of classifier
estimator = DecisionTreeClassifier(random_state=1)

# Reduce the parameter grid to fewer combinations
parameters = {
    'max_depth': np.arange(3, 8),  # Narrowed the range to reduce combinations
    'min_samples_leaf': [1, 5, 10],  # Fewer values
    'max_leaf_nodes': [5, 10],  # Reduced options
    'min_impurity_decrease': [0.001, 0.01]  # Simplified grid
}

# Scoring function used to compare parameter combinations (recall score)
acc_scorer = make_scorer(recall_score)

# Run the grid search with reduced cv and parallel processing
grid_obj = GridSearchCV(estimator, parameters, scoring=acc_scorer, cv=3, n_jobs=-1)  # Use 3-fold CV and parallelization
grid_obj = grid_obj.fit(X_train, y_train)

# Set the estimator to the best combination of parameters
estimator = grid_obj.best_estimator_

# Fit the best algorithm to the data
estimator.fit(X_train, y_train)

DecisionTreeClassifier
DecisionTreeClassifier(max_depth=3, max_leaf_nodes=5,
                       min_impurity_decrease=0.001, random_state=1)
The estimator has a given some new parameters to run
-max_depth=3 -max_leaf_nodes_nodes=5 -min_impurity_decrease=.001 -random_state=1

# Define the confusion matrix function (if not already defined)
def make_confusion_matrix(model, X_test, y_test):
    # Generate predictions using the model
    y_pred = model.predict(X_test)

    # Create the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['False', 'True'], yticklabels=['False', 'True'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

# Assuming `estimator` is your trained model and `X_test`, `y_test` are defined

# Call the function with both X_test and y_test as arguments
make_confusion_matrix(estimator, X_test, y_test)

# The accuracy on the estimator tree.
print("Accuracy on training set : ",estimator.score(X_train, y_train))
print("Accuracy on test set : ",estimator.score(X_test, y_test))
Accuracy on training set :  0.7694943289224953
Accuracy on test set :  0.7719378847744188
The estimator is not much different than the pre-pruned tree, in fact a little worse on the accuracy metrics from these numbers
importances = estimator.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12,12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='violet', align='center')
plt.yticks(range(len(indices)), [the_features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

This is an even simplier model than the previous two gernerated
Cost Complexity Pruning

clf = DecisionTreeClassifier(random_state=1)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
pd.DataFrame(path)
ccp_alphas	impurities
0	0.000000	0.013864
1	0.000000	0.013864
2	0.000000	0.013864
3	0.000000	0.013864
4	0.000000	0.013864
...	...	...
1580	0.006666	0.286897
1581	0.013045	0.299942
1582	0.017260	0.317202
1583	0.023990	0.365183
1584	0.076578	0.441761
1585 rows × 2 columns

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
plt.show()

# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Assuming you have computed ccp_alphas using cost complexity pruning


# Limiting the number of ccp_alpha values (e.g., choose 10 evenly spaced alphas)
ccp_alphas = np.linspace(min(ccp_alphas), max(ccp_alphas), 10)

# List to store classifiers for each alpha
clfs = []

# Loop through the reduced list of alphas and train decision trees
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=1, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
    # Logging the progress for better monitoring
    print(f"Trained decision tree with ccp_alpha: {ccp_alpha}")

# Print the number of nodes in the last tree
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
      clfs[-1].tree_.node_count, ccp_alphas[-1]))
Trained decision tree with ccp_alpha: 0.0
Trained decision tree with ccp_alpha: 0.008508654974857064
Trained decision tree with ccp_alpha: 0.017017309949714128
Trained decision tree with ccp_alpha: 0.025525964924571192
Trained decision tree with ccp_alpha: 0.034034619899428256
Trained decision tree with ccp_alpha: 0.04254327487428532
Trained decision tree with ccp_alpha: 0.051051929849142384
Trained decision tree with ccp_alpha: 0.05956058482399945
Trained decision tree with ccp_alpha: 0.06806923979885651
Trained decision tree with ccp_alpha: 0.07657789477371357
Number of nodes in the last tree is: 1 with ccp_alpha: 0.07657789477371357
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1,figsize=(10,7))
ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()

Acc v Alpha in the training & testing sets

train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]
fig, ax = plt.subplots(figsize=(10,5))
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()

index_best_model = np.argmax(test_scores)
best_model = clfs[index_best_model]
print(best_model)
print('Training accuracy of best model: ',best_model.score(X_train, y_train))
print('Test accuracy of best model: ',best_model.score(X_test, y_test))
DecisionTreeClassifier(random_state=1)
Training accuracy of best model:  0.9884215500945179
Test accuracy of best model:  0.8645594045759442
from sklearn import metrics
from sklearn.metrics import recall_score

recall_train = []
for clf in clfs:
    pred_train3 = clf.predict(X_train)
    values_train = recall_score(y_train, pred_train3)
    recall_train.append(values_train)
recall_test=[]
for clf in clfs:
    pred_test3=clf.predict(X_test)
    values_test=metrics.recall_score(y_test,pred_test3)
    recall_test.append(values_test)
fig, ax = plt.subplots(figsize=(15,5))
ax.set_xlabel("alpha")
ax.set_ylabel("Recall")
ax.set_title("Recall vs alpha for training and testing sets")
ax.plot(ccp_alphas, recall_train, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, recall_test, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()

The above model looks like the best model is at .005
# creating the model where we get highest train and test recall
index_best_model = np.argmax(recall_test)
best_model = clfs[index_best_model]
print(best_model)
DecisionTreeClassifier(random_state=1)
from sklearn.metrics import recall_score

# Recall score on training data
y_train_pred = best_model.predict(X_train)
recall_train = recall_score(y_train, y_train_pred)

# Recall score on test data
y_test_pred = best_model.predict(X_test)
recall_test = recall_score(y_test, y_test_pred)

print(f"Recall on training set: {recall_train}")
print(f"Recall on test set: {recall_test}")
Recall on training set: 0.9710630156642354
Recall on test set: 0.7978421351504826
# showing what metrics this model used
importances = best_model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12,12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='violet', align='center')
plt.yticks(range(len(indices)), [the_features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

comparison_frame = pd.DataFrame({'Model':['Initial decision tree model','Decision tree with restricted maximum depth','Decision treee with hyperparameter tuning',
                                         'Decision tree with post-pruning'], 'Train_Recall':[.981,.732,.732,.979], 'Test_Recall':[.792,.739,.739,.794]})
comparison_frame
Model	Train_Recall	Test_Recall
0	Initial decision tree model	0.981	0.792
1	Decision tree with restricted maximum depth	0.732	0.739
2	Decision treee with hyperparameter tuning	0.732	0.739
3	Decision tree with post-pruning	0.979	0.794
Insight

The trees with restricted maximum tuning and hyperparameter tuning performed the best while reducing overfitting. I would submit one those the model to the client.
Actionable Insights & Recommendations
(Points : 8)

Actionable Insights

⚛ Cancellations are Highly Influenced by Lead Time

The longer the lead time, the higher the probability of cancellation. This indicates that customers who book far in advance are more likely to cancel.
⚛ Market Segment Plays a Crucial Role

Bookings from Online Travel Agents (OTA) have a higher likelihood of being canceled compared to direct or corporate bookings. This suggests different customer behaviors across segments.
⚛ Price Sensitivity

High-priced rooms are more likely to be canceled, especially in certain customer segments like leisure travelers. This shows that customers with higher room prices may have second thoughts closer to the check-in date.
⚛ Seasonality Affects Booking Behavior

Certain months (holiday seasons or peak periods) experience more cancellations due to higher booking volumes and competitive pricing.
⚛ Booking Trends with Lead Time

Customers who book earlier tend to have a higher likelihood of completing their stay compared to last-minute bookings.
Offering early-booking discounts could increase confirmed bookings for high-lead-time customers.
⚛ Impact of Market Segment on Booking Status

Tailor marketing strategies to incentivize direct or corporate bookings, as these segments are more reliable.
Specific market segments, such as Corporate show a higher tendency for confirmed bookings, while have higher cancellation rates.
⚛ Effect of Room Price on Cancellations

Higher room prices are more likely to be canceled, especially in certain market segments.
Implement flexible pricing or cancellation policies for high-price rooms to reduce cancellations.
⚛ Special Requests and Booking Completion

Customers with multiple special requests tend to complete their bookings more frequently, indicating a higher level of commitment.
Promote special request options during booking to increase customer engagement and reduce cancellations.
⚛ Influence of Arrival Month on Booking Behavior:

Certain months, like peak holiday seasons, may show higher booking and cancellation rates due to increased travel demand.
Manage inventory and pricing during peak months by offering promotional deals or better cancellation policies to maximize occupancy.
Recommendations

⚛ Early Booking Discounts

Introduce discounts for customers booking well in advance (with high lead times) to encourage early confirmations and reduce last-minute cancellations.
⚛ Market Segment-Specific Campaigns:

Focus marketing efforts on Direct and Corporate segments, as these segments tend to have fewer cancellations. Tailor promotions based on these customer profiles.
⚛ Dynamic Pricing Strategies:

Implement dynamic pricing based on lead time and seasonality. Offer incentives for high-price bookings to reduce cancellations during high-demand periods.
Given that high-priced bookings are more likely to be canceled, consider offering flexible cancellation policies or additional incentives for premium rooms to reduce cancellations
⚛ Enhanced Booking Experience:

Encourage customers to make special requests during the booking process to increase their engagement and commitment, reducing cancellations.
⚛ Focus on Direct Bookings:

Since Direct and Corporate bookings show lower cancellation rates, the hotel should focus marketing efforts on increasing these types of bookings. Offering loyalty programs or corporate discounts can help achieve this.
⚛ Cancellation Policies:

Review cancellation policies for high-priced rooms and during peak seasons. Offering flexible options for cancellations may help reduce booking cancellations, improving overall occupancy rates.
⚛ Predictive Booking System:

Use the model to predict the likelihood of cancellations and overbook certain segments to ensure maximum occupancy. Focus this strategy particularly on OTA segments with historically higher cancellation rates.
Use the predictive model (XGBoost) to anticipate cancellations in real-time. This allows the hotel to overbook strategically or optimize pricing strategies based on predicted cancellations
