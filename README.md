# MISHMASH-Hackathon---TechCratz
Project developed for the MISHMASH Online Hackathon 

<p>In this competition we are given primarly two datasets namely Test.xlsx and Train.xlsx. The both datasets comprises of 36 attributes. In the Test Data set the time is given as periods, each year have 13 periods and in the train data set 12000 days are given. We are given two objectives:</p>
<ul>
  <li> Find the Drivers that effects the Sales(EQ) </li>
  <li> By identifying the drivers how accuratly can we predict the sales </li>
</ul>

<p> For the primary objective we used the bayasian method: Linear regression and correlation. But the data is not easily identifiable by linear regression lines or scatter plots. So we calculated the co-efficient of correlation and p value. From that we identified 14 attributes that act as a driving forces for the sales (EQ) </p>
<ul>
  <li> Digital Impressions </li>
  <li> Fuel Price </li>
  <li> Inflation </li>
  <li> Trade Invest </li>
  <li> Brand Equity </li>
  <li> Average Sales Price </li>
  <li> Avg_promo_pct_ACV </li>
  <li> Estimated ACV Selling </li>
  <li> pct_ACV </li>
  <li> No of Items sold </li>
  <li> Competitor 2 </li>
  <li> Competitor 3 </li>
  <li> Sales Category </li>
  <li> Sales sub category </li>
</ul>

<p> For the secondary objective we identified the problem as a multivariate time series problem. Then we applied various time series model for identifying trend, seasonality, etc. The models used for identifying these objectives are </p>
<ul>
  <li> Naive Forecasting </li>
  <li> Moving Average </li>
  <li> Exponential Smoothing </li>
  <li> Hot Linear </li>
  <li> Holt Winter </li>
  <li> SARIMAX </li>
</ul>

<b> Each of these objectives code, solutions, snippets,etc. are added in this repository on respective folders </b>
<ol>
  <li> Objective 1 : Identifying Drivers Towards Sales(EQ) </li>
  <li> Various Models : Understanding Time Series Features </li>
  <li> Objective 2 : Multi-step Model </li>
</ol>

<b> <u>  Final predicted file is uploaded as prediction.xlsx and the problem that faced with hurdle 2 given is added as solution_Phase2.png </u> </b>
