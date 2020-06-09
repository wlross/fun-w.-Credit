# Reverse Engineering a Credit Score Model to Guide Prospective Homeowners
Brian Aoyama, Ryan Bolles, Ece Wyrick, Will Ross*  
In collaboration with Landis Technologies

**Executive Summary**

**Problem**

Landis Technologies provides an alternative path to homeownership through a rent-to-own business model. Most customers that apply to Landis’ business have credit issues affecting their  ability to secure a mortgage loan. They may have a history of default or be at risk to miss rent payments. As an intermediary between applicants and lenders, Landis uses an applicant’s credit history, credit score, and publicly available information from credit agencies to determine if an applicant will be profitable for the company. Predictive models that mimic the algorithm used by credit agencies help Landis increase revenue on a per-customer basis by providing Landis a quantitative approach to help customers improve their credit scores.  This approach is further augmented when a model can be interpreted to help Landis identify what aspects of an individual’s credit profile they should focus on.   

**Findings**
An initial linear regression model in R showed that there is a nonlinear relationship between the information captured in a credit report and the credit score. Utilizing a variety of multivariate and logarithmic regression estimation tools (e.g., LASSO) and machine learning algorithms, we were able to create models that accurately predict an applicant’s credit score within approximately 40 points. Critical to our analysis of different modeling approaches was identifying an approach that had both high accuracy and high interpretability in keeping with Landis’ business objective to serve as a “coach”. 

**Recommendation**
Of all approaches assessed, the model created through a Gradient Boosted XGBoost Regressor (RMSE: 41.57), appeared to provide Landis Technologies with the best balance of accuracy and, thanks to the use of Shapley Values, interpretability.  This was in contrast to other non-linear methods such as a Logarithmic LASSO regression (RMSE: 42.4), Deep Learning (RMSE: 38.96), and ensembling (RMSE: 37.22), all of which provided high accuracy but failed with regards to interpretability.  The superior interpretability of the Gradient Boosted approach was validated by using high Shapley value variables to hand-tune a linear model that still achieved high accuracy.  Using the coefficients for the variables from this Shapley-derived regression, Landis can further refine its coaching strategy for providing customers the most effective path to improving their credit score. 
A full summary of numeric results and approach details can be found on the following page.
Experimental Results Summary

 
**Problem Overview**


**Company**
Landis is a social impact company that helps renters transition to homeownership. Customers complete a short survey on Landis’ website, receive pre-qualification for a loan, and then search for and select a house. Landis purchases the home and rents it to the customer. Within 12 months of the move-in date the customer is expected to purchase the home from Landis. Consistent rent payments are used to help improve the customer’s credit score and build up the required down payment on the home. Landis generates revenue from the rent payments, and from a 3% price increase on the purchase price when the home is sold to the renter. 
 
Landis decides what the customer’s home budget should be based on their ability to: (1) make monthly rent payments, (2) save for a down payment, and (3) get a mortgage in the next 12 months. Landis takes several factors into consideration when reviewing an application. The customer could be declined because of their rent-to-income ratio, debt-to-income ratio or credit profile.

The  typical customer has low savings and struggles with credit issues, and can not figure out a viable path to home ownership. Target home price is around $120-200K, with monthly rental payments of $1000-2000/month as opposed to $700/month. Landis acts as a savings mechanism for the consumer, and provides online and in-person coaching over the course of the process. Landis aims to identify the customers who will be able to get approved for their bank mortgages at the end of the 12 months. 

This business model depends on bank loans to fund the purchase of homes. Banks use a borrower’s credit score to determine the approved loan amount and interest rate. Three main firms collect credit reports and publish credit scores: Experian, Equifax, and TransUnion. Although these firms give consumers general guidance to improve credit scores, there is little public knowledge surrounding the model used to derive credit scores from credit reports. 
 
**How do credit reports work?**

Based on the high-level guidance provided from credit score companies, FICO Scores are calculated from credit data based on five categories: payment history (35%), amounts owed (30%), length of credit history (15%), new credit (10%) and credit mix (10%).
 
While FICO provides the guideline breakdown above, the importance of each category actually varies from person to person. For instance, the weights for someone with a long credit history versus someone with a more recent credit history would be calculated differently. Since credit reports get updated very frequently, it is difficult to isolate the effect of one changing variable on the credit score. 
Broadly speaking, credit scores depend on:
●	Payment history - The most significant factor determining credit score. On time payments boost the individual’s score, while missed payments hurt the score. 
●	Accounts owed - Individual’s debt in relation to his available credit. The more the available credit is utilized, the lower are the credit scores, since the credit bureaus worry that the individual may be borrowing beyond his means. 
●	Length of credit history - The longer the credit history, the higher the credit score. Credit bureaus typically look at average length of credit history, which takes into account the ages of both older and newer accounts. 
●	Credit mix - Having a mix of different types of credit typically increases credit score. Different types of credit are credit cards, retail accounts, installment loans, finance company accounts and mortgage loans.
●	New credit - Credit bureaus typically view an individual to be of greater credit risk if they take out several credit accounts in a short period of time. 
At a high level, there are four categories of information included in a credit report:
●	Personally Identifiable Information (PII) - Name, address, SSN, date of birth, employment information. PII does not factor into the credit score. 
●	Credit Accounts - This information comes from the individual’s lenders. Information includes type of account (credit card, auto loan, mortgage, etc.), date opened, credit limit or loan amount, the account balance and payment history. This information is the basis of the credit score.
●	Credit Inquiries - When the individual applies for a loan, the inquiry appears here, regardless of whether the loan was approved or not. Hard inquiries happen when the individual actually applies for a loan. Soft inquiries happen when lenders pre-approve the person for lending offers. Only hard inquiries on the report are available to be viewed by the lenders. The higher the hard inquiries, the lower the credit score. 
●	Public Record and Collections - Overdue debt sent to collections appears on the credit report. The information comes from public records. 
It is important to note that the lenders take into account factors beyond the FICO score in making lending decisions. Those factors may include income, job history, type of credit etc. 
Objective
Landis has accumulated over 2,000 credit reports and corresponding credit scores, and wants a model that can predict a borrower’s credit score based on their credit data. Accurately predicting credit scores will help Landis pre-qualify customers for an appropriate loan amount and interest rate, and help Landis coach customers on how they can improve their credit scores. Our core objective is to build a predictive model that minimizes prediction error.
Using the data collected from each credit report, the associate credit score, and public knowledge published by the credit agencies, we determined that the relationship between the variables and credit score was non-linear. In order to build a model to account for the non-linearity we built a data set that included the numerical values of all the variables in the original data set as well as the natural logarithm of every numerical value. This data was used in a multivariate regression, the Least Absolute Shrinkage and Selector Operator (LASSO), and a machine learning algorithm, XGBoost, to build a predictive model that minimized the predicted error of the customer’s credit score. In addition, while predictive accuracy is of top importance, explainability is also important to Landis’s mission - since their goal is to coach customers to increase their scores. While we did not necessarily need to use more advanced and sophisticated techniques to achieve a low RMSE, we did utilize these techniques to achieve interpretability in the context of the business’s mission.

**Data and Sources**

When a potential customer applies to Landis, he or she gives the company permission to access his or her credit report and credit score. Landis provided our team with a dataset containing 2,231 observations. Each observation contains credit report information for an individual as well as that individual's credit score.

Y Variable
The left-hand side (Y) variable for each observation is the consumer credit score associated with each credit report. 
Dependent Variable	Mean	Median	SD	   Min	 Max
Credit Score	     577.2	 568	 69.680	 405	 843

X Variables
A credit report contains 5-10 pages of data on an individual's credit history. Generally speaking, two types of credit report data are used to generate credit scores. 

Trade Line Data: These account for the bulk of a consumer's credit report and consist of information about every loan the consumer has taken out. For each trade line, the credit report documents loan type, loan amount, date opened, current balance, and payment history.

Derogatory Items: These are long-lasting negative marks on a consumer's credit report, such as late payments, bankruptcies, and referrals to collections agencies. Derogatory items typically harm a consumer's credit score.
The data included in each credit report varies by applicant, depending on the number and mix of credit lines per consumer and the consumer’s behavior. For example, a customer with 10 credit cards, a mortgage, an education loan, and several automotive loans will have more data points associated with their credit report than an individual with 3 credit cards and no other loans.

To standardize the number of right-hand side (X) variables per observation, Landis provided summary statistics for each observation. The initial dataset consisted of 59 X variables for each observation.

**Model and Regression Analysis**

**Data Cleansing**
To prepare the dataset for analysis, we first needed to convert the three date fields in the dataset from text into R date format. We used the ‘anytime’ and ‘lubridate’ packages to create a new variable called ‘DaysOpen’, which calculates the number of days between the most recent credit account opening and the date of the credit report for each customer. 
New Variable Creation
Based on the information FICO publishes about how credit scores are calculated, we created the following new variables in an attempt to create a prediction model that minimizes RMSE:
Days Open: FICO reports that Length of Credit History accounts for 15% of an individual’s credit score. We calculated the DaysOpen variable to measure the number of days between the most recently opened credit account and the date of the credit report. 

Revolving Credit Utilization: FICO shares that credit utilization - especially on revolving credit accounts -  is a significant consideration when calculating consumer credit scores accounts. When consumers spend a high percentage of their available credit, it means they’re close to maxing out their credit cards, which can harm their credit score. To calculate this variable, we divided Balance (revolving accounts) by High Credit (revolving accounts). 

Percent of Credit Accounts in Good Standing: We hypothesized that consumers with a high percentage of their credit accounts in ‘Satisfactory’ standing would have a higher credit score. Therefore, we calculated this variable by dividing the Count of Satisfactory Accounts by the Count of Total Accounts for each observation.

Secured Debt % of Total Debt: Since FICO shares that credit mix figures into credit score calculations, we created a variable that divides total secured debt balance by total outstanding debt balance. 

Revolving Credit to Total Liability High Credit Ratio: A different attempt to calculate approximate credit utilization. Here, we calculated total High Credit (revolving accounts) as a percentage of Total Liability High Credit (all accounts).
Total Debt to Total High Credit Ratio: We calculated this ratio as another imperfect proxy for credit utilization, this time dividing total Outstanding Balance (all accounts) by total High Credit (all accounts).

**Modeling Approach**

Our team’s primary objective (Objective 1) was to build a model that predicted consumer credit scores with minimal error as measured by RMSE. Our team iterated on 9 distinct approaches to build the best prediction model. To assess the predictive accuracy of each model, we split the dataset into a training set consisting of a random sample of 70% of the observations, and a test set, consisting of 30% of observations. We tested each model against the test dataset and used Root Mean Squared Error (RMSE) to measure the out-of-sample prediction error of the model. 

Because Landis’ business involves coaching consumers about how to improve their credit scores before taking out a mortgage, our secondary objective (Objective 2) was to help Landis understand how they could interpret and use the credit score prediction model to coach their customers.  While this secondary objective was, in some ways, inherently in contrast with the goal of maximizing predictive accuracy, predictive power without interpretability leaves Landis without the ability to take targeted action with their customers, a critical aspect of the rent-to-own business model.  

We benchmarked our primary objective of accuracy by targeting a predictive model that could achieve out-of-sample RMSE <45. We benchmarked our secondary objective of high interpretability by assessing our ability to derive 8-12 key features that provide a linear model with RMSE <50 from a performant (RMSE <45) non-linear model.

Metric	Note	Target
RMSE	Measures prediction error; proxy for prediction accuracy of model	< 45
RMSE as % of SD
of ground-truth credit scores	Represents improvement over naive model	< 60%
RMSE as % of Range 
of ground-truth credit scores 	Represents relative scale of potential error, given credit scores in test sample	< 10%

**Pre-Analysis**

We built Reg1 to conduct exploratory analysis. We ran a multivariate linear regression using the entire dataset (i.e., not using the train and test samples used for prediction models). 
The plot of the residuals from this model showed that the mean error is not zero. The shape of the mean line, specifically the “V” shape, shows that the relationship among all dependent variables and the independent variable is non-linear. This understanding carried throughout our analysis and led us to utilize the natural logarithm of our variables, along with machine learning algorithms, in order to fit the variables to a multivariate regression. We did not utilize the natural logarithm of the dependent variable because interpreting a percent change of a customer’s credit score would not be valuable information for Landis to predict credit score.
 
We also plotted the correlation for all of the variables in the data set. This showed high correlation between many of the variables. We expected this result because some of the data points were a summation from other variables. Additionally, the high correlation between many variables demonstrated that we would need to  utilize software tools to help our model determine the best predictive variables. 
 
**Approach 1: Naive Model**

We began our analysis by creating a base ‘Naive’ model that simply calculated the average credit score for the training dataset (576.9) and used that single value to predict all credit scores in the test dataset.  This model serves as a benchmark, and all subsequent models should improve upon this. The RMSE of this model is 66.04 - approximately 100% the standard deviation of the test set. This is because the mean of our training data is expected to be approximately equal to the mean of our test set. 
Metric	Score	Target
RMSE	66.04	< 45
RMSE as % of SD
of ground-truth credit scores	100%	< 60%
RMSE as % of Range 
of ground-truth credit scores 	17%	< 10%
 
**Approach 2: Multivariate Linear Regression (‘Kitchen Sink’)**

As an additional baseline and in order to assess the quality of the data, we then created a linear regression that used all Y variables (Kitchen Sink) without any non-linear transformation, but including the 6 new variables we created (described above). 
Metric	Score	Target
RMSE	56.68	< 45
RMSE as % of SD
of ground-truth credit scores	86%	< 60%
RMSE as % of Range 
of ground-truth credit scores 	14%	< 10%

This approach yielded a RMSE of 56.68, a slight (14%) improvement from our baseline model. However, since this is a multivariate linear regression model and we know both that this is a nonlinear relationship and that many variables exhibit high correlations, we did not expect this model to be a useful predictive tool. Instead, this model serves as a more refined baseline for future models that should solve for these baseline issues using tools such as LASSO and machine learning techniques to account for high correlation, nonlinearity, and pre-existing knowledge that we are trying to reverse engineer a rules-based system. 

**Approach 3: LASSO (linear)**

In order to help prioritize among 62 X variables, we used LASSO to identify the number of X variables that results in the most performant model; i.e., the model with the lowest RMSE. One concern in using the LASSO approach is that it prioritizes variables with large coefficients, but our kitchen sink model indicated that many of the variables with large coefficients were the product of double-counting. Because LASSO’s approach will not select a series of small coefficients while eliminating a large one, we predicted that we would need to hand-tune this model. Likewise, the plot of residuals for Regression 1 indicated non-linearity in the model, so we needed to address this nonlinearity to further improve the model’s predictive accuracy. 
Utilizing LASSO without using the natural logarithm of any variable created a model with a RMSE of 55.70. Interestingly, this is actually worse than our multivariate (‘Kitchen Sink’) linear regression. While this result was more or less expected, it was somewhat surprising as many of the 45 variables this model identified -- for example, the so-called derogatory variables -- appeared to be among those with the largest coefficients, which is what one would expect based on information shared by the credit rating agencies about their processes. 
Metric	Score	Target
RMSE	55.70	< 45
RMSE as % of SD
of ground-truth credit scores	84%	< 60%
RMSE as % of Range 
of ground-truth credit scores 	14%	< 10%
 
**Approach 4: LASSO (with log transformation)**

With 62 X variables and the ability to combine and log any variety of these variables to assess non-linear impacts, we needed to identify methods that would search for performant non-linear models in an automated fashion.  As a result, we elected to Log all of our X variables (designated as name.y) and then run a LASSO that evaluated superset of 122 variables.  While this introduced the risk of further double-counting, we recognized this ultimately just meant a more complex non-linear model -- e.g. b1*variable.x + b2*log(variable.y) can really be thought of as a single non-linear transformation.  
This model ended up using 66% of all available variables, 81 of the 122, and as expected included the original value and the natural logarithm of many of the same variables in the model.  While we were prepared to interpret the more complex non-linear coefficients, the quantity of variables selected by LASSO made this approach difficult to understand, weakening its strength on Landis’ second objective. The predictive performance of the model, however, was very impressive with an RMSE of 42.4, a 36% improvement from our naïve model and 64% of the SD of the test set. With RMSE representing 11% of the range of the model, these results were quickly approaching our target metrics for accuracy. 
Metric	Score	Target
RMSE	42.40	< 45
RMSE as % of SD
of ground-truth credit scores	64%	< 60%
RMSE as % of Range 
of ground-truth credit scores 	11%	< 10%

**Approach 5**

In light of the outputs of both our LASSO approaches and an impressive RMSE of 42.4 for the LASSO approach that incorporated Log-values (Approach 4), we elected to establish a benchmark for our secondary objective of interpretability.  Because we know Landis can only coach its clients on a subset of variables, the 81 coefficients provided by this approach were not in and of themselves particularly helpful.  The team set out to use these results to try and hand-tune an ~8-12 variable model that would still create an RMSE below 50. 

Using our knowledge of the credit scoring process as well as large coefficient variables from the Lasso Log approach, we constructed an eleven coefficient model using the variables SatisfactoryPct, revcreditutil, AutoCount, DerogOtherCount, LiabilityBankruptcyCount, LiabilityCurrentAdverseCount, PublicRecordCount, DisputeCount, log(MortgageCount), log(Day30), and log(InquiryCount). Despite the fact these variables produced a model with a significantly improved residual plot (see below), the high RMSE of 64.39 was only an incremental improvement to the Naive model at 97% of the test set SD.  

Metric	Score	Target
RMSE	64.39	< 45
RMSE as % of SD
of ground-truth credit scores	97%	< 60%
RMSE as % of Range 
of ground-truth credit scores 	16%	< 10%

Due to these results, we did not believe that the Log Lasso Model (Approach 4), which itself is difficult to interpret, could point us to a model that did have high interpretability.  We decided to research non-linear techniques in the burgeoning field of machine learning to see if we could learn a model that generated both high accuracy (measured as RMSE <45) and high interpretability (measured as ability to derive 10 key features that provide a linear model with RMSE <50 from an accurate non-linear model).

**Approach 6**

In researching the field of machine learning interpretability and explainability, conversations with students at Stanford’s Institute for Computational and Mathematical Engineer, professors at Columbia University’s Department of Statistics, and researchers at Microsoft Research, led us to two key techniques -- gradient boosting trees and shapley values -- that showed promise in creating both highly accurate and highly interpretable models for predictive regression models with inputs with 10-100 features. 

Using the R library for XGBoost, we implemented a gradient boosted model with a tree depth of 12, 22 rounds of boosting, and a learning rate of 0.3 (parameters optimized using grid search).  Following identical procedures for train/test split and accuracy measurement, our XGBoost model demonstrated RMSE of 41.57.  This result was incrementally better than our Log Lasso Model (Approach 4, RMSE 42.40), representing ~10% of the range of credit scores and ~63% of the standard deviation of scores in the test set.  With sufficient accuracy for this approach established, we turned to Shapley values to try and assess interpretability.  

Metric	Score	Target
RMSE	41.57	< 45
RMSE as % of SD
of ground-truth credit scores	63%	< 60%
RMSE as % of Range 
of ground-truth credit scores 	12%	< 10%

**Approach 7**

Shapley values attempt to interpret non-linear models by withholding X variables from a model’s training sequence and assessing the resulting change in accuracy.  Coarsely, SHAP posits that the larger the change in accuracy when a variable is withheld, the more important that variable is to the original non-linear model.  Using the R library SHAPforxgboost, we were able to quickly determine the SHAP values for our highly accurate XGBoost model (Approach 6, RMSE 41.57).  

Mimicking our procedure in Approach 5, we used the SHAP output (vs. coefficients in Approach 5) to identify an ~8-12 coefficient linear model that might provide more-than-baseline accuracy (RMSE <60).  We established a ten coefficient model using the variables revcreditutil, SatisfactoryPct, TotalLiabilityPastDue, OpenPastDue, InquiryCount, InstallmentPastDue, EducationPastDue, and LiabilityBankruptcyCount.  While the residuals of this manufactured model were indicative of a poor overall fit, the model did produce an RMSE of 56.33 or about 12% of the range, and 74% of the standard deviation. 

Metric	Score	Target
RMSE	49.02	< 45
RMSE as % of SD
of ground-truth credit scores	74%	< 60%
RMSE as % of Range 
of ground-truth credit scores 	12%	< 10%

Shapley Values of Incorporated Variables (Top 10)
revcreditutil    SatisfactoryPct     TotalLiabilityPastDue      OpenPastDue    LiabilityCurrentAdverseCount
22.71648759    15.75986259                9.28344348                  6.80315965                      4.92675703
InquiryCount   InstallmentPastDue   EducationPastDue    LiabilityBankruptcyCount    RevolvingPastDue    
  6.54156411           6.52060807                6.36318140                    5.60813197                        5.13545806

While Shapley values, and really all methods for interpretability of non-linear models are imperfect, Landis’ mandate is two-fold.  First, identify high accuracy (low RMSE) predictive models that give insight into the cloaked practices of credit scoring agencies.  Second, identify the features/variables of that model that can be “coached” to help improve a score.  Not only does Gradient Boosting provide a high accuracy model, SHAP provides a seemingly helpful method for interpreting that model.  Moreover, this particular SHAP explanation seems to be robust.  First, the variables it identifies are intuitive -- e.g.,  it makes sense that derived variables such as revcreditutil and SatisfactoryPct, both of which are publicly highlighted by scoring agencies, would be among the most influential to overall credit score.  Second, linear models that use these features seem to be a significant improvement on a Naive model in accurately predicting credit score movement.

While the reality of the credit scoring industry is that it makes use of complex, rule-based systems that are purposefully cloaked, we’re optimistic that this gradient boosting technique, combined with its explanatory Shapley can be helpful to Landis in reaching both of their objectives.   

**Approaches 8 and 9**

While Gradient Boosting with Shapley values provided both high predictive accuracy and high interpretability, we felt it was important to be exhaustive in looking for more accurate models, even if there was a risk of low interpretability.  As a result, we elected to use R wrappers for Keras (‘keras’ and ‘kerasR’) and TensorFlow (‘tensorflow) to attempt to beat our results using Deep Learning.  Deep Learning has provided significant advancements in machine learning performance measures over the last decade.  While Deep Learning’s ‘sweetspot’ tends to be higher dimensionality tasks, we implemented a four layer neural net (256 sigmoid, drop .4, 128 softmax, drop .3, 64 relu, drop .2, 64 relu) to predict a single output node representing the continuous value of credit score.  This model architecture, optimized over 130 epochs, ultimately yielded an RMSE of 38.96 or 10% of the range and 59% of the SD.  This was a meaningful but not overwhelming improvement over XGBoost/Approach 6 (RMSE 41.57) and our Log Lasso/Approach 4 (RMSE 42.40). We ultimately concluded that given the difficulty in explaining Deep Learning models, that the combination of Gradient Boosting and Shapley Values was still superior in the context of Landis’ use case.

Metric	Score	Target
RMSE	38.96	< 45
RMSE as % of SD
of ground-truth credit scores	59%	< 60%
RMSE as % of Range 
of ground-truth credit scores 	10%	< 10%


Finally, in order to exhaust our options, we assessed a weighted-average ensemble model, weighting our Deep Learning (Approach 8) at 60% and our XGBoost model at 40% for each predicted value.  While this mix (arrived at through coarse experimentation) did ultimately yield an RMSE of 37.22, the strongest result we saw across all experiments, again the improvement in accuracy was not worth the trade off of reduced interpretability.  

Metric	Score	Target
RMSE	37.22	< 45
RMSE as % of SD of ground-truth credit scores	56%	< 60%
RMSE as % of Range of ground-truth credit scores 	9%	< 10%

**Takeaways & Recommendations**

Our primary objective was to construct a model that minimized prediction error in an applicant’s credit score. This is valuable to Landis because the lenders they work with utilize credit scores to determine the value of a loan to award an applicant. Equipped with a model that predicts the results of a lending agency’s credit rating allows Landis to pre-determine the potential revenue generated by an applicant and the risk of default of an applicant. 

Our secondary objective was interpretability.  Landis also aims to coach customers through systematic ways to improve their credit score and help customers reach a major life milestone - home ownership - that they may not otherwise be able to achieve. This objective requires knowledge around the specific variables that are most important when agencies calculate credit scores from credit reports.

The model we created using Gradient Boosted trees with the XGBoost library provides Landis with a very accurate predictive tool.  And through the use of Shapley values we were able to identify important variables surrounding credit score calculations.  When coaching customers in method to improve credit scores, the model indicated that Landis would do well to focus on the variables: revcreditutil, SatisfactoryPct, TotalLiabilityPastDue, OpenPastDue, LiabilityCurrentAdverseCount, InquiryCount, InstallmentPastDue, EducationPastDue, LiabilityBankruptcyCount, RevolvingPastDue. 

While the XGBoost model does not itself provide coefficients that would make coaching credit score improvement easy, we were able to use the Shapley interpretation of this model to come up with a performant linear model (Approach 7).  This model, while slightly less accurate, can be used in conjunction with the XGBoost model to provide coefficients that Landis can use to explicitly demonstrate how recommended changes will positively impact the customer’s credit score. This improvement will help customers with other loan applications and any refinancing opportunity. Ultimately this will also help Landis generate more revenue and maintain higher profits. 
 
Based on our understanding of how FICO calculates credit scores, we believe it is possible to further improve the model’s predictive accuracy by including additional variables using more granular credit report data. Our team was working with a relatively limited set of 60 summary data points, but Landis has valuable data at the trade-line level within each credit report. As a starting point, we recommend incorporating the average age of trade lines (by credit category, and weighted by balance), trade-line level utilization scores, the time since the most recent adverse credit events, credit mix (by credit type, balance remaining), and the % of accounts where payment has never been late. These variables could reduce the predictive error and are also sufficiently concrete to help Landis coach their customers. 

Last but not least, if an approach for interpreting more complex non-linear models such as our most accurate deep learning models or our ensembled approach could be identified, it would help to balance the trade-off between predictive accuracy and interpretability that we’ve grappled with throughout this project.  We encourage the Landis team to explore techniques such as Layerwise Relevance Propagation (LRP) or the application of Shapley Values to deep learning.  

Ultimately it is an acknowledged fact that the credit scoring industry is purposefully opaque in its methods.  And while this will always be a challenge for Landis’ business, we are optimistic further progress can be made. 
