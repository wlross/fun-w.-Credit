#SYSTEM PREP
#import and view data 
creditdata = read.csv('2020-03-05-credit-data-stanford.csv',fileEncoding='UTF-8-BOM')
View(creditdata)
summary(creditdata)
#get applicable libraries etc.
library(anytime)
library(lubridate)
require(caTools) 
library(Metrics)
library(corrplot)
library(Matrix)
library(glmnet)
library(xgboost)
library(dplyr)
library(SHAPforxgboost)
library(data.table)
library(here)
library(ggplot2)
library(keras)
library(kerasR)
library(tensorflow)


# Data Preparation
# Create 'DaysOpen' Variable
creditdata$OpenedDateFixed = parse_date_time(creditdata$CreditLiabilityAccountOpenedDate,c('mdy'))
creditdata$OpenedDateFixed = anydate(creditdata$OpenedDateFixed)
creditdata$CreatedDatetimeFixed = parse_date_time(creditdata$CreatedDatetime,c('mdy'))
creditdata$CreatedDatetimeFixed = anydate(creditdata$CreatedDatetimeFixed)
creditdata$DaysOpen = difftime(creditdata$CreatedDatetimeFixed, creditdata$OpenedDateFixed,units="days")

# Clean data to include only columns used in regression; get rid of placeholders / unused dates
View(creditdata)
creditdata$DaysOpen = as.integer(creditdata$DaysOpen)
creditdata = creditdata[-c(59:63)]
creditdata = na.omit(creditdata)
summary(creditdata$DaysOpen)

# Create additional variables used in model
# Debt-to-HighCredit Ratio (proxy for credit utilization; total debt / total liability high credit)
creditdata$TotDebtTotHighCred = creditdata$TotalLiabilityBalance / creditdata$TotalLiabilityHighCredit
# Revolving High Credit to Total Liability High Credit
creditdata$RevHCtoTotHighCred = creditdata$RevolvingHighCredit / creditdata$TotalLiabilityHighCredit

# % of credit accts in satisfactory standing
creditdata$SatisfactoryPct = creditdata$LiabilitySatisfactoryCount / creditdata$TotalLiabilityCount
# Secured Debt as % of total debt
creditdata$SecDebtPctTotDebt = creditdata$TotalSecuredLoanBalance / creditdata$TotalLiabilityBalance

# Revolving Credit utilization (problem: 852 values where RevHC = 0)
#Imperfect, making assumption for 850 obs where no revolving high credit line
creditdata$revcreditutil = ifelse(creditdata$RevolvingHighCredit > 0, creditdata$RevolvingBalance / creditdata$RevolvingHighCredit, creditdata$RevolvingBalance / creditdata$TotalLiabilityHighCredit)
creditdata = na.omit(creditdata)

# Remove na values

# Set Seed for any random variables
set.seed(123)

#Split the data into train and test set
# Note: Using different approach than WR approach (sample.split)
n = nrow(creditdata)
trainind = sample(n,0.7*n)
train = creditdata[trainind,]
test = creditdata[-trainind,]

# Set X Variables for full sample population to subset
xVar = creditdata[,3:64]

# SUMMARY REGRESSION & CORRELATION MATRIX
# Note: Using entire dataset, not train/test (used for prediction, later)
# Run regression o full sample population
reg1 = lm(creditdata$Score ~ ., data = xVar)
summary(reg1)
# Take a look at correlation matrix
M = cor(xVar[sapply(xVar, is.numeric)], use='complete.obs')
write.csv(M, 'M.csv')
# corrplot doesn't work well beause it's difficult to draw a pie chart matrix on a 60x60 matrix
# Therefore, we exported to a CSV file and ucsed conditional formatting to highlight cells where correlation >0.7

# Plot of residuals vs Predicted Y Values
plot(reg1, 1)


# APPROACH1 - Naive Model
naivePred = mean(train$Score)
naivePred
rmseNaive = sqrt(mean((test$Score - naivePred)^2))
rmseNaive
rmseNaive/sd(test$Score)

# APPROACH2 - Kitchen Sink Linear Regression Model
#set x variables for train sample to subset
xVarTrain = train[,3:64]
# reun regression on train sample
reg2 = lm(train$Score ~ ., data = xVarTrain)
summary(reg2)
#set x variables for Test sample to subset
xVarTest = test[,3:64]
# get predicted values for yVarTest
yVarTestPred = predict(reg2, newdata = xVarTest)
yVarTestPred[is.na(yVarTestPred)] = naivePred
# calculate RMSE
rmseReg2 = rmse(test$Score, yVarTestPred)
rmseReg2
#establish range of credit scores as benchmark for RMSE
credScoreRange=as.integer(summary(test$Score)[6] - summary(test$Score)[1])
rmseReg2 / credScoreRange
rmseReg2/sd(test$Score)



#APPROACH3 - LASSO Analysis without Logs
trainLassoCreditData = train[-c(1)]
testLassoCreditData = test[-c(1)]

X3 = sparse.model.matrix(Score ~ ., data = trainLassoCreditData)
Y3 = trainLassoCreditData$Score

reg3Lasso = cv.glmnet(X3, Y3)
# number of nonzero coeffs in best Lasso model
sum(coef(reg3Lasso, s='lambda.min') != 0)
# Coefficients for best lasso model
coef(reg3Lasso, s='lambda.min')
-sort(coef(reg3Lasso, s='lambda.min'))
#plot results
plot(reg3Lasso$nzero, sqrt(reg3Lasso$cvm), xlab='Number of X variables', ylab='Out of Sample RMSE', pch=16, type='b', col='red')

# Predicted values for data
testX3 = sparse.model.matrix(Score ~ ., data = testLassoCreditData)
reg3Lasso.fitted = predict(reg3Lasso,newx=testX3,s="lambda.min")
summary(reg3Lasso.fitted)
rmseReg3Lasso = rmse(testLassoCreditData$Score, reg3Lasso.fitted)
rmseReg3Lasso
rmseReg3Lasso / credScoreRange
rmseReg3Lasso/sd(test$Score)

#APPROCH4 - LASSO Analysis with ALL LOGS
# Get Log Vales
logTrainLassoCreditData = train
logTrainLassoCreditData = log(train[,c(3:61)])

logTestLassoCreditData = test
logTestLassoCreditData = log(test[,c(3:61)])

#set negative infinity values to 0
logTrainLassoCreditData[logTrainLassoCreditData < 0] = 0
logTrainLassoCreditData$ID = train$ID

logTestLassoCreditData[logTestLassoCreditData < 0] = 0
logTestLassoCreditData$ID = test$ID

# Merge log and non-log into one df
bothTrainLassoCreditData = merge(train, logTrainLassoCreditData, by.x = "ID", by.y = "ID")
bothTestLassoCreditData = merge(test, logTestLassoCreditData, by.x = "ID", by.y = "ID")

# Remove UID column
bothTrainLassoCreditData = bothTrainLassoCreditData[-c(1)]
bothTestLassoCreditData = bothTestLassoCreditData[-c(1)]

#Build Lasso Model
X4 = sparse.model.matrix(Score ~ ., data = bothTrainLassoCreditData)
Y4 = bothTrainLassoCreditData$Score
reg4Lasso = cv.glmnet(X4, Y4)

# number of nonzero coeffs in best Lasso model
sum(coef(reg4Lasso, s='lambda.min') != 0)# sort and print Coefficients for best lasso model
-sort(coef(reg4Lasso, s='lambda.min'))
coef(reg4Lasso, s='lambda.min')
#plot results
plot(reg4Lasso$nzero, sqrt(reg4Lasso$cvm), xlab='Number of X variables', ylab='Out of Sample RMSE', pch=16, type='b', col='red')

# Predicted values for data
testX4 = sparse.model.matrix(Score ~ ., data = bothTestLassoCreditData)
reg4Lasso.fitted = predict(reg4Lasso,newx=testX4,s="lambda.min")
summary(reg4Lasso.fitted)
rmseReg4Lasso = rmse(bothTestLassoCreditData$Score, reg4Lasso.fitted)
rmseReg4Lasso
rmseReg4Lasso / credScoreRange



#APPROACH5 - ATTEMPT TO HAND TUNE BASED ON LASSO OUTPUTS (only e+01 or greater)
#Debt/HighCredit
#train new model
reg5 = lm(train$Score ~  SatisfactoryPct + revcreditutil + AutoCount.x + DerogOtherCount.x + LiabilityBankruptcyCount.x + LiabilityCurrentAdverseCount.x + PublicRecordCount.x + DisputeCount.x  + MortgageCount.y + Day30.y + InquiryCount.y, data=bothTrainLassoCreditData)
summary(reg5)
plot(reg5,1)
yVarTestPred5 = predict(reg5,bothTestLassoCreditData)
yVarTestPred5 = as.integer(yVarTestPred5)
yVarTestPred5[is.na(yVarTestPred5)] = naivePred
rmseReg5 = rmse(bothTestLassoCreditData$Score,yVarTestPred5)
rmseReg5



#APPROACH6 - LOOK AT NON-LINEAR MODEL USING XGBOOST FOR GRADIENT BOOSTING AND GRID SEARCH FOR PARAMETERS - NO LOGS
#get data into 'DMatrix', equivalent to sparsematrix clunk in Lasso
DTrain = xgb.DMatrix(data=X3, label=Y3)
yVarPred = as.integer(test$Score)
DTest = xgb.DMatrix(data=testX3, label=yVarPred)
# train a model using our training data
XGBmodel = xgboost(data = DTrain, 
                 max.depth = 12, # the maximum depth of each decision tree
                 nround = 22, # max number of boosting iterations                              
                 early_stopping_rounds = 3, # just to save compute power
                 objective = "reg:linear")  

XGBpredict = predict(XGBmodel, DTest)
XGBrmse=rmse(test$Score, XGBpredict)
XGBrmse
XGBrmse / credScoreRange
XGBrmse/sd(test$Score)
plot(XGBmodel,1)



#APPROACH7 - USE SHAP TO DERIVE EXPLAINED XGBOOST CONTRIBUTIONS AND TRY TO MIMIC W/ LM
shap_values=shap.values(xgb_model = XGBmodel, X_train = X3)
shap_values$mean_shap_score
#Top 10 Shap Scores Are:
#... TotalLiabilityPastDue + RevolvingPastDue + OpenBalance + TotalUnsecuredLoanBalance
#... RevolvingHighCredit + LiabilityCurrentAdverseCount + RevolvingBalance
#... MortgageHighCredit + InstallmentPastDue + InquiryCount
reg7 = lm(train$Score ~ revcreditutil + SatisfactoryPct + TotalLiabilityPastDue + OpenPastDue + InquiryCount + InstallmentPastDue + EducationPastDue + LiabilityBankruptcyCount, data=xVarTrain)
summary(reg7)
yVarTestPredShap = predict(reg7,xVarTest)
yVarTestPredShap = as.integer(yVarTestPredShap)
yVarTestPredShap
plot(reg7,1)
rmseReg7 = rmse(test$Score,yVarTestPredShap)
rmseReg7
rmseReg7 / credScoreRange
rmseReg7 /sd(test$Score)


#APPROACH8 - USE KERAS/TENSORFLOW ReLU FOR DEEP LEARNING BASED APPROACH
# Normalize training data
train_data = scale(X3)
train_data
# Use means and standard deviations from training set to normalize test set
col_means_train = attr(train_data, "scaled:center") 
col_stddevs_train = attr(train_data, "scaled:scale")
test_data = scale(testX3, center = col_means_train, scale = col_stddevs_train)
train_labels = Y3
test_labels = test$Score

colnames(train_data) <- NULL
colnames(test_data) <- NULL
train_data = train_data[1:1546,2:63]
test_data = test_data[1:663,2:63]
train_data[1,]
test_data[1,]
#creates virtual environment, need Python TF, Keras running
tensorflow::install_tensorflow(version = "1.13.1")
library(keras)
library(kerasR)
library(tensorflow)
#instantiate DL model
DLmodel = keras_model_sequential() 
#determine DL model architecture
DLmodel%>%
  #firsthiddenlayer
  layer_dense(
    units              = 256, 
    activation         = "sigmoid", 
    input_shape        = c(62)) %>% 
  layer_dropout(rate = 0.4) %>% 
  # Second hidden layer
  layer_dense(
    units              = 128, 
    activation         = "softmax") %>% 
  layer_dropout(rate = 0.3) %>% 
  # Third hidden layer
  layer_dense(
    units              = 64, 
    activation         = "relu") %>% 
  layer_dropout(rate = 0.2) %>% 
  # Fourth hidden layer
  layer_dense(
    units              = 64, 
    activation         = "relu") %>% 
  # Output layer
  layer_dense(
    units= 1) %>% 
  #compile neural network
  compile(
  loss = "mse",
  optimizer = optimizer_rmsprop(),
  metrics = list("mean_absolute_error")
  )
#summarize model architecture
DLmodel %>% summary()
# Display training progress by printing a single dot for each completed epoch.
print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)    
#set numer of epochs
epochs = 130
# Fit the model and store training stats
history = DLmodel %>% fit(
  train_data,
  train_labels,
  epochs = epochs,
  validation_split = 0.2,
  verbose = 0,
  callbacks = list(print_dot_callback)
)
#graph outputs on train set
library(ggplot2)
plot(history, metrics = "mean_absolute_error", smooth = FALSE) +
  coord_cartesian(ylim = c(0, 50))

DLpred = predict(DLmodel, test_data)
DLpred
DLrmse=rmse(test$Score, DLpred)
DLrmse




#APPROACH9 - ENSEMBLE LASSO (All), DEEP LEARNING, AND GRADIENT BOOSTED TECHNIQUES
ensemblepred = (.6*DLpred + .4*XGBpredict) 
ensemblepred
ensemblermse=rmse(test$Score, ensemblepred)
ensemblermse


#APPROACH10 - include credit utilization ratio
#crreate copies of input variables to look at things like creditscore utilization, etc.....
revUtil = xVarTrain
revUtilTest = xVarTest
View(revUtil)
revUtil$DebtHighCred = revUtil$TotalLiabilityBalance/revUtil$TotalLiabilityHighCredit
revUtilTest$DebtHighCred = revUtilTest$TotalLiabilityBalance/revUtilTest$TotalLiabilityHighCredit
revUtil$CredMix = revUtil$RevolvingHighCredit/revUtil$TotalHighCredit
revUtilTest$CredMix = revUtilTest$RevolvingHighCredit/revUtilTest$TotalHighCredit




#COMPARISONS
#Naive Model
rmseNaive
rmseNaive/credScoreRange
rmseNaive/sd(test$Score)
#Kitchen Sink LM Model
rmseReg2
rmseReg2/credScoreRange
rmseReg2/sd(test$Score)
#Lasso Model
rmseReg3Lasso
rmseReg3Lasso/credScoreRange
rmseReg3Lasso/sd(test$Score)
#LogLasso Model
rmseReg4Lasso
rmseReg4Lasso/credScoreRange
rmseReg4Lasso/sd(test$Score)
#Hand Picked Linear/Logistic Mix Model
rmseReg5
rmseReg5/credScoreRange
rmseReg5/sd(test$Score)
#XGBoost Gradient Boosted Model
XGBrmse
XGBrmse/credScoreRange
XGBrmse/sd(test$Score)
#SHAP Derived Linear Model
rmseReg7
rmseReg7/credScoreRange
rmseReg7/sd(test$Score)
#Deep Learning Model
DLrmse
DLrmse/credScoreRange
DLrmse/sd(test$Score)
#Ensemble Model (DL+XGBoost+LASSO)
ensemblermse
ensemblermse/credScoreRange
ensemblermse/sd(test$Score)




