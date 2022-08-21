# The following libraries must be installed if they are not listed in packages
install.packages ("caret")
install.packages ("ROCR")
install.packages ("rpart")
install.packages ("rpart.plot")
install.packages ("randomForest")
install.packages ("forecast")

# 1. Lines 14-19: Load necessary libraries every time you start 
library(caret)
library(ROCR)
library(rpart)
library(rpart.plot)
library(randomForest)
library(forecast)

library(readr)
heart_Disease <- read_csv("~/Desktop/heart_Disease.csv")
View(heart_Disease)

impute_data <- function(vec, mn) {
  ifelse(is.na(vec), mn, vec)
}

data_mean <- sapply(heart_Disease[,1:14],mean, na.rm=TRUE)
# Run this command to look at means for all the columns we found by running the sapply function
(data_mean)

summary(heart_Disease)

set.seed(100)
train <- sample(nrow(heart_Disease), 0.7*nrow(heart_Disease),replace=FALSE)
trainSet <- heart_Disease[train,]
testSet <- heart_Disease[-train,]

summary(trainSet)
summary(testSet)

install.packages("randomForest")
library(randomForest)

set.seed(44)
table(trainSet$target)
ctrf <- randomForest(as.factor(trainSet$target) ~ . ,
                     data = trainSet,
                     ntree = 1000, 
                     mtry = 1, 
                     sampsize = c(97,97), 
                     importance = TRUE, 
                     method = "class",
                     na.action = randomForest::na.roughfix,
                     replace = FALSE)
ctrf

nrow_train <- nrow(trainSet)

# Lines 152-154: Plot the error rate against the number of trees
plot(ctrf, main = "")
legend("topright", c("OOB", "No", "Yes"), 
       text.col = 1:6, lty = 1:3, col = 1:3)


# Compute the number of rows in the smaller training set.
nrow_small_train <- round(nrow(trainSet)*0.75)
nrow_validation <- nrow_train - nrow_small_train

set.seed(9211)
# Sample the row indices in the smaller training set
row_indices_smaller_train <- sample(1:nrow_train, size=nrow_small_train, replace=FALSE)
# Split the training set into the smaller training set and the validation set using these indices. 
smaller_train <- trainSet[row_indices_smaller_train,]
validation <- trainSet[-row_indices_smaller_train,]

heartdisease_linear_model <- lm(target ~ age + sex + cp + trestbps + chol + fbs + restecg + thalach + exang + oldpeak + slope + ca + thal,data=smaller_train)
# Use summary function to see the model summary. The summary gives the coefficients, their p-values, and other regression statistics.
summary(heartdisease_linear_model)

heartdisease_linear_model_stepped <- step(heartdisease_linear_model, direction="backward")
summary(heartdisease_linear_model_stepped)

install.packages("rpart")
library(rpart)


heartdisease_reg_tree <- rpart(target ~ age + sex + cp + trestbps + chol + fbs + restecg + thalach + exang + oldpeak + slope + ca + thal,data=smaller_train)

summary(heartdisease_reg_tree)

print(heartdisease_reg_tree)

plot(heartdisease_reg_tree)
text(heartdisease_reg_tree, use.n=TRUE, cex=0.5)
#######
set.seed(44)
ctfull <- rpart(target ~ . , data = smaller_train,
                method = "class",
                parms = list(split = "information"),
                control = rpart.control(minsplit =0, minbucket =1, cp =0.00001), 
                model = TRUE)
print(ctfull)
min_xerror <- min(ctfull$cptable[,"xerror"])
min_xerror

#  Find the minimum xerror tree by running given command
min_error_tree <- which.min(ctfull$cptable[,"xerror"])
min_error_tree                        

# Find min_error_tree's xstd
min_xerror_xstd <- ctfull$cptable[min_error_tree , "xstd"]
min_xerror_xstd

#  Sum min_xerror + min_xerror_xstd
min_xerror + min_xerror_xstd
# - Find smallest tree with cross-validated error xerror less than 0.327137
# - This corresponds to tree numbered 10 with xerror 0.31890
#                                        which has a CP of 0.00206186
# - We will use this CP to construct best pruned tree in the next step

plotcp(ctfull)

############

library(partykit)
library(party)

plot(as.party(heartdisease_reg_tree), type="extended")
reg_predictions <- predict(heartdisease_linear_model_stepped,validation)
tree_predictions <- predict(heartdisease_linear_model_stepped,validation)
print(reg_predictions)
View(reg_predictions)
print(tree_predictions)

library(forecast)
accuracy(reg_predictions,validation$target)
accuracy(tree_predictions,validation$target)
############
which.min(ctfull$cptable[,"xerror"])
# Lines 134-135: Get the min error tree (met)'s CP and save it in met_cp
met_cp <-ctfull$cptable[which.min(ctfull$cptable[,"xerror"]), "CP"]
met_cp

set.seed(42)
rtmin <- rpart(target ~ . , data = trainSet,
               method = "anova", 
               parms = list(split = "information"),
               control = rpart.control(minsplit = 0, minbucket = 1, 
                                       cp = met_cp), # or write cp = 0.0006
               model = TRUE)

# Line 156: See a summary pruned tree using print function 
print(rtmin)
# Lines Get the predictions for the testing data frame  
rtPred <- predict(rtmin, newdata = testSet, type = "vector")

# 5.1 Performance Evaluation Method 1:
# Compute error measures we have learned when estimating numerical outcomes
# Recall error of obs i is the difference between actual and predicted value
# AE represents average error
# RMSE represents root mean squared error

# Lines 213-216: Calculate and display AE and RMSE
AE <- mean((testSet$taregt - rtPred))
RMSE <- sqrt(mean((testSet$target - rtPred)^2))
AE
RMSE
