#Assignment 4 Tree Based Methods

install.packages("tree")
library(tree)
library(ISLR) 
attach(OJ)

#Create a training set containing a random sample of 800 observations, and a test set containing the
#remaining observations. Take a screenshot of your code. (Hint: set.seed (2), train=sample())

# We split the observations into a training set and a test set, build the tree using the training set, and evaluate its performance on the test data
set.seed(2)

# randomly choose 800 observations (out of the total observations) as the training dataset
train=sample(1:nrow(OJ), 800)

# set the test dataset using Data[-train,]
Oj.test=OJ[-train,]


#Fit a tree to the training data, with Purchase as the response and the other variables as predictors.
#Use the summary( ) function to produce summary statistics about the tree. 

?OJ

#set the DV on the test data
Purchase.test=Purchase[-train]

#run the classification tree on the training dataset
tree.OJ=tree(Purchase~.,OJ,subset=train)

summary(tree.OJ)

#Plot the tree and take a screenshot of the tree (Hint: plot() and text())
plot(tree.OJ)
text(tree.OJ)

#Predict the response on the test data, and produce a confusion matrix comparing the test labels to the
#predicted test labels. What is the accuracy rate?


tree.pred=predict(tree.OJ,Oj.test,type="class")

summary(tree.pred)

#confusion matrix

table(tree.pred,Purchase.test)



# accuracy rate
mean(tree.pred==Purchase.test)



#Apply the cv.tree() function to the training set in order to determine the optimal tree size. 
#(Use set.seed(7)). Print the results (Hint: the results should contain the size, k, method etc).

set.seed(7)

cv.OJ=cv.tree(tree.OJ,FUN=prune.misclass)
names(cv.OJ)
cv.OJ

#Produce a plot with tree size (i.e. size) on the x-axis and cross-validated classification error rate (i.e.dev) on the y-axis.

plot(cv.OJ$size,cv.OJ$dev,type="b")


#Produce a pruned tree corresponding to the optimal tree size obtained using cross-validation. Take a
#screenshot of a pruned tree. 


prune.OJ=prune.misclass(tree.OJ,best=4)
plot(prune.OJ)
text(prune.OJ,pretty=0)

#What is the accuracy rate for the pruned tree? Is it improved compared to the
#accuracy rate in (1.4)?

tree.pred=predict(prune.OJ,Oj.test,type="class")
table(tree.pred,Purchase.test)
mean(tree.pred==Purchase.test)

#If cross-validation does not lead to selection of a pruned tree (i.e. the accuracy rate produced in (1.8)
#is lower than the one in (1.4)), then create a pruned tree with five terminal nodes. What is the accuracy
#rate now?

prune.OJ=prune.misclass(tree.OJ,best=5)
tree.pred=predict(prune.OJ,Oj.test,type="class")
mean(tree.pred==Purchase.test)

#Using the validation-set approach to split the data set into a training set and a test set (Hint: use set.seed(2); 
#validation-set approach: half of the observations are selected as the training dataset while half of observations are treated as the test dataset). 
#Take a screenshot of your code.
attach(Carseats)

set.seed(2)
nrow(Carseats)


# set half of the observations as the training dataset (validation-set approach)

train = sample(1:nrow(Carseats), nrow(Carseats)/2)
tree.carseat=tree(Sales~.,Carseats,subset=train)

summary(tree.carseat)

# we now plot the tree
plot(tree.carseat)
text(tree.carseat,pretty=0)

# true value of DV on the test data
carseat.test=Carseats[-train,"Sales"]

#What test MSE do you obtain?
#MSE (Mean of Squared Errors)


yhat=predict(tree.carseat,newdata=Carseats[-train,])

mean((yhat-carseat.test)^2)


#Use cross-validation in order to determine the optimal level of tree complexity (use set.seed(2)).does pruning the tree improve the test MSE?

set.seed(2)
cv.carseats=cv.tree(tree.carseat)
cv.carseats
plot(cv.carseats$size,cv.carseats$dev,type='b')




#Use the bagging approach in order to analyze this data. Take a screenshot of the results. What test MSE
#do you obtain? (Hint: use set.seed (1); mtry=10 since we have 10 predictors in Carseats dataset and we use all of the predictors in the bagging approach).

install.packages(c("gbm","randomForest"))
library(randomForest)


set.seed(1)
bag.carseats=randomForest(Sales~.,data=Carseats,subset=train,mtry=10,importance=TRUE)
bag.carseats

#2.5 Use random forests to analyze this data. 

#What test MSE do you obtain? (Hint: use set.seed(1); mtry=10/3 since we usually use 1/3 of the 
#predictors when building a random forest of regression trees)


set.seed(1)
bag.carseats=randomForest(Sales~.,data=Carseats,subset=train,mtry=10/3,importance=TRUE)
bag.carseats

#Use the importance() function to determine which variables are most important. Take a screenshot
#of your results.

importance(bag.carseats)

#Plots of these importance measures can be produced using the varImpPlot() function. Take a
#screenshot of your output

varImpPlot(bag.carseats)
