#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees','RRF')

library(rpart)
#install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)

#Reading the input data
churn_data = read.csv("Train_data.csv")


#Removing the commas from the workload column so that we could use it as a number
churn_data$Churn = gsub(" True.", "True", churn_data$Churn)
churn_data$Churn = gsub(" False.", "False", churn_data$Churn)


#Converting necessary predictors to categorical

churn_data$state = as.factor(churn_data$state)
churn_data$area.code = as.factor(churn_data$area.code)
churn_data$phone.number = as.factor(churn_data$phone.number)
churn_data$international.plan = as.factor(churn_data$international.plan)
churn_data$voice.mail.plan = as.factor(churn_data$voice.mail.plan)
churn_data$Churn = as.factor(churn_data$Churn)

### MISSING VALUE ANALYSIS ###

missing_val = data.frame(apply(churn_data,2,function(x){sum(is.na(x))}))

#find about the missing value analysis plot. amelia

### OUTLIER ANALYSIS ###

#Saving the numeric columns first, as outlier analysis is performed on numerical values:

numeric_index = sapply(churn_data,is.numeric)

numeric_data = churn_data[,numeric_index]

cnames = colnames(numeric_data)

#Plot boxplot to visualize outliers:

for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "Churn"), data = subset(churn_data))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i],x="Churn")+
           ggtitle(paste("Box plot for",cnames[i])))
}

#gridExtra::grid.arrange(gn1,gn5,gn2,ncol=3)


#loop to remove all outliers
for(i in cnames){
  print(i)
  val = churn_data[,i][churn_data[,i] %in% boxplot.stats(churn_data[,i])$out]
  churn_data = churn_data[which(!churn_data[,i] %in% val),]
}

### FEATURE SELECTION ###

## Correlation Plot 

corrgram(churn_data[,numeric_index], order = TRUE, lower.panel=panel.shade, upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

churn_data = subset(churn_data, select = -c(total.day.charge,total.eve.charge,total.night.charge,total.intl.charge))

##Chi square test

factor_index = sapply(churn_data,is.factor)
factor_data = churn_data[,factor_index]

for (i in 1:11)
{
  print(names(factor_data)[i])
  print(chisq.test(table(churn_data$Churn,factor_data[,i])))
}

churn_data = subset(churn_data, select = -c(area.code))
churn_data = subset(churn_data, select = -c(state,phone.number))

##MODEL DEVELOPMENT
churn_data$international.plan = gsub("yes", 1, churn_data$international.plan)
churn_data$international.plan = gsub("no", 0, churn_data$international.plan)

churn_data$voice.mail.plan = gsub("yes", 1, churn_data$voice.mail.plan)
churn_data$voice.mail.plan = gsub("no", 0, churn_data$voice.mail.plan)


##########

test_data = read.csv("Test_data.csv")
test_data$state = as.factor(test_data$state)
test_data$area.code = as.factor(test_data$area.code)
test_data$phone.number = as.factor(test_data$phone.number)
test_data$international.plan = as.factor(test_data$international.plan)
test_data$voice.mail.plan = as.factor(test_data$voice.mail.plan)
test_data$Churn = as.factor(test_data$Churn)

test_data = subset(test_data, select = -c(total.day.charge,total.eve.charge,total.night.charge,total.intl.charge))
test_data = subset(test_data, select = -c(area.code))
test_data = subset(test_data, select = -c(state,phone.number))
test_data$international.plan = gsub("yes", 1, test_data$international.plan)
test_data$international.plan = gsub("no", 0, test_data$international.plan)

test_data$voice.mail.plan = gsub("yes", 1, test_data$voice.mail.plan)
test_data$voice.mail.plan = gsub("no", 0, test_data$voice.mail.plan)

test_data$international.plan = as.factor(test_data$international.plan)
test_data$voice.mail.plan = as.factor(test_data$voice.mail.plan)
test_data$Churn = gsub(" True.", "True", test_data$Churn)
test_data$Churn = gsub(" False.", "False", test_data$Churn)

test_data$Churn = as.character(test_data$Churn)
test_data$Churn = as.factor(test_data$Churn)


##Ensuring same levels

test_data$international.plan = as.character(test_data$international.plan)
churn_data$international.plan = as.character(churn_data$international.plan)

test_data$voice.mail.plan = as.character(test_data$voice.mail.plan)
churn_data$voice.mail.plan = as.character(churn_data$voice.mail.plan)

test_data$isTest = rep(1,nrow(test_data))
churn_data$isTest = rep(0,nrow(churn_data))

fullSet = rbind(test_data, churn_data)
fullSet$international.plan = as.factor(fullSet$international.plan)
fullSet$voice.mail.plan = as.factor(fullSet$voice.mail.plan)

test_data = fullSet[fullSet$isTest==1,]
churn_data = fullSet[fullSet$isTest==0,]

churn_data = subset(churn_data, select = -c(isTest))
test_data = subset(test_data, select = -c(isTest))

####


#########

#Logistic Regression

LogModel = glm(Churn ~ .,family=binomial,data=churn_data)
#print(summary(LogModel))

LR_predictions = predict(LogModel,newdata=test_data,type='response')
LR_predictions = ifelse(LR_predictions > 0.5, "True", "False")
LR_confusion_table = table(LR_predictions,test_data$Churn)
 
cat("Accuracy of Logistic Regression model:", sum(diag(LR_confusion_table)/nrow(test_data)) *100, "%")

##Decision Tree Classifier

DT_model = rpart(Churn ~ ., data = churn_data, method = "class")
#summary(DT_model)
DT_predictions = predict(DT_model, test_data, type = "class")
DT_confusion_table = table(DT_predictions,test_data$Churn)

cat("Accuracy of Decision Tree Classifier model:", sum(diag(DT_confusion_table)/nrow(test_data)) *100, "%")

#Random Forest Classifier

RF_model = randomForest(Churn ~ ., churn_data, ntree = 500)
#summary(RF_model)
RF_predictions = predict(RF_model, test_data)
RF_confusion_table = table(RF_predictions,test_data$Churn)

cat("Accuracy of Random Forest Classifier model:", sum(diag(RF_confusion_table)/nrow(test_data)) *100, "%")
