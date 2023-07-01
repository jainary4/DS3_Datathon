setwd("/Users/murphylee10/Downloads")

feat.d = read.csv(file="Featurized Data.csv", header=T)
feat.submit = read.csv(file="Featurized Data Prediction.csv", header=T)

library(tidymodels, rsample)

library(rpart)
library(rpart.plot)
library(stringr)

glimpse(feat.d)

feat.d2 = feat.d %>%
  rename(RUL = Remaining.Useful.Life,
         IDC = initial.discharge.capacity,
         FDC = final.discharge.capacity,
         DCS = discharge.cap..slope,
         DCI = dis..cap..intercept,
         MR = min..resistance,
         DR = Delta.resistance,
         DV = Delta_Variance)

feat.d2 = feat.d2 %>% mutate(IDC2 = as.numeric(str_sub(IDC, start = 2)), DV2 = as.numeric(str_sub(DV, end = -2)))

feat.submit2 = feat.submit %>%
  rename(IDC = initial.discharge.capacity,
         FDC = final.discharge.capacity,
         DCS = discharge.cap..slope,
         DCI = dis..cap..intercept,
         MR = min..resistance,
         DR = Delta.resistance,
         DV = Delta_Variance)

glimpse(feat.submit2)

feat.submit2 = feat.submit2 %>% mutate(IDC2 = as.numeric(str_sub(IDC, start = 2)), DV2 = as.numeric(str_sub(DV, end = -2)))


library(tidymodels)
mpg_split = initial

tree.m = rpart(RUL~IDC2+FDC+DCS+DCI+MR+DR+DV2, data=feat.d2)
feat.submit2$RUL = predict(tree.m, newdata=feat.submit2)

feat.submit2 = feat.submit2 %>% select(Cell.ID, RUL)

write.csv(feat.submit2, file="file_4_submission.csv", row.names=F)

#tree.m = rpart(Remaining.Useful.Life~initial.discharge.capacity+final.discharge.capacity+discharge.cap..slope+dis..cap..intercept+min..resistance+Delta.resistance+Delta_Variance, data=feat.d)

#feat.submit$pred.Remaining.Useful.Life = predict(tree.m, newdata=feat.submit)

glimpse(feat.d2)

feat.d2$pred.RUL = predict(tree.m, newdata=feat.d2)

mean((feat.d2$RUL-feat.d2$pred.RUL)^2)


library(caret)
feat.d2_split <- initial_split(feat.d2, prop = 0.8)
feat.d2_train <- training(feat.d2_split)
feat.d2_test <- testing(feat.d2_split)

# Define a grid of hyperparameters to search
hyper_grid <- expand.grid(cp = seq(0.001, 0.01, by = 0.001))

# Train the decision tree using the training set and the hyperparameter grid
tree.m <- train(RUL ~ IDC2 + FDC + DCS + DCI + MR + DR + DV2, 
                data = feat.d2_train, 
                method = "rpart",
                tuneGrid = hyper_grid,
                trControl = trainControl(method = "cv", number = 10))

# Print the results of the hyperparameter tuning
print(tree.m)

# Plot the cross-validation performance of the different hyperparameters
plot(tree.m)

# Prune the tree using the optimal complexity parameter
pruned_tree <- prune(tree.m$finalModel, cp = tree.m$bestTune$cp)

# Evaluate the pruned tree on the test set
predictions <- predict(pruned_tree, newdata = feat.d2_test)
RMSE <- sqrt(mean((feat.d2_test$RUL - predictions)^2))
print(RMSE)

feat.submit2$RUL = predict(pruned_tree, newdata=feat.submit2)

feat.submit2 = feat.submit2 %>% select(Cell.ID, RUL)

write.csv(feat.submit2, file="file_4_submission.csv", row.names=F)
