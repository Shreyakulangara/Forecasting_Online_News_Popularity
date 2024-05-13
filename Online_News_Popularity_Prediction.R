#-----Section 01 get working directory and load libraries-------------------------------------------
# get the data
# set working directory
setwd(dirname(file.choose()))
getwd()
library(Amelia)
library(TeachingDemos)
library(reshape2)
library(ggplot2)
library(randomForest)
library(vip)
library(e1071)
library(caret)
library(C50)
library(plyr)
library(gmodels)
library(ROCR)
library(pROC)
library(adabag)
library(dplyr)
library(magrittr)
library(xgboost)
library(tidyr)
library(corrplot)
library(factoextra)
library(FactoMineR)
library(gridExtra)
library(gt)
library(nFactors)
library(psych)
library(corrgram)

compute_roc_auc <- function(res, true_labels,model_name) {
  roc_list <- list()
  auc_values <- numeric()
  
  # Iterate over each class
  for (class in levels(true_labels)) {
    auc_values <- numeric(0)
    # Create binary outcome for the class
    class_binary <- as.integer(true_labels == class)
    
    # Create prediction object
    pred <- prediction(res[, class], class_binary)
    
    # Compute ROC curve
    perf_obj <- performance(pred, "tpr", "fpr")
    
    # Store ROC object in the list
    roc_list[[class]] <- perf_obj
    
    tpr <- unlist(perf_obj@y.values[[1]])
    fpr <- unlist(perf_obj@x.values[[1]])
    
    # Compute AUC
    auc_value <- auc(fpr, tpr)
    cat("AUC for class", class, ":", auc_value, "\n")
    
    # Store AUC value
    auc_values <- c(auc_values, auc_value)
    
    
  }
  plot(roc_list[[1]], main = paste0("Multiclass ROC Curve", model_name), col = "blue", lwd = 2, ylim = c(0, 1))
  # Add ROC curves for other classes
  for (i in 2:length(roc_list)) {
    plot(roc_list[[i]], col = i, add = TRUE, lwd = 2)
  }
  lines(c(0, 1), c(0, 1), col = "gray", lty = 2)
  
  legend("bottomright", legend = c("Extremely Bad","Majority","Extremely Good","Random classifier"), 
         col = c(1:length(levels(true_labels)),"gray"), lwd = 2)
  
  return(auc_values)
}

# ---------Function to calculate precision, recall, and F1 score for a given class-------
calculate_metrics <- function(conf_matrix_table, class_index) {
  TP <- conf_matrix_table[class_index, class_index]
  FP <- sum(conf_matrix_table[, class_index]) - TP
  FN <- sum(conf_matrix_table[class_index, ]) - TP
  
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  f1_score <- 2 * precision * recall / (precision + recall)
  
  return(list(precision = precision, recall = recall, f1_score = f1_score))
}

compute_classification_metrics <- function(conf_matrix, predictions, true_labels) {
  # Calculate accuracy
  accuracy <- sum(diag(conf_matrix$table)) / sum(conf_matrix$table)
  
  # Calculate kappa statistic
  kappa_statistic <- conf_matrix$overall['Kappa']
  
  # Convert factors to numeric
  p_numeric <- as.numeric(as.character(predictions))
  true_labels_numeric <- as.numeric(as.character(true_labels))
  
  # Compute RMSE
  rmse <- sqrt(mean((p_numeric - true_labels_numeric)^2))
  
  # Calculate class-wise metrics
  num_classes <- nrow(conf_matrix$table)
  class_metrics <- lapply(1:num_classes, function(i) calculate_metrics(conf_matrix$table, i))
  
  # Calculate macro-average metrics
  macro_precision <- mean(sapply(class_metrics, function(x) x$precision))
  macro_recall <- mean(sapply(class_metrics, function(x) x$recall))
  macro_f1_score <- mean(sapply(class_metrics, function(x) x$f1_score))
  
  # Calculate micro-average metrics
  micro_precision <- sum(diag(conf_matrix$table)) / sum(conf_matrix$table)
  micro_recall <- sum(diag(conf_matrix$table)) / sum(conf_matrix$table)
  micro_f1_score <- 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
  
  # Calculate weighted-average metrics
  weight <- colSums(conf_matrix$table) / sum(conf_matrix$table)
  weighted_precision <- sum(sapply(1:num_classes, function(i) class_metrics[[i]]$precision * weight[i]))
  weighted_recall <- sum(sapply(1:num_classes, function(i) class_metrics[[i]]$recall * weight[i]))
  weighted_f1_score <- sum(sapply(1:num_classes, function(i) class_metrics[[i]]$f1_score * weight[i]))
  
  # Output results
  result <- list(
    class_metrics = class_metrics,
    macro_precision = macro_precision,
    macro_recall = macro_recall,
    macro_f1_score = macro_f1_score,
    micro_precision = micro_precision,
    micro_recall = micro_recall,
    micro_f1_score = micro_f1_score,
    weighted_precision = weighted_precision,
    weighted_recall = weighted_recall,
    weighted_f1_score = weighted_f1_score,
    accuracy = accuracy,
    kappa_statistic = kappa_statistic,
    rmse = rmse
  )
  
  return(result)
}


#-----Section 02 load csv and remove non - predictive features -------------------------------------------
# import data file OnlineNewsPopularity.csv and put relevant variables in a data frame
OnlineNewsPopularity <- read.csv("OnlineNewsPopularity.csv", stringsAsFactors = FALSE)

# drop the url,timedelta variable (non-predictive variable)
OnlineNewsPopularity <- subset(OnlineNewsPopularity, select = -c(url, timedelta))

#-----Section 03 Exploratory Data analysis -------------------------------------------
#### Descriptive statistics ####
# exploring and preparing the data

#number of observations
nrow(OnlineNewsPopularity)
#number of variables
ncol(OnlineNewsPopularity)
#list the names of variables
names(OnlineNewsPopularity)

# examine the structure of the OnlineNewsPopularity dataframe
str(OnlineNewsPopularity)

#Inspect top 4 rows and last 4 rows of the dataframe
head(OnlineNewsPopularity,4)
tail(OnlineNewsPopularity,4)

#Summary Statistics
summary(OnlineNewsPopularity)

#checking for missing values
apply(OnlineNewsPopularity, MARGIN = 2, FUN = function(x) sum(is.na(x)))
#remove any missing values
OnlineNewsPopularity <- na.omit(OnlineNewsPopularity)


#Measures of central tendency and dispersion on OnlineNewsPopularity - shares
max(OnlineNewsPopularity$shares)
min(OnlineNewsPopularity$shares)
range <- max(OnlineNewsPopularity$shares)-min(OnlineNewsPopularity$shares)
mean(OnlineNewsPopularity$shares)
median(OnlineNewsPopularity$shares)
sd(OnlineNewsPopularity$shares)
quantiles <- quantile(OnlineNewsPopularity$shares, c(0.25,0.75), na.rm = TRUE)
iqr_value <- diff(quantiles)
iqr_value


####Univariate analysis ####

#Univariate Analysis Target Variable - shares 
ggplot(OnlineNewsPopularity, aes(x = "", y = shares)) +
  geom_boxplot() +
  labs(title = "Boxplot for Target Variable - shares", x="Shares", y = "Count") +
  theme(axis.text.x = element_blank()) +
  scale_y_continuous(labels = function(x) format(x, scientific = FALSE))

ggplot(OnlineNewsPopularity, aes(x = shares)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 100) +
  labs(title = "Histogram of Target Variable - Shares", 
       x = "Shares", 
       y = "Frequency")

# q-q plot -shares
qqnorm(OnlineNewsPopularity$shares, xlab = "Theoretical Quantiles: Shares",
       main = "Q-Q Plot for Shares")
qqline(OnlineNewsPopularity$shares, col = 2)

# ks test - shares
#Null Hypothesis: The shares is normally distributed.
ks.test(OnlineNewsPopularity$shares,"pnorm", 
        mean(OnlineNewsPopularity$shares), 
        sd(OnlineNewsPopularity$shares))
# p-value < 2.2e-16 < 0.05 - NOT NORMAL

####Multivariate analysis####
# Reshape your data into long format
OnlineNewsPopularity_long <- pivot_longer(OnlineNewsPopularity, cols = everything(), names_to = "variable")

# Create a boxplot for every variable
theme_set(theme_bw())
ggplot(OnlineNewsPopularity_long, aes(x = factor(variable), y = value)) +
  geom_boxplot() +
  facet_wrap(~ variable, scales = "free", dir = "v")+
  labs(title = "Boxplot - OnlineNewsPopularity",x="variable") +
  theme(
    axis.text.x = element_text(size = 6),  # Adjust font size for x-axis text
    axis.text.y = element_text(size = 4),  # Adjust font size for y-axis text
    strip.text = element_text(size = 6)    # Adjust font size for facet labels
  )


# Compute the correlation matrix
correlation_matrix <- cor(OnlineNewsPopularity[, -1])

# Plot the correlation matrix as a heatmap
ggplot(data = reshape2::melt(correlation_matrix), aes(Var2, Var1, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1,1), space = "Lab", name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 8, hjust = 1),
        axis.title = element_blank(), 
        axis.ticks = element_blank(),  
        panel.grid = element_blank()  
  ) +
  coord_fixed()+
  labs(title = "Correlation Heatmap")


ggplot(OnlineNewsPopularity, aes(x = n_tokens_content, y = shares)) +
  geom_point() +
  labs(title = "Number of Words in Content vs Number of Shares",
       x = "Number of Words in Content",
       y = "Number of Shares") +
  theme_minimal()

ggplot(OnlineNewsPopularity, aes(x = n_tokens_title, y = shares)) +
  geom_point() +
  labs(title = "Number of Words in title vs Number of Shares",
       x = "Number of Words in title",
       y = "Number of Shares") +
  theme_minimal()


#--------- Section 04 - Data pre processing --------

#### remove noise ####
ggplot(OnlineNewsPopularity, aes(x = "", y = n_tokens_content)) +
  geom_boxplot() +
  labs(title = "Boxplot - n_tokens_content",
       x = "n_tokens_content",
       y = "Count")
OnlineNewsPopularity <- OnlineNewsPopularity[OnlineNewsPopularity$n_tokens_content != 0, ]
dim(OnlineNewsPopularity)

#### remove outlier from n_non_stop_words ####
ggplot(OnlineNewsPopularity, aes(x = "", y = n_non_stop_words)) +
  geom_boxplot() +
  labs(title = "Boxplot - n_non_stop_words",
       x = "n_non_stop_words",
       y = "Count")

OnlineNewsPopularity = OnlineNewsPopularity[!OnlineNewsPopularity$n_non_stop_words==1042,]

#### Shares Categorization and Problem Transformation	####
# Calculate ECDF
ecdf_function <- ecdf(OnlineNewsPopularity$shares)

# Generate sequence of values
x_values <- seq(min(OnlineNewsPopularity$shares), max(OnlineNewsPopularity$shares), length.out = 1000)

# Compute ECDF for the sequence
y_values <- ecdf_function(x_values)

# Create data frame
ecdf_df <- data.frame(x = x_values, y = y_values)

# Plot using ggplot2
ggplot(ecdf_df, aes(x, y)) +
  geom_line(color = "blue", size = 0.5) +
  labs(x = "shares", y = "%", 
       title = "Cumulative Distribution for shares")+
  scale_x_continuous(breaks = seq(0, 900000, 45000)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.02, hjust = 1, size = 2)) +
  theme_bw()


#It shows the majority of number of shares are less than 15000. 
#Therefore, I filter out the row with shares higher than 15000 to see the distribution of the majority.

# Subset the dataframe to include only rows where shares < 15000
df_majority <- OnlineNewsPopularity[OnlineNewsPopularity$shares < 15000, ]

# Compute ECDF
ecdf_data <- ecdf(df_majority$shares)

# Generate sequence of values for plotting
x_values <- seq(min(df_majority$shares), max(df_majority$shares), length.out = 1000)

# Compute ECDF for the sequence of values
y_values <- ecdf_function(x_values)

# Create data frame for plotting
ecdf_df <- data.frame(x = x_values, y = y_values)

# Plot ECDF
ggplot(ecdf_df, aes(x, y)) +
  geom_line(color = "blue", size = 0.5) +
  labs(x = "shares", y = "%", 
       title = "Cumulative Distribution of Majority Shares") +
  scale_x_continuous(breaks = seq(0, 16000, by = 1000)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme_bw()  

# we can tell that most articles are in the range b/w 600 and 3000
# thus divide the number of shares into 3 levels- 
# extremely bad - shares lower than 600
# majority - shares between 600 to 3000
# extremely good - shares more than 3000.

# Plot histogram
ggplot(df_majority, aes(x = shares)) +
  geom_histogram(binwidth = 200, color = "black", fill = "lightblue") +
  labs(x = "shares", y = "count", 
       title = "Distribution of Shares") +
  scale_x_continuous(breaks = seq(0, 15000, by = 500)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 4, family = "sans")) +
  geom_vline(xintercept = c(600, 3000), color = "red") +
  theme_bw()

# Define bin edges
bin_edges <- c(0, 600, 3000, 1000000)

# Define labels for the levels
bin_names <- c(0, 1, 2)

# Create shares_levels column using cut function
OnlineNewsPopularity$shares_levels <- cut(OnlineNewsPopularity$shares, breaks = bin_edges, labels = bin_names, right = FALSE)

# Convert shares_levels to a factor
OnlineNewsPopularity$shares_levels <- as.factor(OnlineNewsPopularity$shares_levels)

counts <- table(OnlineNewsPopularity$shares_levels)
counts

# Compute the proportion of each group
round(prop.table(table(OnlineNewsPopularity$shares_levels))*100, digits = 1)
level_counts <- table(OnlineNewsPopularity$shares_levels)
level_prop <- prop.table(level_counts)
colors <- c("lightyellow", "lightgreen", "orange")

# Create a pie chart
pie(level_prop, labels = paste(names(level_prop), ": ", round(level_prop * 100, 2), "%"), col = colors)
title(main = "Proportion of Shares Levels", cex.main = 1.2)
# Create a bar graph
barplot(level_counts, col = colors, 
        main = "Proportion of Shares Levels", 
        xlab = "Shares Levels", 
        ylab = "Count",
        ylim = c(0, max(level_counts) + 5000)
)

####One-hot encoding####
OnlineNewsPopularity[0, 12:17] 
OnlineNewsPopularity[0, 30:36] 
OnlineNewsPopularity$weekday <- NULL
OnlineNewsPopularity$data_channel <- NULL

# Loop through each row
for (i in 1:nrow(OnlineNewsPopularity)) {
  # Convert data_channel
  if (OnlineNewsPopularity[i, 12] == 1) {
    OnlineNewsPopularity[i, "data_channel"] <- "lifestyle"
  } else if (OnlineNewsPopularity[i, 13] == 1) {
    OnlineNewsPopularity[i, "data_channel"] <- "entertainment"
  } else if (OnlineNewsPopularity[i, 14] == 1) {
    OnlineNewsPopularity[i, "data_channel"] <- "bus"
  } else if (OnlineNewsPopularity[i, 15] == 1) {
    OnlineNewsPopularity[i, "data_channel"] <- "socmed"
  } else if (OnlineNewsPopularity[i, 16] == 1) {
    OnlineNewsPopularity[i, "data_channel"] <- "tech"
  } else if (OnlineNewsPopularity[i, 17] == 1) {
    OnlineNewsPopularity[i, "data_channel"] <- "world"
  }
  
  # Convert weekday
  if (OnlineNewsPopularity[i, 30] == 1) {
    OnlineNewsPopularity[i, "weekday"] <- "monday"
  } else if (OnlineNewsPopularity[i, 31] == 1) {
    OnlineNewsPopularity[i, "weekday"] <- "tuesday"
  } else if (OnlineNewsPopularity[i, 32] == 1) {
    OnlineNewsPopularity[i, "weekday"] <- "wednesday"
  } else if (OnlineNewsPopularity[i, 33] == 1) {
    OnlineNewsPopularity[i, "weekday"] <- "thursday"
  } else if (OnlineNewsPopularity[i, 34] == 1) {
    OnlineNewsPopularity[i, "weekday"] <- "friday"
  } else if (OnlineNewsPopularity[i, 35] == 1) {
    OnlineNewsPopularity[i, "weekday"] <- "saturday"
  } else if (OnlineNewsPopularity[i, 36] == 1) {
    OnlineNewsPopularity[i, "weekday"] <- "sunday"
  }
}

table(OnlineNewsPopularity$weekday)
table(OnlineNewsPopularity$data_channel)

sum(is.na(OnlineNewsPopularity$data_channel))
sum(is.na(OnlineNewsPopularity$weekday))

# Replace missing values in the 'data_channel' column with "none"
OnlineNewsPopularity$data_channel <- ifelse(is.na(OnlineNewsPopularity$data_channel), "none", OnlineNewsPopularity$data_channel)
OnlineNewsPopularity <- subset(OnlineNewsPopularity, select = -c(data_channel_is_lifestyle, data_channel_is_entertainment, data_channel_is_bus, data_channel_is_socmed, data_channel_is_tech, data_channel_is_world))
OnlineNewsPopularity <- subset(OnlineNewsPopularity, select = -c(weekday_is_monday, weekday_is_tuesday, weekday_is_wednesday, weekday_is_thursday, weekday_is_friday, weekday_is_saturday, weekday_is_sunday))


OnlineNewsPopularity$data_channel <- as.factor(OnlineNewsPopularity$data_channel)
OnlineNewsPopularity$weekday <- as.factor(OnlineNewsPopularity$weekday)
OnlineNewsPopularity$is_weekend <- as.factor(OnlineNewsPopularity$is_weekend)

OnlineNewsPopularity <- subset(OnlineNewsPopularity, select = -shares)

column_index <- which(names(OnlineNewsPopularity) == "shares_levels")
OnlineNewsPopularity <- OnlineNewsPopularity[, c(1:(column_index-1), (column_index+1):(ncol(OnlineNewsPopularity)), column_index)]

# Create a dataframe with the count of shares_levels for each weekday
share_counts <- OnlineNewsPopularity %>%
  group_by(weekday, shares_levels) %>%
  summarise(count = n()) %>%
  mutate(shares_levels = factor(shares_levels, levels = c("0", "1", "2")))

# Plot
ggplot(share_counts, aes(x = weekday, y = count, fill = shares_levels)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Count of shares_levels over different days of the week",
       x = "Day of the Week", y = "Count") +
  scale_fill_manual(values = c("yellow", "lightgreen", "orange"),
                    name = "Shares Levels",
                    labels = c("extremely bad", "majority", "extremely good")) +
  theme_minimal() +
  theme(legend.position = "right") 

share_counts2 <- OnlineNewsPopularity %>%
  group_by(data_channel, shares_levels) %>%
  summarise(count = n()) %>%
  mutate(shares_levels = factor(shares_levels, levels = c("0", "1", "2")))

# Plot
ggplot(share_counts2, aes(x = data_channel, y = count, fill = shares_levels)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Count of shares_levels over different data channels",
       x = "Data Channel", y = "Count") +
  scale_fill_manual(values = c("yellow", "lightgreen", "orange"),
                    name = "Shares Levels",
                    labels = c("extremely bad", "majority", "extremely good")) +
  theme_minimal() +
  theme(legend.position = "right") 


#### normalize the numerical variables using Min Max scaling ####
# Identify numeric variables
numeric_vars <- sapply(OnlineNewsPopularity, is.numeric)

scaled_numeric_data <- as.data.frame(apply(OnlineNewsPopularity[, numeric_vars], MARGIN = 2, FUN = function(x)
  (x - min(x))/diff(range(x))))


scaled_numeric_data_long <- pivot_longer(scaled_numeric_data, cols = everything(), names_to = "variable")

# Create a boxplot for every variable
theme_set(theme_bw())
ggplot(scaled_numeric_data_long, aes(x = factor(variable), y = value)) +
  geom_boxplot() +
  facet_wrap(~ variable, scales = "free")+
  labs(title = "Boxplot - scaled data",x="variable") +
  theme(
    axis.text.x = element_text(size = 6),  # Adjust font size for x-axis text
    axis.text.y = element_text(size = 4),  # Adjust font size for y-axis text
    strip.text = element_text(size = 6)    # Adjust font size for facet labels
  )

summary(scaled_numeric_data)

# Combine scaled numeric data with non-numeric data
OnlineNewsPopularity_scaled <- cbind(scaled_numeric_data, OnlineNewsPopularity[!numeric_vars])

summary(OnlineNewsPopularity_scaled)

#-------------- Dimensionality Reduction -----------

OnlineNewsPopularity_scaled.ind_var<-OnlineNewsPopularity_scaled[, !colnames(OnlineNewsPopularity_scaled) %in% c("shares_levels")]

OnlineNewsPopularity_scaled.ind_var$data_channel <- as.numeric(OnlineNewsPopularity_scaled.ind_var$data_channel)
OnlineNewsPopularity_scaled.ind_var$weekday <- as.numeric(OnlineNewsPopularity_scaled.ind_var$weekday)
OnlineNewsPopularity_scaled.ind_var$is_weekend <- as.numeric(OnlineNewsPopularity_scaled.ind_var$is_weekend)


corrgram(cor(OnlineNewsPopularity_scaled.ind_var), 
         text.panel=function(...) {}, # Suppress text display
         order=FALSE, 
         cor.method = "pearson", 
         lower.panel=panel.conf,
         upper.panel=panel.pie, 
         main="Corrgram - Online New Popularity",
         addrect = 2)

corrplot(cor(OnlineNewsPopularity_scaled.ind_var), main="Corrplot - Online New Popularity  ", mar = c(0,0,3,0),type = "upper", tl.col = "black", tl.srt = 45,tl.pos = "n")

cor_matrix <- cor(OnlineNewsPopularity_scaled.ind_var)

# Set correlation threshold
threshold <- 0.7

# Identify highly correlated variable pairs
highly_correlated <- which(cor_matrix > threshold & cor_matrix < 1, arr.ind = TRUE)

print(highly_correlated) # Print highly correlated variable pairs

highly_correlated_rows <- highly_correlated[, "row"] # Extract the row indices of highly correlated variables

highly_correlated_vars <- names(highly_correlated_rows) # Extract the row names of highly correlated variables

# Remove one variable from each highly correlated pair
df_clean <- OnlineNewsPopularity_scaled.ind_var[, !(names(OnlineNewsPopularity_scaled.ind_var) %in% highly_correlated_vars)]

OnlineNewsPopularity_scaled.ind_var<-df_clean
# Calculate eigenvalues
ev <- eigen(cor(OnlineNewsPopularity_scaled.ind_var))
ev_values <- ev$values

# Plot a scree plot of eigenvalues
plot(ev_values, type="b", col="blue", xlab="Variables", ylab="Eigenvalues", main = "Scree Plot of Eigenvalues")

# Calculate cumulative proportion of eigenvalues and plot
ev_sum <- sum(ev_values)
cumulative_proportion <- cumsum(ev_values) / ev_sum
plot(cumulative_proportion, type="b", col="red", xlab="Number of components", ylab="Cumulative proportion",main="Cumulative Proportion of Variance Explained by PCA Components")

# Identify the "elbow" point to determine the number of factors/components to extract
elbow_point <- which(cumulative_proportion > 0.8)[1]  # Adjust threshold as needed
number_of_factors <- elbow_point
number_of_factors

# Perform PCA with the determined number of factors
pca_result <- principal(OnlineNewsPopularity_scaled.ind_var, nfactors = number_of_factors, rotate = "varimax")
pca_result
# Extract relevant information from pca_result
loadings_matrix <- pca_result$loadings
scores_matrix <- pca_result$scores
summary <- summary(pca_result)


selected_variables <- list()

# Loop through each principal component
for (i in 1:ncol(loadings_matrix)) {
  # Find the variable with the maximum loading (absolute value) for this component
  max_loading_index <- which.max(abs(loadings_matrix[, i]))
  
  # Get the name of the variable with the maximum loading
  selected_variable <- rownames(loadings_matrix)[max_loading_index]
  
  # Store the selected variable for this component
  selected_variables[[i]] <- selected_variable
}

print(selected_variables)

selected_variables<- c(selected_variables, "shares_levels")
OnlineNewsPopularity_scaled <- OnlineNewsPopularity_scaled[, unlist(selected_variables)]

#--------Section 05- train test splitting ------
# Set the seed for reproducibility
set.seed(123)
randomized_indices <- order(runif(nrow(OnlineNewsPopularity_scaled)))
OnlineNewsPopularity_scaled <- OnlineNewsPopularity_scaled[randomized_indices, ]

# Split the data set into 70% training and 30% testing
train_index <- createDataPartition(OnlineNewsPopularity_scaled$shares_levels, p = 0.7, list = FALSE)
train_data <- OnlineNewsPopularity_scaled[train_index, ]
test_data <- OnlineNewsPopularity_scaled[-train_index, ]
# check the proportion of class variable
round(prop.table(table(OnlineNewsPopularity_scaled$shares_levels))*100,digits = 2)
round(prop.table(table(train_data$shares_levels))*100, digits = 2)
round(prop.table(table(test_data$shares_levels))*100, digits = 2)

true_labels <- as.factor(test_data$shares_levels)

#----------section 06 - naive bayes --------------

set.seed(125)
nb_model <- naiveBayes(shares_levels ~ ., data = train_data)
summary(nb_model)

predictions_nb_model <- predict(nb_model, test_data)
crosstable_nb <- CrossTable(test_data$shares_levels, predictions_nb_model,
                            prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
                            dnn = c('Actual shares_levels', 'Predicted  shares_levels'))
conf_matrix_nb <- confusionMatrix(as.factor(predictions_nb_model), as.factor(test_data$shares_levels),mode = "everything")
conf_matrix_nb
prob_nb <- as.data.frame(predict(nb_model, test_data, type = "raw"))
res_nb <- as.data.frame(cbind(predictions_nb_model, prob_nb))
head(res_nb)

output <- capture.output(suppressWarnings(auc_values_nb<-compute_roc_auc(res_nb, true_labels," - Naive Bayes")))
filtered_output <- output[!grepl("Setting levels: control = 0, case =", output)]
cat("\nAUC Values - Naive Bayes",filtered_output, sep = "\n")

metrics_nb <- compute_classification_metrics(conf_matrix_nb, predictions_nb_model, true_labels)


for (i in 1:length(metrics_nb$class_metrics)) {
  if (i == 1) {
    writeLines("\nClass-wise metrics of Naive Bayes:\n")
    cat("Class", i-1, "Precision:", paste0(round(metrics_nb$class_metrics[[i]]$precision*100,2),"%","\n"))
    cat("Class", i-1, "Recall:", paste0(round(metrics_nb$class_metrics[[i]]$recall*100,2),"%", "\n"))
    cat("Class", i-1, "F1 Score:", paste0(round(metrics_nb$class_metrics[[i]]$f1_score*100,2),"%", "\n"))
  } else {
    cat("\nClass", i-1, "Precision:", paste0(round(metrics_nb$class_metrics[[i]]$precision*100,2),"%","\n"))
    cat("Class", i-1, "Recall:", paste0(round(metrics_nb$class_metrics[[i]]$recall*100,2),"%", "\n"))
    cat("Class", i-1, "F1 Score:", paste0(round(metrics_nb$class_metrics[[i]]$f1_score*100,2),"%", "\n"))
  }
}

cat(
  "\n Evaluation Metrics of Naive Bayes:\n \n",
  "Weighted-average Precision:", paste0(round(metrics_nb$weighted_precision*100,2),"%"), "\n",
  "Weighted-average Recall:", paste0(round(metrics_nb$weighted_recall*100,2),"%"), "\n",
  "Weighted-average F1 Score:", paste0(round(metrics_nb$weighted_f1_score*100,2),"%"), "\n",
  "Accuracy:", paste0(round(metrics_nb$accuracy*100,2),"%"), "\n",
  "Kappa Statistic:", paste0(round(metrics_nb$kappa_statistic*100,2),"%"), "\n"
)

#--------Section 07 random forest ----------

set.seed(12345)
rf_model <- randomForest(shares_levels ~ ., data = train_data, importance=TRUE, cores = 6)
#summary of the model
rf_model
varImpPlot(rf_model, main = "rf - variable importance")
vip(rf_model)
p_rf <- predict(rf_model, test_data)
crosstable_rf <- CrossTable(test_data$shares_levels, p_rf,
                            prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
                            dnn = c('Actual shares_levels', 'Predicted  shares_levels'))

confusion_matrix_rf <- confusionMatrix(p_rf, test_data$shares_levels, mode= "everything")
confusion_matrix_rf
prob_rf <- as.data.frame(predict(rf_model, test_data, type = "prob"))
res_rf <- as.data.frame(cbind(p_rf, prob_rf))
head(res_rf)
output <- capture.output(suppressWarnings(auc_values_rf <-compute_roc_auc(res_rf, true_labels," - Random Forest")))
filtered_output <- output[!grepl("Setting levels: control = 0, case =", output)]
cat("\nAUC Values - Random Forest",filtered_output, sep = "\n")

metrics_rf <- compute_classification_metrics(confusion_matrix_rf, p_rf, true_labels)


for (i in 1:length(metrics_rf$class_metrics)) {
  if (i == 1) {
    writeLines("\nClass-wise metrics of Random Forest:\n")
    cat("Class", i-1, "Precision:", paste0(round(metrics_rf$class_metrics[[i]]$precision*100,2),"%","\n"))
    cat("Class", i-1, "Recall:", paste0(round(metrics_rf$class_metrics[[i]]$recall*100,2),"%", "\n"))
    cat("Class", i-1, "F1 Score:", paste0(round(metrics_rf$class_metrics[[i]]$f1_score*100,2),"%", "\n"))
  } else {
    cat("\nClass", i-1, "Precision:", paste0(round(metrics_rf$class_metrics[[i]]$precision*100,2),"%","\n"))
    cat("Class", i-1, "Recall:", paste0(round(metrics_rf$class_metrics[[i]]$recall*100,2),"%", "\n"))
    cat("Class", i-1, "F1 Score:", paste0(round(metrics_rf$class_metrics[[i]]$f1_score*100,2),"%", "\n"))
  }
}
cat(
  "\n Evaluation Metrics of Random Forest:\n \n",
  "Weighted-average Precision:", paste0(round(metrics_rf$weighted_precision*100,2),"%"), "\n",
  "Weighted-average Recall:", paste0(round(metrics_rf$weighted_recall*100,2),"%"), "\n",
  "Weighted-average F1 Score:", paste0(round(metrics_rf$weighted_f1_score*100,2),"%"), "\n",
  "Accuracy:", paste0(round(metrics_rf$accuracy*100,2),"%"), "\n",
  "Kappa Statistic:", paste0(round(metrics_rf$kappa_statistic*100,2),"%"), "\n"
)

#------Section 08 AdaBoost --------------------------------
# Train the AdaBoost model
n_estimators <- 50

# Fit AdaBoost model with parallelization
ada_model <- boosting(shares_levels ~ ., data = train_data,
                  boos = TRUE, mfinal = n_estimators, 
                  control = rpart.control(cp = -1, xval = 10), OOB = TRUE, B = 50, ntree = 100, cores = 8)

summary(ada_model)

# Make predictions on the test data
predictions_ada <- predict(ada_model, test_data)

# Cross-tabulation
crosstable_ada <- CrossTable(test_data$shares_levels, predictions_ada$class,
                             prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
                             dnn = c('Actual shares_levels', 'Predicted shares_levels'))
# Confusion matrix
conf_matrix_ada <- confusionMatrix(as.factor(predictions_ada$class), test_data$shares_levels,
                                   mode = "everything")
conf_matrix_ada
prob_adaboost <- t(sapply(predictions_ada$prob, function(x) c(1 - sum(x), x)))
res_ada <- as.data.frame(cbind(predictions_ada$class, predictions_ada$prob))

print(head(res_ada))

res_ada <- data.frame(
  p_ada = as.factor(res_ada$V1),
  `0` = as.numeric(res_ada$`V2`),
  `1` = as.numeric(res_ada$`V3`),
  `2` = as.numeric(res_ada$`V4`)
)
names(res_ada) <- c("p_ada", "0", "1", "2")

print(head(res_ada))
output <- capture.output(suppressWarnings(auc_values_ada <- compute_roc_auc(res_ada, true_labels," - AdaBoost")))
filtered_output <- output[!grepl("Setting levels: control = 0, case =", output)]
cat("\nAUC Values - AdaBoost",filtered_output, sep = "\n")

metrics_ada <- compute_classification_metrics(conf_matrix_ada, predictions_ada, true_labels)


for (i in 1:length(metrics_ada$class_metrics)) {
  if (i == 1) {
    writeLines("\nClass-wise metrics of AdaBoost:\n")
    cat("Class", i-1, "Precision:", paste0(round(metrics_ada$class_metrics[[i]]$precision*100,2),"%","\n"))
    cat("Class", i-1, "Recall:", paste0(round(metrics_ada$class_metrics[[i]]$recall*100,2),"%", "\n"))
    cat("Class", i-1, "F1 Score:", paste0(round(metrics_ada$class_metrics[[i]]$f1_score*100,2),"%", "\n"))
  } else {
    cat("\nClass", i-1, "Precision:", paste0(round(metrics_ada$class_metrics[[i]]$precision*100,2),"%","\n"))
    cat("Class", i-1, "Recall:", paste0(round(metrics_ada$class_metrics[[i]]$recall*100,2),"%", "\n"))
    cat("Class", i-1, "F1 Score:", paste0(round(metrics_ada$class_metrics[[i]]$f1_score*100,2),"%", "\n"))
  }
}
cat(
  "\n Evaluation Metrics of AdaBoost:\n \n",
  "Weighted-average Precision:", paste0(round(metrics_ada$weighted_precision*100,2),"%"), "\n",
  "Weighted-average Recall:", paste0(round(metrics_ada$weighted_recall*100,2),"%"), "\n",
  "Weighted-average F1 Score:", paste0(round(metrics_ada$weighted_f1_score*100,2),"%"), "\n",
  "Accuracy:", paste0(round(metrics_ada$accuracy*100,2),"%"), "\n",
  "Kappa Statistic:", paste0(round(metrics_ada$kappa_statistic*100,2),"%"), "\n"
)

#------Section 09 XGBoost --------------------------------

str(train_data)
trn<-train_data
tst<-test_data
trn$is_weekend <- as.numeric(trn$is_weekend)
tst$is_weekend <- as.numeric(tst$is_weekend)

trn$weekday <- as.numeric(trn$weekday)
tst$weekday <- as.numeric(tst$weekday)

trn$shares_levels <- as.numeric(trn$shares_levels)-1
tst$shares_levels <- as.numeric(tst$shares_levels)-1

train_matrix <- xgb.DMatrix(data = as.matrix(trn[, !(names(trn) %in% "shares_levels")]), label = trn$shares_levels)
test_matrix <- xgb.DMatrix(data = as.matrix(tst[, !(names(tst) %in% "shares_levels")]), label = tst$shares_levels)


# K-folds Cross-validation to Estimate Error
numberOfClasses <- length(unique(OnlineNewsPopularity_scaled$shares_levels))
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses)
nround    <- 50
cv.nfold  <- 5

cv_model <- xgb.cv(params = xgb_params,
                   data = train_matrix, 
                   nrounds = nround,
                   nfold = cv.nfold,
                   verbose = FALSE,
                   prediction = TRUE)

# Assess Out-of-Fold Prediction Error
OOF_prediction <- data.frame(cv_model$pred) %>%
  mutate(max_prob = max.col(., ties.method = "last"),
         label = trn$shares_levels+1)


confusionMatrix(factor(OOF_prediction$max_prob),
                factor(OOF_prediction$label),
                mode = "everything")

# Train Full Model and Assess Test Set Error
xgb_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = nround)

xgb_model

pred_xgb <- predict(xgb_model, newdata = test_matrix)
xgb_test_prediction <- matrix(pred_xgb, nrow = numberOfClasses,
                          ncol = length(pred_xgb) / numberOfClasses) %>%
  t() %>%
  data.frame() %>%
  mutate(label = tst$shares_levels+1,
         max_prob = max.col(., "last"))


cross_table_xgb <- CrossTable(x = xgb_test_prediction$max_prob, y = xgb_test_prediction$label,
                              prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
                              dnn = c('Actual shares_levels', 'Predicted shares_levels'))

conf_matrix_xgb <- confusionMatrix(factor(xgb_test_prediction$max_prob),
                factor(xgb_test_prediction$label),
                mode = "everything")
conf_matrix_xgb
# Variable Importance
names <- colnames(trn[, -1])
importance_matrix <- xgb.importance(feature_names = names, model = xgb_model)

gp <- xgb.ggplot.importance(importance_matrix)
print(gp)


res_xgb <- matrix(pred_xgb, nrow = numberOfClasses,
                          ncol = length(pred_xgb) / numberOfClasses) %>%
  t() %>%
  data.frame() %>%
  mutate(
         max_prob = max.col(., "last")
         ) %>%
           select(max_prob, everything())
names(res_xgb) <- c("p_xgb", "0", "1", "2")

output <- capture.output(suppressWarnings(auc_values_xgb <- compute_roc_auc(res_xgb, true_labels," - XGBoost")))
filtered_output <- output[!grepl("Setting levels: control = 0, case =", output)]
cat("\nAUC Values - XGBoost",filtered_output, sep = "\n")

metrics_xgb <- compute_classification_metrics(conf_matrix_xgb, pred_xgb, true_labels)

for (i in 1:length(metrics_xgb$class_metrics)) {
  if (i == 1) {
    writeLines("\nClass-wise metrics of XGBoost:\n")
    cat("Class", i-1, "Precision:", paste0(round(metrics_xgb$class_metrics[[i]]$precision*100,2),"%","\n"))
    cat("Class", i-1, "Recall:", paste0(round(metrics_xgb$class_metrics[[i]]$recall*100,2),"%", "\n"))
    cat("Class", i-1, "F1 Score:", paste0(round(metrics_xgb$class_metrics[[i]]$f1_score*100,2),"%", "\n"))
  } else {
    cat("\nClass", i-1, "Precision:", paste0(round(metrics_xgb$class_metrics[[i]]$precision*100,2),"%","\n"))
    cat("Class", i-1, "Recall:", paste0(round(metrics_xgb$class_metrics[[i]]$recall*100,2),"%", "\n"))
    cat("Class", i-1, "F1 Score:", paste0(round(metrics_xgb$class_metrics[[i]]$f1_score*100,2),"%", "\n"))
  }
}
cat(
  "\n Evaluation Metrics of XGBoost:\n \n",
  "Weighted-average Precision:", paste0(round(metrics_xgb$weighted_precision*100,2),"%"), "\n",
  "Weighted-average Recall:", paste0(round(metrics_xgb$weighted_recall*100,2),"%"), "\n",
  "Weighted-average F1 Score:", paste0(round(metrics_xgb$weighted_f1_score*100,2),"%"), "\n",
  "Accuracy:", paste0(round(metrics_xgb$accuracy*100,2),"%"), "\n",
  "Kappa Statistic:", paste0(round(metrics_xgb$kappa_statistic*100,2),"%"), "\n"
)

#---------------Model Comparison -------------------------

models <- c("Naive Bayes","Random Forest", "AdaBoost", "XGBoost")
precision <- c(metrics_nb$weighted_precision,metrics_rf$weighted_precision, metrics_ada$weighted_precision,metrics_xgb$weighted_precision)
recall <- c(metrics_nb$weighted_recall,metrics_rf$weighted_recall, metrics_ada$weighted_recall,metrics_xgb$weighted_recall)
f1_score <- c(metrics_nb$weighted_f1_score,metrics_rf$weighted_f1_score, metrics_ada$weighted_f1_score,metrics_xgb$weighted_f1_score)
accuracy <- c(metrics_nb$accuracy,metrics_rf$accuracy, metrics_ada$accuracy,metrics_xgb$accuracy)
model_metrics <- data.frame(models, precision, recall, accuracy, f1_score)

model_metrics_long <- gather(model_metrics, metric, value, -models)
model_metrics_long$value <- model_metrics_long$value * 100

ggplot(model_metrics_long, aes(x = models, y = value, fill = metric)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.77) +
  geom_text(aes(label = paste0(round(value, 2), "%")), 
            vjust = -0.5, 
            size = 3,
            position = position_dodge(width = 0.8)) +
  labs(title = "Model Evaluation Metrics",
       x = "Models",
       y = "Score") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(breaks = seq(0, 100, by = 10))

#-----Section 11-------------------------------------------
# remove all variables from the environment
rm(list=ls())

