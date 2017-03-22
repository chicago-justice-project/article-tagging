library(tidyverse)

"Models explore"
# Explore results from 
# Load result files from loop in quant_justice_models.R
algorithm_summaries <- read.csv("algorith_summaries_031117.csv", stringsAsFactors = F)
ensemble_summaries <- read.csv("ensemble_summaries_031107.csv", stringsAsFactors = F)
algorithm_summaries_detailed <- read.csv("model_performance_measures_031107_cleaned.csv", stringsAsFactors = F) #NOTE: total_positive and total_negative values off for RANDOM FOREST ENSEMBLE
ensemble_summaries_detailed <- read.csv("ensemble_summaries_more_detail_031117_cleaned.csv", stringsAsFactors = F) 


# Clean
ensemble_summaries_detailed[ensemble_summaries_detailed == 99999] <- NA
algorithm_summaries_detailed[colnames(select(algorithm_summaries_detailed, num_articles_predicted:Recall_for_max_Matt_coef))] <- sapply(algorithm_summaries_detailed[colnames(select(algorithm_summaries_detailed, num_articles_predicted:Recall_for_max_Matt_coef))], as.numeric)
algorithm_summaries_detailed[algorithm_summaries_detailed == 99999.000] <- NA


# Example explorations
View(ensemble_summaries_detailed %>% group_by(crime_category) %>% summarise(mean_F_score = mean(F_Score, na.rm = T),
                                                                     max_F_score = max(F_Score, na.rm = T),
                                                                     mean_accuracy = mean(Accuracy, na.rm = T),
                                                                     max_accuracy = max(Accuracy, na.rm = T)))

View(algorithm_summaries_detailed %>% group_by(model) %>% summarise(mean(AUC, na.rm = T), max(AUC, na.rm = T), 
                                                      mean(Max_F_score, na.rm = T), max(Max_F_score, na.rm = T),
                                                      mean(Max_Accuracy, na.rm = T), max(Max_Accuracy, na.rm = T),
                                                      mean(Max_Matt_coef, na.rm = T), max(Max_Matt_coef, na.rm = T)))

View(algorithm_summaries_detailed %>% group_by(crime_category) %>% summarise(mean(AUC, na.rm = T), max(AUC, na.rm = T), 
                                                      mean(Max_F_score, na.rm = T), max(Max_F_score, na.rm = T),
                                                      mean(Max_Accuracy, na.rm = T), max(Max_Accuracy, na.rm = T),
                                                      mean(Max_Matt_coef, na.rm = T), max(Max_Matt_coef, na.rm = T)))