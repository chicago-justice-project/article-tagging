

#Created for second loop

library(tidyverse)
library(tm)
library(qdap)
library(RTextTools)
library(SnowballC)
library(caret)
library(pROC)
library(ROCR)

"code in this section brings in the data and merges with other dfs from news_categorization db"
#############

#bring articles into R -- note that R will want to set a default value for comment.char (comment.chare = #) -- do not use this!
newsarticles_article <- read.csv("newsarticles_article.csv", header=FALSE, stringsAsFactors=FALSE) #not included because it's 1.24GB
#create column names
colnames(newsarticles_article) <- c("id", "feedname", "url","orig_html","title","bodytext","relevant","created", "last_modified")

#bring in news_article_ids and respective codes
news_id_and_codes <- read.csv("newsarticles_article_categories.csv", header=FALSE, stringsAsFactors=FALSE)
colnames(news_id_and_codes) <- c("V1", "article_id", "category_id")

news_id_and_codes_ag <- news_id_and_codes %>% 
      group_by(article_id) %>%
      summarise(category_ids = paste(category_id, collapse = ","))

dim(news_id_and_codes_ag)

#keep ALL, even if not tagged with a category code -- subset below if needed
df_news <- merge(newsarticles_article[c("id", "title", "bodytext", "relevant", "feedname","url","created","last_modified")], news_id_and_codes_ag, by.x = "id", by.y = "article_id", all.x = T)

rm(newsarticles_article) #too big, get it out of memory

#bring in category names, one per column
newsarticles_category_names <- read.csv("newsarticles_category.csv", header=FALSE, stringsAsFactors=FALSE)
newsarticles_category_names <- newsarticles_category_names[-4]
colnames(newsarticles_category_names) <- c("category_id", "category_name", "category_abbreviation")

#add columns for each category
for (i in 1:nrow(newsarticles_category_names)) {
      col_name <- paste("code", i,newsarticles_category_names[i,"category_name"], sep = "_")
      df_news[col_name] <- ""
}


#this creates a binomial (1 = yes, 2 = no) for each category, using the 'category_ids' column
cat_names <- names(select(df_news, `code_1_Office of Emergency Management & Communications`:`code_38_Police Use of Taser`)) #extract col names needed for loop, below

for (i in cat_names) {
      num_to_find <- strsplit(i, "_")
      num_to_find <- num_to_find[[1]][2]
      df_news[i] <- ifelse(grepl(paste("\\<",num_to_find,"\\>", sep=""),df_news$category_ids), 1, 0) #exact match
}


###############
"CHOOSE DF to use"
#remove unneeded observations
##keep only TAGGED articles 
df_news <- df_news[which(!is.na(df_news$category_ids)),]

##keep all RELEVANT articles -- NOTE: the relevant and UNTAGGED articles can be used as TEST articles
#df_news <- df_news[which(!is.na(df_news$category_ids)),]

gc() #clean up memory

"Quick look at crime category totals"
View(as.data.frame(colSums(select(df_news, `code_1_Office of Emergency Management & Communications`:`code_38_Police Use of Taser`))))

"Create train/test subset"
# create smaller data frame of n articles randomly selected from training/test data
set.seed(101)
test_df <- df_news[sample(1:nrow(df_news),4000), ]
test_df <- test_df[order(test_df$id),] #order so that the IDs match when predicting, below

"Add date features"
# create date var, first split after the "%Y-%d-%m"
test_df$Full_date <- as.Date(test_df$created, format="%Y-%m-%d")
test_df$Year <- as.numeric(format(test_df$Full_date, "%Y"))
test_df$Month <- format(test_df$Full_date,"%B")
test_df$Weekday <- weekdays(test_df$Full_date)
test_df$Week<- format(test_df$Full_date, "%W")  

"Text preprocessing"
# Combine the title and body of article
df <- data.frame(ID = test_df$id,
                 txt = paste(test_df$title,test_df$bodytext, sep=""))
mycorpus <- with(df, as.Corpus(txt, ID))

# Pre-processing prep: make all lowercase, remove punctuation, remove numbers, strip whitespace, remove common skipwords 
skipWords <- function(x) removeWords(x, c(stopwords("english"),qdapDictionaries::Top100Words))
funcs <- list(content_transformer(tolower), removePunctuation, removeNumbers, stripWhitespace, skipWords) 

# Preprocess articles without stemming
a <- tm_map(mycorpus, FUN = tm_reduce, tmFuns = funcs)

# Proprocess articles WITH stemming
## NOTE: In order to work, must load the SnowballC package
b <- tm_map(mycorpus, FUN = tm_reduce, tmFuns = funcs,mc.cores=1)
b <- tm_map(b, stemDocument, mc.cores=1)

"Create document term matrices"

#  
mydtm <- DocumentTermMatrix(a, control = list(wordLengths = c(3,12))) #term freqs no stemming
mydtm_stem <- DocumentTermMatrix(b, control = list(wordLengths = c(3,12))) #term freqs with stemming
mydtm_w_tfidf_norm <- DocumentTermMatrix(a, control = list(wordLengths = c(3,12),weighting = function(x) weightTfIdf(x, normalize = T))) #tf idf no stemming
mydtm_stem_w_tfidf <- DocumentTermMatrix(b, control = list(wordLengths = c(3,12),weighting = function(x) weightTfIdf(x, normalize = T))) #tf idf with stemming

"Loop through containers to create all desired DTM variations by assigned (i) levels of sparsity"
## for first long loop used c(.98,.985, .99, .995, .998) #took 30 hours to run with no real benefit 
## of .998 sparsity (more terms) vs. .98 sparsity and .998 took way longer to run
dflist= list()
count = 0
for (i in .98){ #http://stackoverflow.com/questions/19209604/creating-a-multiple-data-frames-in-for-loop
      count = count + 1
      dflist[[count]] <- assign(paste0("mydtm_sparse_", i),removeSparseTerms(mydtm, i))
      names(dflist)[count]<-paste0("mydtm_sparse_", i)
      count = count + 1
      dflist[[count]] <- assign(paste0("mydtm_stem_sparse_", i),removeSparseTerms(mydtm_stem, i))
      names(dflist)[count]<-paste0("mydtm_stem_sparse_", i)
      count = count + 1
      dflist[[count]] <- assign(paste0("mydtm_w_tfidf_sparse_", i),removeSparseTerms(mydtm_w_tfidf_norm, i))
      names(dflist)[count]<-paste0("mydtm_w_tfidf_sparse_", i)
      count = count + 1
      dflist[[count]] <-  assign(paste0("mydtm_stem_w_tfidf_sparse_", i),removeSparseTerms(mydtm_stem_w_tfidf, i))
      names(dflist)[count]<-paste0("mydtm_stem_w_tfidf_sparse_", i)
}

# check # of features for each df created in the previous loop
sapply(dflist, dim) 



"Function that gives model performance measures"
# RORC library required
get_performance_measures <- function(predicted, actual) { 
      f_score <- performance(prediction(predicted,actual), "f")
      f_score <- data.frame(f_score@x.values, f_score@y.values)
      f_score <- f_score[!is.na(f_score[,2]),]
      f_score <- f_score[is.finite(f_score[,1]),]
      f_score <- if(nrow(f_score)==0) {data.frame(col_1 = 99999, col_2 = 99999)} else {f_score[which(max(f_score[,2])==f_score[2]),]}
      names(f_score) <- c("optimal_cutoff_point", "max_f_score")
      prec_score <- performance(prediction(predicted,actual), "prec")
      prec_score <- data.frame(prec_score@x.values, prec_score@y.values)
      prec_score <- prec_score[!is.na(prec_score[,2]),]
      prec_score <- prec_score[is.finite(prec_score[,1]),]
      prec_score <- if(nrow(prec_score)==0) {data.frame(col_1 = 99999, col_2 = 99999)} else {prec_score}
      #prec_score_for_max_f_score <- prec_score[which(prec_score[,1]==f_score[1,1]),2]
      prec_score_for_max_f_score <- if(prec_score[1,1]==99999 | f_score[1,1]==99999) {99999} else {prec_score[which(prec_score[,1]==f_score[1,1]),2]}
      recall_score <- performance(prediction(predicted,actual), "rec")
      recall_score <- data.frame(recall_score@x.values, recall_score@y.values)
      recall_score <- recall_score[!is.na(recall_score[,2]),]
      recall_score <- recall_score[is.finite(recall_score[,1]),]
      recall_score <- if(nrow(recall_score)==0) {data.frame(col_1 = 99999, col_2 = 99999)} else {recall_score}
      #recall_score_for_max_f_score <- recall_score[which(recall_score[,1]==f_score[1,1]),2]
      recall_score_for_max_f_score <- if(recall_score[1,1]==99999 | f_score[1,1]==99999) {99999} else {recall_score[which(recall_score[,1]==f_score[1,1]),2]}
      acc_score <- performance(prediction(predicted,actual), "acc")
      acc_score <- data.frame(acc_score@x.values, acc_score@y.values)
      acc_score <- acc_score[!is.na(acc_score[,2]),]
      acc_score <- acc_score[is.finite(acc_score[,1]),]
      acc_score <- if(nrow(acc_score)==0) {data.frame(col_1 = 99999, col_2 = 99999)} else {acc_score[which(max(acc_score[,2])==acc_score[2]),]}
      # acc_score <- acc_score[which(max(acc_score[,2])==acc_score[2]),]
      # prec_score_for_max_accuracy <- prec_score[which(prec_score[,1]==acc_score[1,1]),2]
      # recall_score_for_max_accuracy <- recall_score[which(recall_score[,1]==acc_score[1,1]),2]
      prec_score_for_max_accuracy <- if(prec_score[1,1]==99999 | acc_score[1,1]==99999) {99999} else {prec_score[which(prec_score[,1]==acc_score[1,1]),2]}
      recall_score_for_max_accuracy <- if(recall_score[1,1]==99999 | acc_score[1,1]==99999) {99999} else {recall_score[which(recall_score[,1]==acc_score[1,1]),2]}
      names(acc_score) <- c("optimal_cutoff_point", "max_accuracy")
      mat_score <- performance(prediction(predicted,actual), "phi") #Matthews correlation coefficient. same as phi
      mat_score <- data.frame(mat_score@x.values, mat_score@y.values)
      mat_score <- mat_score[!is.na(mat_score[,2]),]
      mat_score <- mat_score[is.finite(mat_score[,1]),]
      mat_score <- if(nrow(mat_score)==0) {data.frame(col_1 = 99999, col_2 = 99999)} else {mat_score[which(max(mat_score[,2])==mat_score[2]),]}
      #mat_score <- mat_score[which(max(mat_score[,2])==mat_score[2]),]
      prec_score_for_max_mat_coef <- if(prec_score[1,1]==99999 | mat_score[1,1]==99999) {99999} else {prec_score[which(prec_score[,1]==mat_score[1,1]),2]}
      recall_score_for_max_mat_coef <- if(recall_score[1,1]==99999 | mat_score[1,1]==99999) {99999} else {recall_score[which(recall_score[,1]==mat_score[1,1]),2]}
      names(mat_score) <- c("optimal_cutoff_point", "max_accuracy")
      auc_score <- if(sum(table(actual)[1:2]==0)>0) {99999} else {performance(prediction(predicted,actual), "auc")} #the `sum(table(actual)[1:2]==0)>0` returns TRUE if there is either all 0s or all 1s
      auc_score <- if(sum(table(actual)[1:2]==0)>0) {auc_score} else {auc_score@y.values}
      auc_score <- if(sum(table(actual)[1:2]==0)>0) {auc_score} else {as.numeric(auc_score)}
      da_measures <- data.frame("model_AUC" = auc_score,
                                "cutoff_for_max_f_score" = f_score[1,1], "max_f_score" = f_score[1,2],
                                "precision_for_max_f_score" = prec_score_for_max_f_score, "recall_for_max_f_score" = recall_score_for_max_f_score,
                                "cutoff_for_max_accuracy" = acc_score[1,1], "max_accuracy" = acc_score[1,2],
                                "precision_for_max_accuracy" = prec_score_for_max_accuracy, "recall_for_max_accuracy" = recall_score_for_max_accuracy,
                                "cutoff_for_max_matt_coef" = mat_score[1,1], "max_matt_coef" = mat_score[1,2],
                                "precision_for_max_matt_coef" = prec_score_for_max_mat_coef, "recall_for_max_matt_coef" = recall_score_for_max_mat_coef)
      da_measures
}  

"Create the probability of positive tag for each model"
# initial probabilty comes from analytics@document_summary output from RTextTools create_analytics function
new_prob <- function (df, label, prob){
      ifelse(df[,label] == 1, round(df[,prob],3), round(1 - df[,prob],3))
}


"Default model hyperparmetersnote for RTextTools train_model function"
# train_model(container, algorithm=c("SVM","SLDA","BOOSTING","BAGGING",
# "RF","GLMNET","TREE","NNET","MAXENT"), method = "C-classification",
# cross = 0, cost = 100, kernel = "radial", maxitboost = 100,
# maxitglm = 10^5, size = 1, maxitnnet = 1000, MaxNWts = 10000,
# rang = 0.1, decay = 5e-04, trace=FALSE, ntree = 200,
# l1_regularizer = 0, l2_regularizer = 0, use_sgd = FALSE,
# set_heldout = 0, verbose = FALSE,
# ...)


"Loop through specified models for each DTM for each crime category"
# NOTE: this with only 4 DTMS this took nearly 10 hours to run

# record start time -- 
ptv <- proc.time()

# create shell dfs 
pm_df <- ""
algorithm_summaries <- ""
ensemble_summaries <- ""
all_measures_stacked<- data.frame(df_name = "",crime_category = "", ensemble_model = "", num_articles_predicted = 222,
                                  Precision = 222, Recall = 2222,
                                  F_Score = 222, Accuracy = 222,
                                  True_Negative_n = 222, False_Negative_n = 222, 
                                  True_Positive_n_ = 222, False_Positive_ = 222) 

count <- 0

# here we go...!
for(i in dflist){
      count <- count + 1
      for(z in colnames(select(df_news, starts_with("code")))) { #
            if(sum(test_df[,z]==1)<10) {next}
            container <- create_container(i, test_df[,z], trainSize=1:(.75*(nrow(i))),testSize=((.75*(nrow(i)))+1):nrow(i), virgin=FALSE)
            print(z)
            ptm <- proc.time()
            models <- train_models(container, algorithms=c("SVM","GLMNET","RF", "MAXENT")) 
            results <- classify_models(container, models)
            analytics <- create_analytics(container, results)
            #summary(analytics)
            topic_summary <- analytics@label_summary
            alg_summary <- analytics@algorithm_summary
            ens_summary <-analytics@ensemble_summary
            doc_summary <- analytics@document_summary
            print(proc.time() - ptm)
            algorithm_summary <- as.data.frame(t(as.data.frame(analytics@algorithm_summary)))
            #algorithm_summary$ids <- paste0(row.names(algorithm_summary),"_",  names(dflist)[count] ,"_", z)#MUST FIX THIS
            algorithm_summary$model_type <- row.names(algorithm_summary)
            algorithm_summary$df_name <- names(dflist)[count]
            algorithm_summary$crime_category <- z
            algorithm_summaries <- rbind(algorithm_summaries, algorithm_summary)
            ensemble_summary <- as.data.frame(analytics@ensemble_summary)
            #TODO -- bring in COUNTS -- see below how I did this with dplyr
            #ensemble_summary$ids <- paste0(row.names(ensemble_summary),"_",  names(dflist)[count] ,"_", z)#MUST FIX THIS
            ensemble_summary$model_type <- row.names(ensemble_summary)
            ensemble_summary$df_name <- names(dflist)[count]
            ensemble_summary$crime_category <- z
            ensemble_summaries <- rbind(ensemble_summaries, ensemble_summary)
            
            #created for loop
            mod_names <- data.frame(m_name = colnames(select(doc_summary, contains("LABEL"))))
            mod_names<-mod_names %>% separate(col = m_name,into = c("name","label"), sep = "_", remove = F)
            for (j in mod_names$name) {
                  doc_summary[,paste(j,"_PROB_1", sep="")] <- new_prob(doc_summary, paste(j,"_LABEL", sep=""), paste(j,"_PROB", sep=""))
            }
            #add performance measures
            model_names <- colnames(select(doc_summary, ends_with("PROB_1")))
            for (k in model_names) { #NEED to add in the dtm name in the pm_df rbind
                  pm <- get_performance_measures(doc_summary[, k],doc_summary$MANUAL_CODE)
                  pm_df <- rbind(pm_df, data.frame(df_name = names(dflist)[count], crime_category = z, Algorithm_name = k, num_articles_predicted = nrow(doc_summary),
                                                   AUC = round(pm$model_AUC,3), Cutoff_for_max_F_score = round(pm$cutoff_for_max_f_score,3), Max_F_score = round(pm$max_f_score,3),
                                                   Precision_for_max_F_score = round(pm$precision_for_max_f_score,3), Recall_for_max_F_score = round(pm$recall_for_max_f_score,3),
                                                   Cutoff_for_max_Accuracy = round(pm$cutoff_for_max_accuracy,3), Max_Accuracy = round(pm$max_accuracy,3),
                                                   Precision_for_max_Accuracy = round(pm$precision_for_max_accuracy,3), Recall_for_max_Accuracy = round(pm$recall_for_max_accuracy,3),
                                                   Cutoff_for_max_matt_coef = round(pm$cutoff_for_max_matt_coef,3), Max_Matt_coef = round(pm$max_matt_coef,3),
                                                   Precision_for_max_Matt_coef = round(pm$precision_for_max_matt_coef,3), Recall_for_max_Matt_coef = round(pm$recall_for_max_matt_coef,3)))
            }
            
            #do random forest on predictions -- much more sophisticated than potentially arbitrary cut-off at greater than 50% probability for 1 or more of the models
            ##create validation (named test) set after cross-validation is done
            set.seed(2031)
            rows <- sample(nrow(doc_summary))
            doc_test <- doc_summary[rows,]
            doc_test$MANUAL_CODE <- as.factor(as.character(doc_test$MANUAL_CODE))
            levels(doc_test$MANUAL_CODE) <- make.names(levels(factor(doc_test$MANUAL_CODE)))
            split <- round(nrow(doc_test) * .75)
            train <- doc_test[1:split,]
            test <- doc_test[(split+1):nrow(doc_test),]
            myControl <- trainControl( 
                  method = "cv",
                  number = 4,
                  summaryFunction = twoClassSummary,
                  classProbs = TRUE, # IMPORTANT!
                  verboseIter = TRUE
            )
            model <- train(
                  MANUAL_CODE ~.,
                  #tuneGrid = data.frame(mtry = c(2,3,7)),
                  data = select(train,one_of("MANUAL_CODE"), ends_with("PROB_1")) %>% mutate_each(funs(as.factor), one_of("MANUAL_CODE")), 
                  method = "ranger",
                  trControl = myControl
            )

            ##use predicted probabilities to be able to get performance measures
            g <- predict(model, select(test,one_of("MANUAL_CODE"), ends_with("PROB_1")) %>% mutate_each(funs(as.factor), one_of("MANUAL_CODE")), type = "prob")  
            ##Add performance measures for the RF ensemble
            rf_pm <-get_performance_measures(g$X1,test$MANUAL_CODE)
            pm_df <- rbind(pm_df, data.frame(df_name = names(dflist)[count], crime_category = z, Algorithm_name = "Random_Forest_ensemble",num_articles_predicted = nrow(test), 
                                             AUC = round(rf_pm$model_AUC,3), Cutoff_for_max_F_score = round(rf_pm$cutoff_for_max_f_score,3), Max_F_score = round(rf_pm$max_f_score,3),
                                             Precision_for_max_F_score = round(rf_pm$precision_for_max_f_score,3), Recall_for_max_F_score = round(rf_pm$recall_for_max_f_score,3),
                                             Cutoff_for_max_Accuracy = round(rf_pm$cutoff_for_max_accuracy,3), Max_Accuracy = round(rf_pm$max_accuracy,3),
                                             Precision_for_max_Accuracy = round(rf_pm$precision_for_max_accuracy,3), Recall_for_max_Accuracy = round(rf_pm$recall_for_max_accuracy,3),
                                             Cutoff_for_max_matt_coef = round(rf_pm$cutoff_for_max_matt_coef,3), Max_Matt_coef = round(rf_pm$max_matt_coef,3),
                                             Precision_for_max_Matt_coef = round(rf_pm$precision_for_max_matt_coef,3), Recall_for_max_Matt_coef = round(rf_pm$recall_for_max_matt_coef,3)))
            
            #creates new ensemble consensus VARS -- in order to ID which thresholds create highest performing ensemble
            for (m in seq(.2, .8, .1)) {
                  doc_summary[paste("at_least_one_model_prob_1_greater_than_",m,sep="")] <- ifelse(rowSums(doc_summary[model_names] > m) >= 1, 1, 0)
                  doc_summary[paste("at_least_two_models_prob_1_greater_than_",m,sep="")] <- ifelse(rowSums(doc_summary[model_names] > m) >= 2, 1, 0)
                  doc_summary[paste("at_least_three_models_prob_1_greater_than_",m,sep="")] <- ifelse(rowSums(doc_summary[model_names] > m) >= 3, 1, 0)
                  doc_summary[paste("all_models_prob_1_greater_than_",m,sep="")] <- ifelse(rowSums(doc_summary[model_names] > m) == 4, 1, 0)
            }
            
            ##loop to create these performance measures for all parameter variations, for each ensemble 
            #names of the different ensemble predictions created above
            ensemble_names <- colnames(select(doc_summary, contains("greater_than_")))
            #shell
            #create df and add in names, then stack each iteration using rbind
            
            for (p in ensemble_names){
                  toto <- confusionMatrix(as.character(doc_summary[,p]),as.character(doc_summary$MANUAL_CODE), positive = "1")
                  the_stack <- data.frame(df_name = names(dflist)[count], crime_category = z,ensemble_model = p, num_articles_predicted = nrow(doc_summary),
                                          Precision = round(toto$byClass["Precision"],3), Recall = round(toto$byClass["Recall"],3),
                                          F_Score = round(toto$byClass["F1"],3), Accuracy = round(toto$overall["Accuracy"],3),
                                          True_Negative_n = toto$table[1], False_Negative_n = toto$table[3], 
                                          True_Positive_n_ = toto$table[4], False_Positive_ = toto$table[2])
                  all_measures_stacked <- rbind(all_measures_stacked, the_stack)
            }

            
            
      }
      
      #algorithm_summaries <- rbind(algorithm_summaries, algorithm_summary)
}

# stop time
print(proc.time() - ptv)

"Add vars to specify algorithm, preprocessing, number of features/terms used, and total positive/negative tags"
algorithm_summaries_detailed <- algorithm_summaries_detailed %>% separate(col = Algorithm_name, into = c("model", "measure", "remove"), sep = "_")
algorithm_summaries_detailed <- algorithm_summaries_detailed %>% select(-one_of("measure", "remove"))
algorithm_summaries_detailed$model <- ifelse(algorithm_summaries_detailed$model == "Random", "RANDOM FOREST ENSEMBLE", algorithm_summaries_detailed$model)
algorithm_summaries_detailed$is_tfidf <- 0
algorithm_summaries_detailed$is_stem <- 0
algorithm_summaries_detailed[grep("tfidf", algorithm_summaries_detailed$df_name), ]["is_tfidf"] <- 1
algorithm_summaries_detailed[grep("stem", algorithm_summaries_detailed$df_name), ]["is_stem"] <- 1
algorithm_summaries_detailed$is_stem_and_tfidf <- ifelse(algorithm_summaries_detailed$is_tfidf==1 & algorithm_summaries_detailed$is_stem==1, 1, 0)

# bring in ncol data to the different dfs
term_totals <- as.data.frame(sapply(dflist, ncol))
term_totals$df_name <- row.names(term_totals)
names(term_totals)[1] <- "num_terms"
changenames <- setNames(term_totals$num_terms, term_totals$df_name) #new, old
algorithm_summaries_detailed$df_term_total <- algorithm_summaries_detailed$df_name
algorithm_summaries_detailed$df_term_total <- changenames[algorithm_summaries_detailed$df_term_total]  


# bring in actual number of 0s and 1s
tag_totals <- as.data.frame(sapply(test_df[(.75*(nrow(test_df))+1):nrow(test_df),colnames(select(df_news, starts_with("code")))], sum))
tag_totals$crime_category <- row.names(tag_totals)
names(tag_totals)[1] <- "total_positives"
tag_totals$total_negatives <- length(((.75*(nrow(test_df))+1):nrow(test_df))) - tag_totals$total_positives
add_crimes_positive <- setNames(tag_totals$total_positives, tag_totals$crime_category) #new, old
algorithm_summaries_detailed$total_positives <- algorithm_summaries_detailed$crime_category
algorithm_summaries_detailed$total_positives <- add_crimes_positive[algorithm_summaries_detailed$total_positives] 
add_crimes_negative <- setNames(tag_totals$total_negatives, tag_totals$crime_category) #new, old
algorithm_summaries_detailed$total_negatives <- algorithm_summaries_detailed$crime_category
algorithm_summaries_detailed$total_negatives <- add_crimes_negative[algorithm_summaries_detailed$total_negatives] 
algorithm_summaries_detailed<-algorithm_summaries_detailed[-which(row.names(algorithm_summaries_detailed)=="1"),]


"Add vars to specify preprocessing, number of features/terms used, and total positive/negative tags"
ensemble_summaries_detailed$is_tfidf <- 0
ensemble_summaries_detailed$is_stem <- 0
ensemble_summaries_detailed[grep("tfidf", ensemble_summaries_detailed$df_name), ]["is_tfidf"] <- 1
ensemble_summaries_detailed[grep("stem", ensemble_summaries_detailed$df_name), ]["is_stem"] <- 1
ensemble_summaries_detailed$is_stem_and_tfidf <- ifelse(ensemble_summaries_detailed$is_tfidf==1 & ensemble_summaries_detailed$is_stem==1, 1, 0)

# bring in ncol data to the different dfs
term_totals <- as.data.frame(sapply(dflist, ncol))
term_totals$df_name <- row.names(term_totals)
names(term_totals)[1] <- "num_terms"
changenames <- setNames(term_totals$num_terms, term_totals$df_name) #new, old
ensemble_summaries_detailed$df_term_total <- ensemble_summaries_detailed$df_name
ensemble_summaries_detailed$df_term_total <- changenames[ensemble_summaries_detailed$df_term_total]  


# bring in actual number of 0s and 1s
tag_totals <- as.data.frame(sapply(test_df[(.75*(nrow(test_df))+1):nrow(test_df),colnames(select(df_news, starts_with("code")))], sum))
tag_totals$crime_category <- row.names(tag_totals)
names(tag_totals)[1] <- "total_positives"
tag_totals$total_negatives <- length(((.75*(nrow(test_df))+1):nrow(test_df))) - tag_totals$total_positives
add_crimes_positive <- setNames(tag_totals$total_positives, tag_totals$crime_category) #new, old
ensemble_summaries_detailed$total_positives <- ensemble_summaries_detailed$crime_category
ensemble_summaries_detailed$total_positives <- add_crimes_positive[ensemble_summaries_detailed$total_positives] 
add_crimes_negative <- setNames(tag_totals$total_negatives, tag_totals$crime_category) #new, old
ensemble_summaries_detailed$total_negatives <- ensemble_summaries_detailed$crime_category
ensemble_summaries_detailed$total_negatives <- add_crimes_negative[ensemble_summaries_detailed$total_negatives] 
ensemble_summaries_detailed<-ensemble_summaries_detailed[-which(row.names(ensemble_summaries_detailed)=="1"),]

