# Instruction:
# To reproduce our results,
# Please download the synthetic data from this link https://drive.google.com/drive/folders/1zq5JiHq77efVdER9fTUVLbJYPvPQLkNa?usp=drive_link .
# Please move the dowloaded csv files under 03_outputs folder.

# Clear workspace
rm(list=ls())

### Define helper functions ###
bigram_var <- function(x, bigrams) {
  n <- length(x)
  x2 <- cbind(x[1:(n-1)], x[2:n])
  x2 <- apply(x2, 1, paste, collapse = ' ')
  sapply(bigrams, function(y) as.numeric(y %in% x2))
}

construct_bigram <- function(x) {
  n <- length(x)
  res <- cbind(x[1:(n-1)], x[2:n])
  apply(res, 1, paste, collapse = ' ')
}

unigram_var <- function(x, unigrams) {
  sapply(unigrams, function(y) as.numeric(y %in% x))
}

split_binary_samples <- function(x, p_train) {
  group1 <- which(x == 1)
  group2 <- which(x == 0)
  
  index_train <- c(sample(group1, round(length(group1) * p_train)),
                   sample(group2, round(length(group2) * p_train)))
  index_test <- setdiff(seq_along(x), index_train)
  
  list(train = index_train, test = index_test)
}

### Load libraries ###
library(ProcData)
library(nnet)
library(ggplot2)
library(patchwork)

### Set file paths ###
# Please change this path based on your local directory structure
path <- paste0("..\\03_outputs\\",
               c("unconditional_synthetic_200_for_500_reps.csv",
                 "unconditional_synthetic_500_for_500_reps.csv",
                 "unconditional_synthetic_1000_for_500_reps.csv"))

### Prepare result containers ###
reps <- 25

# We will store two lists of matrices: one for the three summary metrics (result) 
# and one for the seven additional prediction metrics (result_g)
result <- list()
result_g <- list()
for (d in 1:3) {  # here we use only the first three datasets
  result[[d]] <- matrix(NA, nrow = reps, ncol = 3)
  result_g[[d]] <- matrix(NA, nrow = reps, ncol = 7)
}

### Set up parallel backend ###
library(doParallel)
library(foreach)
numCores <- parallel::detectCores() - 1  # leave one core free
cl <- makeCluster(numCores)
registerDoParallel(cl)

time_0 <- Sys.time()

### Loop over datasets ###
for (d in 1:3) {  # Loop over the first three datasets
  dat <- read.csv(path[d])
  
  # Use foreach to parallelize over the replications
  tmp <- foreach(r = 1:reps, 
                 .packages = c("ProcData", "nnet", "ggplot2", "patchwork")) %dopar% {
                   # Optional: print progress info (note that output from parallel workers may be buffered)
                   print(paste0("data ", d, ": ### ", r, " ### "))
                   
                   temp.dat <- dat[dat$Iter_index == r, ]
                   n <- nrow(temp.dat)
                   
                   # Create list of action sequences from each row
                   actions_list <- lapply(1:nrow(temp.dat), function(i) {
                     unlist(strsplit(temp.dat$ActionSequence[i], split = ","))
                   })
                   
                   # Construct unigram and bigram derived variables
                   unigrams <- unique(unlist(actions_list))
                   unigrams <- unigrams[! unigrams %in% c("START", "END")]
                   bigrams <- unique(unlist(sapply(actions_list, construct_bigram)))
                   
                   x_uni <- t(sapply(actions_list, unigram_var, unigrams = unigrams))
                   x_bi <- t(sapply(actions_list, bigram_var, bigrams = bigrams))
                   
                   # Select only those bigrams that appear in at least 50 sequences
                   select_bigram <- which(apply(x_bi, 2, sum) >= 50)
                   x_bi <- x_bi[, select_bigram, drop = FALSE]
                   bigrams <- bigrams[select_bigram]
                   
                   x_grams <- cbind(x_uni, x_bi)
                   
                   ## MDS feature extraction
                   K_cand <- (1:5) * 10
                   seqs <- proc(action_seqs = actions_list, time_seqs = NULL)
                   K_res <- chooseK_mds(seqs, K_cand, return_dist = TRUE)
                   theta <- seq2feature_mds(K_res$dist_mat, K_res$K)$theta
                   
                   ## Predict unigram and bigrams via logistic regression
                   n_var <- ncol(x_grams)
                   grams_pred_acc <- rep(0, n_var)
                   
                   for (j in 1:n_var) {
                     split_res <- split_binary_samples(x_grams[, j], 0.8)
                     index_train <- split_res$train
                     index_test <- split_res$test
                     
                     mydata <- data.frame(y = x_grams[, j], x = theta)
                     glm_res <- glm(y ~ ., data = mydata, subset = index_train, family = "binomial")
                     grams_pred_acc[j] <- mean(as.numeric(predict(glm_res, newdata = mydata[index_test, ], type = "link") > 0) == mydata$y[index_test])
                   }
                   
                   # Predict country (coded as UK = 1)
                   index_train <- sample(1:n, n * 0.8)
                   index_test <- setdiff(1:n, index_train)
                   mydata2 <- data.frame(x = theta, g = as.numeric(temp.dat$Country == "UK"))
                   glm_g <- glm(g ~ ., data = mydata2, subset = index_train, family = "binomial")
                   pred_g <- mean(as.numeric(predict(glm_g, newdata = mydata2[index_test, ], type = "link") > 0) == mydata2$g[index_test])
                   
                   res1 <- c(mean(grams_pred_acc), min(grams_pred_acc), pred_g)
                   
                   ## Additional predictions using the MDS features
                   mydata2 <- data.frame(x = theta)
                   temp.dat$Country <- as.numeric(temp.dat$Country == "UK")
                   temp.dat$Gender <- as.numeric(temp.dat$Gender == "Female")
                   
                   glm_g1 <- glm(temp.dat$Country[index_train] ~ ., data = mydata2[index_train, ], family = "binomial")
                   glm_g2 <- glm(temp.dat$Gender[index_train] ~ ., data = mydata2[index_train, ], family = "binomial")
                   
                   temp.dat$GradedScore_c <- factor(temp.dat$GradedScore)
                   mm_score <- multinom(temp.dat$GradedScore_c[index_train] ~ ., data = mydata2[index_train, ])
                   mm_age <- multinom(temp.dat$AgeGroup[index_train] ~ ., data = mydata2[index_train, ])
                   
                   lm_g1 <- lm(temp.dat$ResponseTime[index_train] ~ ., data = mydata2[index_train, ])
                   lm_g2 <- lm(temp.dat$LiteracyScore[index_train] ~ ., data = mydata2[index_train, ])
                   lm_g3 <- lm(temp.dat$NumeracyScore[index_train] ~ ., data = mydata2[index_train, ])
                   
                   pred1 <- mean(as.numeric(predict(glm_g1, newdata = mydata2[index_test, ], type = "link") > 0) == temp.dat$Country[index_test])
                   pred2 <- mean(as.numeric(predict(glm_g2, newdata = mydata2[index_test, ], type = "link") > 0) == temp.dat$Gender[index_test])
                   
                   pred3 <- mean(as.numeric(as.character(predict(mm_score, newdata = mydata2[index_test, ]))) == temp.dat$GradedScore[index_test])
                   pred4 <- mean(as.character(predict(mm_age, newdata = mydata2[index_test, ])) == as.character(temp.dat$AgeGroup[index_test]))
                   
                   pred5 <- cor(predict(lm_g1, newdata = mydata2[index_test, ]), temp.dat$ResponseTime[index_test])^2
                   pred6 <- cor(predict(lm_g2, newdata = mydata2[index_test, ]), temp.dat$LiteracyScore[index_test])^2
                   pred7 <- cor(predict(lm_g3, newdata = mydata2[index_test, ]), temp.dat$NumeracyScore[index_test])^2
                   
                   res2 <- c(pred1, pred2, pred3, pred4, pred5, pred6, pred7)
                   
                   list(res = res1, res_g = res2)
                 }
  
  # Combine the results from all replications for this dataset.
  result[[d]] <- do.call(rbind, lapply(tmp, function(x) x$res))
  result_g[[d]] <- do.call(rbind, lapply(tmp, function(x) x$res_g))
}

### Shut down the cluster ###
stopCluster(cl)

# (Optional) View the time taken
time_elapsed <- Sys.time() - time_0
print(time_elapsed)

### Combine results for plotting or further analysis ###
gdat <- rbind(data.frame(SampleSize = "n=200", result[[1]]),
              data.frame(SampleSize = "n=500", result[[2]]),
              data.frame(SampleSize = "n=1000", result[[3]]))

gdat_g <- rbind(data.frame(SampleSize = "n=200", result_g[[1]]),
                data.frame(SampleSize = "n=500", result_g[[2]]),
                data.frame(SampleSize = "n=1000", result_g[[3]]))

# save gdat and gdat_g for analysis
save(gdat, gdat_g, file = "../03_outputs/sim_results.RData")

