# Code for implementing the IV_PILE estimator

# Load the required packages
library(locClass)
library(nnet)
library(dplyr)
library(randomForest)

#########################################################################
# Expit function
expit <- function(x) exp(x)/(exp(x) + 1)

# Estimate P(Y = y, A = a | Z = z, X) and then construct the 
# Balke-Pearl bound for Conditional ATE.
# The function takes as input a dataset with n rows and 
# the following columns: Z (IV), X (Observed covariates), 
# A (Treatment), Y (Oucome).

# We implement two aproaches: 1) fitting multinomial regression
# 2) use a random forest

estimate_BP <- function(dt, method = 'rf'){

  # Re-code 2*2 = 4 different (A, Y) combinations as 4 categorical
  # outcomes.
  dt$Outcome = 1*(dt$Y == 0 & dt$A == 0) + 2*(dt$Y == 0 & dt$A == 1) + 
    3*(dt$Y == 1 & dt$A == 0) + 4*(dt$Y == 1 & dt$A == 1)
  dt_md = dt %>% select(-A, -Y)
  
  # Prepare two datasets for predicting P(Y = y, A = a | Z = 0, X)
  # and P(Y = y, A = a | Z = 1, X)
  dt_predict_z_0 = dt_md
  dt_predict_z_0$Z = 0  
  dt_predict_z_1 = dt_md
  dt_predict_z_1$Z = 1
  
  # Use the random forest method
  if (method == 'rf'){
     md = randomForest(as.factor(Outcome) ~., data = dt_md, nodesize = 5)
     
     # Predict P(Y = y, A = a | Z = 0, X)
     prob_res_z_0 = predict(md, newdata = dt_predict_z_0, type = 'prob')
     
     # Predict P(Y = y, A = a | Z = 1, X)
     prob_res_z_1 = predict(md, newdata = dt_predict_z_1, type = 'prob')
  }
  else {
   # Fit a multinomial logit regression
   md = multinom(Outcome ~., data = dt_md, trace = FALSE)
   prob_res_z_0 = predict(md, newdata = dt_predict_z_0,  'probs')
   prob_res_z_1 = predict(md, newdata = dt_predict_z_1,  'probs')
  }
  # Balke-Pearl lower bound
  l_1 = prob_res_z_0[,1] + prob_res_z_1[,4] - 1
  l_2 = prob_res_z_1[,1] + prob_res_z_1[,4] - 1
  l_3 = prob_res_z_0[,4] + prob_res_z_1[,1] - 1
  l_4 = prob_res_z_0[,1] + prob_res_z_0[,4] - 1
  l_5 = 2*prob_res_z_0[,1] + prob_res_z_0[,4] + prob_res_z_1[,3] + prob_res_z_1[,4] - 2
  l_6 = prob_res_z_0[,1] + 2*prob_res_z_0[,4] + prob_res_z_1[,1] + prob_res_z_1[,2] - 2
  l_7 = prob_res_z_0[,3] + prob_res_z_0[,4] + 2*prob_res_z_1[,1] + prob_res_z_1[,4] - 2
  l_8 = prob_res_z_0[,1] + prob_res_z_0[,2] + prob_res_z_1[,1] + 2*prob_res_z_1[,4] - 2
  
  bk_lower_bound = pmax(l_1, l_2, l_3, l_4, l_5, l_6, l_7, l_8)
  
  # Balke-Pearl upper bound
  u_1 = 1 - prob_res_z_0[,3] - prob_res_z_1[,2]
  u_2 = 1 - prob_res_z_0[,2] - prob_res_z_1[,3]
  u_3 = 1 - prob_res_z_0[,2] - prob_res_z_0[,3]
  u_4 = 1 - prob_res_z_1[,2] - prob_res_z_1[,3]
  u_5 = 2 - 2*prob_res_z_0[,2] - prob_res_z_0[,3] - prob_res_z_1[,3] - prob_res_z_1[,4]
  u_6 = 2 - prob_res_z_0[,2] - 2*prob_res_z_0[,3] - prob_res_z_1[,1] - prob_res_z_1[,2]
  u_7 = 2 - prob_res_z_0[,3] - prob_res_z_0[,4] - 2*prob_res_z_1[,2] - prob_res_z_1[,3]
  u_8 = 2 - prob_res_z_0[,1] - prob_res_z_0[,2] - prob_res_z_1[,2] - 2*prob_res_z_1[,3]
  
  bk_upper_bound = pmin(u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8)
  
  dt$L = bk_lower_bound
  dt$U = bk_upper_bound
  
  return(dt)
}

####################################################################################
# Implement IV-PILE
# The function takes as input a dataset with n rows and 
# the following columns: Z (IV), X (Observed covariates), 
# A (Treatment), Y (Oucome).

IV_PILE <- function(dt){
  n = dim(dt)[1]
  
  # Create labels
  ind_p = which(dt$L > 0)
  ind_n = which(dt$U < 0)
  ind_sp = which((dt$L <= 0) & (dt$U >= 0) & (abs(dt$U) >= abs(dt$L)))
  ind_sn = which((dt$L <= 0) & (dt$U >= 0) & (abs(dt$U) < abs(dt$L)))
  ind_semi = c(ind_sp, ind_sn)
  
  labels = numeric(n)
  labels[ind_p] = 1
  labels[ind_sp] = 1
  labels = as.factor(labels > 0)
  
  # Create weights
  weights = numeric(n)
  weights[ind_p] = abs(dt$U[ind_p])
  weights[ind_n] = abs(dt$U[ind_n])
  weights[ind_semi] = abs(abs(dt$L[ind_semi]) - abs(dt$U[ind_semi]))
  
  # Run weighted SVM
  dt_md = dt%>%select(-Z, -A, -Y, -Outcome, -L, -U)
  iv_pile = wsvm(labels ~ ., data = dt_md, case.weights = weights, 
                 kernel = 'radial', fitted = TRUE)
  return(iv_pile)
}

####################################################################
# An illustrative example
 
 # Simulate some training data
 n = 500
 # Observed covariates 
 X_1 = runif(n, -1, 1)
 X_2 = runif(n, -1, 1)
 X_3 = runif(n, -1, 1)
 X_4 = runif(n, -1, 1)
 X_5 = runif(n, -1, 1)
 
 # Simulate an unmeasured confounder 
 U = runif(n, -1, 1)
  
 # Simualte IV
 Z = rbinom(n, 1, 0.5)
 
 # Simulate treatment received
 lambda = 2 
 A = rbinom(n, 1, prob = expit(8*Z + X_1 - 7*X_2 + lambda*U*(1 + X_1)))
 
 # Simulate outcome 
 g_2 <- function(X_1, X_2, U, A, delta) 
          return(0.442*(1 - X_1 + X_2 + delta*U)*A)
 xi = 1
 delta = 1
 mu = 1 - X_1 + X_2 + xi*U + g_2(X_1, X_2, U, A, delta)
 Y = rbinom(n, 1, prob = expit(mu))
  
 # We only observe data Z, X, A, Y
 # We do NOT have access to U during the training process
 dt = data.frame(Z, X_1, X_2, X_3, X_4, X_5, A, Y)
  
 # Estimate Balke-Pearl bound using random forest
 dt_with_BP_bound = estimate_BP(dt, method = 'rf')
  
 # IV-PILE with naive and random forest
 IV_PILE_result = IV_PILE(dt_with_BP_bound)


  # Simualte some test data
  n_t = 1000000
  X_1 = runif(n_t, -1, 1)
  X_2 = runif(n_t, -1, 1)
  X_3 = runif(n_t, -1, 1)
  X_4 = runif(n_t, -1, 1)
  X_5 = runif(n_t, -1, 1)
  U = runif(n_t, -1, 1)
  
  # Compuate the true CATE and the true label
  C_X_1 = expit(1 - X_1 + X_2 + xi*U + g_2(X_1, X_2, U, A = rep(1, n_t), delta))
  C_X_0 = expit(1 - X_1 + X_2 + xi*U + g_2(X_1, X_2, U, A = rep(0, n_t), delta))
  C_X_true = C_X_1 - C_X_0
  true_label = (C_X_true > 0)
  dt_test = data.frame(X_1, X_2, X_3, X_4, X_5)
  
  # Apply the IV-PILE estimator on trained data
  IV_PILE_predict = predict(IV_PILE_result, newdata = dt_test)
  
  # Compute the true weighted misclassification error
  loss_IV_PILE_rf = sum(abs(C_X_true)[IV_PILE_predict != true_label])/n_t
