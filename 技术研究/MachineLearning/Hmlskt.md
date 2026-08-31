# Chapter 7 Ensemble Learning and Random Forests
集成学习  
A group of predictors is called an ensemble.  
An ensemble of decision trees is called a random forest.  
投票分类器(Voting Classifiers)  
A very simpler way to create an even better classifier is to aggregate the 
predictions of each classifier: the class that gets the most votes is the 
ensemble's prediction.This majority-vote classifier is called a hard voting classifier.  
投票分类器的原理基于大数定律  
Ensemble methods work best when the predictors are as independent from one another as possible.    
soft voting:If all classifiers are able to estimate class probabilities,then you can tell Scikit-Learn 
predict the class with the highest class probability,averaged over all the individual classifiers.  


