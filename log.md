<p>
On the first run, the accuracy was 44.90%, based on the confusion matrix of 
the first result, it is even more clear how unbalanced the dataset is. 
When using the class_weight parameter for the decision tree class, the accuracy did 
not change significantly(44.13%), however the confusion matrix was more balanced.
Switching to the more balanced medium dataset, the accuracy jumps to 58.53%.
Lastly, using the smallest dataset, the accuracy ends up being 47.25%.
</p>
<p>
Considering the unsatisfactory results of an even feature selection, another
approach will be used further. By fixing the MFCC features, the accuracy is 57.33%, and the confusion 
matrices are similar. The smallest dataset yields a 45.88% accuracy.
On the medium dataset with full genre recognition, the accuracy is 32.76%, 
and on the large one, it is 27.46%. This approach ended up with no improvements.
In conclusion, the best approach is the medium dataset with reduction of genres to top-level.
</p>
<p> Moving forward, all models will be trained on the medium dataset, using a top-level
genre taxonomy. </p>
<p> Using 10 features per category other than MFCCs, the accuracy ends up being 60%,
more notably, the diagonal on the confusion matrix is sharper. Using 20, the results are similar; an increase
in 5 features yields 0.20% higher accuracy, up to 30. (Average training time is 12min).</p>