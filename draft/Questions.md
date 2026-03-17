### Check
* Questions to be answerd
    * What affects target performance? [source performance, domain shift, model capacity, model sensitivity, etc.]
        * Measure the source ideal performance, combined ideal performance, and how much of the way the domain adaptation takes us. Scale from (Xs, Ys), (Xs, Ys, Xt), (Xs, Ys, Xt, Yt)
        * Measure the specificity of the performance improvement for the target versus some random domain (is good on all other domains or just the target?) ~ domain generalization vs domain adaptation
        * What aspect of domain shift is detrimental to performance? (covariate shift, feature shift, structural shift)
        * Can any one dare claim that label shift could be solved with no data???
            * Does the class size matter? if there is class imbalance is the performance degradation proportional to that? if so can we solve it by focal loss? 
    What is structural shift [connection pattern or operator response]?
    * 