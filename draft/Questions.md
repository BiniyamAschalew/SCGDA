### Check
* Questions to be answerd
    * What affects target performance? [source performance, domain shift, model capacity, model sensitivity, etc.]
        * Measure the source ideal performance, combined ideal performance, and how much of the way the domain adaptation takes us. Scale from (Xs, Ys), (Xs, Ys, Xt), (Xs, Ys, Xt, Yt)
        * Measure the specificity of the performance improvement for the target versus some random domain (is good on all other domains or just the target?) ~ domain generalization vs domain adaptation
        * What aspect of domain shift is detrimental to performance? (covariate shift, feature shift, structural shift)
        * Can any one dare claim that label shift could be solved with no data???
            * Does the class size matter? if there is class imbalance is the performance degradation proportional to that? if so can we solve it by focal loss? 
    What is structural shift [connection pattern or operator response]?
    

### Concepts recap
* Gradient versus Jacobian
    * Gradient is computed on a scalar loss function, and it has the same dimension as the parameters. 
    * Jacobian 

### Transformers and foundational models
* Grouped Query Attention (GQA) [https://arxiv.org/abs/2205.14135]
    * A more efficient attention mechanism that reduces the computational complexity of self-attention by grouping queries together.
    * This means that instead of computing attention for each query independently, GQA computes attention for groups of queries, which can significantly reduce the computational cost while still maintaining good performance. 
    * In GQA, queries are divided into groups, and each group attends to a subset of the keys and values. This allows for more efficient computation while still capturing important dependencies in the data.


## Lab session prep on GPT2 
The questions should focus on the following ideas:
1) 