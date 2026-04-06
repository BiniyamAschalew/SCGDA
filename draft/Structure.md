* Distribution shift
    * Domain shift: P(X) changes but P(Y|X) remains the same
    * Task shift: P(Y|X) changes

* Claims
    * The task shift could be more influential ~~ we are choosing similar dataset characteristics so may be not..

* Ours
    * Align the structure throught the realized filters ~ to reduce the distribution shift introduced at each message passing layer
    * Constrain the Alignment loss (when the cross domain MMD is below a certain threshold ~ like the intra domain MMD) shut off the loss to prevent over-alignment and negative transfer
    * Entropy maximization to encourage the model to prune out the spurious correlations and learn more invariant representations
    * We compare models in-terms of the target-specific performance vs reference performance
    * Homophily alignment by creating roles based on the maximum class propagation scheme

* Looking at the paper src/invariant_representation_learning.pdf
    * The upper bound to the transfer is given as source error + distribution shift (measured by Wasserstein distance) + optimal joint error while the lower bound to the transfer is given as 

* Structural shift
    *  
    * Operator view: 
        * we start with the identity matrix or random probe matrix
        * first we apply the feature projection (which is just the feature matrix)
        * then we apply the structure projection (message passing) ~ which could be multiple times per layer
* It is about the loss functions than just the architecture.
* The normalizations 




* PFN
    * Pretraining ~ prior fitten
    * Finetuning ~ data fitten (posterior fitten) -- not permanently finetuning (in-context learning)
    * Pros
        * Inference time should be fast
        * Can leverage large amount of data for pretraining
    * Cons
        * in-context learning could require a large number of examples ~ large context window || memory
        * may not be able to leverage the in-context examples effectively, especially when the number of in-context examples is small 