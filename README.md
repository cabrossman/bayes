# bayes
Bayesian Analysis with Python

## Chapter 1
### Probs
- p(A,B) = p(A|B)p(B)
- p(A|B) = p(A) **if** A & B are independent
- p(th|y) = p(y|th)p(th)/p(y) => p(th|y)p(y) === p(y|th)*p(th)
- posterior = (likelihood*prior)/marginal_dist
- iid - timeseries isnt iid due to autocorrelation

### Central Ideas
- prior : should reflect what we know about the value of theta before seeing data. Things about the more likely values, the spread etc. We can start with flat priors if we know nothing but can normally do better
- likelihood : how likely are we to get these values given our prior
- posterier : output distribution of theta - reflects what we know about problem given our data nd model. "A bayesian is one who, vaguely expecting a horse, and catching a glimpse of a donkey, strongly believes he has seen a mule."
- marginal likelihood : mostly a normalizing factor to convert to probabilities
- We assume there is a TRUE dist that is unknown and we get a finate sample by collecting data. We build a probabilistic model using the prior and likelihood to obtain a posterier dist (which captures all info about the problem given our model and data). Everything is derived from the postier
- cycle
-- sample from true dist
-- infer posterier dist
-- sample from postier dist and validate


### Beta Distribution
- models uncertainty in theta
- conjugate prior of the binomial dist 
-- conjugate prior : A prior that when used in como with a likelihood returns a posterier with estimates functional form as the prior

### Highest Posterior Density (HPD) Interval
- HPD is the shortest interval containing a given portion of the probability density


## Chapter 2
### Creating Simple Model & Evaluating Example
- Setup : bernoulli dist (heads/tails) - with real theta @ 0.35, data = 4 observations, 1 success, prior = beta(1,1)
- Kernel Density Estimation (KDE)
- Summary = mean, sd, 94% HDP values (@3% & @97% repsectively)
- Plot postier = gives 94% HDI & post distribution
- ROPE (Region of Practical Equivalence) - If the coin is in range [0.45,0.55] and not exactly 0.5
-- If ROPE no overlap with HPD - coin is not fair
-- If ROPE inside HPD we say coin is fair
-- If ROPE partially overlap we are uncertain

### Gaussians
- Conjugate prior of the Gaussian mean is the Gaussian iteself. 
- Everytime we measure avg of something with large enough sample it will be distributed gaussian 

### Notes/ Thoughts at this point
We are trying to figure out the distribution from which the observations where generated. We do this by 
1. Guessing the distribution of the observations
2. Guessing the parameter distributions that make sense
3. Using data to update those beliefs to arrive at the final model
4. Final model is sample from to "fill in gaps" and reason about

### Effect Size
- Quantify the difference between two groups
- Cohen's d (diff of means / pooled std dev) - puts in scale of z-score
- probability of superiority : cum_norm_dist(cohen's d / sqrt(2))

### Hierarchical models
- pool data together and estimate quality 
- likelihood has shared priors
- shared priors cause shrinkage among all thetas especially at extreemes