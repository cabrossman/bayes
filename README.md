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

## Chapter 3 - Regression
### Simple linear regression
- mean = a + bX with sd = e
- a => normal dist, b => normal dist, e = half cauchy
- frequentist will agree with maximum a posteriori (MAP (the mode of the posterior) from a bayesian simple linear regression with flat priors)

### Problems with correlation of prams
- slope and intercept become correleated with makes the algorithms hard to converge
- demean to fix - also helps with interpretation - as when X = 0 its the mean

### Pearson Coefficient from a multivariate Gaussian
- can estimate covariance matrix - using pm.MvNormal

### Robust Linear Regression
- if you have outliers that dont fit the assumptions of a normal distribution you can model with T distribution
- you'll add a pm.Exponential with updates similar to below. Note output value is close to v=1.2
    ```
        ??_ = pm.Exponential('??_', 1/29)
        ?? = pm.Deterministic('??', ??_ + 1)

        y_pred = pm.StudentT('y_pred', mu=?? + ?? * x_3,
                            sd=??, nu=??, observed=y_3)

    ```

### Hierarchical Linear Regression
- Hierarchical models with `hyper priors` can take knowledge/assumptions about all groups and apply to a particular group
- This has hte effect of shrinking groups with lower observations to the overall mean. UNTIL they have data that says otherwise they are likely most similar to the rest
    ```
    with pm.Model() as hierarchical_model:
        # hyper-priors
        ??_??_tmp = pm.Normal('??_??_tmp', mu=0, sd=10)
        ??_??_tmp = pm.HalfNormal('??_??_tmp', 10)
        ??_?? = pm.Normal('??_??', mu=0, sd=10)
        ??_?? = pm.HalfNormal('??_??', sd=10)

        # priors
        ??_tmp = pm.Normal('??_tmp', mu=??_??_tmp, sd=??_??_tmp, shape=M)
        ?? = pm.Normal('??', mu=??_??, sd=??_??, shape=M)
        ?? = pm.HalfCauchy('??', 5)
        ?? = pm.Exponential('??', 1/30)

        y_pred = pm.StudentT('y_pred', mu=??_tmp[idx] + ??[idx] * x_centered,
                            sd=??, nu=??, observed=y_m)

        ?? = pm.Deterministic('??', ??_tmp - ?? * x_m.mean())
        ??_?? = pm.Deterministic('??_??', ??_??_tmp - ??_?? * x_m.mean())
        ??_?? = pm.Deterministic('??_sd', ??_??_tmp - ??_?? * x_m.mean())

        idata_hm = pm.sample(2000, target_accept=0.99, return_inferencedata=True)
    ```

### Multiple Linear Regression
- X is a matrix instead of a vector. So when setting B we set to same number of columns in Data Matrix

    ```
    with pm.Model() as model_mlr:
        ??_tmp = pm.Normal('??_tmp', mu=0, sd=10)
        ?? = pm.Normal('??', mu=0, sd=1, shape=2)
        ?? = pm.HalfCauchy('??', 5)

        ?? = ??_tmp + pm.math.dot(X_centered, ??)

        ?? = pm.Deterministic('??', ??_tmp - pm.math.dot(X_mean, ??))

        y_pred = pm.Normal('y_pred', mu=??, sd=??, observed=y)

        idata_mlr = pm.sample(2000, return_inferencedata=True)
    ```

### Variable Variance
- Where hetero assumption is violated you can model the error as a funciton of another variables. Below the variance of length is a function of the number of months a baby has been alive. 
- It has its own parameters to grow or shrink the error
    ```
    with pm.Model() as model_vv:
        ?? = pm.Normal('??', sd=10)
        ?? = pm.Normal('??', sd=10)
        ?? = pm.HalfNormal('??', sd=10)
        ?? = pm.HalfNormal('??', sd=10)

        x_shared = shared(data.Month.values * 1.)

        ?? = pm.Deterministic('??', ?? + ?? * x_shared**0.5)
        ?? = pm.Deterministic('??', ?? + ?? * x_shared)

        y_pred = pm.Normal('y_pred', mu=??, sd=??, observed=data.Lenght)

        idata_vv = pm.sample(1000, tune=1000, return_inferencedata=True)


# Chapter 4 - GLM

### Single Predictor, Binary Output
- Predict one of two categories using sepal length
- Demean values
- Chose -alpha/Beta as its the decision boundary when the prediction = 0.5
- Uses Bernoulli with probability sampled from theta
    ```
    with pm.Model() as model_0:
        ?? = pm.Normal('??', mu=0, sd=10)
        ?? = pm.Normal('??', mu=0, sd=10)
        
        ?? = ?? + pm.math.dot(x_c, ??)    
        ?? = pm.Deterministic('??', pm.math.sigmoid(??))
        bd = pm.Deterministic('bd', -??/??)
        
        yl = pm.Bernoulli('yl', p=??, observed=y_0)

        idata_0 = pm.sample(1000, return_inferencedata=True)
    ```


### Multiple Predictors, Binary Output
```
with pm.Model() as model_1: 
    ?? = pm.Normal('??', mu=0, sd=10) 
    ?? = pm.Normal('??', mu=0, sd=2, shape=len(x_n)) 
    
    ?? = ?? + pm.math.dot(x_1, ??) 
    ?? = pm.Deterministic('??', 1 / (1 + pm.math.exp(-??))) 
    bd = pm.Deterministic('bd', -??/??[1] - ??[0]/??[1] * x_1[:,0])
    
    yl = pm.Bernoulli('yl', p=??, observed=y_1) 

    idata_1 = pm.sample(2000, target_accept=0.9, return_inferencedata=True)
```

### Multiple Predictors, Multiple Categories
- Here we use standardize the X values using a z score
- use the Softmax function
- use the categorical distribution (generalized bernoulli)
- Allow betas to same as predictors and 1 less than number of categories (4 predicts, 3 cats). Fix final predictors to zero
    ```
    with pm.Model() as model_sf:
        ?? = pm.Normal('??', mu=0, sd=2, shape=2)
        ?? = pm.Normal('??', mu=0, sd=2, shape=(4,2))
        ??_f = tt.concatenate([[0] ,??])
        ??_f = tt.concatenate([np.zeros((4,1)) , ??], axis=1)
        ?? = ??_f + pm.math.dot(x_s, ??_f)
        ?? = tt.nnet.softmax(??)
        yl = pm.Categorical('yl', p=??, observed=y_s)
        idata_sf = pm.sample(1000, return_inferencedata=True)
    ```

### Generative Model
- LDA - is correct boundary when distributions are normal and std devs are equal
- LDA (or QDA) > Logistic Reg when all features are gaussian
    ```
    with pm.Model() as lda:
        ?? = pm.Normal('??', mu=0, sd=10, shape=2)
        ?? = pm.HalfNormal('??', 10)
        setosa = pm.Normal('setosa', mu=??[0], sd=??, observed=x_0[:50])
        versicolor = pm.Normal('versicolor', mu=??[1], sd=??,
                            observed=x_0[50:])
        bd = pm.Deterministic('bd', (??[0] + ??[1]) / 2)
        idata_lda = pm.sample(1000, return_inferencedata=True)
    ```

### Poisson 
- E[x] = np, V[x] = np
- for zero inflated poisson its easier to assume two processes
-- One modeled by Poisson dist with prob X
-- One given extra zeros with prob 1-X
    ```
    with pm.Model() as ZIP_reg:
        ?? = pm.Beta('??', 1, 1)
        ?? = pm.Normal('??', 0, 10)
        ?? = pm.Normal('??', 0, 10, shape=2)
        ?? = pm.math.exp(?? + ??[0] * fish_data['child'] + ??[1] * fish_data['camper'])
        yl = pm.ZeroInflatedPoisson('yl', ??, ??, observed=fish_data['count'])
        idata_ZIP_reg = pm.sample(1000, return_inferencedata=True)
    ```


# Chap 5 - Model Eval
### Posterior Predictive Checks
- Sample from the posterior on the parametric values and compare means & IQR (m1 vs m2 vs data)
- Can compute pvalue as comparing simulated data to actual by counting the proportion of time the simulation is equal or greater than the one computed from data. If they agree we expect a p-value around 0.5 otherwise it may indicate prob of disagreement.
### Occam's razor - simplicity and accuracy
- we want simple models with high predictive abilities
### Inormation criteria
- WAIC (similar to AIC, but bayesian - also similar to LOO-CV)
- Lower the better
### Bayes Factor
- BF = p(y | M0) / p(y | M1)
-- 1-3 Anecdotal
-- 3-10 Moderate
-- 10-30 Strong
-- 30-100 Very Strong
-- >100 Extreme
### Regularizing prior
- weakly informative and informative priors restrict the model to overfit the data and have a regularization effect. 
- Lasso & Ridge are similar to adding priors on hyper parameters

# Chap 6 - Mixture Models
- Many dataets cannot be properly described using a single probability dist - but can be described as a mixture of many dists. These are mixture models
### Finite mixture models
- weighted sum of probability density for k subgroups of the data
- p(y|theta) = sum(weights-i * p-i(y|theta-i))
- Assume weightes sum up to 1
- p-i(y|theta-i) - can be any dist or hierarchical
- we used the beta when we were unsure of two outcomes as a prior. For mixture models instead of two outcomes we have k outcomes. The generalization of the bernouli to K outcomes is the categorical distribution and the generalization of the beta distribution is the Dirichlet dist. 
- Notice the use of "NormalMixture" instead of passing the vector: y = pm.Normal('y', mu=means[z], sd=sd, observed=cs_exp)
    ```
    clusters = 2
    with pm.Model() as model_mg:
        p = pm.Dirichlet('p', a=np.ones(clusters))
        means = pm.Normal('means', mu=cs_exp.mean(), sd=10, shape=clusters)
        sd = pm.HalfNormal('sd', sd=10)
        y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=cs_exp)
        idata_mg = pm.sample(random_seed=123, return_inferencedata=True)
    ```
- Non idetnifibility/Label Switching. Problem is similar to high collinearity - cant determine the independent influence. Solutions are 1) forece components to be ordered (arrange means in strictly increasing order for example) 2) use informative priors

fix
```
clusters = 2
with pm.Model() as model_mgp:
    p = pm.Dirichlet('p', a=np.ones(clusters))
    means = pm.Normal('means', mu=np.array([.9, 1]) * cs_exp.mean(), sd=10, shape=clusters)
    sd = pm.HalfNormal('sd', sd=10)
    order_means = pm.Potential('order_means',tt.switch(means[1]-means[0] < 0, -np.inf, 0))
    y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=cs_exp)
    idata_mgp = pm.sample(1000, random_seed=123, return_inferencedata=True)
```
- outcome is much more certain means

### Non-finite mixture models
- when you dont know k, you can use Dirichlet process
- uses non-parametric model (ie we dont know the number of parameters, like we did before, but we let data collapse infinite params to few most likely)
- DP draw is a distribution

### Continuous mixture models
- when you directly mix two types of models
-- like zero inflated: poisson dist and zero generating process
-- random and logistic regression

# Chap 7 Gaussian Process
### Gaussians
- With enough gaussians you can model any dist
- They have nice mathmatical properties in practice

### Kernals
- Kernel is a symmetric function to build the cov matrix
- K(x, x*) = exp(-||x-x*||^2 / 2l^2)
- l is length scale or bandwidth variance. Higher the l the higher the smoothness of the fuction

### GP regresion
f ~ gp(u, k(x,x*))
y ~ N(u = f(x), sigma = eta) where eta ~ N(0,sigma)

- In the example you can use your knowledge of the distance between islands to inform the kernal of the gp
    ```
    with pm.Model() as model_islands:
        ?? = pm.HalfCauchy('??', 1)
        ??? = pm.HalfCauchy('???', 1)
        
        cov = ?? * pm.gp.cov.ExpQuad(1, ls=???)
        gp = pm.gp.Latent(cov_func=cov)
        f = gp.prior('f', X=islands_dist_sqr)

        ?? = pm.Normal('??', 0, 10)
        ?? = pm.Normal('??', 0, 1)
        ?? = pm.math.exp(?? + f[index] + ?? * log_pop)
        tt_pred = pm.Poisson('tt_pred', ??, observed=total_tools)
        idata_islands = pm.sample(1000, tune=1000, target_accept=0.9, return_inferencedata=True)
    ```

### GP Classification
```
with pm.Model() as model_iris:
    ??? = pm.Gamma('???', 2, 0.5)
    cov = pm.gp.cov.ExpQuad(1, ???) + pm.gp.cov.WhiteNoise(1e-5)
    gp = pm.gp.Latent(cov_func=cov)
    f = gp.prior("f", X=X_1)
    # logistic inverse link function and Bernoulli likelihood
    y_ = pm.Bernoulli("y", p=pm.math.sigmoid(f), observed=y)
    idata_iris = pm.sample(1000, return_inferencedata=True)
```

now can better handle uncertainty at the edges

```
with pm.Model() as model_iris2:
    ??? = pm.Gamma('???', 2, 0.5)
    c = pm.Normal('c', x_1.min())
    ?? = pm.HalfNormal('??', 5)
    cov = (pm.gp.cov.ExpQuad(1, ???) +
        ?? * pm.gp.cov.Linear(1, c) +
        pm.gp.cov.WhiteNoise(1E-5))
    gp = pm.gp.Latent(cov_func=cov)
    f = gp.prior("f", X=X_1)
    # logistic inverse link function and Bernoulli likelihood
    y_ = pm.Bernoulli("y", p=pm.math.sigmoid(f), observed=y)
    idata_iris2 = pm.sample(1000, chains=1, compute_convergence_checks=False, return_inferencedata=True)
```

model U shape
```
with pm.Model() as model_space_flu:
??? = pm.HalfCauchy('???', 1)
cov = pm.gp.cov.ExpQuad(1, ???) + pm.gp.cov.WhiteNoise(1E-5)
gp = pm.gp.Latent(cov_func=cov)
f = gp.prior('f', X=age)
y_ = pm.Bernoulli('y', p=pm.math.sigmoid(f), observed=space_flu)
idata_space_flu = pm.sample(1000, chains=1, compute_convergence_checks=False, return_inferencedata=True)
```

Examples using Poisson & Two Dimensional multi variate classification exist as well

# Chap 8 - Inference Machines
## Non Markovian Methods
### Grid Computing
- define a reasonable interval for param
- place a grid of point on that interval
- for each point multiply the likelidhood and the prior
- scales poorly with number of params

### Quadratic / Laplace / Normal Approx
- Find the mode of the distribution - this will be the mean also
- Compute the Hessian matrix - from this we cana compute the standard deviation
- does ok - but is unbounded (although some distributions are bounded - like binomial). There are tricks such as half normal and log it

### Variational Methods
- Used for cases with LARGE data
- Estimate using simple dist (like Laplace) and measure closeness using KL divergance
- for every parameter we pick a distribution. We pick the distribution that is easy (Normal, exponential, beta, etc) and try to minimize the difference of means between the dist and actual
- Also known as mean field

## Markovian Methods
- Take more samples from higher probability areas. It visits each area proportional to its probability

### Metropolis Hastings MC
- simple implementation
- used for discrete distributions
```
def metropolis(func, draws=10000):
    """A very simple Metropolis implementation"""
    trace = np.zeros(draws)
    old_x = 0.5  # func.mean()
    old_prob = func.pdf(old_x)

    delta = np.random.normal(0, 0.5, draws)
    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = func.pdf(new_x)
        acceptance = new_prob / old_prob
        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x

    return trace
```
### Hamilton MC
- extends the idea to calculate gradient at each step
- HMC is more expense to compute gradient, but has higher acceptance rate so it explores mores
- used for continuous distributions

### Sequential MC
- works in cases where where have multi modal dist
- us tempering factor which is range to use the liklihood or prior, ranges [0,1]


## Diagnosing the samples
### Solutions to bad sampling
- increase samples
- remove a number of samples from the beginning of the trace
- modify sampler parameters, such as increasing the length of the tuning phase
- re-parametrize the model
- transform the data (like centering)

### Diagnosing issues
- Look at the plot_trace function to find good mixing
- RHAT stat - compare the variance between chains with the variance within chains - we expect a value of one, but are ok with values below 1.1
- MCerror = std-dev(x)/sqrt(n)
- Autocorrelation - shouldnt see much if any auto correlation
- Effective sample size --> what sample do you need to overcome any issues with autocorrelation?
