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

## Chapter 3
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
        ν_ = pm.Exponential('ν_', 1/29)
        ν = pm.Deterministic('ν', ν_ + 1)

        y_pred = pm.StudentT('y_pred', mu=α + β * x_3,
                            sd=ϵ, nu=ν, observed=y_3)

    ```

### Hierarchical Linear Regression
- Hierarchical models with `hyper priors` can take knowledge/assumptions about all groups and apply to a particular group
- This has hte effect of shrinking groups with lower observations to the overall mean. UNTIL they have data that says otherwise they are likely most similar to the rest
    ```
    with pm.Model() as hierarchical_model:
        # hyper-priors
        α_μ_tmp = pm.Normal('α_μ_tmp', mu=0, sd=10)
        α_σ_tmp = pm.HalfNormal('α_σ_tmp', 10)
        β_μ = pm.Normal('β_μ', mu=0, sd=10)
        β_σ = pm.HalfNormal('β_σ', sd=10)

        # priors
        α_tmp = pm.Normal('α_tmp', mu=α_μ_tmp, sd=α_σ_tmp, shape=M)
        β = pm.Normal('β', mu=β_μ, sd=β_σ, shape=M)
        ϵ = pm.HalfCauchy('ϵ', 5)
        ν = pm.Exponential('ν', 1/30)

        y_pred = pm.StudentT('y_pred', mu=α_tmp[idx] + β[idx] * x_centered,
                            sd=ϵ, nu=ν, observed=y_m)

        α = pm.Deterministic('α', α_tmp - β * x_m.mean())
        α_μ = pm.Deterministic('α_μ', α_μ_tmp - β_μ * x_m.mean())
        α_σ = pm.Deterministic('α_sd', α_σ_tmp - β_μ * x_m.mean())

        idata_hm = pm.sample(2000, target_accept=0.99, return_inferencedata=True)
    ```

### Multiple Linear Regression
- X is a matrix instead of a vector. So when setting B we set to same number of columns in Data Matrix

    ```
    with pm.Model() as model_mlr:
        α_tmp = pm.Normal('α_tmp', mu=0, sd=10)
        β = pm.Normal('β', mu=0, sd=1, shape=2)
        ϵ = pm.HalfCauchy('ϵ', 5)

        μ = α_tmp + pm.math.dot(X_centered, β)

        α = pm.Deterministic('α', α_tmp - pm.math.dot(X_mean, β))

        y_pred = pm.Normal('y_pred', mu=μ, sd=ϵ, observed=y)

        idata_mlr = pm.sample(2000, return_inferencedata=True)
    ```

### Variable Variance
- Where hetero assumption is violated you can model the error as a funciton of another variables. Below the variance of length is a function of the number of months a baby has been alive. 
- It has its own parameters to grow or shrink the error
    ```
    with pm.Model() as model_vv:
        α = pm.Normal('α', sd=10)
        β = pm.Normal('β', sd=10)
        γ = pm.HalfNormal('γ', sd=10)
        δ = pm.HalfNormal('δ', sd=10)

        x_shared = shared(data.Month.values * 1.)

        μ = pm.Deterministic('μ', α + β * x_shared**0.5)
        ϵ = pm.Deterministic('ϵ', γ + δ * x_shared)

        y_pred = pm.Normal('y_pred', mu=μ, sd=ϵ, observed=data.Lenght)

        idata_vv = pm.sample(1000, tune=1000, return_inferencedata=True)


# Chapter 4

### Single Predictor, Binary Output
- Predict one of two categories using sepal length
- Demean values
- Chose -alpha/Beta as its the decision boundary when the prediction = 0.5
- Uses Bernoulli with probability sampled from theta
    ```
    with pm.Model() as model_0:
        α = pm.Normal('α', mu=0, sd=10)
        β = pm.Normal('β', mu=0, sd=10)
        
        μ = α + pm.math.dot(x_c, β)    
        θ = pm.Deterministic('θ', pm.math.sigmoid(μ))
        bd = pm.Deterministic('bd', -α/β)
        
        yl = pm.Bernoulli('yl', p=θ, observed=y_0)

        idata_0 = pm.sample(1000, return_inferencedata=True)
    ```


### Multiple Predictors, Binary Output
    ```
    with pm.Model() as model_1: 
        α = pm.Normal('α', mu=0, sd=10) 
        β = pm.Normal('β', mu=0, sd=2, shape=len(x_n)) 
        
        μ = α + pm.math.dot(x_1, β) 
        θ = pm.Deterministic('θ', 1 / (1 + pm.math.exp(-μ))) 
        bd = pm.Deterministic('bd', -α/β[1] - β[0]/β[1] * x_1[:,0])
        
        yl = pm.Bernoulli('yl', p=θ, observed=y_1) 
    
        idata_1 = pm.sample(2000, target_accept=0.9, return_inferencedata=True)
    ```

### Multiple Predictors, Multiple Categories
- Here we use standardize the X values using a z score
- use the Softmax function
- use the categorical distribution (generalized bernoulli)
- Allow betas to same as predictors and 1 less than number of categories (4 predicts, 3 cats). Fix final predictors to zero
    ```
    with pm.Model() as model_sf:
        α = pm.Normal('α', mu=0, sd=2, shape=2)
        β = pm.Normal('β', mu=0, sd=2, shape=(4,2))
        α_f = tt.concatenate([[0] ,α])
        β_f = tt.concatenate([np.zeros((4,1)) , β], axis=1)
        μ = α_f + pm.math.dot(x_s, β_f)
        θ = tt.nnet.softmax(μ)
        yl = pm.Categorical('yl', p=θ, observed=y_s)
        idata_sf = pm.sample(1000, return_inferencedata=True)
    ```

### Generative Model
- LDA - is correct boundary when distributions are normal and std devs are equal
- LDA (or QDA) > Logistic Reg when all features are gaussian
    ```
    with pm.Model() as lda:
        μ = pm.Normal('μ', mu=0, sd=10, shape=2)
        σ = pm.HalfNormal('σ', 10)
        setosa = pm.Normal('setosa', mu=μ[0], sd=σ, observed=x_0[:50])
        versicolor = pm.Normal('versicolor', mu=μ[1], sd=σ,
                            observed=x_0[50:])
        bd = pm.Deterministic('bd', (μ[0] + μ[1]) / 2)
        idata_lda = pm.sample(1000, return_inferencedata=True)
    ```

### Poisson 
- E[x] = np, V[x] = np
- for zero inflated poisson its easier to assume two processes
-- One modeled by Poisson dist with prob X
-- One given extra zeros with prob 1-X
    ```
    with pm.Model() as ZIP_reg:
        ψ = pm.Beta('ψ', 1, 1)
        α = pm.Normal('α', 0, 10)
        β = pm.Normal('β', 0, 10, shape=2)
        θ = pm.math.exp(α + β[0] * fish_data['child'] + β[1] * fish_data['camper'])
        yl = pm.ZeroInflatedPoisson('yl', ψ, θ, observed=fish_data['count'])
        idata_ZIP_reg = pm.sample(1000, return_inferencedata=True)
    ```

