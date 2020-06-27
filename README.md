# HLM

Hierarchical linear models (aka mutlievel (mixed-effects) models) are a new way of analysis to consider the intra- within subject variablities.

Here we have an example for loading sleep data before and after COVID19 with following variables/sections:
- IDs: keep track of individuals IDs
- time: 0 for before COVID19 and 1 during COVID19
- To remove outliers we did z-score conversion and removing everything beyond 3*sigma
- Fitting the model to estimate total sleep time (TST).
- To access the predicted values we use .fittedvalues
- And finally we plot the spagetti chart for TST

To interpreting the results you can using following information:
- User ID var: var of Intercept as a random effect 
- Week Var: var of slope as a random effect
Equations:
- Before Covid19: TST = Intercept + week * (x=week) //// time = 0
- During Covid19: TST = Intercept + time + (week + time:week) * (x=week) //// time = 1
