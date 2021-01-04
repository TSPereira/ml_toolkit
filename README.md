## Objective
The objective of the repository is to share generic functions that everybody can use on the day to day work.


## Contribution Flow Summary
To contribute one should clone the repo, add the functions to share and then perform a pull request into the develop
branch.

Pull requests must be well documented and fully working. A test example should be provided such that the code added can 
be tested on review.


## Package Main Structure
```
ml_toolkit
|--clustering
|--db_interaction
|--deep_learning
|--feature_encoding
|--time_series
|--utils
|--webtools
```


## Installation
First clone the repository and navigate to it
```
git clone https://github.com/TSPereira/ml_toolkit.git
cd ml_toolkit
```

Then do one of the following  
`pip install .`  (minimum installation)  
`pip install .[all] -f https://download.pytorch.org/whl/torch_stable.html`  (all extras)  
`pip install .[<extra>]`  (specific extra)

Note: to install torch dependency one needs to add the -f flag as above.