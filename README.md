## Objective
The objective of the repository is to share generic functions that everybody can use on the day to day work.


## Contribution Flow Summary
To contribute one should clone the repo, add the functions to share and then perform a pull request into the develop
branch.

Pull requests must by well documented and fully working. A test example should be provided such that the code added can 
be tested on review.


## Package Main Structure
```
src
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
minimum install: `pip install .`  
all extras: `pip install .[all] -f https://download.pytorch.org/whl/torch_stable.html`  
specific extra: `pip install .[<extra>]`

Note: to install torch dependancy one needs to had the -f flag as above.