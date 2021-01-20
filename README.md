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
|--clustering [extra]
|--db_interaction [extra]
|--deep_learning [extra]
|--feature_encoding [extra]
|--time_series [extra]
|--utils
|--webtools [extra]
```


## Installation
There are two options:  

Stable version:  
    `pip install ml-toolkit --extra-index-url https://api.packagr.app/public`

Directly from the repository  
&nbsp;&nbsp;&nbsp;&nbsp;Clone the repository, navigate to it and install
    
```
git clone https://github.com/TSPereira/ml_toolkit.git
cd ml_toolkit
pip install .
```

Please note that some extra options might need for you to define a `find-links` flag.  
Options for installation as below.  

`pip install .`  (minimum installation)  
`pip install .[all] -f https://download.pytorch.org/whl/torch_stable.html`  (all extras)  
`pip install .[deep_learning] -f https://download.pytorch.org/whl/torch_stable.html`    
`pip install .[<extra>]`  (specific extra)
`pip install .[<extra1>, <extra2>]`  (multiple extras)

To include in a requirements.txt you must add the flags before

```
numpy==1.18.5
--extra-index-url https://api.packagr.app/public -f https://download.pytorch.org/whl/torch_stable.html
ml-toolkit[all]==0.1.1
```

Note: Some internal packages might need additional dependencies (hdbscan needs either MVSC14.0++ or gcc for example)
