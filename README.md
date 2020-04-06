# README
```
The objective of the repository is to share generic functions that everybody can use on the day to day work.
```

## Contribution Flow
```
To contribute one should clone the repo, add the functions to share and then perform a pull request into the develop
branch.

This pull request will have to be approved by at least two people.
Pull requests must by well documented and fully working. A test example should be provided such that the code added can 
be tested on review.

Once the develop branch reaches a certain maturity level, one of the reviewers will merge it to master branch (add tag!).

Eventually a new release to master can deploy the package on Nexus so that everybody can update via "pip install".
```

## Usage
```
Ideally you should use the master branch, since it has been reviewed and should be free of bugs.

However it is normal that one might need to use some function that might be still on the develop branch, so this branch
can also be used on the day to day work. 

Try that any new "functionality tool" that uses this package only use functions existing in the master branch!
```

## Repository Structure
```
standardpackage
|  .gitignore
|  README.md
|
|--utils
|    datetime_utils
|    log_utils
|    os_utils
|    other_utils
|    text_utils
|  
```
