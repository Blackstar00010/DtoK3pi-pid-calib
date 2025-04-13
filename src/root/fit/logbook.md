# Trials
## 1 Original
### Workflow
1. `n_ss`, `n_sb`, `n_bb` all floating
2. `model` = `n_ss` $f_{X, ss} \otimes f_{Y, ss}$ + `n_sb` $f_{X, sb} \otimes f_{Y, sb}$ + `n_bb` $f_{X, bb} \otimes f_{Y, bb}$
3. fit model
4. set all except `n_xx` constant
5. `SPlot()`

### Problem
- weighted DM looking inverted
- weighted deltaM really wierd
- Fitting not done properly
    - `CovarianceStatus: 0; Covariance and correlation matrix: matrix is not present or not valid`
    - no errors on variables

### What is good
- residuals look nice
- yield = sum of sWeights work

### Result
![[logbookthing/ver1_res.png]]
![[logbookthing/ver1_2dweighted.png]]
![[logbookthing/ver1_dmweighted.png]]
![[logbookthing/ver1_deltamweighted.png]]

## 2
setting `n_bb` as a dependent parameter

### Workflow
1. *`n_ssTemp`, `n_sbTemp` floating and `n_bbTemp = nEntries - n_ss - n_sb`*
2. `model` = `n_ssTemp` $f_{X, ss} \otimes f_{Y, ss}$ + `n_sbTemp` $f_{X, sb} \otimes f_{Y, sb}$ + `n_bbTemp` $f_{X, bb} \otimes f_{Y, bb}$
3. fit `modelTemp`
4. set all except `n_xxTemp` constant
5. *`n_ss`, `n_sb`, `n_bb` all floating, with initialisation from the `n_xxTemp` variables*
6. *second fit*
    - Because `SPlot()` need to use `n_xx` which are not fitted
5. `SPlot()`

### Problem
- weighted DM still not getting to sub-zero
- weighted deltaM still wierd
- fitting not done properly
    - still no covariance for the first fit 

### What is good
- residuals look nice
- covariance and errors exists for the second fit
- yield = sum of sWeights work

### Result
![[logbookthing/ver2_res.png]]
![[logbookthing/ver2_2dweighted.png]]
![[logbookthing/ver2_dmweighted.png]]
![[logbookthing/ver2_deltamweighted.png]]

## 3
reducing the number of sub-models

### Workflow
1. *`n_ssTemp`, `n_sbTemp` floating and `n_bbTemp = nEntries - n_ss - n_sb`*
2. `model` = `n_ssTemp` $f_{X, s} \otimes f_{Y, s}$ + `n_sbTemp` $f_{X, s} \otimes f_{Y, b}$ + `n_bbTemp` $f_{X, b} \otimes f_{Y, b}$
    - instead of 2(X, Y) * 3(ss, sb, bb) sub-models, use 2 * 2(s, b) sub-models
3. fit `modelTemp`
4. set all except `n_xxTemp` constant
5. *`n_ss`, `n_sb`, `n_bb` all floating, with initialisation from the `n_xxTemp` variables*
6. *second fit*
    - Because `SPlot()` need to use `n_xx` which are not fitted
5. `SPlot()`

### Problem
- no improvement in output quality over the second try
    - still problems with DM, deltaM
    - still no covariance for the first fit

### What is good
- shorter execution time

### Result
![[logbookthing/ver2_res.png]]
![[logbookthing/ver3_2dweighted.png]]
![[logbookthing/ver3_dmweighted.png]]
![[logbookthing/ver3_deltamweighted.png]]