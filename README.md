# nelder-mead
A basic Python implementation of the Nelder-Mead Method - zero-order optimization algorithm.

## Usage

``` python
from nelder_mead import *

optimizer = NelderMeadOptimizer(e_area=10e-7, e_value=-50, max_iters=15, obj_func=mishra_bird_func)
optimizer.optimize()

print(f'Best simplex:\n{optimizer.all_simplexes[-1]}')
print(f'Best value:\n{optimizer.simplex_values[-1]}')
```
<img src="https://github.com/spaiker7/nelder-mead/assets/70488161/c286060e-4d97-4c14-be08-dfe13e6434b1" height=300%>
