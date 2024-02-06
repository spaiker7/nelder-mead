import numpy as np

import itertools
from copy import deepcopy

class NelderMeadOptimizer():

    def __init__(self,
        obj_func,
        e_area, e_value, max_iters=200,
        alpha=1, beta=0.5, gamma=2,
        num_dimensions=2,
        init_simplex=None
    
        ):
        """
        Parameteres
        -----------
        num_dimensions: number of variables function utilize

        alpha: > 0, coefficient of the reflection
        beta: > 0, coefficient of the contraction
        gamma: > 0, coefficient of the expansion

        stopping criteria:
            e_value: minimal values difference between simplex points
            e_area: minimal area between simplex points
            max_iters: maximum iterations of the algorithm
        """
        super().__init__()

        self.obj_func = obj_func
        self.n = num_dimensions

        self.simplex_values = np.empty((3,))

        # zero-order oracle calls list to track at each step
        self.oracle_calls = [0]

        if not init_simplex:
            self.simplex_points = self.init_simplex()
        else:
            self.simplex_points = init_simplex
            # if any of the values is outside feasibility domain (nan)
            if any(np.isnan(self.simplex_values)):
                raise Exception('NaN values in the initial simplex values')
            for i, point in enumerate(self.simplex_points):
                self.simplex_values[i] = self.obj_func(point)
                self.oracle_calls[-1] += 1


        
        # list for tracking all simplexes
        self.all_simplexes = []

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        if any(param <=0 for param in [alpha, beta, gamma]):
            print([alpha, beta, gamma])
            raise Exception(f'Some coefficient(s) is non-positive')
        
        self.iters = 0

        self.e_area = e_area
        self.e_value = e_value
        self.max_iters = max_iters    

    def init_simplex(self):
        """
        Randomly pick the points for initial simplex.
        """
        initial_points = []
        for i in range(self.n+1):
            func_value = np.nan
            while np.isnan(func_value):

                # ADJUST TO YOUR FUNCTION FEASIBILITY DOMAIN
                point = np.random.uniform(-10, 0, self.n)

                func_value = self.obj_func(point)
                self.oracle_calls[-1] +=1
            self.simplex_values[i] = func_value
            initial_points.append(point)

        if self.oracle_calls[-1] > 3:
            print(f'{self.oracle_calls[-1]} oracle calls were used to initialize simplex points \
                (if the number high - adjust the domain search or provide your own points).')

        return np.array(initial_points)


    def sort_points_by_values(self):
        """
        Sort simplex points and values by values
        """

        self.simplex_points = self.simplex_points[np.argsort(self.simplex_values)]
        self.simplex_values = np.sort(self.simplex_values)
        

    def center_of_gravity(self, points):
        """
        Returns the center of gravity of points without the worst one.

        x_c: center of the gravity
        """
        x_c = np.sum(points[:-1], axis=0) / points[:-1].shape[0]

        return x_c

        
    def reflection(self, x_h, x_c, alpha):
        """
        Reflects the worst point to the center of the gravity by parameter alpha > 0:

        x_r: reflected point
        x_c: center of the gravity
        x_h: worst point (corresponds to highest obj function value)
        """

        x_r = x_c + alpha * (x_c - x_h)

        f_r = self.obj_func(x_r)
        self.oracle_calls[-1] += 1

        return x_r, f_r
    
    def expansion(self, x_c, x_r, gamma):
        """
        Expanse the reflected point further by parameter gamma > 0:

        x_e: expanded point
        x_r: reflected point
        x_c: center of the gravity
        x_h: worst point (corresponds to highest obj function value)
        """

        x_e = x_c + gamma*(x_r - x_c)
        f_e = self.obj_func(x_e)
        self.oracle_calls[-1] += 1

        return x_e, f_e

    def contraction(self, x_c, x_h):
        """
        Make contraction back from the center point by parameter beta > 0:

        x_s: contracted point
        x_c: center of the gravity
        x_h: worst point (corresponds to highest obj function value)
        """

        x_s = x_c + self.beta*(x_h - x_c)
        f_s = self.obj_func(x_s)
        self.oracle_calls[-1] += 1

        return x_s, f_s


    def shrinkage(self):
        
        """
        Simplex contraction of remaining points besides the new best one at index -1

        """
        
        for i, x_i in enumerate(self.simplex_points[:-1]):
            self.simplex_points[i] = self.simplex_points[-1] + (x_i - self.simplex_points[-1]) / 2
            self.simplex_values[i] = self.obj_func(x_i)
            self.oracle_calls[-1] += 1
    
    def calculate_area(self, points):
        """
        Returns current area between points stopping critera
        """
        return 0.5 * np.abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(np.roll(points[:, 0], 1), points[:, 1]))
        

    def calculate_max_value_diff(self, values):
        """
        Returns current max difference between func values for stopping critera
        """
        all_value_diffs = []
        for subset in itertools.combinations(values, 2):
            all_value_diffs.append(np.linalg.norm(np.array(subset[0]) - np.array(subset[1])))
        return np.max(all_value_diffs)
    
    def update_state(self):

        self.all_simplexes.append(self.simplex_points.copy())
        self.area = self.calculate_area(self.simplex_points)
        self.max_value_diff = self.calculate_max_value_diff(self.simplex_values)
        

    def optimize(self):
        
        self.update_state()

        # while stopping criteria has not been meeted:
        while  self.area > self.e_area and \
                self.max_value_diff > self.e_value and \
                self.iters < self.max_iters:
            
            self.oracle_calls.append(0)
            self.iters +=1

            self.sort_points_by_values()
            x_c = self.center_of_gravity(self.simplex_points)
            x_h = self.simplex_points[-1]

            f_l = self.simplex_values[0]
            f_g = self.simplex_values[1]
            f_h = self.simplex_values[-1]
            
            current_alpha = deepcopy(self.alpha)
            f_r = np.nan
            while np.isnan(f_r):
                x_r, f_r = self.reflection(x_h, x_c, current_alpha)
                # reduce the power of reflection if the value outside fesibility domain
                current_alpha /= 2
                if self.oracle_calls[-1] / self.iters > self.n * 10:
                    print('Could not reflect outside the fesibility domain')
                    break

            if f_r < f_l:

                current_gamma = deepcopy(self.gamma)
                f_e = np.nan
                while np.isnan(f_e):
                    x_e, f_e = self.expansion(x_c, x_r, current_gamma)
                    # reduce the power of reflection if the value outside fesibility domain
                    current_gamma /= 2
                    if self.oracle_calls[-1] / self.iters > self.n * 10:
                        print('Could not reflect outside the fesibility domain')
                        break
                
                if f_e < f_l:
                    self.simplex_points[-1] = x_e
                    self.simplex_values[-1] = f_e
                    self.update_state()


                else:
                    self.simplex_points[-1] = x_r
                    self.simplex_values[-1] = f_r
                    self.update_state()

            else:

                if f_r < f_g:
                    self.simplex_points[-1] = x_r
                    self.simplex_values[-1] = f_r
                    self.update_state()

                else:

                    if f_r < f_h:
                        self.simplex_points[-1] = x_r
                        self.simplex_values[-1] = f_r

                    x_s, f_s = self.contraction(x_c, x_h)

                    if f_s < f_h:
                        self.simplex_points[-1] = x_s
                        self.simplex_values[-1] = f_s
                        self.update_state()
                
                    else:
                        self.shrinkage()
                        self.update_state()