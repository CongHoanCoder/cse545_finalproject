import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml, sys, os
import pickle as pkl
import pulp, shutil, random

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)


import warnings
warnings.filterwarnings('ignore')

# read in the parameters for genetic algorithm from inputs.yml
with open('inputs.yml', 'rb') as f:
    params = yaml.safe_load(f.read())

problem_size = params['n']
preload_data_flag = params['data_load']


class Initializer:
    def __init__(self, n, n_constraints=10):
        self.n = n
        self.n_constraints = n_constraints
    def create_vector_c(self):
        return np.random.randint(-10, 10, self.n)
    def create_matrix_A(self):
        return np.random.randint(-5, 10, (self.n_constraints, self.n))
    def create_vector_b(self):
        return np.random.randint(self.n*2, self.n*10, self.n_constraints)
    def save_initial_data(self, c, A, b, filename=None):
        with open(filename, 'wb') as f:
            pkl.dump({'c': c, 'A': A, 'b': b}, f)

def load_initial_data(filename):
    with open(filename, 'rb') as f:
        data = pkl.load(f)
    return data['c'], data['A'], data['b']

if preload_data_flag:
    c, A, b = load_initial_data(filename=f'data/linear_programming_data_{problem_size}.pkl')
    print("Data loaded from file.")
    print(f"c: {c}\nA: {A}\nb: {b}")
else:
    num_constraints = max(int(problem_size/16), 5)
    initializer = Initializer(problem_size, num_constraints)
    c = initializer.create_vector_c()
    A = initializer.create_matrix_A()
    b = initializer.create_vector_b()
    initializer.save_initial_data(c, A, b, filename=f'data/linear_programming_data_{problem_size}.pkl')
    sys.exit("Data initialized and saved. Please set 'data_load' to True in inputs.yml to load the data next time.")

print(A.shape)
print(b.shape)
print(c.shape)

def solve_integer_programming(c, A, b, method='cbc'):
    """"
    Maximize cᵀx
    subject to Ax ≤ b
    and x ≥ 0, x ∈ Z
    """
    c = np.array(c)
    A = np.array(A)
    b = np.array(b)
    n_vars = len(c)
    n_constraints = len(b)
    # 1. Create a linear programming model (using PuLP)
    model = pulp.LpProblem("MatrixForm_LP", pulp.LpMaximize)
    # 2. Create optimization, integer variables 0 <= x0, x1, x2, ... <= 50
    x = pulp.LpVariable.dicts("x", range(n_vars), lowBound=0, upBound=50, cat="Integer")
    # 3. Objective: maximize cᵀx
    model += pulp.lpSum(c[j] * x[j] for j in range(n_vars))
    # 4. Constraints: A_i * x ≤ b_i
    for i in range(n_constraints):
        model += pulp.lpSum(A[i, j] * x[j] for j in range(n_vars)) <= b[i]
    # 5. Solve
    if method=='cbc':
        # Dual/Primal Simplex
        # Use default solver without specifying path for cross-platform compatibility
        model.solve(pulp.PULP_CBC_CMD(msg=False))
    elif method=='glpk':
        # simplex + Interior Point
        model.solve(pulp.GLPK_CMD(msg=False))
    y = pulp.value(model.objective)
    return x, y

# Solve the integer programming:
# Maximize cᵀx
# subject to Ax ≤ b
# and x ≥ 0, x ∈ Z
x, obj_value = solve_integer_programming(c, A, b, method='cbc')
print("Objective value =", obj_value)
exact_obj = obj_value  # ← this is the global optimum!

# prepare data
results_data = {
    "Variable": [f"x{j}" for j in range(problem_size)],
    "Value": [x[j].value() for j in range(problem_size)]
}
df = pd.DataFrame(results_data)
# df = df.sort_values("Variable").reset_index(drop=True)
df.to_pickle(f"data/optimal_solution_{problem_size}.pkl")
df[df["Value"] > 0].head(problem_size)


class GeneticAlgorithmWOC():
    def __init__(self, c, A, b, lb, ub):
        self.c = c
        self.A = A
        self.b = b
        with open('inputs.yml', 'rb') as f:
            params = yaml.safe_load(f.read())
        self.population_size = params['population_size']
        self.n_gen = params['num_generations']
        self.tournament_size = params['tournament_size']
        self.inv_p = params['inv_p']
        self.swap_p= params['swap_p']
        self.topsoln_frac = params['topsoln_frac']
        self.keep_children = params['keep_children']
        self.stall_patience = params['stall_patience']
        self.n_vars = len(c)
        self.n_constraints = len(b)
        self.lower_bound = lb
        self.upper_bound = ub

    @staticmethod
    def is_x_feasible(x, A, b):
        """ Quick check to see if Ax <= b (constraints) """
        return np.all(A @ np.array(x) <= b + 1e-9)
    
    def eval_fitness(self, solution):
        """
        Calculate the fitness of a solution, x: fitness = np.dot(c,x)
        Fitness is defined as the objective value if constraints are satisfied,
        otherwise a large negative penalty is applied.
        """
        # Check if solution is constrained feasible i.e., whether A*solution <= b,
        # If yes, return optimal c^T * x, else return large negative number
        if self.is_x_feasible(solution, self.A, self.b):
            return np.dot(self.c, solution)
        else:
            return -1e6
        
    ### HELPER FUNCTION, useful in ox and pmx crossover implementation and other functions
    def max_feasible_for_var(self, slack, col, lb, ub):
        """
        Inputs:
            slack  is the slack variable when having non-equality constraint, originally, s = b - Ax
            col    is a column of the constraint matrix A
            lb, ub are lower and upper bounds of each entry in solution vector x
        
        We will use this function for each x_j in the solution vector x

        Given slack with initially should be b - Ax or b[i] - SUM_k A[i,k] x[k]
        the following helps us evaluate how far can we increase x_j
        we know that constraint is: SUM_k A[i,k] x[k] <= b[i]
        currently, slack is:        s[i] = b[i] - SUM_k A[i,k] x[k]
        so increasing x[j] by delta adds A[i,j] * delta to left-hand side
        to stay feasible with constraint, we need A[i,j] * delta <= s[i]
        which means delta <= s[i] / A[i,j]
        => x[j] < min_i s[i] / A[i,j]
        we find the position 'pos' in col such that increasing x[pos] will reduce slack
        """
        pos = col > 0   # extract positions/indices in a column of A that are positive
        if np.any(pos):
            vmax_list = np.floor(slack[pos] / col[pos]).astype(int)
            vmax = int(vmax_list.min()) if vmax_list.size > 0 else ub
        else:
            # If all coefficients <= 0, increasing this variable doesn't hurt feasibility
            vmax = ub
        return min(ub, max(lb, vmax))   # choose feasible value for x[j]

    def generate_feasible_individual(self):
        """
        Greedy constructive heuristic to generate feasible solution x in [lb, ub]^n s.t. A x <= b is satisfied.
        """
        n = self.n_vars
        lb = int(self.lower_bound)
        ub = int(self.upper_bound)
        x = np.zeros(n, dtype=int)
        slack = self.b.astype(float) - self.A @ x   # start with full slack = b - Ax
        order = np.random.permutation(n)            # random permutation from 0 to n-1

        # for each column in matrix A,
        #   we check if col is positive
        for j in order:
            col = self.A[:, j].astype(float)    # this is how much constrait i's LHS changes when increasing x[j]
            vmax = self.max_feasible_for_var(slack, col, lb, ub)
            vmin = lb
            # choose a value for x[j] within [vmin, vmax] 
            if vmax < vmin:
                # No room to increase without violating constraints; pick the safest value
                val = vmin if vmin == 0 else 0
            else:
                val = np.random.randint(vmin, vmax + 1)
            # update slack
            x[j] = int(val)
            slack = slack - col * val
        return x.tolist()

    def create_init_popn(self):
        """
        Create an initial population of feasible solutions (A x <= b) with x integer in [lb, ub].
        """
        initial_pop = []
        tries_per_individual = 10
        for _ in range(self.population_size):
            x = None
            # Try greedy construction multiple times with different orders
            for __ in range(tries_per_individual):
                candidate = self.generate_feasible_individual()
                if self.is_x_feasible(candidate, self.A, self.b):
                    x = candidate
                    break
            initial_pop.append(x)
        return initial_pop
    
    def choose_best_individuals(self, population, fitnesses):
        """
        Select the top fraction of individuals based on their fitness values.
        Top fraction is given by self.topsoln_frac.
        This function will be used in selecting the first few candidates in the next_population.
        """
        num_top = max(1, int(self.topsoln_frac * len(population)))  # num_top = number of top-candidates to be selected
        sorted_indices = np.argsort(fitnesses)[::-1]                # indices of num_top candidates that have best fitnesses
        best_indices = sorted_indices[:num_top]                     # sort the indices
        best_individuals = [population[i] for i in best_indices]    # extract best solutions (these should have largest fitness-values)
        return best_individuals
    
    def randomize_good_solution(self, population, fitnesses, k=10):
        """
        Choose a reasonably good solution out of "k" randomly selected solutions
        """
        n = len(population)
        idx = random.sample(range(n), k)                # randomly select k solutions
        best = max(idx, key=lambda i: fitnesses[i])     # choose the best one out of these k-solutions.
        return population[best]
    
    ################################# CROSSOVER IMPLEMENTATION #################################
    def ox(self,p1,p2):
        """ Order Crossover """
        n = self.n_vars
        assert len(p1) == n and len(p2) == n, "Parents must have length n_vars"
        lb = int(self.lower_bound)
        ub = int(self.upper_bound)

        # Find the slice range randomly for ox cross over
        slice_start, slice_end = sorted(np.random.choice(np.arange(n), size=2, replace=False))

        def build_child(p1, p2):
            """
            p1 = parent 1, p2 = parent 2
            Note: we will take a slice from p1 i.e., p1[slice_start, slice_end] and capped with constraints feasibility
                  then fill remaining indices of child by p2, also capped with constraints feasibility
            """
            x = np.zeros(n, dtype=int)
            # first compute slack
            slack = self.b.astype(float) - self.A @ x
            # 1) copy the middle slice from 'p1' but cap by feasibility
            for k in range(slice_start, slice_end+1):
                col = self.A[:, k].astype(float)        # extract column k which has an effect on x[k]
                gene = int(p1[k])                       # extract the wanted 'gene' from p1
                # then compute the max feasible value for x[k], called vmax
                vmax = self.max_feasible_for_var(slack, col, lb, ub)
                val = max(lb, min(gene, vmax))          # choose x[k]'s value between [lb, gene]
                x[k] = val
                slack = slack - col * val               # update slack after choosing x[k]
            # 2) fill remaining positions in cyclic order using 'p2'
            for k in list(range(slice_end + 1, n)) + list(range(0, slice_start)):
                col = self.A[:, k].astype(float)        # extract column k which has an effect on x[k]
                gene = int(p2[k])                       # extract 'gene' from p2
                # then compute the max feasible value for x[k], called vmax
                # then follow the same logic as above for-loop to fill the remaining slot for our child
                vmax = self.max_feasible_for_var(slack, col, lb, ub)
                val = max(lb, min(gene, vmax))
                x[k] = val
                slack = slack - col * val
            return x.tolist()

        # return 2 children
        return build_child(p1, p2), build_child(p2, p1)
    
    def pmx(self,p1,p2):
        """ Partially Mapped Crossover """
        n = self.n_vars
        assert len(p1) == n and len(p2) == n, "Parents must have length n_vars"
        lb = int(self.lower_bound)
        ub = int(self.upper_bound)

        # find the slice range randomly
        slice_start, slice_end = sorted(np.random.choice(np.arange(n), size=2, replace=False))
        # mapping from P2 -> P1 over the segment [slice_start, slice_end+1]:
        # seg_map_12: on slice: [slice_start, slice_end+1], we create a mapping of element in p1 to p2
        seg_map_12 = {k: (int(p1[k]), int(p2[k])) for k in range(slice_start, slice_end+1)}  # index -> (p1_val, p2_val)
        # map_p2_to_p1: value mapping (by PMX): a value from p2 maps to corresponding value in p1 in the segment
        map_p2_to_p1 = {v: u for (_, (u, v)) in seg_map_12.items()}
        # segment_vals_p1: the set of values in p1 that will be used by perform_mapping()
        segment_vals_p1 = set(int(p1[k]) for k in range(slice_start, slice_end+1))

        def perform_mapping(g):
            """
            Follow mapping P2->P1 until we get a value not in the P1 segment
            This is PMX logic: if the desired value g has conflict with segment that's been copied,
            then we need to follow the mapping chain: g <- map_p2_to_p1[g] <- ... 
            until we get a value that is not in p1 segment
            """
            seen = 0; n = self.n_vars
            while g in segment_vals_p1 and g in map_p2_to_p1:
                g = map_p2_to_p1[g]
                seen += 1
                if seen > n:  # avoid pathological loops
                    break
            return g
        
        def build_child(p1, p2):
            """
            p1 = parent1, p2 = parent2
            To create a child from p1 & p2:
            - Copy a random segment from p1 i.e., p1[slice_start, slice_end]
            - Fill the rest by doing a mapping from p2 -> p1.
            - Enforce constraints feasibility
            """
            x = np.zeros(n, dtype=int)
            slack = self.b.astype(float) - self.A @ x
            # 1) copy the middle segment from 'p1' with feasibility clamp
            for k in range(slice_start, slice_end+1):
                col = self.A[:, k].astype(float)
                gene = int(p1[k])
                vmax = self.max_feasible_for_var(slack, col, lb, ub)
                val = max(lb, min(gene, vmax))
                x[k] = val
                slack = slack - col * val
            # 2) fill outside segment positions using PMX mapping from 'p2'
            for k in list(range(0, slice_start)) + list(range(slice_end+1, n)):
                col = self.A[:, k].astype(float)
                gene = int(p2[k])
                gene = perform_mapping(gene)
                vmax = self.max_feasible_for_var(slack, col, lb, ub)
                val = max(lb, min(gene, vmax))
                x[k] = val
                slack = slack - col * val
            return x.tolist()

        # return 2 childrens
        return build_child(p1, p2), build_child(p2, p1)
    
    ###################################################################################################


    ##################################### MUTATION IMPLEMENTATION #####################################
    
    # def mutation(self, x, p=0.15):

    ###################################################################################################



    def genetic_alg_main(self):
        population = self.create_init_popn()
        fitnesses = [self.eval_fitness(ind) for ind in population]

        best_fitness = max(fitnesses)                 # MAXIMISATION
        best_solution = population[fitnesses.index(best_fitness)]

        history = []

        stall = 0
        for gen in range(self.n_gen):
            # ---- record stats ----
            history.append((gen, best_fitness, np.mean(fitnesses)))

            # ---- elitism ----
            next_pop = self.choose_best_individuals(population, fitnesses)

            # ---- fill the rest ----
            while len(next_pop) < self.population_size:
                p1 = self.randomize_good_solution(population, fitnesses,
                                                k=self.tournament_size)
                p2 = self.randomize_good_solution(population, fitnesses,
                                                k=self.tournament_size)
                if p1 is p2:               # avoid identical parents
                    continue

                # crossover
                ox1, ox2 = self.ox(p1, p2)
                pmx1, pmx2 = self.pmx(p1, p2)

                # mutation (apply to each child with probability swap_p / inv_p)
                children = [ox1, ox2, pmx1, pmx2]
                mutated = []
                for child in children:
                    if random.random() < self.swap_p:
                        child = self.mutate_swap(child)
                    if random.random() < self.inv_p:
                        child = self.mutate_inverse(child)
                    mutated.append(child)

                # keep only feasible (they already are, but safeguard)
                next_pop.extend([c for c in mutated if self.is_x_feasible(c, self.A, self.b)])

                # avoid overflow
                if len(next_pop) > self.population_size:
                    next_pop = next_pop[:self.population_size]

            population = next_pop
            fitnesses = [self.eval_fitness(ind) for ind in population]

            # ---- update best ----
            gen_best = max(fitnesses)
            if gen_best > best_fitness:
                best_fitness = gen_best
                best_solution = population[fitnesses.index(gen_best)]
                stall = 0
            else:
                stall += 1
                if stall >= self.stall_patience:
                    print(f"Early stop at generation {gen}")
                    break

        # ---- final result ----
        return {
            "best_solution": best_solution,
            "best_fitness": best_fitness,
            "history": history
        }

# Feasibility-aware swap and inverse mutations for integer-vector GA
import numpy as _np
import random as _random

def _feasible_from_template(self, template, priority_indices=None):
    n = self.n_vars
    lb = int(self.lower_bound)
    ub = int(self.upper_bound)
    x = _np.zeros(n, dtype=int)
    slack = self.b.astype(float) - self.A @ x
    seen = set()
    order = []
    if priority_indices is not None:
        for idx in priority_indices:
            if 0 <= idx < n and idx not in seen:
                order.append(int(idx)); seen.add(int(idx))
    for j in range(n):
        if j not in seen:
            order.append(j)
    for j in order:
        col = self.A[:, j].astype(float)
        desired = int(template[j])
        vmax = self.max_feasible_for_var(slack, col, lb, ub)
        val = max(lb, min(desired, vmax))
        x[j] = val
        slack = slack - col * val
    return x.tolist()


def _mutate_swap(self, x, p=0.15):
    """
    Swap mutation: swap two positions, then rebuild a feasible child guided by the swapped template.
    """
    if _random.random() >= p:
        return x.copy()
    n = self.n_vars
    i, j = _np.random.choice(_np.arange(n), size=2, replace=False)
    template = x.copy()
    template[i], template[j] = template[j], template[i]
    return _feasible_from_template(self, template, priority_indices=[i, j])


def _mutate_inverse(self, x, p=0.15):
    """
    Inverse mutation: reverse a random segment [i, j], then rebuild a feasible child guided by the template.
    """
    if _random.random() >= p:
        return x.copy()
    n = self.n_vars
    i, j = sorted(_np.random.choice(_np.arange(n), size=2, replace=False))
    template = x.copy()
    template[i:j+1] = template[i:j+1][::-1]
    priority = list(range(i, j+1))
    return _feasible_from_template(self, template, priority_indices=priority)

# Attach to class
GeneticAlgorithmWOC._feasible_from_template = _feasible_from_template
GeneticAlgorithmWOC.mutate_swap = _mutate_swap
GeneticAlgorithmWOC.mutate_inverse = _mutate_inverse


ga = GeneticAlgorithmWOC(c, A, b, lb=0, ub=50)
init_pop = initial_population = ga.create_init_popn()
# ga.genetic_alg_main()

result = ga.genetic_alg_main()
gens, bests, means = zip(*result["history"])
plt.plot(gens, bests, label="Best")
plt.plot(gens, means, label="Mean")
plt.axhline(y=exact_obj, color='r', linestyle='--', label="Exact")
plt.legend(); plt.show()