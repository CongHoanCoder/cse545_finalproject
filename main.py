

# ----------------------------------------------------------------------
# 1. IMPORTS
# ----------------------------------------------------------------------
import os
import sys
import time
import random
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import pickle as pkl
import pulp
from scipy.optimize import linprog

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)

# ----------------------------------------------------------------------
# 2. CONFIGURATION (inputs.yml)
# ----------------------------------------------------------------------
with open("inputs.yml", "rb") as f:
    params = yaml.safe_load(f.read())

problem_size = params["n"]
preload_data_flag = params["data_load"]

# ----------------------------------------------------------------------
# 3. DATA INITIALISER / LOADER
# ----------------------------------------------------------------------
class Initializer:
    def __init__(self, n, n_constraints=10):
        self.n = n
        self.n_constraints = n_constraints

    def create_vector_c(self):
        return np.random.randint(-10, 10, self.n)

    def create_matrix_A(self):
        return np.random.randint(-5, 10, (self.n_constraints, self.n))

    def create_vector_b(self):
        return np.random.randint(self.n * 2, self.n * 10, self.n_constraints)

    def save_initial_data(self, c, A, b, filename=None):
        with open(filename, "wb") as f:
            pkl.dump({"c": c, "A": A, "b": b}, f)


def load_initial_data(filename):
    with open(filename, "rb") as f:
        data = pkl.load(f)
    return data["c"], data["A"], data["b"]


# ----------------------------------------------------------------------
if preload_data_flag:
    c, A, b = load_initial_data(filename=f"data/linear_programming_data_{problem_size}.pkl")
else:
    num_constraints = max(int(problem_size / 16), 5)
    initializer = Initializer(problem_size, num_constraints)
    c = initializer.create_vector_c()
    A = initializer.create_matrix_A()
    b = initializer.create_vector_b()
    initializer.save_initial_data(
        c, A, b, filename=f"data/linear_programming_data_{problem_size}.pkl"
    )
    sys.exit(
        "Data initialized and saved. Please set 'data_load' to True in inputs.yml."
    )

print(A.shape, b.shape, c.shape)

# ----------------------------------------------------------------------
# 4. EXACT INTEGER PROGRAMMING SOLVER (PuLP + CBC)
# ----------------------------------------------------------------------
def solve_integer_programming(c, A, b, method="cbc"):
    c = np.array(c)
    A = np.array(A)
    b = np.array(b)
    n_vars = len(c)
    n_constraints = len(b)

    model = pulp.LpProblem("MatrixForm_LP", pulp.LpMaximize)
    x = pulp.LpVariable.dicts(
        "x", range(n_vars), lowBound=0, upBound=50, cat="Integer"
    )
    model += pulp.lpSum(c[j] * x[j] for j in range(n_vars))
    for i in range(n_constraints):
        model += pulp.lpSum(A[i, j] * x[j] for j in range(n_vars)) <= b[i]

    if method == "cbc":
        model.solve(pulp.PULP_CBC_CMD(msg=False))
    elif method == "glpk":
        model.solve(pulp.GLPK_CMD(msg=False))

    return x, pulp.value(model.objective)


# Solve exact IP (only for small n)
if problem_size >= 255:
    print("Problem size too large for exact solver – reading pre-computed solution")
    filename = f"data/optimal_solution_{problem_size}.pkl"
    with open(filename, "rb") as f:
        df = pkl.load(f)
    obj_ip = sum(df.loc[j, "Value"] * c[j] for j in range(problem_size))
    print("IP objective =", obj_ip)
else:
    x_ip_vars, obj_ip = solve_integer_programming(c, A, b, method="cbc")
    print("IP objective =", obj_ip)
    results_data = {
        "Variable": [f"x{j}" for j in range(problem_size)],
        "Value": [x_ip_vars[j].value() for j in range(problem_size)],
    }
    df = pd.DataFrame(results_data)
    df.to_pickle(f"data/optimal_solution_{problem_size}.pkl")

x_ip = df["Value"].values

# ----------------------------------------------------------------------
# 5. GENETIC ALGORITHM + LP + WoAC
# ----------------------------------------------------------------------
class GeneticAlgorithmWOC:
    def __init__(self, c, A, b, lb, ub):
        self.c = np.asarray(c, dtype=float)
        self.A = np.asarray(A, dtype=float)
        self.b = np.asarray(b, dtype=float)

        # ----- GA parameters ------------------------------------------------
        self.population_size = params["population_size"]
        self.n_gen = params["num_generations"]
        self.tournament_size = params["tournament_size"]
        self.topsoln_frac = params["topsoln_frac"]
        self.keep_children = params["keep_children"]
        self.stall_patience = params["stall_patience"]
        self.delta_mutate_p = params["delta_mutate_p"]

        # ----- LP parameters ------------------------------------------------
        self.use_lp_relaxation = params["use_lp_relaxation"]
        self.lp_reinject_every = params["lp_reinject_every"]
        self.n_lp_solutions_frac = params["n_lp_solutions_frac"]

        # ----- WoAC parameters -----------------------------------------------
        self.use_woac = params.get("use_woac", True)
        self.woac_frac = params.get("woac_frac", 0.2)
        self.woac_every = params.get("woac_every", 25)

        # ----- problem size --------------------------------------------------
        self.n_vars = len(c)
        self.lower_bound = lb
        self.upper_bound = ub

    # ------------------------------------------------------------------
    # 5.1  Feasibility & fitness
    # ------------------------------------------------------------------
    @staticmethod
    def is_x_feasible(x, A, b):
        return np.all(A @ np.array(x) <= b + 1e-9)

    def eval_fitness(self, solution):
        if self.is_x_feasible(solution, self.A, self.b):
            return np.dot(self.c, solution)
        return -1e6

    # ------------------------------------------------------------------
    # 5.2  Greedy constructive heuristic (used for init & repair)
    # ------------------------------------------------------------------
    def max_feasible_for_var(self, slack, col, lb, ub):
        pos = col > 0
        if np.any(pos):
            vmax = np.floor(slack[pos] / col[pos]).min()
        else:
            vmax = ub
        return int(min(ub, max(lb, vmax)))

    def generate_feasible_individual(self):
        x = np.zeros(self.n_vars, dtype=int)
        slack = self.b - self.A @ x
        order = np.random.permutation(self.n_vars)
        lb = int(self.lower_bound)
        ub = int(self.upper_bound)
        for j in order:
            col = self.A[:, j]
            vmax = self.max_feasible_for_var(slack, col, lb, ub)
            val = np.random.randint(lb, vmax + 1) if vmax >= lb else lb
            x[j] = val
            slack -= col * val
        return x.tolist()

    # ------------------------------------------------------------------
    # 5.3  Population creation
    # ------------------------------------------------------------------
    def create_init_popn(self):
        pop = []
        tries = 20
        while len(pop) < self.population_size:
            for _ in range(tries):
                cand = self.generate_feasible_individual()
                if self.is_x_feasible(cand, self.A, self.b):
                    pop.append(cand)
                    break
            else:  # fallback
                rand = np.random.randint(self.lower_bound, self.upper_bound + 1, self.n_vars)
                pop.append(self.solution_improve(rand.tolist()))
        return pop

    # ------------------------------------------------------------------
    # 5.4  Selection helpers
    # ------------------------------------------------------------------
    def choose_best_individuals(self, population, fitnesses):
        n_keep = max(1, int(self.topsoln_frac * len(population)))
        idx = np.argsort(fitnesses)[::-1][:n_keep]
        return [population[i] for i in idx]

    def randomize_good_solution(self, population, fitnesses):
        idx = np.random.choice(len(population), size=self.tournament_size, replace=False)
        return population[max(idx, key=lambda i: fitnesses[i])]

    # ------------------------------------------------------------------
    # 5.5  Crossovers (all feasibility-aware)
    # ------------------------------------------------------------------
    def one_point_crossover(self, p1, p2):
        cut = np.random.randint(1, self.n_vars)
        c1 = np.array(p1, dtype=int).copy()
        c2 = np.array(p2, dtype=int).copy()
        c1[cut:], c2[cut:] = c2[cut:], c1[cut:]
        c1 = np.clip(c1, self.lower_bound, self.upper_bound)
        c2 = np.clip(c2, self.lower_bound, self.upper_bound)
        return c1.tolist(), c2.tolist()

    def uniform_crossover(self, p1, p2):
        mask = np.random.rand(self.n_vars) < 0.5
        child = np.where(mask, np.asarray(p1, int), np.asarray(p2, int))
        return np.clip(child, self.lower_bound, self.upper_bound).tolist()

    def blend_crossover(self, p1, p2, alpha=0.5):
        lb, ub = int(self.lower_bound), int(self.upper_bound)
        p1f = np.asarray(p1, float)
        p2f = np.asarray(p2, float)
        w = np.random.uniform(-alpha, 1 + alpha, self.n_vars)
        child = (1 - w) * p1f + w * p2f
        return np.rint(np.clip(child, lb, ub)).astype(int).tolist()

    # ------------------------------------------------------------------
    # 5.6  Mutation
    # ------------------------------------------------------------------
    def delta_mutate(self, x, p=None, max_changes=4, step=4):
        if p is None:
            p = self.delta_mutate_p
        if np.random.rand() > p:
            return x.copy()
        x = np.array(x, dtype=int)
        idxs = np.random.choice(
            self.n_vars, size=np.random.randint(1, max_changes + 1), replace=False
        )
        for j in idxs:
            delta = np.random.randint(-step, step + 1)
            x[j] = int(np.clip(x[j] + delta, self.lower_bound, self.upper_bound))
        return x.tolist()

    # ------------------------------------------------------------------
    # 5.7  Local improvement (repair + fill slack + tiny perturbations)
    # ------------------------------------------------------------------
    def solution_improve(self, x):
        """Greedy repair of infeasibility."""
        x = np.clip(np.array(x, float), self.lower_bound, self.upper_bound)
        lb, ub = float(self.lower_bound), float(self.upper_bound)
        A, b, c = self.A, self.b, self.c
        for _ in range(min(50, self.n_vars * 2)):
            viol = A @ x - b
            if np.all(viol <= 1e-9):
                break
            violated = viol > 1e-9
            pos = A > 0
            help = np.sum(A * (violated[:, None] & pos), axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(help > 1e-12, np.maximum(c, 0) / help, np.inf)
            j = np.argmin(ratio)
            if not np.isfinite(ratio[j]) or help[j] <= 1e-12:
                cand = np.where(help > 1e-12)[0]
                if len(cand) == 0:
                    break
                j = np.random.choice(cand)
            col = A[:, j]
            pos_rows = (col > 0) & violated
            delta_needed = np.max(viol[pos_rows] / col[pos_rows]) if np.any(pos_rows) else 0.0
            new_val = max(lb, x[j] - max(1.0, np.ceil(delta_needed)))
            if new_val == x[j]:
                help[j] = 0.0
                continue
            x[j] = new_val
        return np.rint(np.clip(x, lb, ub)).astype(int).tolist()

    def fill_slack_improve(self, x):
        """Greedily increase high-profit variables that still have slack."""
        x = np.array(x, dtype=float)
        slack = self.b - self.A @ x
        pos = self.A > 0
        score = np.zeros(self.n_vars)
        for j in range(self.n_vars):
            denom = np.sum(np.where(pos[:, j], self.A[:, j], 0.0))
            score[j] = self.c[j] / (denom + 1e-10)
        order = np.argsort(-score)
        for j in order:
            col = self.A[:, j]
            p = col > 0
            if not np.any(p):
                if self.c[j] > 0:
                    x[j] = min(self.upper_bound, x[j] + 1)
                continue
            delta_max = np.min(slack[p] / col[p])
            if delta_max > 0:
                delta = min(delta_max, self.upper_bound - x[j])
                if delta > 0.5 and self.c[j] > 0:
                    delta = np.floor(delta)
                    x[j] += delta
                    slack -= col * delta
        return np.rint(np.clip(x, self.lower_bound, self.upper_bound)).astype(int).tolist()

    def local_improve(self, x, max_steps=5):
        """Small ±1/2/3 perturbations on the highest-c variables."""
        x = np.array(x, dtype=int)
        lhs = self.A @ x
        deltas = [-3, -2, -1, 1, 2, 3]
        K = min(32, self.n_vars)
        top_c_idx = np.argsort(-self.c)[:K]
        for _ in range(max_steps):
            best_gain = 0
            best_j, best_d = None, 0
            for j in top_c_idx:
                for d in deltas:
                    nx = x[j] + d
                    if nx < self.lower_bound or nx > self.upper_bound:
                        continue
                    new_lhs = lhs + self.A[:, j] * d
                    if np.all(new_lhs <= self.b + 1e-9):
                        gain = self.c[j] * d
                        if gain > best_gain:
                            best_gain, best_j, best_d = gain, j, d
            if best_gain <= 0:
                break
            x[best_j] += best_d
            lhs = lhs + self.A[:, best_j] * best_d
        return x.tolist()

    def perform_soln_improvement(self, x):
        x = self.solution_improve(x)
        x = self.fill_slack_improve(x)
        x = self.local_improve(x, max_steps=5)
        return x

    # ------------------------------------------------------------------
    # 5.8  LP relaxation helpers
    # ------------------------------------------------------------------
    def solve_lp_relaxation(self):
        n = self.n_vars
        lb = np.full(n, self.lower_bound, dtype=float)
        ub = np.full(n, self.upper_bound, dtype=float)
        res = linprog(
            c=-self.c,
            A_ub=self.A,
            b_ub=self.b,
            bounds=list(zip(lb, ub)),
            method="highs",
        )
        if res.success:
            return np.asarray(res.x, dtype=float), -res.fun, None
        return None, None, None

    def round_lp_soln(self, x_lp):
        frac = x_lp - np.floor(x_lp)
        toss = np.random.rand(self.n_vars)
        x = np.floor(x_lp) + (toss < frac)
        return np.clip(x, self.lower_bound, self.upper_bound).astype(int)

    def generate_some_lp_solutions(self, n_soln=12, trials_per_seed=3):
        x_lp, _, _ = self.solve_lp_relaxation()
        if x_lp is None:
            return []
        sols = []
        for _ in range(n_soln):
            best, best_f = None, -np.inf
            for _ in range(trials_per_seed):
                cand = self.round_lp_soln(x_lp).tolist()
                cand = self.perform_soln_improvement(cand)
                f = self.eval_fitness(cand)
                if f > best_f:
                    best_f, best = f, cand
            sols.append(best)
        sols.sort(key=lambda s: self.eval_fitness(s), reverse=True)
        return sols

    def reinitialize_population_using_lprelax_random(self, population):
        n_lp = max(1, int(self.n_lp_solutions_frac * self.population_size))
        lp_sols = self.generate_some_lp_solutions(n_soln=n_lp)
        if not lp_sols:
            return population
        pop_fit = np.array([self.eval_fitness(ind) for ind in population])
        lp_fit = np.array([self.eval_fitness(ind) for ind in lp_sols])
        k = min(len(lp_sols), len(population))
        worst_idx = np.argpartition(pop_fit, k - 1)[:k]
        best_lp_idx = np.argsort(-lp_fit)[:k]
        seen = {tuple(ind) for ind in population}
        for wi, li in zip(worst_idx, best_lp_idx):
            if lp_fit[li] > pop_fit[wi] and tuple(lp_sols[li]) not in seen:
                population[wi] = lp_sols[li]
                pop_fit[wi] = lp_fit[li]
                seen.add(tuple(lp_sols[li]))
        return population

    def reinject_lp_solutions(self, population, n_new=5):
        lp_sols = self.generate_some_lp_solutions(n_soln=n_new, trials_per_seed=3)
        pop_fit = np.array([self.eval_fitness(ind) for ind in population])
        lp_fit = np.array([self.eval_fitness(ind) for ind in lp_sols])
        worst_idx = np.argsort(pop_fit)[:n_new]
        best_lp_idx = np.argsort(-lp_fit)[:n_new]
        for wi, li in zip(worst_idx, best_lp_idx):
            if lp_fit[li] > pop_fit[wi]:
                population[wi] = lp_sols[li]
                pop_fit[wi] = lp_fit[li]
        return population

    # ------------------------------------------------------------------
    # 5.9  Wisdom of Artificial Crowds (WoAC)
    # ------------------------------------------------------------------
    def aggregate_woac(self, candidates):
        """Component-wise majority vote (median tie-breaker)."""
        if not candidates:
            return None
        cand_arr = [np.array(s, dtype=int) for s in candidates]
        agg = np.zeros(self.n_vars, dtype=int)
        for j in range(self.n_vars):
            col = [c[j] for c in cand_arr]
            cnt = Counter(col)
            maj_val, maj_cnt = cnt.most_common(1)[0]
            if maj_cnt < (len(cand_arr) / 2 + 1e-6):  # weak majority → median
                maj_val = int(np.median(col))
            agg[j] = np.clip(maj_val, self.lower_bound, self.upper_bound)
        return agg.tolist()

    # ------------------------------------------------------------------
    # 5.10  MAIN GA LOOP
    # ------------------------------------------------------------------
    def genetic_alg_main(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # ---- initialise population -------------------------------------------------
        population = self.create_init_popn()
        if self.use_lp_relaxation:
            population = self.reinitialize_population_using_lprelax_random(population)
        population = [self.perform_soln_improvement(ind) for ind in population]

        fitness = [self.eval_fitness(ind) for ind in population]
        best_fit = max(fitness)
        best_sol = population[fitness.index(best_fit)]
        stall = 0
        hist_best, hist_mean = [], []

        for gen in range(self.n_gen):
            # ---- record statistics -------------------------------------------------
            cur_best = max(fitness)
            cur_mean = np.mean(fitness)
            hist_best.append(cur_best)
            hist_mean.append(cur_mean)

            if cur_best > best_fit:
                best_fit, best_sol = cur_best, population[fitness.index(cur_best)]
                stall = 0
            else:
                stall += 1
                if stall >= self.stall_patience:
                    print(f"Early stopping at gen {gen} (stall={self.stall_patience})")
                    break

            # ---- elitism -----------------------------------------------------------
            next_pop = self.choose_best_individuals(population, fitness)
            next_pop = [self.perform_soln_improvement(ind) for ind in next_pop]

            # ---- generate children -------------------------------------------------
            while len(next_pop) < self.population_size:
                p1 = self.randomize_good_solution(population, fitness)
                p2 = self.randomize_good_solution(population, fitness)
                while p1 == p2:
                    p2 = self.randomize_good_solution(population, fitness)

                # 7 children from 3 crossover types
                c1, c2 = self.one_point_crossover(p1, p2)
                c3 = self.uniform_crossover(p1, p2)
                c4 = self.uniform_crossover(p2, p1)
                c5 = self.blend_crossover(p1, p2, alpha=0.3)
                c6 = self.blend_crossover(p1, p2, alpha=0.4)
                c7 = self.blend_crossover(p1, p2, alpha=0.5)

                children = []
                for ch in [c1, c2, c3, c4, c5, c6, c7]:
                    ch = self.delta_mutate(ch, max_changes=4, step=4)
                    ch = self.perform_soln_improvement(ch)
                    children.append(ch)

                child_fit = [self.eval_fitness(ch) for ch in children]
                best_idx = np.argsort(child_fit)[-self.keep_children :][::-1]
                for i in best_idx:
                    next_pop.append(children[i])

            # ---- LP reinjection ----------------------------------------------------
            if self.use_lp_relaxation and gen > 0 and gen % self.lp_reinject_every == 0:
                next_pop = self.reinject_lp_solutions(next_pop, n_new=5)

            # ---- WoAC injection ----------------------------------------------------
            if (
                self.use_woac
                and gen > 0
                and gen % self.woac_every == 0
            ):
                crowd_sz = max(1, int(self.woac_frac * len(next_pop)))
                sorted_idx = np.argsort([self.eval_fitness(ind) for ind in next_pop])[::-1]
                crowd = [next_pop[i] for i in sorted_idx[:crowd_sz]]
                agg = self.aggregate_woac(crowd)
                if agg is not None:
                    agg = self.perform_soln_improvement(agg)
                    agg_f = self.eval_fitness(agg)
                    worst_i = np.argmin([self.eval_fitness(ind) for ind in next_pop])
                    if agg_f > self.eval_fitness(next_pop[worst_i]):
                        next_pop[worst_i] = agg
                        print(f"Gen {gen}: WoAC injected (fit={agg_f:.2f})")

            # ---- next generation ---------------------------------------------------
            population = next_pop
            fitness = [self.eval_fitness(ind) for ind in population]

        # ---- final result -------------------------------------------------------
        return {
            "optimal_solution": best_sol,
            "optimal_objective": best_fit,
            "history_of_best": hist_best,
            "history_of_fitness": hist_mean,
        }


# ----------------------------------------------------------------------
# 6. MULTI-RUN + ENSEMBLE WoAC
# ----------------------------------------------------------------------
list_of_seeds = random.sample(range(10 * params["n_ga_runs"]), params["n_ga_runs"])
print("Seeds:", list_of_seeds)

best_obj = -np.inf
best_results = None
all_best_solutions = []
all_histories = []          # for plotting all runs
elapsed_times = []
objective_values = []

for sd in list_of_seeds:
    ga = GeneticAlgorithmWOC(c, A, b, lb=0, ub=50)
    t0 = time.perf_counter()
    res = ga.genetic_alg_main(seed=sd)
    dt = time.perf_counter() - t0

    elapsed_times.append(dt)
    objective_values.append(res["optimal_objective"])
    all_best_solutions.append(res["optimal_solution"])
    all_histories.append(res["history_of_best"])

    print(f"seed {sd:4d} | obj {res['optimal_objective']:8.2f} | time {dt:5.2f}s")

    if res["optimal_objective"] > best_obj:
        best_obj = res["optimal_objective"]
        best_results = res

# ----------------------------------------------------------------------
# 7. ENSEMBLE WoAC ACROSS ALL RUNS
# ----------------------------------------------------------------------
if params.get("use_woac", True) and len(all_best_solutions) > 1:
    ens = GeneticAlgorithmWOC(c, A, b, lb=0, ub=50)  # reuse aggregation method
    ens_sol = ens.aggregate_woac(all_best_solutions)
    ens_sol = ens.perform_soln_improvement(ens_sol)
    ens_obj = ens.eval_fitness(ens_sol)
    if ens_obj > best_obj:
        best_obj = ens_obj
        best_results["optimal_solution"] = ens_sol
        best_results["optimal_objective"] = ens_obj
        print(f"ENSEMBLE WoAC improved final objective → {ens_obj:.2f}")

# ----------------------------------------------------------------------
# 8. FINAL COMPARISON
# ----------------------------------------------------------------------
x_ga = np.array(best_results["optimal_solution"], dtype=float)
obj_ga = np.dot(c, x_ga)
rel_err = abs(obj_ga - obj_ip) / abs(obj_ip) if obj_ip != 0 else 0.0

print("\n=== FINAL RESULTS ===")
print(f"IP  objective : {obj_ip: .6f}")
print(f"GA  objective : {obj_ga: .6f}")
print(f"Relative error: {rel_err:.6%}")
print(f"Feasible?      {np.all(A @ x_ga <= b + 1e-9)}")

# ----------------------------------------------------------------------
# 9. VISUALISATION FUNCTIONS
# ----------------------------------------------------------------------
os.makedirs("plots", exist_ok=True)

def plot_convergence(hist_best, hist_mean, title, save_path):
    plt.figure(figsize=(10, 6))
    gens = range(len(hist_best))
    plt.plot(gens, hist_best, label="Best", color="green", lw=2)
    plt.plot(gens, hist_mean, label="Mean", color="steelblue", alpha=0.7)
    plt.axhline(obj_ip, color="red", ls="--", lw=2, label=f"IP = {obj_ip:.1f}")
    plt.title(title, fontsize=14)
    plt.xlabel("Generation")
    plt.ylabel("Objective")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()

def plot_all_runs(histories, title, save_path):
    plt.figure(figsize=(11, 6))
    for i, h in enumerate(histories):
        plt.plot(h, lw=1.2, alpha=0.6, label=f"Run {i+1}" if i < 5 else None)
    plt.axhline(obj_ip, color="red", ls="--", lw=2, label=f"IP = {obj_ip:.1f}")
    plt.title(title, fontsize=14)
    plt.xlabel("Generation")
    plt.ylabel("Best Objective")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()

def plot_solution_heatmap(x_ga, x_ip, title, save_path):
    df = pd.DataFrame(
        {"Variable": [f"x{i}" for i in range(len(x_ga))], "GA": x_ga, "IP": x_ip}
    ).set_index("Variable")
    plt.figure(figsize=(14, 4))
    sns.heatmap(df.T, annot=True, fmt=".0f", cmap="RdYlGn", center=0, cbar_kws={"label": "Value"})
    plt.title(title, fontsize=14)
    plt.ylabel("")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()

def plot_constraint_slack(A, b, x, title, save_path):
    slack = b - A @ x
    df = pd.DataFrame(
        {"Constraint": [f"c{i}" for i in range(len(slack))], "Slack": slack}
    )
    plt.figure(figsize=(9, 5))
    colors = ["red" if s < 1e-3 else "lightgreen" for s in slack]
    bars = plt.bar(df["Constraint"], df["Slack"], color=colors, edgecolor="black")
    plt.axhline(0, color="black", lw=0.8)
    for bar, val in zip(bars, slack):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.05 * abs(val) if val >= 0 else -0.05 * abs(val)),
            f"{val:.1f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9,
        )
    plt.title(title, fontsize=14)
    plt.ylabel("Slack = b - Ax")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()

def plot_objective_histogram(vals, title, save_path):
    plt.figure(figsize=(9, 5))
    plt.hist(vals, bins=8, color="skyblue", edgecolor="black", alpha=0.8)
    plt.axvline(obj_ip, color="red", ls="--", lw=2, label=f"IP = {obj_ip:.1f}")
    plt.axvline(np.mean(vals), color="green", lw=2, label=f"Mean = {np.mean(vals):.1f}")
    plt.title(title, fontsize=14)
    plt.xlabel("Final Objective")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()

# ----------------------------------------------------------------------
# 10. GENERATE ALL PLOTS
# ----------------------------------------------------------------------
plot_convergence(
    best_results["history_of_best"],
    best_results["history_of_fitness"],
    f"GA Convergence – Best Run (n={problem_size})",
    "plots/convergence_best.png",
)

plot_all_runs(
    all_histories,
    f"All {params['n_ga_runs']} GA Runs (n={problem_size})",
    "plots/all_runs.png",
)

plot_solution_heatmap(
    x_ga, x_ip, "GA vs Exact IP Solution", "plots/solution_heatmap.png"
)

plot_constraint_slack(
    A, b, x_ga, "Constraint Slack (GA Solution)", "plots/constraint_slack.png"
)

plot_objective_histogram(
    objective_values,
    f"Final Objective Distribution ({params['n_ga_runs']} runs)",
    "plots/objective_histogram.png",
)

# ----------------------------------------------------------------------
# 11. TEXT SUMMARY
# ----------------------------------------------------------------------
summary = f"""
=== ILP GA + WoAC + LP Summary ===
Problem size : n = {problem_size}, m = {A.shape[0]}
IP optimal   : {obj_ip:.6f}
GA best      : {obj_ga:.6f}
Relative err : {rel_err:.6%}   (feasible = {np.all(A @ x_ga <= b + 1e-9)})
Runs         : {params['n_ga_runs']}  (avg time {np.mean(elapsed_times):.2f}s)
WoAC used    : {params.get('use_woac', False)}
"""
with open("plots/SUMMARY.txt", "w") as f:
    f.write(summary)
print(summary)
print("All figures saved to ./plots/")