# Translated by Github Copilot (o3-mini) from the original code in path_form.py
import os
import pickle
import re
from collections import defaultdict
import time
from tqdm import tqdm
from itertools import product
import ray
import multiprocessing as mp
from tqdm.contrib.concurrent import process_map as parallel_map

import numpy as np
import cvxpy as cp
from scipy.sparse import csr_array

from ..config import TOPOLOGIES_DIR
from ..constants import NUM_CORES
from ..graph_utils import path_to_edge_list
from ..lp_solver import LpSolver, CvxpySolver  # note: this file now uses CvxpSolver internally
from ..path_utils import find_paths, graph_copy_with_edge_weights, remove_cycles
from .abstract_formulation import AbstractFormulation, Objective

PATHS_DIR = os.path.join(TOPOLOGIES_DIR, "paths", "path-form")

# @ray.remote
def _compute_paths_worker(args):
    s_k, t_k, num_paths, edge_disjoint, G = args
    paths = find_paths(G, s_k, t_k, num_paths, edge_disjoint)
    paths_no_cycles = [remove_cycles(path) for path in paths]
    return ((s_k, t_k), paths_no_cycles)

class PathFormulationCVXPY(AbstractFormulation):
    @classmethod
    def new_total_flow(
        cls, num_paths, edge_disjoint=True, dist_metric="inv-cap", out=None
    ):
        return cls(
            objective=Objective.TOTAL_FLOW,
            num_paths=num_paths,
            edge_disjoint=edge_disjoint,
            dist_metric=dist_metric,
            DEBUG=False,
            VERBOSE=False,
            out=out,
        )

    @classmethod
    def new_max_concurrent_flow(
        cls, num_paths, edge_disjoint=True, dist_metric="inv-cap", out=None
    ):
        return cls(
            objective=Objective.MAX_CONCURRENT_FLOW,
            num_paths=num_paths,
            edge_disjoint=edge_disjoint,
            dist_metric=dist_metric,
            DEBUG=False,
            VERBOSE=False,
            out=out,
        )

    @classmethod
    def new_min_max_link_util(
        cls, num_paths, edge_disjoint=True, dist_metric="inv-cap", out=None
    ):
        return cls(
            objective=Objective.MIN_MAX_LINK_UTIL,
            num_paths=num_paths,
            edge_disjoint=edge_disjoint,
            dist_metric=dist_metric,
            DEBUG=False,
            VERBOSE=False,
            out=out,
        )

    @classmethod
    def compute_demand_scale_factor(
        cls, num_paths, edge_disjoint=True, dist_metric="inv-cap", out=None
    ):
        return cls(
            objective=Objective.COMPUTE_DEMAND_SCALE_FACTOR,
            num_paths=num_paths,
            edge_disjoint=edge_disjoint,
            dist_metric=dist_metric,
            DEBUG=False,
            VERBOSE=False,
            out=out,
        )

    @classmethod
    def get_pf_for_obj(cls, objective, num_paths, **kargs):
        if objective == Objective.TOTAL_FLOW:
            return cls.new_total_flow(num_paths, **kargs)
        elif objective == Objective.MAX_CONCURRENT_FLOW:
            return cls.new_max_concurrent_flow(num_paths, **kargs)
        elif objective == Objective.MIN_MAX_LINK_UTIL:
            return cls.new_min_max_link_util(num_paths, **kargs)
        elif objective == Objective.COMPUTE_DEMAND_SCALE_FACTOR:
            return cls.compute_demand_scale_factor(num_paths, **kargs)
        else:
            print('objective "{}" not found'.format(objective))

    def __init__(
        self,
        *,
        objective,
        num_paths,
        edge_disjoint=True,
        dist_metric="inv-cap",
        DEBUG=False,
        VERBOSE=False,
        out=None
    ):
        super().__init__(objective, DEBUG, VERBOSE, out)
        if dist_metric != "inv-cap" and dist_metric != "min-hop":
            raise Exception(
                'invalid distance metric: {}; only "inv-cap" and "min-hop" are valid choices'.format(
                    dist_metric
                )
            )
        self._num_paths = num_paths
        self.edge_disjoint = edge_disjoint
        self.dist_metric = dist_metric

    # flow caps = [((k1, ..., kn), f1), ...]
    def _construct_path_lp(self, G, edge_to_paths, num_total_paths, sat_flows=[]):
        self._print("Constructing Path LP (using CVXPY)")

        # Create cvxpy variable for each path
        path_vars = cp.Variable(num_total_paths, nonneg=True, name="f")
        print(f"# Paths: {num_total_paths}")
        constraints = []
        additional_vars = {}

        if (self._objective == Objective.MIN_MAX_LINK_UTIL
            or self._objective == Objective.COMPUTE_DEMAND_SCALE_FACTOR):
            self._print("{} objective".format(self._objective))
            # Create variable for maximum link utilization.
            z = cp.Variable(nonneg=True, name="z")
            additional_vars["z"] = z
            # For MIN_MAX_LINK_UTIL, cap z at 1.
            if self._objective == Objective.MIN_MAX_LINK_UTIL:
                constraints.append(z <= 1)

            # Edge utilization constraints
            for u, v, c_e in G.edges.data("capacity"):
                if (u, v) in edge_to_paths:
                    paths = edge_to_paths[(u, v)]
                    expr = cp.sum(cp.hstack([path_vars[p] for p in paths]))
                    if c_e == 0.0:
                        constraints.append(expr <= 0)
                    else:
                        constraints.append(expr / c_e <= z)

            # Demand equality constraints
            commod_id_to_path_inds = {}
            self._demand_constrs = []
            for k, d_k, path_ids in self.commodities:
                commod_id_to_path_inds[k] = path_ids
                constraints.append(cp.sum(cp.hstack([path_vars[p] for p in path_ids])) == d_k)

            objective_expr = z
            prob = cp.Problem(cp.Minimize(objective_expr), constraints)
        else:
            if self._objective == Objective.TOTAL_FLOW:
                self._print("TOTAL FLOW objective")
                objective_expr = cp.sum(path_vars)
                obj_direction = cp.Maximize(objective_expr)
            elif self._objective == Objective.MAX_CONCURRENT_FLOW:
                self._print("MAX CONCURRENT FLOW objective")
                alpha = cp.Variable(nonneg=True, name="a")
                additional_vars["alpha"] = alpha
                constraints.append(alpha <= 1)
                for k, d_k, path_ids in self.commodities:
                    constraints.append(
                        cp.sum(cp.hstack([path_vars[p] for p in path_ids])) / d_k >= alpha
                    )
                objective_expr = alpha
                obj_direction = cp.Maximize(objective_expr)
            else:
                raise Exception("Invalid objective")

            # Edge capacity constraints
            num_edges = 0
            for u, v, c_e in G.edges.data("capacity"):
                num_edges += 1
                if (u, v) in edge_to_paths:
                    paths = edge_to_paths[(u, v)]
                    constraints.append(cp.sum(cp.hstack([path_vars[p] for p in paths])) <= c_e)
            print(f"# Edges: {num_edges}")
            # Demand constraints
            print(f"# Commodities: {len(self.commodities)}")
            commod_id_to_path_inds = {}
            self._demand_constrs = []
            for k, d_k, path_ids in self.commodities:
                commod_id_to_path_inds[k] = path_ids
                constraints.append(
                    cp.sum(cp.hstack([path_vars[p] for p in path_ids])) <= d_k
                )
            prob = cp.Problem(obj_direction, constraints)

        # Flow cap constraints
        for fixed_commods, flow_value in sat_flows:
            indices = []
            for k in fixed_commods:
                indices.extend(commod_id_to_path_inds[k])
            constraints.append(cp.sum(cp.hstack([path_vars[i] for i in indices])) >= 0.99 * flow_value)

        if self.DEBUG:
            print("CVXPY problem formulation:")
            print(prob)

        return CvxpySolver(prob, path_vars, additional_vars, self.DEBUG, self.VERBOSE, self.out)
    
    # flow caps = [((k1, ..., kn), f1), ...]
    def _construct_path_lp_matrix(self, G, edge_to_paths, num_total_paths, sat_flows=[]):
        self._print("Constructing Path LP (using CVXPY)")

        # Create cvxpy variable for each path
        path_vars = cp.Variable(num_total_paths, nonneg=True, name="f")
        print(f"# Paths: {num_total_paths}")
        constraints = []
        additional_vars = {}

        assert self._objective == Objective.TOTAL_FLOW, "Matrix formulation only supports TOTAL_FLOW objective"
        assert len(sat_flows) == 0, "Matrix formulation only supports empty satisfied flows list"

        self._print("TOTAL FLOW objective")
        objective_expr = cp.sum(path_vars)
        obj_direction = cp.Maximize(objective_expr)

        def get_sparse_representation(paths, num_vars, row_num=0):
            row_vals = [row_num] * len(paths)
            col_vals = [p for p in paths]
            data = [1] * len(paths)
            return row_vals, col_vals, data

        mat_row, mat_col, mat_data = [], [], []
        rhs_vec = []
        row_id = 0
        # Edge capacity constraints
        num_edges = 0
        for u, v, c_e in G.edges.data("capacity"):
            num_edges += 1
            if (u, v) in edge_to_paths:
                paths = edge_to_paths[(u, v)]
                row, col, data = get_sparse_representation(paths, num_total_paths, row_num=row_id)
                row_id += 1
                mat_row.extend(row)
                mat_col.extend(col)
                mat_data.extend(data)
                rhs_vec.append(c_e)
                # constraints.append(cp.sum(cp.multiply(path_vars, sparse_paths)) <= c_e)
                # constraints.append(sparse_paths @ path_vars <= c_e)
        print(f"# Edges: {num_edges}")

        # Demand constraints
        print(f"# Commodities: {len(self.commodities)}")
        commod_id_to_path_inds = {}
        self._demand_constrs = []
        for k, d_k, path_ids in self.commodities:
            commod_id_to_path_inds[k] = path_ids
            row, col, data = get_sparse_representation(path_ids, num_total_paths, row_num=row_id)
            row_id += 1
            mat_row.extend(row)
            mat_col.extend(col)
            mat_data.extend(data)
            rhs_vec.append(d_k)
            # constraints.append(cp.sum(cp.multiply(path_vars, sparse_paths)) <= d_k)
            # constraints.append(sparse_paths @ path_vars <= d_k)
        
        # Create a sparse csr matrix for the constraints
        A_sparse = csr_array((mat_data, (mat_row, mat_col)), shape=(len(rhs_vec), num_total_paths))
        constraints.append(A_sparse @ path_vars <= rhs_vec)
        prob = cp.Problem(obj_direction, constraints)

        if self.DEBUG:
            print("CVXPY problem formulation:")
            print(prob)

        return CvxpySolver(prob, path_vars, additional_vars, self.DEBUG, self.VERBOSE, self.out)

    @staticmethod
    def paths_full_fname(problem, num_paths, edge_disjoint, dist_metric):
        return os.path.join(
            PATHS_DIR,
            "{}-{}-paths_edge-disjoint-{}_dist-metric-{}-dict.pkl".format(
                problem.name, num_paths, edge_disjoint, dist_metric
            ),
        )
    
    
    @staticmethod
    def compute_paths_parallel(problem, num_paths, edge_disjoint, dist_metric):
        paths_dict = {}
        G = graph_copy_with_edge_weights(problem.G, dist_metric)
        nodes = list(G.nodes)
        # Create a list of tasks for each (s_k,t_k) pair where s_k != t_k
        tasks = [
            (s, t, num_paths, edge_disjoint, G)
            for s, t in product(nodes, nodes) if s != t
        ]
        print(f"# Tasks: {len(tasks)}")
        results = parallel_map(_compute_paths_worker, tasks, max_workers=NUM_CORES, chunksize=128)
        # Collect the results into a dictionary
        for key, paths_no_cycles in results:
            paths_dict[key] = paths_no_cycles
        return paths_dict

    @staticmethod
    def compute_paths(problem, num_paths, edge_disjoint, dist_metric):
        paths_dict = {}
        G = graph_copy_with_edge_weights(problem.G, dist_metric)
        for s_k in tqdm(G.nodes):
            for t_k in G.nodes:
                if s_k == t_k:
                    continue
                paths = find_paths(G, s_k, t_k, num_paths, edge_disjoint)
                paths_no_cycles = [remove_cycles(path) for path in paths]
                paths_dict[(s_k, t_k)] = paths_no_cycles
        return paths_dict

    @staticmethod
    def read_paths_from_disk_or_compute(problem, num_paths, edge_disjoint, dist_metric):
        paths_fname = PathFormulationCVXPY.paths_full_fname(
            problem, num_paths, edge_disjoint, dist_metric
        )
        print("Loading paths from pickle file", paths_fname)

        try:
            with open(paths_fname, "rb") as f:
                paths_dict = pickle.load(f)
                for key, paths in paths_dict.items():
                    paths_no_cycles = [remove_cycles(path) for path in paths]
                    paths_dict[key] = paths_no_cycles
                print("paths_dict size:", len(paths_dict))
                return paths_dict
        except FileNotFoundError:
            print("Unable to find {}".format(paths_fname))
            print("Computing paths...")
            paths_dict = PathFormulationCVXPY.compute_paths_parallel(
                problem, num_paths, edge_disjoint, dist_metric
            )
            print("Saving paths to pickle file")
            with open(paths_fname, "wb") as w:
                pickle.dump(paths_dict, w)
            return paths_dict

    def get_paths(self, problem):
        if not hasattr(self, "_paths_dict"):
            self._paths_dict = PathFormulationCVXPY.read_paths_from_disk_or_compute(
                problem, self._num_paths, self.edge_disjoint, self.dist_metric
            )
        return self._paths_dict

    ###############################
    # Override superclass methods #
    ###############################

    def solve(self, problem, num_threads=NUM_CORES):
        self._problem = problem
        self._solver = self._construct_lp([], )
        return self._solver.solve_lp()

    def pre_solve(self, problem=None):
        if problem is None:
            problem = self.problem

        self.commodity_list = (
            problem.sparse_commodity_list
            if self._warm_start_mode
            else problem.commodity_list
        )
        self.commodities = []
        edge_to_paths = defaultdict(list)
        self._path_to_commod = {}
        self._all_paths = []

        paths_dict = self.get_paths(problem)
        path_i = 0
        for k, (s_k, t_k, d_k) in self.commodity_list:
            paths = paths_dict[(s_k, t_k)]
            path_ids = []
            for path in paths:
                self._all_paths.append(path)
                for edge in path_to_edge_list(path):
                    edge_to_paths[edge].append(path_i)
                path_ids.append(path_i)
                self._path_to_commod[path_i] = k
                path_i += 1
            self.commodities.append((k, d_k, path_ids))
        if self.DEBUG:
            assert len(self._all_paths) == path_i

        self._print("pre_solve done")
        return dict(edge_to_paths), path_i

    def _construct_lp(self, sat_flows=[]):
        edge_to_paths, num_paths = self.pre_solve()
        # return self._construct_path_lp(self._problem.G, edge_to_paths, num_paths, sat_flows)
        return self._construct_path_lp_matrix(self._problem.G, edge_to_paths, num_paths, sat_flows)

    @property
    def sol_dict(self):
        if not hasattr(self, "_sol_dict"):
            sol_dict_def = defaultdict(list)
            # Loop over indices of the cvxpy variable (named "f")
            for p, val in enumerate(self._solver.path_vars.value):
                if abs(val) > 1e-6:
                    commod = self.commodity_list[self._path_to_commod[p]]
                    for edge in path_to_edge_list(self._all_paths[p]):
                        sol_dict_def[commod].append((edge, val))
            self._sol_dict = {}
            sol_dict_def = dict(sol_dict_def)
            for commod_key in self.problem.commodity_list:
                self._sol_dict[commod_key] = sol_dict_def.get(commod_key, [])
        return self._sol_dict

    @property
    def sol_mat(self):
        edge_idx = self.problem.edge_idx
        sol_mat = np.zeros((len(edge_idx), len(self._path_to_commod)), dtype=np.float32)
        for p, val in enumerate(self._solver.path_vars.value):
            if abs(val) > 1e-6:
                k = self._path_to_commod[p]
                for edge in path_to_edge_list(self._all_paths[p]):
                    sol_mat[edge_idx[edge], k] += val
        return sol_mat

    @classmethod
    # Return total number of fib entries and max for any node in topology
    # NOTE: problem has to have a full TM matrix
    def fib_entries(cls, problem, num_paths, edge_disjoint, dist_metric):
        assert problem.is_traffic_matrix_full
        pf = cls.new_total_flow(
            num_paths=num_paths, edge_disjoint=edge_disjoint, dist_metric=dist_metric
        )
        pf.pre_solve(problem)
        return pf.num_fib_entries_for_path_set()

    def num_fib_entries_for_path_set(self):
        self.fib_dict = defaultdict(dict)
        for k, _, path_ids in self.commodities:
            commod_id_str = "k-{}".format(k)
            src = list(path_to_edge_list(self._all_paths[path_ids[0]]))[0][0]
            # For a given TM, we would store weights for each path id. For demo
            # purposes, we just store the path ids
            self.fib_dict[src][commod_id_str] = path_ids

            for path_id in path_ids:
                for u, v in path_to_edge_list(self._all_paths[path_id]):
                    assert path_id not in self.fib_dict[u]
                    self.fib_dict[u][path_id] = v

        self.fib_dict = dict(self.fib_dict)
        fib_dict_counts = [len(self.fib_dict[k]) for k in self.fib_dict.keys()]
        return sum(fib_dict_counts), max(fib_dict_counts)

    @property
    def runtime(self):
        if not hasattr(self, "_runtime"):
            # If available, extract solve time from the solver statistics.
            self._runtime = (
                self._solver.problem.solver_stats.solve_time
                if hasattr(self._solver.problem, "solver_stats")
                else None
            )
        return self._runtime