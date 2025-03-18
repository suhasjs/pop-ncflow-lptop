from collections import defaultdict
import time

import cvxpy as cp
import numpy as np
from scipy.sparse import csr_array

from ..graph_utils import path_to_edge_list
from .abstract_formulation import Objective
from .path_formulation_cvxpy import PathFormulationCVXPY

import pylpsparse as lps

class PathFormulationALCD(PathFormulationCVXPY):
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
    super().__init__(objective=objective, num_paths=num_paths, 
                      edge_disjoint=edge_disjoint, dist_metric=dist_metric, 
                      DEBUG=DEBUG, VERBOSE=VERBOSE, out=out)

  
  # flow caps = [((k1, ..., kn), f1), ...]
  def _construct_path_lp_matrix(self, G, edge_to_paths, num_total_paths, sat_flows=[]):
    self._print("Constructing Path LP for ALCD solver")

    # Create cvxpy variable for each path
    self._path_vars = np.zeros(num_total_paths)
    print(f"# Paths: {num_total_paths}")

    assert self._objective == Objective.TOTAL_FLOW, "Matrix formulation only supports TOTAL_FLOW objective"
    assert len(sat_flows) == 0, "Matrix formulation only supports empty satisfied flows list"

    # maximize total flow = minimize (negative total flow)
    stdform_c = -1 * np.ones(num_total_paths)

    mat_cols, mat_data = [], []
    rhs_vec = []
    # mi = # inequalities, me = # equalities
    # nnz = number of non-zero entries in the constraint matrix
    mi, me, nnz = 0, 0, 0
    # nb = num non-basic variables, nf = num free variables
    nb, nf = num_total_paths, 0
    # Edge capacity constraints
    num_edges = 0
    for u, v, c_e in G.edges.data("capacity"):
      num_edges += 1
      if (u, v) in edge_to_paths:
        paths = edge_to_paths[(u, v)]
        col, data = paths, [1] * len(paths)
        mi += 1
        nnz += len(data)
        mat_cols.append(col)
        mat_data.append(data)
        rhs_vec.append(c_e)
    print(f"# Edges: {num_edges}")

    # Demand constraints
    print(f"# Commodities: {len(self.commodities)}")
    commod_id_to_path_inds = {}
    self._demand_constrs = []
    for k, d_k, path_ids in self.commodities:
      commod_id_to_path_inds[k] = path_ids
      col, data = path_ids, [1] * len(path_ids)
      mi += 1
      nnz += len(data)
      mat_cols.append(col)
      mat_data.append(data)
      rhs_vec.append(d_k)
    
    print(f"ALCD Constraint matrix: {mi} inequalities, {me} equalities, {nnz} non-zero entries")
    print(f"ALCD Variables: {nb} non-basic, {nf} free")

    # Create a lpsparse Matrix object
    Amat = lps.Matrix(mi + me)
    bvec = np.asarray(rhs_vec)
    for i, (col, data) in enumerate(zip(mat_cols, mat_data)):
      Amat.setrow(i, list(zip(col, data)))
    
    # Create primal and dual problems
    self.primalA = Amat
    self.primalA_shape = (mi + me, nb + nf)
    self.primalb = bvec
    self.primalc = stdform_c
    self.dualAt = Amat.transpose()
    self.dualA_shape = (nb + nf, mi + me)
    self.dualb = stdform_c
    self.dualc = bvec
    self.nb, self.nf = nb, nf
    self.mi, self.me = mi, me
    self.m = mi
    return self

  def solve(self, problem, num_threads=8):
    self._problem = problem
    start_t = time.time()
    self._solver = self._construct_lp([], )
    self._setup_time = time.time() - start_t
    ret = self._solver.solve_lp()
    self._solve_time = time.time() - start_t - self._setup_time
    print(f"Solver times -- setup: {self._setup_time:.2f}s, solve: {self._solve_time:.2f}s")
    # TODO (suhasjs): adapt this for ALCD
    self._runtime = self._solve_time + self._setup_time
    print(f"Total solver time: {self.runtime:.2f}s, objective: {self.obj_val:.2f}")
    return ret
  
  def get_primal_alcd_format(self):
    return (self.primalA, self.primalb, self.primalc, self.nb, self.nf, self.m, self.me)
  
  def get_dual_alcd_format(self):
    ## TODO (suhasjs) --> Check if nb, nf, m, me are correctly returned???
    return (self.dualAt, self.dualb, self.dualc, self.me, self.m, self.nb, self.nf)

  def solve_lp(self):
    primal_args = self.get_primal_alcd_format()
    dual_lpargs = self.get_dual_alcd_format()
    
    # Create args for ALCD solver
    lpcfg = lps.LP_Param()
    lpcfg.solve_from_dual = True
    # TODO (suhasjs) --> Why does reducing eta help us here??
    lpcfg.eta = 1
    lpcfg.verbose = True
    lpcfg.tol_trans = 5e-2
    lpcfg.tol = 5e-2
    # lpcfg.tol_sub = args.alcd_tol
    lpcfg.tol_sub = 1e-2
    lpcfg.use_CG = True
    lpcfg.max_iter = 1000
    lpcfg.inner_max_iter = 5
    lpcfg.primal_max_iter = 10
    lpcfg.primal_inner_max_iter = 10
    lpcfg.dual_max_iter = 50
    lpcfg.dual_inner_max_iter = 5
    lpcfg.corrector_max_iter = 2
    lpcfg.penalty_alpha = 0

    # Initialize ALCD solver
    A, b, c, nb, nf, m, me = primal_args
    # A.printrows()
    # print(b)
    At = dual_lpargs[0]
    # At.printrows()
    x0 = np.zeros(len(c))
    w0 = np.ones(len(b))
    init_start_time = time.time()
    print(f"Initalizing ALCD solver")
    if lpcfg.solve_from_dual is False:
      h2jj = np.zeros(nb + nf)
      hjj_ubound = np.zeros(nb + nf)
      lps.init_state(x0, w0, h2jj, hjj_ubound, nb, nf, m, me, A, b, c, lpcfg.eta) 
    else:
      h2jj = np.zeros(m + me)
      hjj_ubound = np.zeros(m + me)
      lps.init_state(w0, x0, h2jj, hjj_ubound, m, me, nb, nf, At, c, b, lpcfg.eta)
    init_end_time = time.time()
    # print(f"h2jj: {h2jj}\nhjj_ubound: {hjj_ubound}")
    # print(f"h2jj: {list(zip(*np.histogram(h2jj, bins=10)))}")
    # print(f"hjj_ubound: {list(zip(*np.histogram(hjj_ubound, bins=10)))}")
    # Solve via ALCD solver
    x0[:] = 1
    w0[:] = 1
    
    # Solve using ALCD
    lpinfo = lps.LP_Info()
    solve_start_time = time.time()
    print(f"Solving using ALCD solver")
    # lps.solve_alcd(A, b, c, x0, w0, h2jj, hjj_ubound, nb, nf, m, me, lpcfg, lpinfo)
    lps.solve_alcd_corrector(A, b, c, x0, w0, h2jj, hjj_ubound, nb, nf, m, me, lpcfg, lpinfo)
    solve_end_time = time.time()
    print(f"ALCD solver stats: {lps.lpinfo_to_dict(lpinfo)}")
    print(f"ALCD Solver: Init: {(init_end_time - init_start_time)*1000:.1f}ms, Solve time: {(solve_end_time - solve_start_time)*1000:.1f}ms")
    print(f"ALCD solver finished in {solve_end_time - init_start_time:.2f}s")

    # copy solution into _path_vars
    self._path_vars[:] = x0
    self._obj_val = -1 * np.dot(c, x0)
    return self

  @property
  def sol_dict(self):
    if not hasattr(self, "_sol_dict"):
      sol_dict_def = defaultdict(list)
      # Loop over indices of the cvxpy variable (named "f")
      for p, val in enumerate(self._path_vars):
        if abs(val) > 1e-6:
          commod = self.commodity_list[self._path_to_commod[p]]
          for edge in path_to_edge_list(self._all_paths[p]):
            sol_dict_def[commod].append((edge, val))
      self._sol_dict = {}
      sol_dict_def = dict(sol_dict_def)
      for commod_key in self.problem.commodity_list:
        self._sol_dict[commod_key] = sol_dict_def.get(commod_key, [])
    return self._sol_dict