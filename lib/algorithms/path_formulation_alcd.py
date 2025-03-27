from collections import defaultdict
import time

import cvxpy as cp
import numpy as np
from scipy.sparse import csr_array

from ..graph_utils import path_to_edge_list
from .abstract_formulation import Objective
from .path_formulation_cvxpy import PathFormulationCVXPY

import pylpsparse as lps

SCALING_CONSTANT = 10

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
    print("Constructing Path LP for ALCD solver")
    start_t = time.time()

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
        mi += 1
        if 'Amat' not in self.state:
          paths = edge_to_paths[(u, v)]
          col, data = paths, [SCALING_CONSTANT] * len(paths)
          nnz += len(data)
          mat_cols.append(col)
          mat_data.append(data)
        rhs_vec.append(c_e * SCALING_CONSTANT)
    print(f"# Edges: {num_edges}")
    print(f"t={time.time() - start_t:.2f}s: Processed edge capacity constraints..")

    # Demand constraints
    print(f"# Commodities: {len(self.commodities)}")
    commod_id_to_path_inds = {}
    for k, d_k, path_ids in self.commodities:
      mi += 1
      if 'Amat' not in self.state:
        commod_id_to_path_inds[k] = path_ids
        col, data = path_ids, [SCALING_CONSTANT] * len(path_ids)
        nnz += len(data)
        mat_cols.append(col)
        mat_data.append(data)
      rhs_vec.append(d_k * SCALING_CONSTANT)
    print(f"t={time.time() - start_t:.2f}s: Processed demand constraints..")
    
    print(f"ALCD Constraint matrix: {mi} inequalities, {me} equalities, {nnz} non-zero entries")
    print(f"ALCD Variables: {nb} non-basic, {nf} free")

    # Create a lpsparse Matrix object
    if 'Amat' not in self.state:
      Amat = lps.Matrix(mi + me)
      for i, (col, data) in enumerate(zip(mat_cols, mat_data)):
        Amat.setrow(i, list(zip(col, data)))
    else:
      print(f"Re-using Amat from previous run...")
      Amat = self.state['Amat']
    bvec = np.asarray(rhs_vec)
    
    # Create primal and dual problems
    self.state['Amat'] = Amat
    self.primalA = Amat
    self.primalA_shape = (mi + me, nb + nf)
    self.primalb = bvec
    self.primalc = stdform_c
    self.dualAt = Amat.transpose() if 'Atmat' not in self.state else self.state['Atmat']
    self.state['Atmat'] = self.dualAt
    # self.dualAt = Amat.transpose()
    self.dualA_shape = (nb + nf, mi + me)
    self.dualb = stdform_c
    self.dualc = bvec
    self.nb, self.nf = nb, nf
    self.mi, self.me = mi, me
    self.m = mi
    print(f"t={time.time() - start_t:.2f}s: Finished all constraints..")
    return self
  
  def solve(self, problem, num_threads=8, state={}, alcd_params={}):
    self.state = state
    self._problem = problem
    start_t = time.time()
    construct_stats, ret = self._construct_lp([], )
    self._setup_time = time.time() - start_t
    obj_val = self.solve_lp(alcd_params=alcd_params)
    self._solve_time = time.time() - start_t - self._setup_time
    print(f"Solver times -- setup: {self._setup_time:.2f}s, solve: {self._solve_time:.2f}s")
    # TODO (suhasjs): adapt this for ALCD
    self._runtime = self._solve_time + self._setup_time
    solve_stats = {
            "setup_time": self._setup_time,
            "solve_time": self._solve_time,
            "num_commodities": len(self.commodity_list),
            "num_paths": len(self._all_paths),
            "num_edges": len(problem.G.edges),
            "num_nodes": len(problem.G.nodes),
            "num_threads": num_threads,
            "objective": self._obj_val,
            "alcd_stats" : self._alcd_stats,
    }
    solve_stats.update(construct_stats)
    print(f"Total solver time: {self.runtime:.2f}s, objective: {self.obj_val:.2f}")
    return solve_stats, obj_val, self.state
  
  def get_primal_alcd_format(self):
    return (self.primalA, self.primalb, self.primalc, self.nb, self.nf, self.m, self.me)
  
  def get_dual_alcd_format(self):
    ## TODO (suhasjs) --> Check if nb, nf, m, me are correctly returned???
    return (self.dualAt, self.dualb, self.dualc, self.me, self.m, self.nb, self.nf)

  def solve_lp(self, alcd_params):
    state = self.state if alcd_params['warm_start'] else {}
    primal_args = self.get_primal_alcd_format()
    dual_lpargs = self.get_dual_alcd_format()
    
    # Create args for ALCD solver
    lpcfg = lps.LP_Param()
    lpcfg.solve_from_dual = False
    # TODO (suhasjs) --> Why does reducing eta help us here??
    lpcfg.eta = 1
    lpcfg.verbose = True
    lpcfg.tol_trans = alcd_params.get('tol_trans', 0.1)
    lpcfg.tol = alcd_params.get('tol', 0.1)
    # lpcfg.tol_sub = args.alcd_tol
    lpcfg.tol_sub = 5e-2
    lpcfg.use_CG = True
    lpcfg.max_iter = 100
    lpcfg.inner_max_iter = 3
    lpcfg.primal_max_iter = alcd_params.get('primal_max_iter', 10)
    lpcfg.primal_inner_max_iter = alcd_params.get('primal_inner_max_iter', 3)
    lpcfg.pinf_dinf_ratio = alcd_params.get('pinf_dinf_ratio', 10)
    lpcfg.dual_max_iter = alcd_params.get('dual_max_iter', 15)
    lpcfg.dual_inner_max_iter = alcd_params.get('dual_inner_max_iter', 3)
    lpcfg.corrector_max_iter = alcd_params.get('corrector_max_iter', 5)
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
    if 'x0' in state:
      print(f"Initializing x0 from previous run")
      x0[:] = state['x0']
    if 'w0' in state:
      print(f"Initializing w0 from previous run")
      w0[:] = state['w0']
    
    # Solve using ALCD
    lpinfo = lps.LP_Info()
    solve_start_time = time.time()
    print(f"Solving using ALCD solver")
    # lps.solve_alcd(A, b, c, x0, w0, h2jj, hjj_ubound, nb, nf, m, me, lpcfg, lpinfo)
    lps.solve_alcd_corrector(A, b, c, x0, w0, h2jj, hjj_ubound, nb, nf, m, me, lpcfg, lpinfo)
    solve_end_time = time.time()
    self._alcd_stats = lps.lpinfo_to_dict(lpinfo)
    print(f"ALCD solver stats: {self._alcd_stats}")
    print(f"ALCD Solver: Init: {(init_end_time - init_start_time)*1000:.1f}ms, Solve time: {(solve_end_time - solve_start_time)*1000:.1f}ms")
    print(f"ALCD solver finished in {solve_end_time - init_start_time:.2f}s")

    # copy solution into _path_vars
    self._path_vars[:] = x0
    self._obj_val = np.sum(x0)

    # save to state for next run
    self.state['x0'] = x0.copy()
    self.state['w0'] = w0.copy()
    return self._obj_val

  @property
  def sol_x(self):
    if not hasattr(self, "_sol_x"):
      self._sol_x = np.zeros(len(self._path_vars))
      self._sol_x[:] = self._path_vars
    return self._sol_x
  
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