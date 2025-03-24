import os
import sys
import time
import numpy as np
sys.path.append("..")

import pickle
from lib.algorithms import PathFormulation, Objective, PathFormulationCVXPY, PathFormulationALCD, POP
from lib.constants import NUM_CORES
from lib.graph_utils import check_feasibility
from lib.problem import Problem
from lib.traffic_matrix import TrafficMatrix
from argparse import ArgumentParser

# set numpy random seed for reproducibility
np.random.seed(42)

num_paths, edge_disjoint, dist_metric = (4, True, "min-hop")
POP_SPLIT_METHOD = "random"
POP_SPLIT_FRACTION = 0.25

PROBLEMS = {
  "UsCarrier" : {
    "topo_fname": "../../topologies/UsCarrier.json",
    "tm_fname": "../../traffic-matrices/toy/UsCarrier.json_toy_0_1.0_traffic-matrix.pkl"
  },
  "Kdl" : {
    "topo_fname": "../../topologies/Kdl.json",
    "tm_fname": "../../traffic-matrices/toy/Kdl.json_toy_0_1.0_traffic-matrix.pkl"
  },
  "ASN" : {
    "topo_fname": "../../topologies/ASN2k.json",
    "tm_fname": "../../traffic-matrices/toy/ASN2k.json_toy_0_1.0_traffic-matrix.pkl"
  }
}

argparser = ArgumentParser(description="Path Formulation with deterministic traffic matrix updates")
argparser.add_argument(
    "--topo",
    choices=PROBLEMS.keys(),
    default="UsCarrier",
    help="problem to run (default: UsCarrier)"
)
argparser.add_argument(
    "--num-rounds",
    type=int,
    default=1,
    help="number of rounds to run (default: 1)"
)
argparser.add_argument(
    "--solver",
    type=str,
    default="ALCD",
    choices=["ALCD", "CVXPY", "POP", "LPALL"],
    help="which solver to use: ALCD, CVXPY, POP, LPALL (LPALL uses original formulation with GUROBI solver)"
)
argparser.add_argument(
    "--num-subproblems",
    type=int,
    default=4,
    help="number of subproblems for POP solver (default: 4)"
)
argparser.add_argument(
  "--update-alpha",
  type=float,
  default=0.1,
  help="maximum perturbation in demands (default: 0.1)"
)
argparser.add_argument(
  "--dump-tms",
  default=None,
  help="where to dump traffic matrices on disk for repeatable results(default: '')"
)
argparser.add_argument(
  "--load-tms",
  default=None,
  help="where to load traffic matrices from disk (default: '')"
)
argparser.add_argument(
  "--save-results",
  default=None,
  help="where to save results of the end-to-end run on disk in .pkl format (default: '')"
)

args = argparser.parse_args()
topo_fname = PROBLEMS[args.topo]["topo_fname"]
tm_fname = PROBLEMS[args.topo]["tm_fname"]
num_rounds = args.num_rounds
update_alpha = args.update_alpha

results = {
  "meta" : {"topo" : args.topo, "solver": args.solver}, "solver_stats" : [], "violations" : []
}

# Load traffic matrices for repeatable results
if args.load_tms is not None:
  print(f"Loading traffic matrices from {args.load_tms}...")
  TRAFFIC_MATRICES = pickle.load(open(args.load_tms, "rb"))
  print(f"Loaded {len(TRAFFIC_MATRICES)} traffic matrices")
  num_rounds = min(len(TRAFFIC_MATRICES), num_rounds)
  results["meta"]["num_rounds"] = num_rounds
  results["meta"]["tm_filename"] = args.load_tms
else:
  TRAFFIC_MATRICES = []
  print(f"Loading initial traffic matrix from {tm_fname}...")
  tm = TrafficMatrix.from_file(tm_fname)
  print(f"Creating {num_rounds} traffic matrices with alpha={update_alpha*100}% perturbation each round ...")
  TRAFFIC_MATRICES.append(tm._tm.copy())
  for i in range(1, num_rounds):
    tm._update(1, "uniform", alpha=update_alpha)
    TRAFFIC_MATRICES.append(tm._tm.copy())
  results["meta"]["update-alpha"] = args.update_alpha
  results["meta"]["num_rounds"] = num_rounds
  if args.dump_tms is not None:
    print(f"Dumping created traffic matrices to {args.dump_tms}...")
    with open(args.dump_tms, "wb") as f:
      pickle.dump(TRAFFIC_MATRICES, f)
assert len(TRAFFIC_MATRICES) >= num_rounds, f"Not enough traffic matrices created. Expected {num_rounds}, got {len(TRAFFIC_MATRICES)}"

# Create a problem object
problem = Problem.from_file(topo_fname, tm_fname)
# use the first traffic matrix for the initial problem
problem._traffic_matrix.tm = TRAFFIC_MATRICES[0]
traffic_seed = problem.traffic_matrix.seed
print("traffic seed: {}".format(traffic_seed))
print("traffic scale factor: {}".format(problem.traffic_matrix.scale_factor))
print("traffic matrix model: {}".format(problem.traffic_matrix.model))
print(f"traffic matrix class: {problem.traffic_matrix.__class__.__name__}")

log = open("/tmp/path_formulation.txt", "w")
state = {}
# Instantiate a PathFormulation object
if args.solver in ["ALCD", "LPALL", "CVXPY"]:
  if args.solver == "ALCD":
    PATH_FORMULATION_BACKEND = PathFormulationALCD
    print("Using ALCD solver")
  elif args.solver == "LPALL":
    PATH_FORMULATION_BACKEND = PathFormulation
    print("Using LPALL solver")
  else:
    PATH_FORMULATION_BACKEND = PathFormulationCVXPY
    print("Using CVXPY solver")
  pf = PATH_FORMULATION_BACKEND(objective=Objective.get_obj_from_str("total_flow"), 
                                num_paths=num_paths, edge_disjoint=edge_disjoint, 
                                dist_metric=dist_metric,out=log, VERBOSE=True)
elif args.solver == "POP":
  pop_args = {
    "num_subproblems": args.num_subproblems, "split_method": POP_SPLIT_METHOD,
    "split_fraction": POP_SPLIT_FRACTION, "algo_cls": PathFormulation,
    "num_paths": num_paths,
  }
  results["meta"]["split_method"] = POP_SPLIT_METHOD
  results["meta"]["split_fraction"] = POP_SPLIT_FRACTION
  results["meta"]["num_subproblems"] = args.num_subproblems
  pf = POP(objective=Objective.get_obj_from_str("total_flow"), **pop_args, 
           edge_disjoint=edge_disjoint, dist_metric=dist_metric, out=log)
else:
  raise ValueError(f"Unknown solver {args.solver}. Must be one of ALCD, CVXPY, POP, LPALL")

state = {}
# Solve each traffic matrix in TRAFFIC_MATRICES
for i in range(num_rounds):
  print(f"Solving problem with traffic matrix {i+1}/{num_rounds}...")
  print(f"# commodities: {len(problem.commodity_list)}")
  print(f"Total demand: {problem.total_demand}")
  if args.solver in ["ALCD", "LPALL", "CVXPY"]:
    solver_stats, obj_val, state = pf.solve(problem, state=state)
  elif args.solver == "POP":
    start_t = time.time()
    solve_stats, obj_val, state = pf.solve(problem, state=state)
    st_runtime = time.time() - start_t
    est_runtime = pf.runtime_est(NUM_CORES)
    solver_stats = {
        "subproblem_stats": solve_stats,
        "est_runtime" : est_runtime, 
        "num_subproblems" : args.num_subproblems, 
        "split_method" : POP_SPLIT_METHOD, 
        "split_fraction" : POP_SPLIT_FRACTION, 
        "num_cores" : NUM_CORES,
        "objective" : obj_val,
        "single_threaded_runtime" : st_runtime,
    }
  print(f"Solver stats: {solver_stats}, Objective value: {obj_val}")
  # check feasibility of the solution
  pf_sol_dict = pf.sol_dict
  violations = check_feasibility(problem, [pf_sol_dict], no_assert=True)
  print(f"Feasibility violations: {violations}")
  # save results to results dict
  results["solver_stats"].append(solver_stats)
  results["violations"].append(violations)

  # update the problem with the next traffic matrix
  if i == num_rounds - 1:
    break
  problem._traffic_matrix.tm = TRAFFIC_MATRICES[i+1]

# save results to disk
if args.save_results is not None:
  print(f"Saving run results to {args.save_results}...")
  with open(args.save_results, "wb") as f:
    pickle.dump(results, f)