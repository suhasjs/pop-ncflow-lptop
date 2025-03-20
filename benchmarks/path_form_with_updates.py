import os
import sys
sys.path.append("..")

import pickle
from lib.algorithms import PathFormulation, Objective, PathFormulationCVXPY, PathFormulationALCD
from lib.problem import Problem
from lib.traffic_matrix import TrafficMatrix
from argparse import ArgumentParser
num_paths, edge_disjoint, dist_metric = (4, True, "min-hop")


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

argparser = ArgumentParser(description="Path Formulation with Updates")
argparser.add_argument(
    "--topo",
    choices=PROBLEMS.keys(),
    default="UsCarrier",
    help="problem to run (default: UsCarrier)"
)
argparser.add_argument(
    "--num_rounds",
    type=int,
    default=1,
    help="number of rounds to run (default: 1)"
)
argparser.add_argument(
  "--update_alpha",
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

args = argparser.parse_args()
topo_fname = PROBLEMS[args.topo]["topo_fname"]
tm_fname = PROBLEMS[args.topo]["tm_fname"]
num_rounds = args.num_rounds
update_alpha = args.update_alpha

# Load traffic matrices for repeatable results
if args.load_tms is not None:
  print(f"Loading traffic matrices from {args.load_tms}...")
  TRAFFIC_MATRICES = pickle.load(open(args.load_tms, "rb"))
  print(f"Loaded {len(TRAFFIC_MATRICES)} traffic matrices")
  num_rounds = len(TRAFFIC_MATRICES)
else:
  TRAFFIC_MATRICES = []
  print(f"Loading initial traffic matrix from {tm_fname}...")
  tm = TrafficMatrix.from_file(tm_fname)
  print(f"Creating {num_rounds} traffic matrices with alpha={update_alpha*100}% perturbation each round ...")
  TRAFFIC_MATRICES.append(tm._tm.copy())
  for i in range(1, num_rounds):
    tm._update(1, "uniform", alpha=update_alpha)
    TRAFFIC_MATRICES.append(tm._tm.copy())
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
pf = PathFormulationCVXPY(objective=Objective.get_obj_from_str("total_flow"), num_paths=num_paths,
                         edge_disjoint=edge_disjoint, dist_metric=dist_metric,out=log, VERBOSE=True)

# Solve each traffic matrix in TRAFFIC_MATRICES
for i in range(num_rounds):
  print(f"Solving problem with traffic matrix {i+1}/{num_rounds}...")
  print(f"# commodities: {len(problem.commodity_list)}")
  print(f"Total demand: {problem.total_demand}")
  ret, state = pf.solve(problem, state=state)
  # update the problem with the next traffic matrix
  if i == num_rounds - 1:
    break
  problem._traffic_matrix.tm = TRAFFIC_MATRICES[i+1]