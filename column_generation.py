import collections
from copy import deepcopy
from heapq import heappop, heappush

import gurobipy as gp
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from tqdm import tqdm

# # Settings
# FLEET_CAPACITY = 200000
# NUM_OF_TIMES = 60
# MAX_TT = 10
# T_LIMIT = NUM_OF_TIMES / 2 + MAX_TT

# MAX_0F_NUM = 50

# EPSILON = 0.1

# Settings
FLEET_CAPACITY = 500000
NUM_OF_TIMES = 52
MAX_TT = 5
T_LIMIT = NUM_OF_TIMES / 2 + MAX_TT

MAX_0F_NUM = 50

EPSILON = 1e-1


def is_less_than_zero(value):
    if value <= -EPSILON:
        return True
    else:
        return False


def is_greater_than_zero(value):
    if value >= EPSILON:
        return True
    else:
        return False


def is_equal_to_zero(value):
    if -EPSILON <= value <= EPSILON:
        return True
    else:
        return False


def i_(arc):
    return arc[0], arc[1]


def j_(arc):
    return arc[2], arc[3]


def k_(arc):
    return arc[4]


def e_(arc):
    return i_(arc), j_(arc), k_(arc)


def get_t_2(t_1, t_12):
    """
    t_1 -> t_2.

    Args:
        t_1 (int): start time
        t_12 (int): start time -> end time

    Returns:
        (int): end time
    """
    t_2 = t_1 + t_12
    if t_2 > NUM_OF_TIMES:
        t_2 = t_2 - NUM_OF_TIMES
    return t_2


def name_(name, key):
    if type(key) is tuple:
        return "%s[%s]" % (name, ",".join([str(_) for _ in key]))
    else:
        return f"{name}[{key}]"


def fake_node(j):
    return j[0], j[1] + NUM_OF_TIMES


def real_node(node):
    n1 = node[1] if node[1] <= NUM_OF_TIMES else node[1] - NUM_OF_TIMES
    return node[0], n1


def e_fake_j(arc):
    return i_(arc), fake_node(j_(arc)), k_(arc)


def e_fake_ij(arc):
    return fake_node(i_(arc)), fake_node(j_(arc)), k_(arc)


def get_delta(arcs, columns):
    delta = {arc: [] for arc in arcs}

    for key, value in columns.items():
        for arc in value["path"]:
            delta[arc].append(key)
    return delta


def is_dup_node(node):
    if node[1] <= MAX_TT:
        return True
    else:
        return False


def gen_init_paths(sn):
    paths = sn.init_paths

    def holding_path(p):
        path = []
        for t in sn.times:
            node_i = (p, t)
            node_j = (p, get_t_2(t, 1))
            arc_ij = (*node_i, *node_j, "h")
            path.append(arc_ij)

        # check_cyclic_path(path)
        return tuple(path)

    paths.append(holding_path(sn.ports[0]))
    return paths


def generate_initial_paths(sn):
    """
    Two kinds of initial paths:

    - All arcs are holding arcs (for the feasibility).
    - Most arcs arc transferring arcs (to cover most arcs).

    Args:
        sn (ShippingNetwork): the ShippingNetwork object

    Returns:
        (list): generated paths
    """
    paths = []

    def holding_path(p):
        path = []
        for t in sn.times:
            node_i = (p, t)
            node_j = (p, get_t_2(t, 1))
            arc_ij = (*node_i, *node_j, "h")
            path.append(arc_ij)

        # check_cyclic_path(path)
        return tuple(path)

    paths.append(holding_path(sn.ports[0]))

    def transferring_path(p_1, p_2):
        t_12 = sn.base_network[p_1][p_2]["t_od"]
        t_121 = 2 * t_12

        t = NUM_OF_TIMES

        path = []

        p_i, t_i = p_1, 1
        p_j, t_j = p_2, get_t_2(t_i, t_12)
        p_k, t_k = p_i, get_t_2(t_j, t_12)
        node_i = (p_i, t_i)
        node_j = (p_j, t_j)
        node_k = (p_k, t_k)
        arc_ij = (*node_i, *node_j, "g")
        arc_jk = (*node_j, *node_k, "g")
        path.append(arc_ij)
        path.append(arc_jk)
        t -= t_121

        while t > t_121:
            p_i, t_i = node_k
            p_j, t_j = p_2, get_t_2(t_i, t_12)
            p_k, t_k = p_i, get_t_2(t_j, t_12)
            node_i = (p_i, t_i)
            node_j = (p_j, t_j)
            node_k = (p_k, t_k)
            arc_ij = (*node_i, *node_j, "g")
            arc_jk = (*node_j, *node_k, "g")
            path.append(arc_ij)
            path.append(arc_jk)
            t -= t_121

        # Holding arcs
        p_i, t_i = node_k
        while t > 0:
            p_j, t_j = p_i, get_t_2(t_i, 1)
            node_i = (p_i, t_i)
            node_j = (p_i, t_j)
            arc_ij = (*node_i, *node_j, "h")
            path.append(arc_ij)
            t -= 1

            p_i, t_i = node_j

        # check_cyclic_path(path)
        return tuple(path)

    for p1 in sn.ports:
        for p2 in sn.ports:
            if p1 == p2:
                pass
            else:
                paths.append(transferring_path(p1, p2))

    return paths


def reduced_cost_(G, beta, path):
    rc = -beta
    for arc in path:
        rc += G.edges[e_(arc)]["weight"]
    return rc


class ShippingNetwork:
    """
    This is a class for a shipping network.

    Attributes:
        idx (int): Index of this instance
        base_network (nx.Graph): p_o, p_d, t_od, the basic structure of shipping network
        times (list): [1, 2, 3, ..., NUM_OF_TIMES (int))
        ports (list): [id1, id2, ..., id10 (int)]
        capacity_of_port (dict): {id: capacity (int/float)}
        cargos (list): [(p_o, p_d, t_1, t_2), ...]
        demand_of_cargo (dict): {(p_o, p_d, t_1, t_2): demand (int/float)}
        network (nx.Graph): the super-network
        arcs (gp.tuplelist): the arcs of shipping network
        y_keys (gp.tuplelist): the keys of Y variables
    """

    def __init__(self, model_dir, idx):
        self.idx = idx

        def get_base_network() -> nx.Graph:
            df = pd.read_csv(f"{model_dir}/{self.idx}/od.csv", sep="|", header=0)
            return nx.from_pandas_edgelist(df, "p_o", "p_d", "t_od")

        # def get_base_network() -> nx.Graph:
        #     df = pd.read_csv(f"{model_dir}/od_3day_unit.csv", sep="|", header=0)
        #     return nx.from_pandas_edgelist(df, "p_o", "p_d", "t_od")

        self.base_network = get_base_network()

        self.times = [i + 1 for i in range(NUM_OF_TIMES)]

        # def get_ports():
        #     df = pd.read_csv(f"{model_dir}/{self.idx}/capacity.csv", sep="|", header=0)

        #     info_dict = {}
        #     for row in df.itertuples(index=False):
        #         info_dict[row.port_id] = [row.capacity]

        #     return gp.multidict(info_dict)

        # self.ports, self.capacity_of_port = get_ports()

        def get_ports():
            df = pd.read_csv(f"{model_dir}/{self.idx}/capacity.csv", sep="|", header=0)

            ports = []
            info_dict = {}
            for row in df.itertuples(index=False):
                for t in self.times:
                    # if t in [1, 2, 3, 4] and row.port_id == 963:
                    #     info_dict[row.port_id, t] = [row.capacity * (1 - 0.75)]
                    # else:
                    #     info_dict[row.port_id, t] = [row.capacity]
                    info_dict[row.port_id, t] = [row.capacity]

                ports.append(row.port_id)

            return ports, gp.multidict(info_dict)[1]

        self.ports, self.capacity_of_port = get_ports()

        def get_cargos():
            df = pd.read_csv(f"{model_dir}/{self.idx}/demand.csv", sep="|", header=0)

            info_dict = {}
            for row in df.itertuples(index=False):
                info_dict[(row.p_o, row.p_d, row.t_1, row.t_2)] = [row.demand]

            return gp.multidict(info_dict)

        self.cargos, self.demand_of_cargo = get_cargos()

        def get_shipping_network():
            df = pd.read_csv(f"{model_dir}/{self.idx}/profit.csv", sep="|", header=0)
            # df = pd.read_csv(f"{model_dir}/{self.idx}/network.csv", sep="|", header=0)

            g = nx.MultiDiGraph()
            arcs = []

            for row in df.itertuples(index=False):
                g.add_edge(
                    u_for_edge=(row.p_i, row.t_i),
                    v_for_edge=(row.p_j, row.t_j),
                    key=row.key,
                    travel_time=row.tt,
                    profit=row.pf,
                )

                arcs.append((row.p_i, row.t_i, row.p_j, row.t_j, row.key))

            return g, gp.tuplelist(arcs)

        self.network, self.arcs = get_shipping_network()

        def get_y_keys():
            y_keys = []
            for p_o, p_d, t_1, t_2 in self.cargos:
                t_ij = self.base_network[p_o][p_d]["t_od"]

                p_i, t_i = p_o, t_1
                p_j, t_j = p_d, get_t_2(t_i, t_ij)
                i = (p_i, t_i)
                j = (p_j, t_j)
                y_keys.append((*i, *j, t_1, t_2))

                while t_i != t_2:
                    t_i = get_t_2(t_i, 1)
                    t_j = get_t_2(t_j, 1)
                    i = (p_i, t_i)
                    j = (p_j, t_j)
                    y_keys.append((*i, *j, t_1, t_2))
            return y_keys

        self.y_keys = get_y_keys()
        print("")


class RestrictedMasterProblem:
    """
    This is the class of restricted master problem (RMP).

    Attributes:
        model (gp.Model): The Gurobi model.
    """

    def __init__(self, sn, paths=None):
        """
        Initialize the RMP.

        Args:
            sn (ShippingNetwork): shipping network.
        """
        # Columns in this RMP model
        self.columns = collections.OrderedDict()
        self.current_cols = set()

        # Save the results of this RMP model
        self.num_of_drop = 0
        self.num_of_add = 0

        # Init columns
        if paths:
            pass
        else:
            paths = generate_initial_paths(sn)
        # paths = gen_init_paths(sn)

        for idx, path in enumerate(paths):
            self.columns[f"0_{idx}"] = {"path": path}
            self.columns[f"0_{idx}"]["0f_num"] = 0
            self.current_cols.add(path)

        self.num_of_add = len(self.columns)

        # Objective coefficient
        delta = get_delta(arcs=sn.arcs, columns=self.columns)

        P = {
            i: -sum(sn.network.edges[e_(arc)]["profit"] for arc in value["path"])
            for i, value in self.columns.items()
        }

        # Initialization
        self.model = gp.Model("RMP")

        # Vars
        X = self.model.addVars([i for i in sn.arcs], name="x")
        Y = self.model.addVars([i for i in sn.y_keys], name="y")
        # F = self.model.addVars([i for i in self.columns.keys()], obj=P, name="f")
        F = self.model.addVars([i for i in self.columns.keys()], name="f")
        # Slack variables
        # COM1 = self.model.addVars([c for c in sn.cargos], name='sv_com1')
        # CAPA = self.model.addVars([(p, t) for p in sn.ports for t in sn.times], name='sv_capa')

        # Constrs
        self.model.addConstrs(
            (F.sum(delta[arc]) - X[arc] == 0 for arc in sn.arcs),
            name="a",
        )

        self.model.addConstr(
            F.sum() == FLEET_CAPACITY,
            name="b",
        )

        self.model.addConstrs(
            (
                # Y.sum(p_o, "*", p_d, "*", t_1, t_2) + COM1[p_o, p_d, t_1, t_2]
                # == sn.demand_of_cargo[p_o, p_d, t_1, t_2]
                Y.sum(p_o, "*", p_d, "*", t_1, t_2)
                <= sn.demand_of_cargo[p_o, p_d, t_1, t_2]
                for p_o, p_d, t_1, t_2 in sn.cargos
            ),
            name="com1",  # commodity 1
        )

        self.model.addConstrs(
            (
                Y.sum(arc[0], arc[1], arc[2], arc[3], "*", "*") - X[arc] == 0
                for arc in sn.arcs.select("*", "*", "*", "*", "g")
            ),
            name="com2",  # commodity 2
        )

        self.model.addConstrs(
            (
                # X.sum(p, t, "*", "*", "g") + X.sum("*", "*", p, t, "g") + CAPA[(p, t)] == sn.capacity_of_port[p]
                X.sum(p, t, "*", "*", "g") + X.sum("*", "*", p, t, "g")
                <= sn.capacity_of_port[p, t]  #! Here
                for p in sn.ports
                for t in sn.times
            ),
            name="capa",
        )

        self.model.setObjective(F.prod(P), gp.GRB.MINIMIZE)

    def set_warm_start_attrs(self, results):
        var_name = "f"
        for k in self.columns.keys():
            var = self.model.getVarByName(name_(var_name, k))
            # var.VBasis = -2
            var.PStart = 0

        for var_name in ["x", "y", "f", "a", "com1", "com2", "capa", "b"]:
            if var_name in ["x", "y", "f"]:
                for k in results["primal_sols"][var_name].keys():
                    var = self.model.getVarByName(name_(var_name, k))
                    if var:
                        # var.VBasis = self.results['vbasis'][var_name][k]
                        var.PStart = results["primal_sols"][var_name][k]
            elif var_name in ["a", "com1", "com2", "capa"]:
                for k in results["dual_sols"][var_name].keys():
                    var = self.model.getConstrByName(name_(var_name, k))
                    # var.CBasis = self.results['cbasis'][var_name][k]
                    var.DStart = results["dual_sols"][var_name][k]
            elif var_name == "b":
                var = self.model.getConstrByName(var_name)
                # var.CBasis = self.results['cbasis'][var_name]
                var.DStart = results["dual_sols"][var_name]

    def solve(self, sn, results):
        """
        Solve the current RMP model.

        Args:
            sn (ShippingNetwork):

        Returns:

        """

        self.model.Params.LPWarmStart = 0
        self.model.Params.OutputFlag = 0
        self.model.Params.Method = 3

        if results:
            self.set_warm_start_attrs(results)

        self.model.optimize()

        # Optimal objective function
        obj = self.model.ObjVal

        # Optimal solution value
        x = {i: self.model.getVarByName(name_("x", i)).X for i in sn.arcs}
        y = {i: self.model.getVarByName(name_("y", i)).X for i in sn.y_keys}
        f = {i: self.model.getVarByName(name_("f", i)).X for i in self.columns.keys()}

        for k, v in f.items():
            if is_greater_than_zero(v):
                self.columns[k]["0f_num"] = 0
            else:
                self.columns[k]["0f_num"] += 1

        # VBasis
        x_vb = {i: self.model.getVarByName(name_("x", i)).VBasis for i in sn.arcs}
        y_vb = {i: self.model.getVarByName(name_("y", i)).VBasis for i in sn.y_keys}
        f_vb = {
            i: self.model.getVarByName(name_("f", i)).VBasis
            for i in self.columns.keys()
        }

        # Optimal dual solution value
        alphas = {i: self.model.getConstrByName(name_("a", i)).Pi for i in sn.arcs}
        beta = self.model.getConstrByName("b").Pi
        com1 = {i: self.model.getConstrByName(name_("com1", i)).Pi for i in sn.cargos}
        com2 = {
            i: self.model.getConstrByName(name_("com2", i)).Pi
            for i in sn.arcs.select("*", "*", "*", "*", "g")
        }
        capa = {
            (p, t): self.model.getConstrByName(name_("capa", (p, t))).Pi
            for p in sn.ports
            for t in sn.times
        }

        # CBasis
        alphas_cb = {
            i: self.model.getConstrByName(name_("a", i)).CBasis for i in sn.arcs
        }
        beta_cb = self.model.getConstrByName("b").CBasis
        com1_cb = {
            i: self.model.getConstrByName(name_("com1", i)).CBasis for i in sn.cargos
        }
        com2_cb = {
            i: self.model.getConstrByName(name_("com2", i)).CBasis
            for i in sn.arcs.select("*", "*", "*", "*", "g")
        }
        capa_cb = {
            (p, t): self.model.getConstrByName(name_("capa", (p, t))).CBasis
            for p in sn.ports
            for t in sn.times
        }

        # print("Finish RMP")
        return {
            "obj": obj,
            "primal_sols": {"x": x, "y": y, "f": f},
            "vbasis": {"x": x_vb, "y": y_vb, "f": f_vb},
            "dual_sols": {
                "a": alphas,
                "b": beta,
                "com1": com1,
                "com2": com2,
                "capa": capa,
            },
            "cbasis": {
                "a": alphas_cb,
                "b": beta_cb,
                "com1": com1_cb,
                "com2": com2_cb,
                "capa": capa_cb,
            },
        }

    def add_columns(self, iter_num, paths, features, labels):
        """Add new columns generated by subproblem."""
        for idx, value in enumerate(zip(paths, features, labels)):
            path, feature, label = value
            if label == 1:
                # Create a new Gurobi Column object
                c = gp.Column()
                for arc in path:
                    c.addTerms(1, self.model.getConstrByName(name_("a", arc)))
                c.addTerms(1, self.model.getConstrByName("b"))

                key = f"{iter_num}_{idx}"
                # profit = sum(sn.network.edges[e_(arc)]["profit"] for arc in path)
                profit = feature[1]
                self.model.addVar(obj=-profit, name=name_("f", key), column=c)
                self.columns[key] = {"path": path}
                self.columns[key]["0f_num"] = 0
                self.current_cols.add(path)
            else:
                pass

        self.num_of_add = sum(labels)
        self.model.ModelName = "RMP_%d" % iter_num
        self.model.update()

    def drop_columns(self):
        pop_keys = []
        self.num_of_drop = 0
        for key, value in self.columns.items():
            if value["0f_num"] >= MAX_0F_NUM:
                self.model.remove(self.model.getVarByName(name_("f", key)))
                pop_keys.append(key)
                self.num_of_drop += 1
                if self.num_of_drop >= 10:
                    break

        for key in pop_keys:
            pop_item = self.columns.pop(key)
            self.current_cols.remove(pop_item["path"])

        self.model.update()


class SubproblemDivide:
    def __init__(self, sn):
        self.g = {}
        self.beta = None

        def get_two_subnetworks():
            g_a = nx.MultiDiGraph()
            g_b = nx.MultiDiGraph()

            cut = int(NUM_OF_TIMES / 2)
            src_cut_1 = 1
            src_cut_2 = src_cut_1 + MAX_TT
            dst_cut_1 = src_cut_1 + cut
            dst_cut_2 = dst_cut_1 + MAX_TT

            def add_edge_(G):
                G.add_edge(u_for_edge=i, v_for_edge=j, key=key, profit=pr)

            def is_in_g_src():
                return t_o <= dst_cut_2 and t_d <= dst_cut_2 and t_o < t_d

            def is_in_g_dst():
                return (
                    (t_o >= dst_cut_1 and t_d >= dst_cut_1 and t_o < t_d)
                    or (t_o >= dst_cut_1 and t_d <= src_cut_2 and t_o > t_d)
                    or (t_o <= src_cut_2 and t_d <= src_cut_2 and t_o < t_d)
                )

            for i, j, key in sn.network.edges(keys=True):
                p_o, t_o = i
                p_d, t_d = j
                pr = sn.network.edges[i, j, key]["profit"]

                if is_in_g_src():
                    add_edge_(g_a)
                if is_in_g_dst():
                    add_edge_(g_b)

            return g_a, g_b

        def get_networks_for_subproblem():
            start = 1
            stop = start + MAX_TT - 1
            src_nodes = [(p, t) for p in sn.ports for t in range(start, stop + 1)]

            start = 1 + int(NUM_OF_TIMES / 2)
            stop = start + MAX_TT - 1
            dst_nodes = [(p, t) for p in sn.ports for t in range(start, stop + 1)]

            g_a, g_b = get_two_subnetworks()

            def get_network_info(g, name):
                nodelist = list(g)
                indices = {n: i for i, n in enumerate(nodelist)}
                src = np.array([indices[n] for n in src_nodes])
                dst = np.array([indices[n] for n in dst_nodes])
                src_mat = np.concatenate(
                    [src.reshape(-1, 1)] * dst.shape[0], axis=0
                ).reshape(-1)
                dst_mat = np.concatenate(
                    [dst.reshape(-1, 1)] * dst.shape[0], axis=1
                ).reshape(-1)
                if name == "a":
                    return {
                        "network": g,
                        "nodelist": nodelist,
                        "src": src,
                        "dst": dst,
                        "src_mat": src_mat,
                        "dst_mat": dst_mat,
                        "indices": indices,
                    }
                else:
                    return {
                        "network": g,
                        "nodelist": nodelist,
                        "src": dst,
                        "dst": src,
                        "src_mat": dst_mat,
                        "dst_mat": src_mat,
                        "indices": indices,
                    }

            return {"a": get_network_info(g_a, "a"), "b": get_network_info(g_b, "b")}

        self.g = get_networks_for_subproblem()

    def update_weights(self, results):
        alphas, beta = results["dual_sols"]["a"], results["dual_sols"]["b"]

        # update dual solutions value
        self.beta = beta

        cut = int(NUM_OF_TIMES / 2)
        src_cut_1 = 1
        src_cut_2 = src_cut_1 + MAX_TT
        dst_cut_1 = src_cut_1 + cut
        dst_cut_2 = dst_cut_1 + MAX_TT

        def update_weight_(G):
            G.edges[e]["weight"] = -n - G.edges[e]["profit"]

        def is_in_a():
            return t_o <= dst_cut_2 and t_d <= dst_cut_2 and t_o < t_d

        def is_in_b():
            return (
                (t_o >= dst_cut_1 and t_d >= dst_cut_1 and t_o < t_d)
                or (t_o >= dst_cut_1 and t_d <= src_cut_2 and t_o > t_d)
                or (t_o <= src_cut_2 and t_d <= src_cut_2 and t_o < t_d)
            )

        for arc, n in alphas.items():
            p_o, t_o = i_(arc)
            p_d, t_d = j_(arc)
            e = e_(arc)
            if is_in_a():
                update_weight_(self.g["a"]["network"])
            if is_in_b():
                update_weight_(self.g["b"]["network"])

        for _, n in self.g.items():
            n["matrix"] = nx.to_numpy_array(
                n["network"],
                nodelist=n["nodelist"],
                multigraph_weight=min,
                nonedge=np.nan,
            )

    def get_shortest_cyclic_path(self):
        def get_dist(n):
            dist_mat, predecessors = floyd_warshall(
                csgraph=csr_matrix(n["matrix"]), directed=True, return_predecessors=True
            )
            return dist_mat, predecessors

        self.g["a"]["dist_mat"], self.g["a"]["predecessors"] = get_dist(self.g["a"])
        self.g["b"]["dist_mat"], self.g["b"]["predecessors"] = get_dist(self.g["b"])

        rc_array = (
            self.g["a"]["dist_mat"][self.g["a"]["src_mat"], self.g["a"]["dst_mat"]]
            + self.g["b"]["dist_mat"][self.g["b"]["src_mat"], self.g["b"]["dst_mat"]]
        )

        idx = rc_array.argmin()
        rc = rc_array[idx] - self.beta

        results = {
            "a": {"src": None, "dst": None, "row": None},
            "b": {"src": None, "dst": None, "row": None},
        }

        def update_results(n):
            results[n]["src"] = self.g[n]["src_mat"][idx]
            results[n]["dst"] = self.g[n]["dst_mat"][idx]
            results[n]["row"] = self.g[n]["predecessors"][results[n]["src"]]

        update_results("a")
        update_results("b")

        indices = {"a": [], "b": []}
        cur = results["b"]["dst"]
        indices["b"].append(cur)
        while cur != results["b"]["src"]:
            cur = results["b"]["row"][cur]
            indices["b"].append(cur)

        cur = results["a"]["dst"]
        indices["a"].append(cur)
        while cur != results["a"]["src"]:
            cur = results["a"]["row"][cur]
            indices["a"].append(cur)

        return rc, {
            "a": {
                "i": indices["a"][1:],
                "j": indices["a"][:-1],
            },
            "b": {
                "i": indices["b"][1:],
                "j": indices["b"][:-1],
            },
        }

    def get_cyclic_paths(self, sn, max_num):
        def get_arcs_of_path(n):
            arcs = []
            for i_idx, j_idx in zip(indices[n]["i"], indices[n]["j"]):
                i, j = self.g[n]["nodelist"][i_idx], self.g[n]["nodelist"][j_idx]

                if i[0] == j[0]:
                    key = "h"
                else:
                    weight_d = self.g[n]["network"].edges[i, j, "d"]["weight"]
                    weight_g = self.g[n]["network"].edges[i, j, "g"]["weight"]
                    if weight_d < weight_g:
                        key = "d"
                    else:
                        key = "g"

                # i, j = n['nodelist'][i_idx], n['nodelist'][j_idx]
                arcs.append((*i, *j, key))

            return arcs

        def check_cyclic_path(path):
            for idx, arc in enumerate(path):
                try:
                    nxt = path[idx + 1]
                except IndexError:
                    nxt = path[0]

                if j_(arc) != i_(nxt):
                    print("Error path")

        def update_matrix_remove(name):
            if name == "a":
                name_0 = "b"
            else:
                name_0 = "a"

            for arc in arcs_of_path[name]:
                node_i = i_(arc)
                node_j = j_(arc)

                ind_i = self.g[name]["indices"][node_i]
                ind_j = self.g[name]["indices"][node_j]
                self.g[name]["matrix"][ind_i, ind_j] = np.inf

                try:
                    ind_i_0 = self.g[name_0]["indices"][node_i]
                    ind_j_0 = self.g[name_0]["indices"][node_j]
                    self.g[name_0]["matrix"][ind_i_0, ind_j_0] = np.inf
                except KeyError:
                    pass

        def update_matrix(name):
            if name == "a":
                name_0 = "b"
            else:
                name_0 = "a"

            for arc in arcs_of_path[name]:
                if k_(arc) == "g":
                    # self.g[name]['network'].edges[e_(arc)]['capacity'] -= capacity_min
                    # try:
                    #     self.g[name_0]['network'].edges[e_(arc)]['capacity'] -= capacity_min
                    # except KeyError:
                    #     pass

                    # if self.g[name]['network'].edges[e_(arc)]['capacity'] == 0:

                    # set the weight of this arc to type "d"
                    node_i = i_(arc)
                    node_j = j_(arc)

                    if node_i[0] == node_j[0]:
                        pass
                    else:
                        ind_i = self.g[name]["indices"][node_i]
                        ind_j = self.g[name]["indices"][node_j]
                        self.g[name]["matrix"][ind_i, ind_j] = self.g[name][
                            "network"
                        ].edges[node_i, node_j, "d"]["weight"]

                        try:
                            ind_i_0 = self.g[name_0]["indices"][node_i]
                            ind_j_0 = self.g[name_0]["indices"][node_j]
                            self.g[name_0]["matrix"][ind_i_0, ind_j_0] = self.g[name_0][
                                "network"
                            ].edges[node_i, node_j, "d"]["weight"]
                        except KeyError:
                            pass

        min_rc = float("inf")
        cyclic_paths = []
        features = []

        for _ in range(max_num):
            rc, indices = self.get_shortest_cyclic_path()

            if rc < min_rc:
                min_rc = rc

            if is_less_than_zero(rc):
                arcs_of_path = {
                    "a": get_arcs_of_path("a"),
                    "b": get_arcs_of_path("b"),
                }
                path = arcs_of_path["a"] + arcs_of_path["b"]
                path = tuple(sorted(path, key=lambda x: x[1]))

                # path = []
                # path.extend(get_arcs_of_path(self.networks['a'], indices['a']))
                # path.extend(get_arcs_of_path(self.networks['b'], indices['b']))
                # path = tuple(sorted(path, key=lambda x: x[1]))

                check_cyclic_path(path)
                cyclic_paths.append(path)

                profit = sum(sn.network.edges[e_(a)]["profit"] for a in path)
                features.append((rc, profit))

                is_stop = True
                for arc in path:
                    if arc[4] == "g":
                        is_stop = False
                        break

                if is_stop:
                    break

                # update_matrix('a')
                # update_matrix('b')
                update_matrix_remove("a")
                update_matrix_remove("b")
                # indices_0 = {'a': {'i': [], 'j': []}, 'b': {'i': [], 'j': []}}
                #
                # def get_indices(n):
                #     if n == 'a':
                #         n_0 = 'b'
                #     else:
                #         n_0 = 'a'
                #
                #     for i, j in zip(indices[n]['i'], indices[n]['j']):
                #         node_i = self.networks[n]['nodelist'][i]
                #         node_j = self.networks[n]['nodelist'][j]
                #         try:
                #             i_0 = self.networks[n_0]['indices'][node_i]
                #             j_0 = self.networks[n_0]['indices'][node_j]
                #             indices_0[n_0]['i'].append(i_0)
                #             indices_0[n_0]['j'].append(j_0)
                #         except KeyError:
                #             pass
                #
                # get_indices('b')
                # get_indices('a')
                #
                # self.networks['a']['matrix'][
                #     indices['a']['i'] + indices_0['a']['i'],
                #     indices['a']['j'] + indices_0['a']['j']] = np.inf
                # self.networks['b']['matrix'][
                #     indices['b']['i'] + indices_0['b']['i'],
                #     indices['b']['j'] + indices_0['b']['j']] = np.inf
            else:
                break

            # (199, 115, 469, 117, 'g'), (469, 117, 811, 1, 'g')
            # (199, 115, 190, 116, 'g'), (190, 116, 199, 117, 'g'), (199, 117, 811, 1, 'g')

            # t3 = perf_counter()
            # print(f"Floyd: {t2 - t1:.2f}")
            # print(f"Update info: {t3 - t3:.2f}")

        # def check_dup(c1, c2):
        #     # print('arc:', c1)
        #     # for i, arc in enumerate(cyclic_paths[c1]):
        #     #     if arc in cyclic_paths[c2]:
        #     #         print(i)
        #     print('arc:', c2)
        #     for i, arc in enumerate(cyclic_paths[c2]):
        #         if k_(arc) in ['h', 'd'] and (i_(arc), j_(arc), 'g') in cyclic_paths[c1]:
        #             print(i)
        return cyclic_paths, features, min_rc


class Label:
    __slots__ = ["w", "t", "i", "j", "a_ij", "wx", "p", "idx"]

    def __init__(self, w, t, p, i, j, a_ij, idx):
        """
        Initialize the attribute of a label.

        Args:
            w (int): Sum of weight.
            t (int): Sum of travel time.
            wx (float): weight_ij * (xbar_ij - x_ij).
            p (float): Sum of profit (one unit) of this path (label).
        """
        self.w = w
        self.t = t
        # self.wx = wx
        self.p = p
        self.i = i
        self.j = j
        self.a_ij = a_ij
        self.idx = idx

    def __lt__(self, nxt):
        """
        "A<B" means "A is dominated by B". Dominated label have

        - larger reduced cost
        - larger link flow
        - smaller total profit

        Args:
            nxt (Label): Another label belonging to the same node.

        Returns:
            (bool): Return true if this label is dominated by the label "nxt", otherwise false.
        """
        # if self.w > nxt.w and self.wx > nxt.wx and self.p < nxt.p:
        if self.w > nxt.w:
            return True
        else:
            return False


class LabelSetting:
    def __init__(self, sn):
        self.G = deepcopy(sn.network)
        self.nodes = [(p, t) for p in sn.ports for t in sn.times]

    def update_weights(self, results):
        alphas, beta = results["dual_sols"]["a"], results["dual_sols"]["b"]

        for arc, value in alphas.items():
            e = e_(arc)
            self.G.edges[e]["weight"] = -value - self.G.edges[e]["profit"]

        self.beta = beta

        self.G_succ = self.G._adj

    def single_source_label_setting(self, source: tuple):
        """
        Generate labels from the source node, end at the half of total time.

        Args:
            source (tuple): Starting node for path.

        Returns:
            (dict): {node: [label]}
        """
        H = []
        heappush(H, (0, source))

        # Label of node
        # w, t, p, i, j, a_ij, idx
        L = {source: [(Label(0, 0, 0, (0, 0), source, 0, 0))]}

        while H:
            # Choose a node
            i_tt, i = heappop(H)

            # Generate labels for successor node
            for j, e in self.G_succ[i].items():
                for key, attr in e.items():
                    t_ij = attr["travel_time"]
                    w_ij = attr["weight"]
                    a_ij = key
                    p_ij = attr["profit"]

                    for idx, label_i in enumerate(L[i]):
                        w_j = label_i.w + w_ij
                        t_j = label_i.t + t_ij
                        p_j = label_i.p + p_ij

                        if t_j >= NUM_OF_TIMES:
                            break

                        label_j = Label(w_j, t_j, p_j, i, j, a_ij, idx)

                        try:
                            L_j_new = []
                            is_dominated = False

                            for label in L[j]:
                                if label_j < label:
                                    heappush(L_j_new, label)
                                    is_dominated = True

                            if not is_dominated:
                                heappush(L_j_new, label_j)

                            L[j] = L_j_new
                        except KeyError:  # No label in node j
                            L[j] = [label_j]
                            heappush(H, (t_j, j))

        def backward_induction(source, L, beta):
            reduced_cost = np.inf
            profit = None
            pred = None
            arc = None

            for u, v, k in self.G.in_edges(source, keys=True):
                label = L[u][0]
                weight_sum = label.w + self.G.edges[u, v, k]["weight"]
                if weight_sum < reduced_cost:
                    reduced_cost = weight_sum
                    profit = label.p + self.G.edges[u, v, k]["profit"]
                    pred = u
                    arc = (*u, *v, k)

            path = [arc]

            label = L[pred][0]
            while label.j != source:
                arc = (label.i[0], label.i[1], label.j[0], label.j[1], label.a_ij)
                path.append(arc)
                label = L[label.i][label.idx]

            reduced_cost -= beta
            path = tuple(sorted(path, key=lambda x: x[1]))
            return path, profit, reduced_cost

        return backward_induction(source, L, self.beta)

    def get_cyclic_paths(self, max_num):
        min_rc = 0
        cyclic_paths = []
        features = []

        def check_cyclic_path(path):
            for idx, arc in enumerate(path):
                try:
                    nxt = path[idx + 1]
                except IndexError:
                    nxt = path[0]

                if j_(arc) != i_(nxt):
                    print("Error path")

        num = 0
        for source in self.nodes:
            path, profit, reduced_cost = self.single_source_label_setting(source)
            check_cyclic_path(path)
            if reduced_cost <= min_rc:
                min_rc = reduced_cost

                cyclic_paths.append(path)
                features.append((reduced_cost, profit))

                num += 1

            if num >= max_num:
                break

        return cyclic_paths, features, min_rc

        # def backward_induction(label, L):
        #     sum_of_weight = label.w
        #     path = []
        #     while label.i[0] != 0:
        #         arc = (label.i[0], label.i[1], label.j[0], label.j[1], label.a_ij)
        #         path.append(arc)
        #         label = L[label.i][label.idx]
        #     return sum_of_weight, path

        # def generate_paths(nodes_1, nodes_2):
        #     p = {}
        #     for node_a in nodes_1:
        #         L = self.single_source_label_setting(node_a)
        #         for node_b in nodes_2:
        #             p[(node_a, node_b)] = []
        #             for label in L[node_b]:
        #                 heappush(p[(node_a, node_b)], backward_induction(label, L))
        #     return p

        # paths = {}
        # paths.update(generate_paths(self.nodes_a, self.nodes_b))
        # paths.update(generate_paths(self.nodes_b, self.nodes_a))

        # cycles = []
        # for n_a in self.nodes_a:
        #     for n_b in self.nodes_b:
        #         for path_a, path_b in zip(paths[(n_a, n_b)], paths[(n_b, n_a)]):
        #             try:
        #                 sum_of_weight = path_a[0] + path_b[0] + self.beta
        #                 cycle = path_a[1] + path_b[1]
        #                 heappush(cycles, (sum_of_weight, cycle))
        #             except KeyError:
        #                 pass

        # return cycles


class Subproblem:
    def __init__(self, sn):
        self.beta = None

        def get_one_subnetwork():
            g = nx.MultiDiGraph()

            # edges_d = []
            # edges_g = []

            def add_edge_(G):
                G.add_edge(u_for_edge=u, v_for_edge=v, key=key, profit=pr)

            # def append_node_():
            #     if key != "d":
            #         edges_g.append((u, v, key))
            #     elif key != "g":
            #         edges_d.append((u, v, key))

            for i, j, key in sn.network.edges(keys=True):
                # tt = sn.network.edges[i, j, key]["travel_time"]
                pr = sn.network.edges[i, j, key]["profit"]

                if is_dup_node(j):
                    if not is_dup_node(i):
                        u, v = i, fake_node(j)
                        add_edge_(g)
                        # append_node_()
                    else:
                        u, v = fake_node(i), fake_node(j)
                        add_edge_(g)
                        # append_node_()

                        u, v = i, j
                        add_edge_(g)
                        # append_node_()
                else:
                    u, v = i, j
                    add_edge_(g)
                    # append_node_()

            # return G, edges_d, edges_g
            return g

        def get_networks_for_subproblem():
            start = 1
            stop = start + MAX_TT - 1
            src_nodes = [(p, t) for p in sn.ports for t in range(start, stop + 1)]
            dst_nodes = [
                (p, t + NUM_OF_TIMES) for p in sn.ports for t in range(start, stop + 1)
            ]

            g = get_one_subnetwork()

            def get_network_info(g):
                nodelist = list(g)
                indices = {n: i for i, n in enumerate(nodelist)}
                src = np.array([indices[n] for n in src_nodes])
                dst = np.array([indices[n] for n in dst_nodes])

                return {
                    "network": g,
                    "nodelist": nodelist,
                    "src": src,
                    "dst": dst,
                    "nodelist_real": [real_node(n) for n in nodelist],
                    "indices": indices,
                }

            return get_network_info(g)

        self.networks = {"g": get_networks_for_subproblem()}

        # self.G, self.edges_d, self.edges_g = get_one_subnetwork()

        # self.nodelist = list(self.G)
        # self.nodelist_real = [real_node(n) for n in self.nodelist]

        # indices = {n: i for i, n in enumerate(self.nodelist)}

        # self.src = np.array(
        #     [indices[(p, t)] for p in sn.ports for t in range(1, MAX_TT + 1)]
        # )
        # self.dst = np.array(
        #     [
        #         indices[(p, t + NUM_OF_TIMES)]
        #         for p in sn.ports
        #         for t in range(1, MAX_TT + 1)
        #     ]
        # )

    def update_weights(self, results):
        alphas, beta = results["dual_sols"]["a"], results["dual_sols"]["b"]

        # update dual solutions value
        self.beta = beta

        def update_weight_():
            self.networks["g"]["network"].edges[e]["weight"] = (
                -n - self.networks["g"]["network"].edges[e]["profit"]
            )

        for arc, n in alphas.items():
            if is_dup_node(j_(arc)):
                if not is_dup_node(i_(arc)):
                    e = e_fake_j(arc)
                    update_weight_()
                else:
                    e = e_fake_ij(arc)
                    update_weight_()
                    e = e_(arc)
                    update_weight_()
            else:
                e = e_(arc)
                update_weight_()

        for n in self.networks.values():
            n["matrix"] = nx.to_numpy_array(
                n["network"],
                nodelist=n["nodelist"],
                multigraph_weight=min,
                nonedge=np.inf,
            )

    def get_shortest_path(self, n):
        g = csr_matrix(n["matrix"])
        dist_matrix, predecessors = floyd_warshall(
            csgraph=g, directed=True, return_predecessors=True
        )

        idx = dist_matrix[n["src"], n["dst"]].argmin()
        src, dst = n["src"][idx], n["dst"][idx]
        row = predecessors[src]
        rc = dist_matrix[src, dst] - self.beta

        indices = []

        cur = dst
        indices.append(cur)
        while cur != src:
            cur = row[cur]
            indices.append(cur)

        return rc, {"i": indices[1:], "j": indices[:-1]}

    def get_cyclic_paths(self, sn, max_num):
        def get_arcs_of_path(n, indices):
            arcs = []
            for i_idx, j_idx in zip(indices["i"], indices["j"]):
                i, j = n["nodelist"][i_idx], n["nodelist"][j_idx]
                if i[0] == j[0]:
                    key = "h"
                else:
                    try:
                        weight_d = n["network"].edges[i, j, "d"]["weight"]
                        weight_g = n["network"].edges[i, j, "g"]["weight"]
                    except KeyError:
                        j = fake_node(j)
                        weight_d = n["network"].edges[i, j, "d"]["weight"]
                        weight_g = n["network"].edges[i, j, "g"]["weight"]
                    if weight_d < weight_g:
                        key = "d"
                    else:
                        key = "g"

                i, j = n["nodelist_real"][i_idx], n["nodelist_real"][j_idx]
                arcs.append((*i, *j, key))

            return arcs

            # if is_dup_node(j) and not is_dup_node(i):
            #     u, v = i, fake_node(j)
            # else:
            #     u, v = i, j
            # if u[0] == v[0]:
            #     key = "h"
            # else:
            #     if G.edges[u, v, "d"]["weight"] < G.edges[u, v, "g"]["weight"]:
            #         key = "d"
            #     else:
            #         key = "g"

            # arcs = tuple(sorted(arcs, key=lambda x: x[1]))

        # 1 path needs 0.09s for 10P 60T
        min_rc = float("inf")
        cyclic_paths = []
        features = []

        for _ in range(max_num):
            rc, indices = self.get_shortest_path(self.networks["g"])

            if rc < min_rc:
                min_rc = rc

            if is_less_than_zero(rc):
                path = get_arcs_of_path(self.networks["g"], indices)
                path = tuple(sorted(path, key=lambda x: x[1]))
                cyclic_paths.append(path)
                profit = sum(sn.network.edges[e_(a)]["profit"] for a in path)
                features.append((rc, profit))
                self.networks["g"]["matrix"][indices["i"], indices["j"]] = np.inf
            else:
                break

            # sp.networks['g']['indices'][(811, 121)]
            # 1190
            # sp.networks['g']['indices'][(811, 1)]
            # 1246
            # sp.networks['g']['indices'][(469, 117)]
            # 1178
            # t3 = perf_counter()
            # print(f"Floyd: {t2 - t1:.2f}")
            # print(f"Update info: {t3 - t3:.2f}")

        return cyclic_paths, features, min_rc


class InfoLogger:
    def __init__(self, model_dir, instance_idx, log_filename="iter.csv"):
        self.datadir = f"{model_dir}/{instance_idx}"

        self.log_path = f"{self.datadir}/{log_filename}"

        with open(self.log_path, "w") as f:
            f.write(
                "Iter|Gen RMP time|Sol RMP time|Sol SP time|Col- |Col+ |RMP col|SP path|Flow > 0|Obj|RC\n"
            )

        self.iter = 0

    def write_info(self, t, info):
        t1, t2, t3, t4 = t
        # num_of_drop = rmp.num_of_drop
        # num_of_add = rmp.num_of_add
        # num_of_rmp_cols = len(rmp.columns)
        # num_of_sp_paths = len(sp.paths)
        # num_of_positive_F = sum(
        #     [1 for value in rmp.results["primal_sols"]["f"].values() if value > 0]
        # )
        # obj = rmp.results["obj"]
        # rc = sp.min_rc
        with open(self.log_path, "a") as f:
            f.write(
                "%d|%.2f|%.2f|%.2f|%d|%d|%d|%d|%d|%.2f|%.2f\n"
                % (
                    self.iter,
                    t2 - t1,
                    t3 - t2,
                    t4 - t3,
                    info["Col-"],
                    info["Col+"],
                    info["RMP col"],
                    info["SP path"],
                    info["Flow > 0"],
                    info["Obj"],
                    info["RC"],
                )
            )

        self.iter += 1

    def write_gurobi_model(self, model):
        """Save the Gurobi model."""
        # model.write(f"{self.datadir}/data/model_{self.iter}.mps")
        model.write(f"{self.datadir}/data/model_{self.iter}.lp")

    def write_gurobi_sol(self, model):
        """Save the Gurobi results."""
        with open(f"{self.datadir}/data/sol_{self.iter}.json", "w") as f:
            f.write(model.getJSONSolution())


class ExpertMILP:
    def __init__(self, m, paths, features, results=None):
        # Copy the previous problem
        m.update()
        self.model = m.copy()
        # m.Params.LPWarmStart = 2  # Presolve
        self.model.ModelName = "ColumnSelection"

        for i, value in enumerate(zip(paths, features)):
            path, feature = value
            # Generate the new column
            c = gp.Column()
            for arc in path:
                c.addTerms(1, self.model.getConstrByName(name_("a", arc)))
            c.addTerms(1, self.model.getConstrByName("b"))

            profit = feature[1]

            # Update objective function
            column_var = self.model.addVar(
                obj=-profit,
                name=name_("f", f"new_{i}"),
                column=c,
            )

            expert_var = self.model.addVar(
                ub=1,
                obj=0.01,
                vtype=gp.GRB.BINARY,
                name=name_("e", i),
            )

            # Generate constraint (8)
            self.model.addConstr(
                column_var - FLEET_CAPACITY * expert_var <= 0,
                # column_var - expert_var <= 0,
                name=name_("expert", i),
            )

        self.model.update()

        if results:
            for var_name in ["x", "y", "f"]:
                for k in results["primal_sols"][var_name].keys():
                    var = self.model.getVarByName(name_(var_name, k))
                    if var:
                        var.Start = results["primal_sols"][var_name][k]
            for i in range(len(features)):
                self.model.getVarByName(name_("e", i)).Start = 1

        # flows = [
        #     m.getVarByName(name_("f", f"new_{i}")).X for i in range(len(features))
        # ]

        self.model.optimize()

        # print("MILP obj: %.2f" % self.model.ObjVal)
        # print("MILP F[0]: %.2f" % self.model.getVarByName(name_("f", "new_0")).X)
        # print("MILP P[0]: %.2f" % self.model.getVarByName(name_("f", "new_0")).Obj)

        self.labels = [
            self.model.getVarByName(name_("e", i)).X for i in range(len(features))
        ]


class InitHoldingArcs:
    def __init__(self, sn):
        paths = []

        def holding_path(p):
            path = []
            for t in sn.times:
                node_i = (p, t)
                node_j = (p, get_t_2(t, 1))
                arc_ij = (*node_i, *node_j, "h")
                path.append(arc_ij)

            # check_cyclic_path(path)
            return tuple(path)

        for p in sn.ports:
            paths.append(holding_path(p))

        self.cyclic_paths = paths


class InitPath:
    def __init__(self, sn):
        self.g = None

        def get_two_subnetworks():
            g_a = nx.MultiDiGraph()
            g_b = nx.MultiDiGraph()

            cut = int(NUM_OF_TIMES / 2)
            src_cut_1 = 1
            src_cut_2 = src_cut_1 + MAX_TT
            dst_cut_1 = src_cut_1 + cut
            dst_cut_2 = dst_cut_1 + MAX_TT

            def add_edge_(G):
                G.add_edge(
                    u_for_edge=i,
                    v_for_edge=j,
                    key=key,
                    profit=profit,
                    capacity=capacity,
                )

            def in_g_a():
                return t_o <= dst_cut_2 and t_d <= dst_cut_2 and t_o < t_d

            def in_g_b():
                return (
                    (t_o >= dst_cut_1 and t_d >= dst_cut_1 and t_o < t_d)
                    or (t_o >= dst_cut_1 and t_d <= src_cut_2 and t_o > t_d)
                    or (t_o <= src_cut_2 and t_d <= src_cut_2 and t_o < t_d)
                )
                # return (
                #     t_o >= dst_cut_1 and (t_d >= dst_cut_1 or t_d <= src_cut_2)
                # ) or (t_o <= src_cut_2 and t_d <= src_cut_2)

            for i, j, key in sn.network.edges(keys=True):
                p_o, t_o = i
                p_d, t_d = j

                profit = -sn.network.edges[i, j, key]["profit"]

                if key == "g":
                    # 为什么这里有一个地板除？为了让 capacity 减少的时候更整一些，不影响结果，没问题
                    capacity = sn.demand_of_cargo.select(p_o, p_d, t_o, "*")[0] // 1000
                else:
                    capacity = np.inf

                if in_g_a():
                    add_edge_(g_a)
                if in_g_b():
                    add_edge_(g_b)

            return g_a, g_b

        def get_g():
            start = 1
            stop = start + MAX_TT - 1
            src_nodes = [(p, t) for p in sn.ports for t in range(start, stop + 1)]

            start = 1 + int(NUM_OF_TIMES / 2)
            stop = start + MAX_TT - 1
            dst_nodes = [(p, t) for p in sn.ports for t in range(start, stop + 1)]

            g_a, g_b = get_two_subnetworks()

            def get_network_info(g, name):
                nodelist = list(g)
                indices = {n: i for i, n in enumerate(nodelist)}
                src = np.array([indices[n] for n in src_nodes])
                dst = np.array([indices[n] for n in dst_nodes])
                src_mat = np.concatenate(
                    [src.reshape(-1, 1)] * dst.shape[0], axis=0
                ).reshape(-1)
                dst_mat = np.concatenate(
                    [dst.reshape(-1, 1)] * dst.shape[0], axis=1
                ).reshape(-1)
                if name == "a":
                    return {
                        "network": g,
                        "nodelist": nodelist,
                        "src": src,
                        "dst": dst,
                        "src_mat": src_mat,
                        "dst_mat": dst_mat,
                        "indices": indices,
                        "matrix": nx.to_numpy_array(
                            g,
                            nodelist=nodelist,
                            multigraph_weight=min,
                            nonedge=np.nan,
                            weight="profit",
                        ),
                    }
                else:
                    return {
                        "network": g,
                        "nodelist": nodelist,
                        "src": dst,
                        "dst": src,
                        "src_mat": dst_mat,
                        "dst_mat": src_mat,
                        "indices": indices,
                        "matrix": nx.to_numpy_array(
                            g,
                            nodelist=nodelist,
                            multigraph_weight=min,
                            nonedge=np.nan,
                            weight="profit",
                        ),
                    }

            return {"a": get_network_info(g_a, "a"), "b": get_network_info(g_b, "b")}

        self.g = get_g()

    def get_shortest_cyclic_path(self):
        def get_dist(n):
            dist_mat, predecessors = floyd_warshall(
                csgraph=csr_matrix(n["matrix"]), directed=True, return_predecessors=True
            )
            return dist_mat, predecessors

        self.g["a"]["dist_mat"], self.g["a"]["predecessors"] = get_dist(self.g["a"])
        self.g["b"]["dist_mat"], self.g["b"]["predecessors"] = get_dist(self.g["b"])

        rc_array = (
            self.g["a"]["dist_mat"][self.g["a"]["src_mat"], self.g["a"]["dst_mat"]]
            + self.g["b"]["dist_mat"][self.g["b"]["src_mat"], self.g["b"]["dst_mat"]]
        )

        idx = rc_array.argmin()
        rc = rc_array[idx]

        results = {
            "a": {"src": None, "dst": None, "row": None},
            "b": {"src": None, "dst": None, "row": None},
        }

        def update_results(n):
            results[n]["src"] = self.g[n]["src_mat"][idx]
            results[n]["dst"] = self.g[n]["dst_mat"][idx]
            results[n]["row"] = self.g[n]["predecessors"][results[n]["src"]]

        update_results("a")
        update_results("b")

        indices = {"a": [], "b": []}
        cur = results["b"]["dst"]
        indices["b"].append(cur)
        while cur != results["b"]["src"]:
            cur = results["b"]["row"][cur]
            indices["b"].append(cur)

        cur = results["a"]["dst"]
        indices["a"].append(cur)
        while cur != results["a"]["src"]:
            cur = results["a"]["row"][cur]
            indices["a"].append(cur)

        return rc, {
            "a": {
                "i": indices["a"][1:],
                "j": indices["a"][:-1],
            },
            "b": {
                "i": indices["b"][1:],
                "j": indices["b"][:-1],
            },
        }

    def get_cyclic_paths(self, max_num):
        def get_arcs_of_path(n):
            arcs = []
            for i_idx, j_idx in zip(indices[n]["i"], indices[n]["j"]):
                i, j = self.g[n]["nodelist"][i_idx], self.g[n]["nodelist"][j_idx]

                if i[0] == j[0]:
                    key = "h"
                else:
                    if self.g[n]["network"].edges[i, j, "g"]["capacity"] == 0:
                        key = "d"
                    else:
                        key = "g"

                arcs.append((*i, *j, key))

            return arcs

        def check_cyclic_path(path):
            for idx, arc in enumerate(path):
                try:
                    nxt = path[idx + 1]
                except IndexError:
                    nxt = path[0]

                if j_(arc) != i_(nxt):
                    print("Error path")

        def find_min_capacity():
            capacities = []

            for name in ["a", "b"]:
                for arc in arcs_of_path[name]:
                    if k_(arc) == "g":
                        capacities.append(
                            self.g[name]["network"].edges[e_(arc)]["capacity"]
                        )

            return min(capacities)

        def update_matrix(name):
            if name == "a":
                name_0 = "b"
            else:
                name_0 = "a"

            for arc in arcs_of_path[name]:
                if k_(arc) == "g":
                    self.g[name]["network"].edges[e_(arc)]["capacity"] -= capacity_min
                    try:
                        self.g[name_0]["network"].edges[e_(arc)]["capacity"] -= (
                            capacity_min
                        )
                    except KeyError:
                        pass

                    if self.g[name]["network"].edges[e_(arc)]["capacity"] == 0:
                        # set the profit of this arc to type "d"
                        node_i = i_(arc)
                        node_j = j_(arc)

                        if node_i[0] == node_j[0]:
                            pass
                        else:
                            ind_i = self.g[name]["indices"][node_i]
                            ind_j = self.g[name]["indices"][node_j]
                            self.g[name]["matrix"][ind_i, ind_j] = self.g[name][
                                "network"
                            ].edges[node_i, node_j, "d"]["profit"]

                            try:
                                ind_i_0 = self.g[name_0]["indices"][node_i]
                                ind_j_0 = self.g[name_0]["indices"][node_j]
                                self.g[name_0]["matrix"][ind_i_0, ind_j_0] = self.g[
                                    name_0
                                ]["network"].edges[node_i, node_j, "d"]["profit"]
                            except KeyError:
                                pass

        min_rc = float("inf")
        cyclic_paths = []
        features = []

        for _ in tqdm(range(max_num)):
            # if _ in [384, 385, 386]:
            #     mzj = '1'

            rc, indices = self.get_shortest_cyclic_path()

            if rc < min_rc:
                min_rc = rc

            arcs_of_path = {
                "a": get_arcs_of_path("a"),
                "b": get_arcs_of_path("b"),
            }
            path = arcs_of_path["a"] + arcs_of_path["b"]
            path = tuple(sorted(path, key=lambda x: x[1]))

            check_cyclic_path(path)
            cyclic_paths.append(path)

            features.append(rc)

            is_stop = True
            for arc in path:
                if arc[4] == "g":
                    is_stop = False
                    break

            if is_stop:
                break

            capacity_min = find_min_capacity()

            update_matrix("a")
            update_matrix("b")

        return cyclic_paths, features, min_rc
