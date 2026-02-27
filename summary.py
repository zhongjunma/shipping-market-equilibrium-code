import json
import re
from pathlib import Path

import gurobipy as gp
import pandas as pd

DATA_BASE_DIR = Path("./data")


def read_sol(model_dir, num_iteration):
    filepath = model_dir / f"sol_{num_iteration}.json"
    with open(filepath, "r") as f:
        sol = json.load(f)

    sols = {}
    for var in sol["Vars"]:
        sols[var["VarName"]] = var["X"]
    return sols


def destringizer(s):
    m = re.match(r"([xyfa])\[(.+)\]", s)
    t = m.group(1)
    k = m.group(2)
    if t == "x" or t == "a":
        k = re.match(r"(\d+),(\d+),(\d+),(\d+),([dgh])", k)
        k = tuple([int(k.group(i)) for i in range(1, 5)] + [k.group(5)])
    elif t == "y":
        k = re.match(r"(\d+),(\d+),(\d+),(\d+),(\d+),(\d+)", k)
        k = tuple([int(k.group(i)) for i in range(1, 7)])
    elif t == "f":
        k = re.match(r"(\d+)_(\d+)", k)
        k = tuple([int(k.group(i)) for i in range(1, 3)])
    return t, k


def get_variables(model, sols):
    vars = model.getVars()
    X = gp.tupledict()
    Y = gp.tupledict()
    F = gp.tupledict()
    f = gp.tupledict()

    for var in vars:
        varname = var.getAttr(gp.GRB.Attr.VarName)
        t, k = destringizer(varname)
        if t == "x":
            try:
                X[k] = sols[varname]
            except Exception:
                X[k] = 0
        elif t == "y":
            try:
                Y[k] = sols[varname]
            except Exception:
                Y[k] = 0
        elif t == "f":
            f[k] = var
            try:
                F[k] = sols[varname]
            except Exception:
                F[k] = 0

    return X, Y, F, f


def get_paths(model, f):
    paths = {}
    path_ids = {}

    for key, var in f.items():
        col = model.getCol(var)
        constrs = [col.getConstr(i) for i in range(col.size() - 1)]
        path = [
            destringizer(constr.getAttr(gp.GRB.Attr.ConstrName))[1]
            for constr in constrs
        ]
        path = tuple(sorted(path, key=lambda x: x[1]))
        paths[key] = path

        if path not in path_ids.keys():
            path_ids[path] = []
        path_ids[path].append(key)
    return paths, path_ids


def get_gp_solution(model_dir, num_iteration):
    m = gp.read(str(model_dir / f"model_{num_iteration}.lp"))
    s = read_sol(model_dir, num_iteration)

    X, Y, F, f = get_variables(m, s)
    paths, path_ids = get_paths(m, f)
    gp_solution = {
        "X": X,
        "Y": Y,
        "F": F,
        "paths": paths,
        "path_ids": path_ids,
    }
    return gp_solution


if __name__ == "__main__":
    scenarios = {
        "Baseline": "real-world-1",
        "No Heuristic": "real-world-no-h",
        "LSA": "real-world-lsa",
        "Cost^{-25\%}": "cost-minus-25",
        "Cost^{-50\%}": "cost-minus-50",
        "Cost^{+25\%}": "cost-plus-25",
        "Cost^{+50\%}": "cost-plus-50",
        "Income$^{-25\%}": "income-minus-25",
        "Income$^{-50\%}": "income-minus-50",
        "Income$^{+25\%}": "income-plus-25",
        "Income$^{+50\%}": "income-plus-50",
        "Demand$^{-25\%}": "demand-minus-25",
        "Demand$^{-50\%}": "demand-minus-50",
        "Demand$^{+25\%}": "demand-plus-25",
        "Demand$^{+50\%}": "demand-plus-50",
        "Demand$^{-75\%}": "demand-75",
        "Capacity$^{-50\%}": "capacity-50",
        "Capacity$^{-75\%}": "capacity-75",
    }

    indices = [str(_) for _ in range(10)]

    result_df = []
    for scenario, scenario_folder in scenarios.items():
        for index in indices:
            instance_dir = DATA_BASE_DIR / scenario_folder / index
            df = pd.read_csv(instance_dir / "iter.csv", sep="|", header=0)

            cpu_mp = (
                df.loc[:, ["Gen RMP time", "Sol RMP time", "Sol SP time"]].sum().sum()
            )
            cpu_rmp = df.loc[:, ["Gen RMP time", "Sol RMP time"]].sum().sum()
            cpu_sp = df["Sol SP time"].sum()
            num_iteration = df.shape[0]
            idx_last_row = num_iteration - 1
            num_column = int(df.iloc[idx_last_row]["RMP col"])
            num_positive_flow = int(df.iloc[idx_last_row]["Flow > 0"])

            lower_bound_df = pd.read_csv(
                instance_dir / "iter_lower_bound.csv", sep="|", header=0
            )
            lower_bound = -lower_bound_df.iloc[0]["Obj"]

            # df["runtime"] = df["Gen RMP time"] + df["Sol RMP time"] + df["Sol SP time"]
            # df["cumsum"] = df["runtime"].cumsum()

            optimal_obj_value = -df.iloc[idx_last_row]["Obj"]
            # try:
            #     obj_1hr = -df.loc[
            #         (df["cumsum"] <= 3600) & (df["cumsum"].shift(-1) > 3600), "Obj"
            #     ].values[0]
            # except IndexError:
            #     obj_1hr = obj_final

            # gap_1hr = (obj_final - obj_1hr) / (obj_final - lower_bound) * 100

            initial_obj_value = -df.iloc[0]["Obj"]
            gap_h = (
                (optimal_obj_value - initial_obj_value)
                / (optimal_obj_value - lower_bound)
                * 100
            )

            gp_solution = get_gp_solution(instance_dir / "data", str(num_iteration))
            flow_distribution = {"h": 0, "d": 0, "g": 0}
            for key, value in gp_solution["X"].items():
                if value > 0:
                    flow_distribution[key[-1]] += value
            sum_of_link_flow = sum(flow_distribution.values())
            for key, value in flow_distribution.items():
                flow_distribution[key] = value / sum_of_link_flow

            result_df.append(
                (
                    scenario,
                    index,
                    cpu_mp,
                    cpu_rmp,
                    cpu_sp,
                    num_iteration,
                    num_column,
                    num_positive_flow,
                    # gap_1hr,
                    optimal_obj_value,
                    initial_obj_value,
                    lower_bound,
                    gap_h,
                    flow_distribution["h"],
                    flow_distribution["d"],
                    flow_distribution["g"],
                )
            )

    result_df = pd.DataFrame(
        result_df,
        columns=[
            "instance",
            "index",
            "cpu_mp",
            "cpu_rmp",
            "cpu_sp",
            "num_iteration",
            "num_column",
            "num_positive_flow",
            "optimal_obj_value",
            "initial_obj_value",
            "lower_bound",
            "gap_h",
            "flow_h_ratio",
            "flow_d_ratio",
            "flow_g_ratio",
        ],
    )
    result_df.to_excel(Path("tables/summary_raw.xlsx"), index=False)
    summary_df = result_df.groupby("instance").mean(numeric_only=True).reset_index()
    summary_df.to_excel(Path("tables/summary.xlsx"), index=False)
