from pathlib import Path
from time import perf_counter

from tqdm import tqdm

from column_generation import (
    InfoLogger,
    InitPath,
    RestrictedMasterProblem,
    ShippingNetwork,
    SubproblemDivide,
    is_greater_than_zero,
    is_less_than_zero,
)


def new_init_path(model_dir, idx):
    t1 = perf_counter()

    # Initialize object, generate RMP
    sn = ShippingNetwork(model_dir, idx)

    ip = InitPath(sn)
    cyclic_paths, features, min_rc = ip.get_cyclic_paths(100000)
    # cyclic_paths = InitHoldingArcs(sn).cyclic_paths

    sp = SubproblemDivide(sn)
    rmp = RestrictedMasterProblem(sn, paths=cyclic_paths)
    info_logger = InfoLogger(model_dir, idx)

    t2 = perf_counter()

    # Solve RMP
    results = rmp.solve(sn, results=None)

    t3 = perf_counter()

    # Solve SP
    sp.update_weights(results)

    # t31 = perf_counter()
    # print("update weight: %.2fs" % (t31 - t3))

    max_num = 50
    cyclic_paths, features, min_rc = sp.get_cyclic_paths(sn, max_num)

    t4 = perf_counter()
    # print("get path: total %.2fs, average %.2fs" % (t4 - t31, (t4 - t31) / max_num))
    # print("min reduced cost: %.2f" % min_rc)

    # Save results of this iteration
    info = {
        "Col-": rmp.num_of_drop,
        "Col+": rmp.num_of_add,
        "RMP col": len(rmp.current_cols),
        "SP path": len(features),
        "Flow > 0": sum(
            [1 for v in results["primal_sols"]["f"].values() if is_greater_than_zero(v)]
        ),
        "Obj": results["obj"],
        "RC": min_rc,
    }
    info_logger.write_info((t1, t2, t3, t4), info)
    rmp.num_of_drop = 0
    rmp.num_of_add = 0

    while is_less_than_zero(min_rc):
        # while min_rc >= -1:
        # print("\n\nIter %d" % info_logger.iter)
        # Update RMP columns, add or drop
        t1 = perf_counter()

        # Check if RMP need drop columns
        # if info["RMP col"] / info["Flow > 0"] >= 1.5:
        #     rmp.drop_columns()
        # rmp.drop_columns()

        labels = [1 if p not in rmp.current_cols else 0 for p in cyclic_paths]
        rmp.add_columns(info_logger.iter, cyclic_paths, features, labels=labels)

        t2 = perf_counter()

        # Solve RMP
        results = rmp.solve(sn, results)
        # print("RMP obj: %.2f" % results['obj'])
        # print("RMP F[0]: %.2f" % results['primal_sols']['f']['%d_%d' % (info_logger.iter, 0)])

        t3 = perf_counter()

        # Solve SP
        sp.update_weights(results)

        # t31 = perf_counter()
        # print('update weight: %.2fs' % (t31 - t3))

        # max_num = 50
        # if info_logger.iter <= 30:
        #     max_num = 50
        # else:
        #     max_num = 30

        cyclic_paths, features, min_rc = sp.get_cyclic_paths(sn, max_num)

        t4 = perf_counter()
        # print('get path: total %.2fs, average %.2fs' % (t4 - t31, (t4 - t31) / max_num))
        # print('min reduced cost: %.2f' % reduced_costs[0])
        # mzj = 0
        # Save results of this iteration
        info = {
            "Col-": rmp.num_of_drop,
            "Col+": rmp.num_of_add,
            "RMP col": len(rmp.current_cols),
            "SP path": len(features),
            "Flow > 0": sum(
                [
                    1
                    for v in results["primal_sols"]["f"].values()
                    if is_greater_than_zero(v)
                ]
            ),
            "Obj": results["obj"],
            "RC": min_rc,
        }
        info_logger.write_info((t1, t2, t3, t4), info)
        # rmp.num_of_drop = 0
        # rmp.num_of_add = 0

        if info_logger.iter % 50 == 0 or info["RC"] >= -1:
            info_logger.write_gurobi_model(rmp.model)
            info_logger.write_gurobi_sol(rmp.model)
        # info_logger.write_gurobi_model(rmp.model)
        # info_logger.write_gurobi_sol(rmp.model)

    # save the result of final iteration
    info_logger.write_gurobi_model(rmp.model)
    info_logger.write_gurobi_sol(rmp.model)


if __name__ == "__main__":
    base_dir = Path("./data")
    model_dirs = [
        "capacity-minus-25",
        "capacity-minus-50",
        "capacity-50",
        "capacity-75",
        "demand+75",
        "demand-75",
        "real-world-1",
        "cost-minus-25",
        "cost-minus-50",
        "cost-plus-25",
        "cost-plus-50",
        "income-minus-25",
        "income-minus-50",
        "income-plus-25",
        "income-plus-50",
        "demand-minus-25",
        "demand-minus-50",
        "demand-plus-25",
        "demand-plus-50",
    ]
    idx_list = [i for i in range(10)]
    for md in tqdm(model_dirs):
        for idx in idx_list:
            new_init_path(base_dir / md, idx)
