"""Score LCB execution results using the paper's calc_best_of_n algorithm."""
import json
import sys
from tqdm import tqdm

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f]

def calc_best_of_n(dataset, sol_num, ut_num, task_sol_results, task_sol_ut_results):
    accuracy = 0
    for data in dataset:
        task_id = data['task_id']
        sol_pass_ut_set = {}
        for sol_id in range(sol_num):
            for ut_id in range(ut_num):
                if sol_id not in sol_pass_ut_set:
                    sol_pass_ut_set[sol_id] = set()
                key = f"{task_id}-{sol_id}-{ut_id}"
                if key in task_sol_ut_results and task_sol_ut_results[key] == 'pass':
                    sol_pass_ut_set[sol_id].add(ut_id)

        sol_pass_ut_set = sorted(sol_pass_ut_set.items(), key=lambda item: len(item[1]), reverse=True)
        top_pass_ut_num = len(sol_pass_ut_set[0][1])
        top_sol_list = [v for v in sol_pass_ut_set if len(v[1]) == top_pass_ut_num]

        select_sol_ids = []
        max_consistency = 0
        for v1 in top_sol_list:
            consistency = sum(1 for v2 in top_sol_list if v1[1] == v2[1])
            if consistency > max_consistency:
                select_sol_ids = [v1[0]]
                max_consistency = consistency
            elif consistency == max_consistency:
                select_sol_ids.append(v1[0])

        num = sum(1 for v in select_sol_ids if task_sol_results.get(f'{task_id}-{v}') == 'pass')
        accuracy += num / len(select_sol_ids)

    return round(accuracy / len(dataset), 4)

def score(execution_result_path, label):
    dataset = load_jsonl('/home/ubuntu/workspace/coderm_reproduce_code/data/benchmark/input_livecodebench_sol.jsonl')
    sol_anno = load_jsonl('/home/ubuntu/workspace/coderm_reproduce_code/data/result/livecodebench/sol_llama3-8b_100_anno.jsonl')

    task_sol_results = {}
    for data in sol_anno:
        for sol_id in range(len(data['solutions'])):
            task_sol_results[f"{data['task_id']}-{sol_id}"] = data['solutions'][sol_id]['result']

    task_sol_ut_results = {}
    ut_result = load_jsonl(execution_result_path)
    for data in tqdm(ut_result, desc=f"Loading {label}"):
        task_sol_ut_results[f"{data['task_id']}-{data['sol_id']}-{data['ut_id']}"] = data['result']

    acc = calc_best_of_n(dataset, 100, 100, task_sol_results, task_sol_ut_results)
    print(f"{label}: pass@1 = {acc} ({acc*100:.2f}%)")
    return acc

if __name__ == '__main__':
    results = {}
    import os
    paths = {
        "Baseline (CodeRM-8B)": "/home/ubuntu/workspace/coderm_reproduce_code/output/2026-03-13_05-57-11/livecodebench/execution/100_sol_100_ut_result.jsonl",
        "Exp4 (full, LR=5e-6)": "/home/ubuntu/workspace/coderm_reproduce_code/output/full-training-exp4-livecodebench-ut100-fixed/2026-03-28_18-23-34/livecodebench/execution/100_sol_100_ut_result.jsonl",
        "Exp8-step25": "/home/ubuntu/workspace/coderm_reproduce_code/output/exp8-step25-lcb/2026-03-30_17-14-43/livecodebench/execution/100_sol_100_ut_result.jsonl",
        "Exp8-step50": "/home/ubuntu/workspace/coderm_reproduce_code/output/exp8-step50-lcb/2026-03-30_17-16-13/livecodebench/execution/100_sol_100_ut_result.jsonl",
    }

    for label, path in paths.items():
        if os.path.exists(path):
            results[label] = score(path, label)
        else:
            print(f"{label}: file not found")

    print("\n=== Summary ===")
    for label, acc in results.items():
        print(f"  {label}: {acc*100:.2f}%")
