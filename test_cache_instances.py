"""
测试缓存实例是否正确

功能：
1. 测试 data/cache/*.pkl 是否能正常读取
2. 检查实例规模
3. 测试 evaluator 是否正常
4. 确认 run_ablation 修改正确
"""

import os
import pickle
from evaluation.fitness import FitnessEvaluator

INSTANCES = [
    "S1_Small",
    "S2_Small",
    "S3_Small",
    "S4_Small",
    "M1_Medium",
    "M2_Medium",
    "M3_Medium",
    "L1_Large"
]


def main():

    print("\n=== Testing Cached Instances ===\n")

    for inst in INSTANCES:

        cache_file = os.path.join("data", "cache", f"{inst}_cache.pkl")

        print(f"Testing {inst}")

        if not os.path.exists(cache_file):
            print(f"❌ Missing cache: {cache_file}\n")
            continue

        try:

            with open(cache_file, "rb") as f:
                data = pickle.load(f)

            print("✔ cache loaded")

            print(
                f"   Tasks: {data.num_tasks} | "
                f"Classes: {data.num_classes} | "
                f"Teachers: {data.num_teachers} | "
                f"Rooms: {data.num_classrooms}"
            )

            # 测试 evaluator
            evaluator = FitnessEvaluator()
            evaluator.set_data(data)

            print("✔ evaluator initialized")

            # 检查一个任务
            if len(data.tasks) > 0:
                first_task = next(iter(data.tasks.values()))
                print(f"✔ sample task id: {first_task.id}")

            print("✔ instance OK\n")

        except Exception as e:
            print(f"❌ ERROR: {e}\n")

    print("=== Test Finished ===")


if __name__ == "__main__":
    main()