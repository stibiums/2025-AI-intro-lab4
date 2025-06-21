import optuna
import subprocess
import re
import os
import shutil

# --- 从 autograder.py 导入 q3 函数 ---
try:
    from autograder import q3 as get_autograder_q3_score_func
    print("Successfully imported q3 from autograder.")
except ImportError:
    print("ERROR: Could not import q3 from autograder.py. Make sure autograder.py is in the same directory and has a q3 function.")
    exit(1)
except Exception as e:
    print(f"ERROR: An unexpected error occurred during import from autograder.py: {e}")
    exit(1)
# --- /从 autograder.py 导入 q3 函数 ---


# --- 配置 ---
ANSWER_PLANNING_PY = "answerPlanning.py"
# 围绕上一轮最佳参数进行精细调整，并固定MAX_ITER
PARAM_RANGES = {
    "STEP_DISTANCE": {"type": "float", "low": 0.8, "high": 1.2},
    "GOAL_BIAS": {"type": "float", "low": 0.1, "high": 0.2},
}
N_TRIALS = 300
# --- /配置 ---

original_answer_planning_content = None
backup_file_path = ANSWER_PLANNING_PY + ".bak"
IS_GIT_REPO = False

def check_git_repo():
    global IS_GIT_REPO
    IS_GIT_REPO = os.path.isdir(".git")
    if not IS_GIT_REPO:
        print(f"警告: 当前目录不是一个Git仓库。调参结果将不会被自动提交到Git。")
    else:
        print("检测到Git仓库，将尝试为每次试验提交结果。")

def backup_original_file():
    global original_answer_planning_content
    try:
        if os.path.exists(backup_file_path):
            print(f"警告: 备份文件 {backup_file_path} 已存在。")
            if not original_answer_planning_content:
                 with open(backup_file_path, 'r', encoding='utf-8') as f:
                    original_answer_planning_content = f.read()
        
        with open(ANSWER_PLANNING_PY, 'r', encoding='utf-8') as f:
            current_content = f.read()
            if original_answer_planning_content is None:
                original_answer_planning_content = current_content
        
        with open(backup_file_path, 'w', encoding='utf-8') as f:
            f.write(original_answer_planning_content)
        print(f"脚本启动时的 {ANSWER_PLANNING_PY} 状态已备份到 {backup_file_path}")

    except FileNotFoundError:
        print(f"错误: {ANSWER_PLANNING_PY} 未找到。无法开始调参。")
        exit(1)
    except Exception as e:
        print(f"备份文件时出错: {e}")
        exit(1)

def restore_original_file():
    if original_answer_planning_content:
        try:
            with open(ANSWER_PLANNING_PY, 'w', encoding='utf-8') as f:
                f.write(original_answer_planning_content)
            print(f"{ANSWER_PLANNING_PY} 已恢复到脚本运行前的状态。")
        except Exception as e:
            print(f"恢复 {ANSWER_PLANNING_PY} 时出错: {e}")
            print(f"请从 {backup_file_path} 或 Git历史 手动恢复。")
    else:
        print(f"没有原始文件内容可恢复。请从 {backup_file_path} 或 Git历史 手动恢复。")

def modify_hyperparameters_in_file(params):
    try:
        with open(ANSWER_PLANNING_PY, 'r', encoding='utf-8') as f:
            content = f.readlines()

        new_content = []
        for line in content:
            # 只修改需要调整的参数
            if re.match(r"^\s*STEP_DISTANCE\s*=\s*[\d\.]+", line):
                new_content.append(f"STEP_DISTANCE = {params['STEP_DISTANCE']:.4f}         # RRT 每次扩展的步长 (auto-tuned)\n")
            elif re.match(r"^\s*GOAL_BIAS\s*=\s*[\d\.]+", line):
                new_content.append(f"GOAL_BIAS = {params['GOAL_BIAS']:.4f}             # RRT 采样时朝向目标的概率 (auto-tuned)\n")
            else:
                new_content.append(line)

        with open(ANSWER_PLANNING_PY, 'w', encoding='utf-8') as f:
            f.writelines(new_content)
    except Exception as e:
        print(f"修改超参数时出错: {e}")
        raise

def run_autograder_and_get_q3_score():
    try:
        print("  Calling autograder.q3() to get score...")
        score = get_autograder_q3_score_func()
        print(f"  Score returned by autograder.q3(): {score}")
        return float(score)
    except Exception as e:
        print(f"  An unexpected error occurred while running autograder.q3(): {e}")
        return 0.0

def git_commit_trial(trial_number, params, score):
    if not IS_GIT_REPO:
        return
    
    param_str_parts = []
    for k, v in params.items():
        if isinstance(v, float):
            param_str_parts.append(f"{k}={v:.3f}")
        else:
            param_str_parts.append(f"{k}={v}")
    short_param_str = ",".join(param_str_parts)

    commit_message = f"Optuna Planning Trial {trial_number}: Score={score:.2f}, Params=({short_param_str})"
    
    try:
        status_process = subprocess.run(['git', 'status', '--porcelain', ANSWER_PLANNING_PY], capture_output=True, text=True, check=True)
        if not status_process.stdout.strip():
            print(f"  Git: {ANSWER_PLANNING_PY} 没有改动，跳过提交。")
            return

        subprocess.run(['git', 'add', ANSWER_PLANNING_PY], check=True, capture_output=True, text=True)
        subprocess.run(['git', 'commit', '-m', commit_message], check=True, capture_output=True, text=True)
        print(f"  已将试验 {trial_number} 的结果提交到Git。")
    except Exception as e:
        print(f"  Git操作失败 (trial {trial_number}): {e}")


def objective(trial):
    params = {}
    for p_name, p_config in PARAM_RANGES.items():
        if p_config["type"] == "float":
            params[p_name] = trial.suggest_float(p_name, p_config["low"], p_config["high"], log=p_config.get("log", False))
        elif p_config["type"] == "int":
            params[p_name] = trial.suggest_int(p_name, p_config["low"], p_config["high"])
    
    print(f"\nTrial {trial.number}: Testing params {params}")
    
    score = 0.0
    try:
        modify_hyperparameters_in_file(params)
        score = run_autograder_and_get_q3_score()
        if score > 0.0: 
            git_commit_trial(trial.number, params, score)
        else:
            print(f"  试验 {trial.number} 得分为0或解析失败，不进行Git提交。")

    except Exception as e:
        print(f"  Trial {trial.number} failed due to error during setup or execution: {e}")
        return 0.0 

    return score

if __name__ == "__main__":
    check_git_repo()
    backup_original_file()

    study = optuna.create_study(direction="maximize")
    try:
        study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)
    except KeyboardInterrupt:
        print("\n调参被用户中断。")
    except Exception as e:
        print(f"\n调参过程中发生意外错误: {e}")
    finally:
        print("\n正在恢复原始文件...")
        restore_original_file()

    print("\n调参完成。")
    try:
        if study.trials: 
            print("最佳试验:")
            best_trial = study.best_trial
            print(f"  分数 (Q3 Score): {best_trial.value}")
            print("  参数: ")
            for key, value in best_trial.params.items():
                print(f"    {key}: {value}")
        else:
            print("没有试验成功完成。")
    except Exception as e:
        print(f"打印最佳试验时出错: {e}")