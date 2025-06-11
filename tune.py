import optuna
import subprocess
import re
import os
import shutil
import time # 用于可能的重试或时间戳

# --- 从 autograder.py 导入 q1 函数 ---
try:
    from autograder import q1 as get_autograder_q1_score_func # 重命名以避免潜在冲突
    print("Successfully imported q1 from autograder.")
except ImportError:
    print("ERROR: Could not import q1 from autograder.py. Make sure autograder.py is in the same directory and has a q1 function.")
    exit(1)
except Exception as e:
    print(f"ERROR: An unexpected error occurred during import from autograder.py: {e}")
    exit(1)
# --- /从 autograder.py 导入 q1 函数 ---


# --- 配置 ---
ANSWER_LOCALIZATION_PY = "answerLocalization.py"
# AUTOGRADER_PY = "autograder.py" # 不再需要通过subprocess调用整个autograder
PARAM_RANGES = {
    "W": {"type": "float", "low": 0.01, "high": 0.5, "log": True},
    "ALPHA": {"type": "float", "low": 1.0, "high": 1.5},
    "sigma_P": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
    "sigma_T": {"type": "float", "low": 0.001, "high": 0.05, "log": True},
    "K": {"type": "int", "low": 1, "high": 10}
}
N_TRIALS = 500
# AUTOGRADER_TIMEOUT = 120 # 不再直接控制autograder的超时，由autograder内部的programcall控制
# --- /配置 ---

original_answer_localization_content = None
backup_file_path = ANSWER_LOCALIZATION_PY + ".bak"
IS_GIT_REPO = False

def check_git_repo():
    global IS_GIT_REPO
    IS_GIT_REPO = os.path.isdir(".git")
    if not IS_GIT_REPO:
        print("警告: 当前目录不是一个Git仓库。调参结果将不会被自动提交到Git。")
        print("建议在该目录下初始化Git仓库 (git init) 并将 answerLocalization.py 添加到跟踪中，以保存调参历史。")
    else:
        print("检测到Git仓库，将尝试为每次试验提交结果。")

def backup_original_file():
    global original_answer_localization_content
    try:
        if os.path.exists(backup_file_path):
            print(f"警告: 备份文件 {backup_file_path} 已存在。如果脚本中断，请检查此文件。")
            if not original_answer_localization_content:
                 with open(backup_file_path, 'r', encoding='utf-8') as f:
                    original_answer_localization_content = f.read()
        
        with open(ANSWER_LOCALIZATION_PY, 'r', encoding='utf-8') as f:
            current_content = f.read()
            if original_answer_localization_content is None:
                original_answer_localization_content = current_content
        
        with open(backup_file_path, 'w', encoding='utf-8') as f:
            f.write(original_answer_localization_content)
        print(f"脚本启动时的 {ANSWER_LOCALIZATION_PY} 状态已备份到 {backup_file_path}")

    except FileNotFoundError:
        print(f"错误: {ANSWER_LOCALIZATION_PY} 未找到。无法开始调参。")
        exit(1)
    except Exception as e:
        print(f"备份文件时出错: {e}")
        exit(1)

def restore_original_file():
    if original_answer_localization_content:
        try:
            with open(ANSWER_LOCALIZATION_PY, 'w', encoding='utf-8') as f:
                f.write(original_answer_localization_content)
            print(f"{ANSWER_LOCALIZATION_PY} 已恢复到脚本运行前的状态。")
        except Exception as e:
            print(f"恢复 {ANSWER_LOCALIZATION_PY} 时出错: {e}")
            print(f"请从 {backup_file_path} 或 Git历史 手动恢复。")
    else:
        print(f"没有原始文件内容可恢复。请从 {backup_file_path} 或 Git历史 手动恢复。")

def modify_hyperparameters_in_file(params):
    try:
        with open(ANSWER_LOCALIZATION_PY, 'r', encoding='utf-8') as f:
            content = f.readlines()

        new_content = []
        for line in content:
            stripped_line = line.strip()
            if re.match(r"^\s*W\s*=\s*[\d\.]+", line):
                new_content.append(f"W = {params['W']:.4f}   # 权重常数 (auto-tuned)\n")
            elif re.match(r"^\s*ALPHA\s*=\s*[\d\.]+", line):
                new_content.append(f"ALPHA = {params['ALPHA']:.4f}  # 每个点重采样比占比例多的倍数 (auto-tuned)\n")
            elif re.match(r"^\s*sigma_P\s*=\s*[\d\.]+", line):
                new_content.append(f"sigma_P = {params['sigma_P']:.4f}  # 位置方差 (auto-tuned)\n")
            elif re.match(r"^\s*sigma_T\s*=\s*[\d\.]+", line):
                new_content.append(f"sigma_T = {params['sigma_T']:.4f}  # 角度方差 (auto-tuned)\n")
            elif re.match(r"^\s*K\s*=\s*[\d]+", line):
                new_content.append(f"K = {params['K']}  # 采样点数量 (auto-tuned)\n")
            else:
                new_content.append(line)

        with open(ANSWER_LOCALIZATION_PY, 'w', encoding='utf-8') as f:
            f.writelines(new_content)
    except Exception as e:
        print(f"修改超参数时出错: {e}")
        raise

def run_autograder_and_get_q1_score():
    try:
        # 直接调用从 autograder.py 导入的 q1 函数
        print("  Calling autograder.q1() to get score...")
        # autograder.py 中的 q1 函数应该返回一个数值（分数）
        # 它内部会调用 taskLocalization.py 并处理输出
        score = get_autograder_q1_score_func()
        print(f"  Score returned by autograder.q1(): {score}")
        return float(score)  # 确保返回的是浮点数
    except FileNotFoundError as e:
        # 这种情况可能发生在 autograder.py 内部尝试运行 taskLocalization.py 时
        # 或者 PYTHON_PATH 配置不正确
        print(f"  Error during autograder.q1() execution (FileNotFound): {e}")
        print("  请检查 autograder.py 中的 PYTHON_PATH 以及 taskLocalization.py 是否存在。")
        return 0.0
    except IndexError as e:
        # 这种情况可能发生在 autograder.py 的 filterq1 函数中，如果输出格式不符合预期
        print(f"  Error during autograder.q1() execution (IndexError, likely parsing taskLocalization output): {e}")
        print("  请检查 autograder.py 中的 filterq1 函数以及 taskLocalization.py 的输出格式。")
        return 0.0
    except subprocess.CalledProcessError as e:
        # 如果 taskLocalization.py 返回非零退出码
        print(f"  Error: A call to taskLocalization.py within autograder.q1() failed with CalledProcessError: {e}")
        if e.output:
            print(f"  Output from failed process: {e.output}")
        if e.stderr:
            print(f"  Stderr from failed process: {e.stderr}")
        return 0.0
    except subprocess.TimeoutExpired as e:
        # 超时由 autograder.py 内部的 programcall 控制
        print(f"  Error: A call to taskLocalization.py within autograder.q1() timed out: {e}")
        return 0.0
    except Exception as e:
        print(f"  An unexpected error occurred while running autograder.q1(): {e}")
        # import traceback # 取消注释以获取更详细的错误信息
        # traceback.print_exc()
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

    commit_message = f"Optuna Trial {trial_number}: Score={score:.2f}, Params=({short_param_str})"
    
    try:
        status_process = subprocess.run(['git', 'status', '--porcelain', ANSWER_LOCALIZATION_PY], capture_output=True, text=True, check=True)
        if not status_process.stdout.strip():
            print(f"  Git: {ANSWER_LOCALIZATION_PY} 没有改动，跳过提交。")
            return

        subprocess.run(['git', 'add', ANSWER_LOCALIZATION_PY], check=True, capture_output=True, text=True)
        subprocess.run(['git', 'commit', '-m', commit_message], check=True, capture_output=True, text=True)
        print(f"  已将试验 {trial_number} 的结果提交到Git。")
    except subprocess.CalledProcessError as e:
        print(f"  Git操作失败 (trial {trial_number}): {e.stderr or e.stdout or e}")
        print(f"  尝试提交的信息: {commit_message}")
    except FileNotFoundError:
        print("  Git命令未找到。跳过Git提交。")
    except Exception as e:
        print(f"  Git提交过程中发生未知错误: {e}")


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
        score = run_autograder_and_get_q1_score()
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
            print(f"  分数 (Q1 Score): {best_trial.value}")
            print("  参数: ")
            for key, value in best_trial.params.items():
                print(f"    {key}: {value}")
        else:
            print("没有试验成功完成。")
    except optuna.exceptions.OptunaError: 
        print("未能找到最佳试验（可能所有试验都失败了或没有试验完成）。")
    except Exception as e:
        print(f"打印最佳试验时出错: {e}")