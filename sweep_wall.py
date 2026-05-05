"""Sweep planning evaluations across wall background conditions.

Usage:
    python sweep_wall.py --model-name wall_jepa_img80_v4_128 --n-evals 50 --gpu 2
"""

import subprocess
import json
import os
import argparse


def extract_planning_result_dir(output_string):
    for line in output_string.split('\n'):
        if "Planning result saved dir:" in line:
            return line.split("Planning result saved dir:")[-1].strip()
    return None


def parse_logs_json(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()

        lines = content.strip().split('\n')
        final_success_rate = None
        total_steps = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "final_eval/success_rate" in data:
                    final_success_rate = data["final_eval/success_rate"]
                if "step" in data:
                    total_steps = max(total_steps, data["step"])
            except json.JSONDecodeError:
                continue

        return {
            "final_success_rate": final_success_rate,
            "total_steps": total_steps,
            "file_path": file_path
        }
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def write_results_to_json(filename, results):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run wall planning experiments across background conditions"
    )
    parser.add_argument("--model-name", type=str, required=True,
                        help="Name of the trained wall model")
    parser.add_argument("--n-evals", type=int, default=50,
                        help="Number of evaluation episodes (default: 50)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device to use (default: 0)")
    parser.add_argument("--model-epoch", type=str, default="latest",
                        help="Model epoch to load (default: latest)")
    parser.add_argument("--timeout", type=int, default=6000,
                        help="Timeout per job in seconds (default: 6000)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    backgrounds = ['default', 'slight_change', 'color', 'large_color', 'large_color_gradient']

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    base_args = [
        "python", "plan.py",
        "--config-name", "plan_wall.yaml",
        f"model_name={args.model_name}",
        f"n_evals={args.n_evals}",
        f"model_epoch={args.model_epoch}",
    ]

    all_results = {}

    for background in backgrounds:
        result_logs_list = []
        output_filename = f'{args.model_name}_wall_sweep_{background}.json'
        print(f"\n{'=' * 75}")
        print(f"Background: {background}")
        print(f"Output: {output_filename}")
        print(f"{'=' * 75}\n")

        cmd = base_args + [f"+wall_env.background={background}"]
        print(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=args.timeout, env=env)
        except subprocess.TimeoutExpired:
            print(f"ERROR: Timed out after {args.timeout}s")
            result_logs = {
                'model_name': args.model_name,
                'background': background,
                'error': f'Timed out after {args.timeout}s'
            }
            result_logs_list.append(result_logs)
            write_results_to_json(output_filename, result_logs_list)
            continue

        if result.returncode != 0:
            print(f"ERROR: Failed with return code {result.returncode}")
            print(f"STDERR:\n{result.stderr[-500:] if result.stderr else 'None'}")
            result_logs = {
                'model_name': args.model_name,
                'background': background,
                'error': f'Failed with return code {result.returncode}',
                'stderr': result.stderr[:500] if result.stderr else None
            }
            result_logs_list.append(result_logs)
            write_results_to_json(output_filename, result_logs_list)
            continue

        result_dir = extract_planning_result_dir(result.stdout)
        if result_dir is None:
            print(f"ERROR: Could not extract result directory")
            result_logs = {
                'model_name': args.model_name,
                'background': background,
                'error': 'Could not extract result directory'
            }
            result_logs_list.append(result_logs)
            write_results_to_json(output_filename, result_logs_list)
            continue

        print(f"Result dir: {result_dir}")
        logs_path = os.path.join(result_dir, "logs.json")
        logs_data = parse_logs_json(logs_path) if os.path.exists(logs_path) else None

        result_logs = {'model_name': args.model_name, 'background': background}
        if logs_data:
            result_logs.update(logs_data)
        else:
            result_logs['error'] = 'Could not parse logs.json'
            result_logs['result_dir'] = result_dir

        result_logs_list.append(result_logs)
        all_results[background] = result_logs.get('final_success_rate')
        print(f"Result: {result_logs}")
        write_results_to_json(output_filename, result_logs_list)

    # Print summary
    print(f"\n{'=' * 75}")
    print(f"SWEEP SUMMARY — {args.model_name}")
    print(f"{'=' * 75}")
    for bg, sr in all_results.items():
        sr_str = f"{sr:.2%}" if sr is not None else "FAILED"
        print(f"  {bg:30s} {sr_str}")
