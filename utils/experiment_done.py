import os
import time
from typing import Optional, List, Dict, Any


def check_experiment_done(output_dir: str) -> bool:
    done_file_path = os.path.join(output_dir, "experiment_done.txt")

    if os.path.exists(done_file_path):
        print(f"Experiment already completed. Done file found: {done_file_path}")
        print("Skipping execution. Remove the done file to re-run the experiment.")
        return True

    return False


def create_experiment_done_file(
    output_dir: str,
    csv_log_path: str,
    all_tta_results: Optional[List[Dict[str, Any]]] = None,
    logger=None
) -> None:
    done_file_path = os.path.join(output_dir, "experiment_done.txt")

    os.makedirs(output_dir, exist_ok=True)

    with open(done_file_path, 'w') as f:
        f.write(f"Experiment completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total files processed: {len(all_tta_results) if all_tta_results else 0}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"CSV log file: {csv_log_path}\n")

        if all_tta_results:
            mean_dsc = sum(result['mean_dsc'] for result in all_tta_results) / len(all_tta_results)
            f.write(f"mean_dsc: {mean_dsc:.6f}\n")

    print(f"Done file created: {done_file_path}")
    if logger:
        logger.info(f"Done file created: {done_file_path}")


def get_done_file_path(output_dir: str) -> str:
    return os.path.join(output_dir, "experiment_done.txt")


def remove_done_file(output_dir: str) -> bool:
    done_file_path = get_done_file_path(output_dir)

    if os.path.exists(done_file_path):
        os.remove(done_file_path)
        print(f"Done file removed: {done_file_path}")
    return True
