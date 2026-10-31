import os
import json
import csv
import numpy as np
from collections import OrderedDict, defaultdict


FINAL_SUMMARY_METRICS = ('mean_dsc',)


def calculate_directory_stats(dsc_scores):
    if not dsc_scores:
        return {
            'count': 0,
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0
        }

    scores = np.array(dsc_scores)
    return {
        'count': len(scores),
        'mean': np.mean(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores)
    }


def calculate_hierarchical_directory_stats(file_results, data_dir):
    root = os.path.abspath(data_dir)
    groups = defaultdict(list)
    for result in file_results:
        directory = os.path.abspath(os.path.join(root, result['directory']))
        if os.path.commonpath([root, directory]) != root:
            raise ValueError(f"Result directory is outside data_dir: {directory}")
        while True:
            groups[directory].append(result['mean_dsc'])
            if directory == root:
                break
            directory = os.path.dirname(directory)
    return {directory: calculate_directory_stats(scores) for directory, scores in groups.items()}


def sort_directories_postorder(hierarchical_stats, data_dir):
    data_dir = os.path.abspath(data_dir)

    depth_groups = defaultdict(list)
    for dir_path in hierarchical_stats.keys():
        if dir_path == data_dir:
            depth = 0
        else:
            rel_path = os.path.relpath(dir_path, data_dir)
            depth = len(rel_path.split(os.sep))
        depth_groups[depth].append(dir_path)

    sorted_dirs = []
    for depth in sorted(depth_groups.keys(), reverse=True):
        sorted_dirs.extend(sorted(depth_groups[depth]))

    return sorted_dirs


def sort_directories_preorder(hierarchical_stats, data_dir):
    data_dir = os.path.abspath(data_dir)

    depth_groups = defaultdict(list)
    for dir_path in hierarchical_stats.keys():
        if dir_path == data_dir:
            depth = 0
        else:
            rel_path = os.path.relpath(dir_path, data_dir)
            depth = len(rel_path.split(os.sep))
        depth_groups[depth].append(dir_path)

    sorted_dirs = []
    for depth in sorted(depth_groups.keys()):
        sorted_dirs.extend(sorted(depth_groups[depth]))

    return sorted_dirs


def get_directory_tree_structure(data_dir):
    tree_structure = defaultdict(list)

    for root, dirs, files in os.walk(data_dir):
        rel_path = os.path.relpath(root, data_dir)
        if rel_path == ".":
            depth = 0
        else:
            depth = len(rel_path.split(os.sep))

        tree_structure[depth].append(root)

    return tree_structure


def _format_directory_stats_line(dir_path, stats, data_dir):
    data_dir = os.path.abspath(data_dir)
    rel_path = os.path.relpath(dir_path, data_dir)
    if rel_path == ".":
        rel_path = os.path.basename(data_dir)

    depth = 0 if dir_path == data_dir else len(rel_path.split(os.sep))
    prefix = "  " * depth

    lines = [
        f"{prefix}{rel_path}/",
        f"{prefix}   Total Files: {stats['count']}"
    ]

    if stats['count'] > 0:
        lines.extend([
            f"{prefix}   mean_dsc: {stats['mean']:.4f} +/- {stats['std']:.4f}",
            f"{prefix}   Range: [{stats['min']:.4f}, {stats['max']:.4f}]"
        ])

    lines.append("")
    return lines


def print_directory_tree_stats(hierarchical_stats, data_dir, logger, use_preorder=True):
    if not hierarchical_stats:
        logger.warning("No hierarchical statistics available to display")
        return

    if use_preorder:
        sorted_dirs = sort_directories_preorder(hierarchical_stats, data_dir)
        traversal_type = "Pre-order traversal (root -> leaves)"
    else:
        sorted_dirs = sort_directories_postorder(hierarchical_stats, data_dir)
        traversal_type = "Post-order traversal (leaves -> root)"

    logger.info(f"DIRECTORY STATISTICS ({traversal_type}):")
    logger.info(f"=" * 60)

    for dir_path in sorted_dirs:
        if dir_path in hierarchical_stats:
            lines = _format_directory_stats_line(dir_path, hierarchical_stats[dir_path], data_dir)
            for line in lines:
                logger.info(line)


def print_directory_tree_stats_legacy(dir_stats, data_dir, logger, indent=0):
    prefix = "  " * indent

    for dir_path in sorted(dir_stats.keys()):
        if dir_path.startswith(data_dir):
            rel_path = os.path.relpath(dir_path, data_dir)
            if rel_path == ".":
                rel_path = os.path.basename(data_dir)

            stats = dir_stats[dir_path]
            logger.info(f"{prefix}{rel_path}/")
            logger.info(f"{prefix}   Files: {stats['count']}")
            if stats['count'] > 0:
                logger.info(f"{prefix}   mean_dsc: {stats['mean']:.4f} +/- {stats['std']:.4f}")
                logger.info(f"{prefix}   Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            logger.info(f"{prefix}")


def export_tta_directory_tree_stats_json(hierarchical_stats, data_dir, output_path, use_preorder=True):
    sort_dirs = sort_directories_preorder if use_preorder else sort_directories_postorder
    directory_stats = OrderedDict()
    root = os.path.abspath(data_dir)
    for directory in sort_dirs(hierarchical_stats, root):
        relative = os.path.relpath(directory, root)
        stats = hierarchical_stats[directory]
        directory_stats[relative] = {
            'relative_path': relative,
            'file_count': int(stats['count']),
            'mean_dsc': float(stats['mean']),
            'std': float(stats['std']),
            'min': float(stats['min']),
            'max': float(stats['max']),
        }
    result = {
        'metadata': {'data_dir': data_dir, 'metric': 'mean_dsc', 'total_directories': len(directory_stats)},
        'directory_stats': directory_stats,
    }
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=2, ensure_ascii=False)


def _as_finite_float(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _mean_or_none(values):
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return None
    return float(np.mean(finite_values))


def _normalize_relative_directory(result, data_dir, path_key='file_path', directory_key='directory'):
    data_dir_abs = os.path.abspath(data_dir) if data_dir else None

    directory = result.get(directory_key)
    if directory:
        directory = str(directory)
        if data_dir_abs and os.path.isabs(directory):
            rel_dir = os.path.relpath(directory, data_dir_abs)
        else:
            rel_dir = directory
    else:
        file_path = str(result.get(path_key, ''))
        if data_dir_abs and os.path.isabs(file_path):
            rel_path = os.path.relpath(os.path.abspath(file_path), data_dir_abs)
        else:
            rel_path = file_path
        rel_dir = os.path.dirname(rel_path)

    if rel_dir in ('', '.'):
        return '.'
    return rel_dir.replace(os.sep, '/')


def _split_task_from_subtask(subtask):
    if subtask in ('', '.'):
        return '.'
    return subtask.replace('\\', '/').split('/')[0]


def _aggregate_metric_records(records, metric_names):
    metrics = OrderedDict()
    metric_counts = OrderedDict()
    for metric in metric_names:
        values = [record['metrics'].get(metric) for record in records]
        finite_values = [value for value in values if value is not None]
        metrics[metric] = _mean_or_none(finite_values)
        metric_counts[metric] = len(finite_values)

    return OrderedDict([
        ('sample_count', int(len(records))),
        ('metric_counts', metric_counts),
        ('metrics', metrics),
    ])


def _aggregate_summary_entries(entries, metric_names):
    metrics = OrderedDict()
    metric_counts = OrderedDict()
    for metric in metric_names:
        values = [entry['metrics'].get(metric) for entry in entries]
        finite_values = [value for value in values if value is not None]
        metrics[metric] = _mean_or_none(finite_values)
        metric_counts[metric] = len(finite_values)

    return OrderedDict([
        ('sample_count', int(sum(entry.get('sample_count', 0) for entry in entries))),
        ('metric_counts', metric_counts),
        ('metrics', metrics),
    ])


def calculate_final_metric_summary(
    file_results,
    data_dir,
    metric_names=FINAL_SUMMARY_METRICS,
    path_key='file_path',
    directory_key='directory',
    metadata=None,
):
    metric_names = tuple(metric_names)
    normalized_records = []

    for result in file_results:
        subtask = _normalize_relative_directory(result, data_dir, path_key, directory_key)
        task = _split_task_from_subtask(subtask)
        metrics = OrderedDict()
        for metric in metric_names:
            metrics[metric] = _as_finite_float(result.get(metric))

        normalized_records.append(OrderedDict([
            ('task', task),
            ('subtask', subtask),
            ('metrics', metrics),
        ]))

    subtask_groups = defaultdict(list)
    for record in normalized_records:
        subtask_groups[record['subtask']].append(record)

    subtask_summaries = OrderedDict()
    for subtask in sorted(subtask_groups.keys()):
        stats = _aggregate_metric_records(subtask_groups[subtask], metric_names)
        task = _split_task_from_subtask(subtask)
        stats['task'] = task
        stats['subtask'] = subtask
        subtask_summaries[subtask] = stats

    task_groups = defaultdict(list)
    for subtask, subtask_stats in subtask_summaries.items():
        task_groups[subtask_stats['task']].append(subtask_stats)

    task_summaries = OrderedDict()
    for task in sorted(task_groups.keys()):
        stats = _aggregate_summary_entries(task_groups[task], metric_names)
        stats['task'] = task
        stats['subtask_count'] = int(len(task_groups[task]))
        stats['aggregation'] = 'unweighted_mean_of_subtask_averages'
        task_summaries[task] = stats

    if task_summaries:
        hierarchical_overall = _aggregate_summary_entries(list(task_summaries.values()), metric_names)
    else:
        hierarchical_overall = _aggregate_summary_entries([], metric_names)
    hierarchical_overall['task_count'] = int(len(task_summaries))
    hierarchical_overall['subtask_count'] = int(len(subtask_summaries))
    hierarchical_overall['aggregation'] = 'unweighted_mean_of_task_averages'

    global_average = _aggregate_metric_records(normalized_records, metric_names)
    global_average['aggregation'] = 'sample_weighted_mean'

    summary_metadata = OrderedDict([
        ('data_dir', data_dir),
        ('sample_count', int(len(normalized_records))),
        ('task_count', int(len(task_summaries))),
        ('subtask_count', int(len(subtask_summaries))),
        ('metrics', list(metric_names)),
    ])
    if metadata:
        summary_metadata.update(metadata)

    return OrderedDict([
        ('metadata', summary_metadata),
        ('global_average', global_average),
        ('hierarchical_average', OrderedDict([
            ('overall', hierarchical_overall),
            ('tasks', task_summaries),
            ('subtasks', subtask_summaries),
        ])),
    ])


def _format_final_summary_metric(metric, value):
    return '' if value is None else f"{value:.6f}"


def export_final_metric_summary(
    summary,
    output_dir,
    json_filename='final_summary_stats.json',
    csv_filename='final_summary_stats.csv',
):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, json_filename)
    csv_path = os.path.join(output_dir, csv_filename)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    metric_names = tuple(summary['metadata']['metrics'])
    header = [
        'average_type',
        'task',
        'subtask',
        'sample_count',
        'task_count',
        'subtask_count',
        *metric_names,
    ]

    def row_for(average_type, stats, task='', subtask=''):
        return [
            average_type,
            task,
            subtask,
            stats.get('sample_count', ''),
            stats.get('task_count', ''),
            stats.get('subtask_count', ''),
            *[
                _format_final_summary_metric(metric, stats.get('metrics', {}).get(metric))
                for metric in metric_names
            ],
        ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row_for('global_sample_average', summary['global_average']))
        writer.writerow(row_for('hierarchical_overall_average', summary['hierarchical_average']['overall']))

        for task, stats in summary['hierarchical_average']['tasks'].items():
            writer.writerow(row_for('task_average', stats, task=task))

        for subtask, stats in summary['hierarchical_average']['subtasks'].items():
            writer.writerow(row_for('subtask_average', stats, task=stats.get('task', ''), subtask=subtask))

    return json_path, csv_path


def _format_metric_summary_line(stats, metric_names):
    parts = []
    for metric in metric_names:
        value = stats['metrics'].get(metric)
        parts.append(f"{metric}=N/A" if value is None else f"{metric}={value:.6f}")
    return ', '.join(parts)


def log_final_metric_summary(summary, logger, title='FINAL METRIC SUMMARY'):
    metric_names = tuple(summary['metadata']['metrics'])
    global_stats = summary['global_average']
    hierarchical_overall = summary['hierarchical_average']['overall']

    logger.info(f"\n{title}:")
    logger.info("=" * 80)
    logger.info(
        f"Global sample average ({global_stats.get('sample_count', 0)} samples): "
        f"{_format_metric_summary_line(global_stats, metric_names)}"
    )
    logger.info(
        "Hierarchical overall average "
        f"({hierarchical_overall.get('task_count', 0)} tasks, "
        f"{hierarchical_overall.get('subtask_count', 0)} sub-tasks): "
        f"{_format_metric_summary_line(hierarchical_overall, metric_names)}"
    )

    logger.info("Task-level hierarchical averages:")
    for task, stats in summary['hierarchical_average']['tasks'].items():
        logger.info(
            f"  - {task} ({stats.get('subtask_count', 0)} sub-tasks, "
            f"{stats.get('sample_count', 0)} samples): "
            f"{_format_metric_summary_line(stats, metric_names)}"
        )
    logger.info("=" * 80)


def save_detailed_results(output_dir, dir_stats, file_results, hierarchical_stats=None, data_dir=None, use_preorder=True):
    if dir_stats is not None:
        dir_stats_for_json = {}
        for dir_path, stats in dir_stats.items():
            dir_stats_for_json[dir_path] = {
                'count': int(stats['count']),
                'mean_dsc': float(stats['mean']),
                'std': float(stats['std']),
                'min': float(stats['min']),
                'max': float(stats['max'])
            }

        with open(os.path.join(output_dir, 'directory_stats.json'), 'w') as f:
            json.dump(dir_stats_for_json, f, indent=2)

    if hierarchical_stats is not None:
        from collections import OrderedDict

        if data_dir:
            data_dir_abs = os.path.abspath(data_dir)
            if use_preorder:
                sorted_dirs = sort_directories_preorder(hierarchical_stats, data_dir)
            else:
                sorted_dirs = sort_directories_postorder(hierarchical_stats, data_dir)

            hierarchical_stats_for_json = OrderedDict()
            for dir_path in sorted_dirs:
                stats = hierarchical_stats[dir_path]
                rel_path = os.path.relpath(dir_path, data_dir_abs)
                if rel_path == ".":
                    rel_path = "."
                hierarchical_stats_for_json[rel_path] = {
                    'count': int(stats['count']),
                    'mean_dsc': float(stats['mean']),
                    'std': float(stats['std']),
                    'min': float(stats['min']),
                    'max': float(stats['max'])
                }
        else:
            hierarchical_stats_for_json = {}
            for dir_path, stats in hierarchical_stats.items():
                hierarchical_stats_for_json[dir_path] = {
                    'count': int(stats['count']),
                    'mean_dsc': float(stats['mean']),
                    'std': float(stats['std']),
                    'min': float(stats['min']),
                    'max': float(stats['max'])
                }

        with open(os.path.join(output_dir, 'hierarchical_directory_stats.json'), 'w') as f:
            json.dump(hierarchical_stats_for_json, f, indent=2)

    with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
        json.dump(file_results, f, indent=2)
