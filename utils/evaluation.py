import os
import numpy as np

from .statistics import (
    calculate_directory_stats, 
    print_directory_tree_stats,
    print_directory_tree_stats_legacy,
    save_detailed_results
)


def calculate_and_display_statistics(args, logger, dir_stats, file_results, all_dsc_scores, hierarchical_stats):
    total_stats = calculate_directory_stats(all_dsc_scores)

    logger.info(f'\nFINAL RESULTS SUMMARY')
    logger.info(f'=' * 60)
    logger.info(f'Total files processed: {total_stats["count"]}')

    if total_stats['count'] == 0:
        logger.warning('No files were successfully processed!')
        return

    logger.info(f'mean_dsc: {total_stats["mean"]:.4f} +/- {total_stats["std"]:.4f}')
    logger.info(f'mean_dsc range: [{total_stats["min"]:.4f}, {total_stats["max"]:.4f}]')

    logger.info(f'\nHIERARCHICAL DIRECTORY RESULTS:')
    logger.info(f'=' * 60)
    print_directory_tree_stats(hierarchical_stats, args.data_dir, logger, use_preorder=True)

    save_detailed_results(
        output_dir=args.output_dir,
        dir_stats=None,
        file_results=file_results,
        hierarchical_stats=hierarchical_stats,
        data_dir=args.data_dir,
        use_preorder=True
    )

    np.save(os.path.join(args.output_dir, 'dsc_scores.npy'), np.array(all_dsc_scores))

    saved_files = [
        (os.path.join(args.output_dir, "hierarchical_directory_stats.json"), "hierarchical statistics"),
        (os.path.join(args.output_dir, "detailed_results.json"), "file details (JSON)"),
        (os.path.join(args.output_dir, "results.csv"), "detailed results (CSV)"),
        (os.path.join(args.output_dir, "dsc_scores.npy"), "DSC scores array")
    ]

    logger.info(f'\nDetailed results saved to:')
    for file_path, description in saved_files:
        logger.info(f'  - {file_path} ({description})')
