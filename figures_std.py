import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path


def plot_method_performance(base_result_dir=None):
    """
    Reads performance data (MCC vs. number of features) from CSV files in a 
    specified directory and generates a comparative point plot.
    
    Args:
        base_result_dir (str, optional): The base directory containing the 
                                         '{dataset}.csv' files. Defaults to a 
                                         pre-defined path if None.
    """

    datasets = ["citeseq", "meth", "lawlor", "pbmc", "tcga", "proteomics"]
    features_to_include = [1, 3, 5, 10, 15, 20]
    
    if base_result_dir:
        result_dir = base_result_dir
    else:
        
        result_dir = "/result" 

    
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)

    
    all_data = []
    
    if not Path(result_dir).is_dir():
        print(f"Error: The specified directory does not exist: {result_dir}")
        print("Please ensure the path is correct or pass it via command line.")
        return

    for dataset in datasets:
        file_path = os.path.join(result_dir, f"{dataset}.csv")
        
        if not os.path.exists(file_path):
            print(f"Warning: file {file_path} not found, skipping.")
            continue
        
        try:
            df = pd.read_csv(file_path, index_col=0)
            df['dataset'] = dataset 
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue


    if not all_data:
        print("No data was loaded. Please check the 'result_dir' path and file names.")
        return

    combined_df = pd.concat(all_data)
    
    
    filtered_df = combined_df[combined_df['nb_genes'].isin(features_to_include)]


    plt.figure(figsize=(14, 8))

    ax = sns.pointplot(
        data=filtered_df,
        x="nb_genes",
        y="mcc",
        hue="method",
        palette="colorblind",
        markers=["o", "s", "D", "v", "^", "<", ">"],
        linestyles=["-", "--", "-.", ":", "-", "--", "-."],
        errorbar="sd", 
        capsize=.1
    )

    plt.title("Average Performance Across All Datasets", fontsize=18, weight="bold")
    plt.xlabel("Number of Top Features", fontsize=16, weight="bold")
    plt.ylabel("Matthews Correlation Coefficient (MCC)", fontsize=16, weight="bold")
    
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        loc="best",
        ncol=2,
        fontsize=12,
        title="Method",
        title_fontsize=13,
        frameon=True,
        facecolor='white',
        edgecolor='black',
        framealpha=1
    )

    plt.tight_layout()
    plt.savefig("average_comparison_all_datasets.svg", dpi=300)
    print("Plot saved as 'average_comparison_all_datasets.svg'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a point plot comparing method performance (MCC) across feature counts.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "filepath", 
        type=str, 
        nargs='?', 
        default=None, 
        help=(
            "Optional: The base directory path containing the '{dataset}.csv' files.\n"
            "If not provided, the default path in the script will be used."
        )
    )
    
    args = parser.parse_args()
    
    if args.filepath:
        print(f"INFO: Using provided path for results: {args.filepath}")
        plot_method_performance(base_result_dir=args.filepath)
    else:
        print("WARNING: No path provided. Using the default path set inside the script.")
        plot_method_performance()