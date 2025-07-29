import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import logging
import numpy as np
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from fcsparser import parse as parse_fcs
    FCS_PARSER_AVAILABLE = True
except ImportError:
    FCS_PARSER_AVAILABLE = False

try:
    from flowio import FlowData
    FLOWIO_AVAILABLE = True
except ImportError:
    FLOWIO_AVAILABLE = False


def read_fcs_file_for_compare(filepath):
    if FCS_PARSER_AVAILABLE:
        try:
            _, data = parse_fcs(filepath)
            if isinstance(data, pd.DataFrame):
                return data
            else:
                return pd.DataFrame(data)
        except Exception as e:
            logger.warning(f"fcsparser failed for {filepath} during comparison load: {e}")

    if FLOWIO_AVAILABLE:
        try:
            flow_data = FlowData(filepath, ignore_offset_error=True)
            data = np.reshape(flow_data.events, (-1, flow_data.channel_count))
            
            channels = []
            for i in range(flow_data.channel_count):
                channel_name = flow_data.text.get(f'$P{i+1}N') or flow_data.text.get(f'$P{i+1}S') or f'Channel_{i+1}'
                channels.append(channel_name)
            return pd.DataFrame(data, columns=channels)
        except Exception as e:
            logger.warning(f"flowio failed for {filepath} during comparison load: {e}")
    
    raise Exception(f"Could not read FCS file {filepath} - no working FCS parsers available for comparison.")


def load_processed_data(results_dir="data/processed"):
    # fcs_files = glob.glob(f"{results_dir}/*_processed.fcs")
    csv_files = glob.glob(f"{results_dir}/*_processed.csv")
    
    all_datasets_loaded = {}
    
    """for fcs_file in fcs_files:
        sample_name = os.path.basename(fcs_file).replace('_processed.fcs', '')
        logger.info(f"Attempting to load FCS: {fcs_file} for {sample_name}...")
        try:
            df = read_fcs_file_for_compare(fcs_file)
            all_datasets_loaded[sample_name] = df
            logger.info(f"  - Loaded FCS: {len(df)} cells, {len(df.columns)} features")
        except Exception as e:
            logger.error(f"  - Failed to load FCS {fcs_file}. Error: {e}")"""

    for csv_file in csv_files:
        sample_name = os.path.basename(csv_file).replace('_processed.csv', '')
        if sample_name not in all_datasets_loaded:
            logger.info(f"Loading CSV: {csv_file} for {sample_name}...")
            try:
                df = pd.read_csv(csv_file)
                all_datasets_loaded[sample_name] = df
                logger.info(f"  - Loaded CSV: {len(df)} cells, {len(df.columns)} features")
            except Exception as e:
                logger.error(f"  - Error loading CSV {csv_file}: {e}")
        else:
            logger.info(f"CSV {csv_file} skipped for {sample_name} as FCS already loaded.")


    datasets = {}
    sorted_sample_names = sorted(all_datasets_loaded.keys())
    for name in sorted_sample_names:
        datasets[name] = all_datasets_loaded[name]
    
    return datasets

def compare_cluster_distributions(datasets):
    fig, axes = plt.subplots(2, 2, figsize=(15, 14))
    fig.suptitle('Cluster Analysis Comparison', fontsize=16)
    
    cluster_counts = {}
    for name, df in datasets.items():
        if 'Cluster_ID' in df.columns:
            cluster_counts[name] = df['Cluster_ID'].value_counts().sort_index()
    
    if cluster_counts:
        cluster_df = pd.DataFrame(cluster_counts).fillna(0)
        ax00_plot = cluster_df.plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Cell Count per Cluster')
        axes[0,0].set_xlabel('Cluster ID')
        axes[0,0].set_ylabel('Number of Cells')
        axes[0,0].legend(title='Sample', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(ax00_plot.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor") 
    else:
        axes[0,0].set_title('Cell Count per Cluster (No Cluster_ID found)')
        axes[0,0].text(0.5, 0.5, 'No cluster data available', horizontalalignment='center', verticalalignment='center', transform=axes[0,0].transAxes)

    # Plot 2: Cluster proportions
    if cluster_counts:
        cluster_props = cluster_df.div(cluster_df.sum(axis=0), axis=1)
        ax01_plot = cluster_props.plot(kind='bar', stacked=True, ax=axes[0,1])
        axes[0,1].set_title('Cluster Proportions')
        axes[0,1].set_xlabel('Cluster ID')
        axes[0,1].set_ylabel('Proportion')
        axes[0,1].legend(title='Sample', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(ax01_plot.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")
    else:
        axes[0,1].set_title('Cluster Proportions (No Cluster_ID found)')
        axes[0,1].text(0.5, 0.5, 'No cluster data available', horizontalalignment='center', verticalalignment='center', transform=axes[0,1].transAxes)
    
    # Plot 3: Total cell counts
    total_cells = {name: len(df) for name, df in datasets.items()}
    if total_cells:
        axes[1,0].bar(total_cells.keys(), total_cells.values())
        axes[1,0].set_title('Total Cells per Sample')
        axes[1,0].set_ylabel('Number of Cells')
        plt.setp(axes[1,0].get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor") 
    else:
        axes[1,0].set_title('Total Cells per Sample (No Data)')
        axes[1,0].text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center', transform=axes[1,0].transAxes)

    # Plot 4: UMAP comparison
    sample_names = list(datasets.keys())  
    
    if len(sample_names) > 0:
        if len(sample_names) > 1:
            umap_fig, umap_axes = plt.subplots(1, min(len(sample_names), 4), figsize=(5 * min(len(sample_names), 4), 6)) # Max 4 UMAP plots per row
            if isinstance(umap_axes, plt.Axes):
                 umap_axes = [umap_axes]
            else:
                 umap_axes = umap_axes.flatten() 
            
            for i, name in enumerate(sample_names[:min(len(sample_names), 4)]):
                df = datasets[name]
                ax = umap_axes[i]
                if 'UMAP1' in df.columns and 'UMAP2' in df.columns and 'Cluster_ID' in df.columns:
                    sns.scatterplot(x='UMAP1', y='UMAP2', hue='Cluster_ID', data=df,
                                    s=5, alpha=0.7, palette='tab10', ax=ax, legend='full')
                    ax.set_title(f'UMAP: {name}')
                    ax.set_xlabel('UMAP1')
                    ax.set_ylabel('UMAP2')
                    ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor") 
                    plt.setp(ax.get_yticklabels(), rotation=0)
                else:
                    ax.set_title(f'UMAP: {name} (No Data)')
                    ax.text(0.5, 0.5, 'No UMAP/Cluster data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            
            plt.tight_layout()
            umap_fig_path = 'plots/umap_comparison_individual.png'
            ensure_output_dir(umap_fig_path)
            plt.savefig(umap_fig_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved individual UMAP comparison plot: {umap_fig_path}")
            plt.close(umap_fig)

            axes[1,1].set_title('UMAP Comparison (See Individual Plots)')
            axes[1,1].text(0.5, 0.5, f'Multiple UMAPs saved to {os.path.basename(umap_fig_path)}', horizontalalignment='center', verticalalignment='center', transform=axes[1,1].transAxes)
            axes[1,1].set_xticks([])
            axes[1,1].set_yticks([])

        else:
            name = sample_names[0]
            df = datasets[name]
            if 'UMAP1' in df.columns and 'UMAP2' in df.columns and 'Cluster_ID' in df.columns:
                sns.scatterplot(x='UMAP1', y='UMAP2', hue='Cluster_ID', data=df,
                                s=5, alpha=0.7, palette='tab10', ax=axes[1,1])
                axes[1,1].set_title(f'UMAP Embedding: {name}')
                axes[1,1].set_xlabel('UMAP1')
                axes[1,1].set_ylabel('UMAP2')
                axes[1,1].legend(title='Cluster')
                plt.setp(axes[1,1].get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")
                plt.setp(axes[1,1].get_yticklabels(), rotation=0)
            else:
                axes[1,1].set_title(f'UMAP: {name} (No UMAP/Cluster Data)')
                axes[1,1].text(0.5, 0.5, 'No UMAP/Cluster data available', horizontalalignment='center', verticalalignment='center', transform=axes[1,1].transAxes)
    else:
        axes[1,1].set_title('UMAP Comparison (No Samples)')
        axes[1,1].text(0.5, 0.5, 'No UMAP data available', horizontalalignment='center', verticalalignment='center', transform=axes[1,1].transAxes)
    
    plt.tight_layout()
    comparison_fig_path = 'plots/results_comparison.png'
    ensure_output_dir(comparison_fig_path)
    plt.savefig(comparison_fig_path, dpi=300, bbox_inches='tight')
    logger.info("Saved comparison plot: plots/results_comparison.png")
    plt.close(fig)

def analyze_marker_expression(datasets, markers=None):
    if not datasets:
        logger.warning("No datasets provided for marker expression analysis.")
        return

    all_numeric_cols = set()
    for df in datasets.values():
        all_numeric_cols.update(df.select_dtypes(include=np.number).columns)
    
    exclusion_patterns = ['umap', 'time', 'event', 'cluster', 'id', 'bkgd', 'dna', 'ir_']
    potential_markers = [col for col in all_numeric_cols
                         if not any(x in col.lower() for x in exclusion_patterns)]
    
    if not potential_markers:
        logger.warning("No potential marker channels found across datasets for expression analysis.")
        return

    if markers:
        final_markers = [m for m in markers if m in potential_markers]
    else:
        final_markers = sorted(potential_markers)[:10]
    
    if not final_markers:
        logger.warning("No valid markers to analyze after filtering.")
        return

    logger.info(f"Analyzing expression for markers: {final_markers[:5]}{'...' if len(final_markers) > 5 else ''}")
    
    num_markers_to_plot = min(len(final_markers), 6)
    if num_markers_to_plot == 0:
        logger.info("No markers to plot for expression analysis.")
        return

    n_rows = (num_markers_to_plot + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows * 6))
    axes = axes.flatten()
    
    for i, marker in enumerate(final_markers[:num_markers_to_plot]):
        marker_data = []
        labels_for_plot = []
        has_data = False
        for name, df in datasets.items():
            if marker in df.columns:
                marker_data.append(df[marker])
                labels_for_plot.append(name)
                has_data = True
        
        if has_data:
            axes[i].boxplot(marker_data, labels=labels_for_plot)
            axes[i].set_title(f'{marker} Expression')
            plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor") 
            axes[i].set_ylabel('Expression Level')
        else:
            axes[i].set_title(f'{marker} (No Data)')
            axes[i].text(0.5, 0.5, 'No data for this marker', horizontalalignment='center', verticalalignment='center', transform=axes[i].transAxes)

    for j in range(num_markers_to_plot, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    marker_expression_path = 'plots/marker_expression_comparison.png'
    ensure_output_dir(marker_expression_path)
    plt.savefig(marker_expression_path, dpi=300, bbox_inches='tight')
    logger.info("Saved marker expression plot: plots/marker_expression_comparison.png")
    plt.close(fig)


def create_summary_table(datasets):
    if not datasets:
        logger.warning("No datasets provided for summary table creation.")
        return None

    summary_data = []
    
    for name, df in datasets.items():
        summary = {
            'Sample': name,
            'Total_Cells': len(df),
            'Num_Clusters': df['Cluster_ID'].nunique() if 'Cluster_ID' in df.columns else 0,
        }
        
        if 'UMAP1' in df.columns and 'UMAP2' in df.columns:
            summary['UMAP_Range_X'] = f"{df['UMAP1'].min():.2f} to {df['UMAP1'].max():.2f}"
            summary['UMAP_Range_Y'] = f"{df['UMAP2'].min():.2f} to {df['UMAP2'].max():.2f}"
        else:
            summary['UMAP_Range_X'] = 'N/A'
            summary['UMAP_Range_Y'] = 'N/A'
        
        if 'Cluster_ID' in df.columns:
            cluster_dist = df['Cluster_ID'].value_counts().sort_index()
            for cluster_id in cluster_dist.index:
                summary[f'Cluster_{cluster_id}_Count'] = cluster_dist[cluster_id]
        
        summary_data.append(summary)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df_path = 'results/analysis_summary.csv'
    ensure_output_dir(summary_df_path)
    summary_df.to_csv(summary_df_path, index=False)
    logger.info(f"Saved summary table: {summary_df_path}")
    logger.info("\nSummary:")
    logger.info(summary_df.to_string(index=False))
    
    return summary_df

def ensure_output_dir(filepath):
    if filepath:
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created dir: {output_dir}")


def main():
    logger.info("Loading processed results...")
    datasets = load_processed_data()
    
    if not datasets:
        logger.error("No processed data found in data/processed/ dir.")
        return
    
    logger.info(f"\nFound {len(datasets)} datasets:")
    for name in datasets.keys():
        logger.info(f"  - {name}")
    
    logger.info("\nCreating comparison plots...")
    compare_cluster_distributions(datasets)
    
    logger.info("\nAnalyzing marker expression...")
    analyze_marker_expression(datasets)
    
    logger.info("\nCreating summary table...")
    create_summary_table(datasets)
    
    logger.info("\nAnalysis complete! Check plots/ and results/ dirs for outputs.")

if __name__ == "__main__":
    main()