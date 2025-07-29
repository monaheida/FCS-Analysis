import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import fcswrite

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from fcsparser import parse as parse_fcs
    FCS_PARSER_AVAILABLE = True
    logger.info("fcsparser is available")
except ImportError as e:
    logger.error(f"fcsparser not available: {e}. FCS file reading might be limited.")
    FCS_PARSER_AVAILABLE = False

try:
    from flowio import FlowData
    FLOWIO_AVAILABLE = True
    logger.info("flowio is available")
except ImportError as e:
    logger.error(f"flowio not available: {e}. FCS file reading might be limited.")
    FLOWIO_AVAILABLE = False

try:
    import umap
    from sklearn.cluster import KMeans
    ML_LIBRARIES_AVAILABLE = True
    logger.info("ML libraries (umap, sklearn) are available")
except ImportError as e:
    logger.error(f"Required ML libraries (umap-learn, scikit-learn) not available: {e}. Analysis steps will fail.")
    ML_LIBRARIES_AVAILABLE = False


def read_fcs_file_robust(filepath):
    logger.info(f"Reading FCS file: {filepath}")

    if FCS_PARSER_AVAILABLE:
        try:
            logger.info("Trying fcsparser...")
            meta, data = parse_fcs(filepath)
            
            if isinstance(data, pd.DataFrame):
                df = data
            else:
                df = pd.DataFrame(data)
                
            logger.info(f"Successfully read FCS file with {len(df)} events and {len(df.columns)} channels using fcsparser")
            return df, meta
            
        except Exception as e:
            logger.warning(f"fcsparser failed for {filepath}: {e}")
    
    if FLOWIO_AVAILABLE:
        try:
            logger.info("Trying flowio as fallback...")
            flow_data = FlowData(filepath, ignore_offset_error=True)
            
            data = np.reshape(flow_data.events, (-1, flow_data.channel_count))
            
            channels = []
            for i in range(flow_data.channel_count):
                channel_name = None
                
                # PnN (long name), then PnS (short name), then generic
                if f'$P{i+1}N' in flow_data.text:
                    channel_name = flow_data.text[f'$P{i+1}N']
                elif f'$P{i+1}S' in flow_data.text:
                    channel_name = flow_data.text[f'$P{i+1}S']
                elif i < len(flow_data.channels):
                    channel_info = flow_data.channels[i]
                    channel_name = (channel_info.get('PnN') or 
                                  channel_info.get('PnS') or 
                                  channel_info.get('name'))
                
                if not channel_name or channel_name.strip() == '':
                    channel_name = f'Channel_{i+1}'
                    
                channels.append(channel_name)
            
            df = pd.DataFrame(data, columns=channels)
            
            logger.info(f"flowio success: {len(df)} events and {len(df.columns)} channels")
            return df, flow_data.text
            
        except Exception as e:
            logger.warning(f"flowio failed for {filepath}: {e}")
    
    raise Exception(f"Could not read FCS file {filepath} - no working FCS parsers available")


def write_fcs_file(df, output_fcs_path):
    csv_fallback_path = output_fcs_path.replace('.fcs', '.csv')

    try:
        df.to_csv(csv_fallback_path, index=False)
        logger.info(f"Saved processed data as CSV fallback: {csv_fallback_path}")
    except Exception as e:
        logger.error(f"Failed to save CSV fallback file: {csv_fallback_path}. Error: {e}")
        raise

    try:
        numeric_df = df.select_dtypes(include=np.number)
        data_matrix = numeric_df.values

        channel_names = list(numeric_df.columns)

        logger.info(f"Attempting to write FCS file to: {output_fcs_path}")
        fcswrite.write_fcs(
            filename=output_fcs_path,
            chn_names=channel_names,
            data=data_matrix
        )
        logger.info(f"FCS file written successfully (using fcswrite's basic format): {output_fcs_path}")

    except Exception as e:
        logger.error(f"Failed to write FCS file using fcswrite: {e}")
        logger.warning(f"FCS output not reliably generated. The CSV at {csv_fallback_path} contains the processed data.")
        raise


def select_channels_from_file(channels_file, fcs_data_df):
    logger.info(f"Reading channels from file: {channels_file}")
    
    try:
        channels_df = pd.read_csv(channels_file, sep='\t')
        
        logger.info(f"Channels file columns: {list(channels_df.columns)}")
        logger.info(f"Channels file shape: {channels_df.shape}")
        
        selected_channel_names = []
        if 'use' in channels_df.columns:
            if 'desc' in channels_df.columns:
                selected_channel_names = channels_df[channels_df['use'] == 1]['desc'].tolist()
                logger.info(f"Using 'desc' column for channel names, found {len(selected_channel_names)} channels marked for use")
            elif 'name' in channels_df.columns:
                selected_channel_names = channels_df[channels_df['use'] == 1]['name'].tolist()
                logger.info(f"Using 'name' column for channel names, found {len(selected_channel_names)} channels marked for use")
            else:
                logger.warning("Expected columns 'desc' or 'name' not found alongside 'use', trying to infer first column.")
                selected_channel_names = channels_df[channels_df['use'] == 1].iloc[:, 0].tolist()
        else:
            logger.warning("Column 'use' not found. Assuming last column indicates use (1) and first column is channel name.")
            selected_channel_names = channels_df[channels_df.iloc[:, -1] == 1].iloc[:, 0].tolist()
        
        available_fcs_channels = list(fcs_data_df.columns)
        final_channels = [ch for ch in selected_channel_names if ch in available_fcs_channels]
        
        logger.info(f"Selected {len(final_channels)} channels from file that match FCS data.")
        if final_channels:
            logger.info(f"Selected channels (first 5): {final_channels[:5]}{'...' if len(final_channels) > 5 else ''}")
        
        if not final_channels:
            logger.error("No matching channels found between channels.txt and FCS file!")
            logger.error(f"Channels requested from file (first 5): {selected_channel_names[:5]}")
            logger.error(f"Available FCS channels (first 5): {available_fcs_channels[:5]}")
            
            logger.info("Looking for potential fuzzy matches (case-insensitive, partial match)...")
            unmatched_requested = [ch for ch in selected_channel_names if ch not in final_channels]
            for req_ch in unmatched_requested:
                found_match = False
                for fcs_ch in available_fcs_channels:
                    if req_ch.lower() == fcs_ch.lower():
                        logger.info(f"Exact case-insensitive match: '{req_ch}' <-> '{fcs_ch}'")
                        if fcs_ch not in final_channels:
                            final_channels.append(fcs_ch)
                        found_match = True
                        break
                    elif req_ch.lower() in fcs_ch.lower() or fcs_ch.lower() in req_ch.lower():
                        logger.info(f"Partial match: '{req_ch}' <-> '{fcs_ch}'")
                if not found_match:
                    logger.info(f"No match found for '{req_ch}'")
            
            final_channels = list(set(final_channels) & set(available_fcs_channels))
            if not final_channels and selected_channel_names:
                logger.error("Despite fuzzy match attempts, no valid channels were confirmed.")
            elif not final_channels:
                logger.warning("No channels were selected at all from channels.txt.")

        return final_channels
        
    except Exception as e:
        logger.error(f"Error reading channels file '{channels_file}': {e}")
        return []

def ensure_output_dir(filepath):
    if filepath:
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
            logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        elif not output_dir:
            logger.debug("No output directory specified, using current directory.")

def main():
    parser = argparse.ArgumentParser(description="FCS Data Analysis Pipeline")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input FCS file path (required)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output processed data file path (.fcs or .csv)")
    parser.add_argument("-p", "--plot", type=str, default=None,
                        help="Output UMAP plot image file path (e.g., .png)")
    parser.add_argument("-c", "--channels_file", type=str, required=True,
                        help="Path to channels.txt file (required, specifies channels for analysis)")
    parser.add_argument("--asinh_cofactor", type=float, default=5.0,
                        help="Cofactor for asinh transformation [default %(default)s]")
    parser.add_argument("--num_clusters", type=int, default=5,
                        help="Number of clusters for k-means [default %(default)s]")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input FCS file not found: {args.input}")
        raise FileNotFoundError(f"Input FCS file not found: {args.input}")

    if not os.path.exists(args.channels_file):
        logger.error(f"Channels file not found: {args.channels_file}")
        raise FileNotFoundError(f"Channels file not found: {args.channels_file}")

    if args.output:
        ensure_output_dir(args.output)
    if args.plot:
        ensure_output_dir(args.plot)

    try:
        fcs_data_df, original_fcs_metadata = read_fcs_file_robust(args.input)
        logger.info(f"Loaded FCS data: {len(fcs_data_df)} events, {len(fcs_data_df.columns)} channels")
        logger.info(f"FCS columns (first 10): {list(fcs_data_df.columns)[:10]}")
    except Exception as e:
        logger.error(f"Failed to read input FCS file with robust reader: {e}")
        raise

    selected_channels = select_channels_from_file(args.channels_file, fcs_data_df)
    
    if not selected_channels:
        logger.error("No valid channels found for analysis based on channels.txt. Please check your channels file and FCS data.")
        raise ValueError("No valid channels for analysis.")

    logger.info(f"Final selected channels for analysis: {', '.join(selected_channels)}")

    expr_data = fcs_data_df[selected_channels].values
    logger.info(f"Expression data shape: {expr_data.shape}")

    logger.info(f"Applying asinh transformation with cofactor: {args.asinh_cofactor}")
    expr_trans = np.arcsinh(expr_data / args.asinh_cofactor)

    logger.info("Computing 2D UMAP embedding...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_embedding = reducer.fit_transform(expr_trans)
    logger.info(f"UMAP embedding shape: {umap_embedding.shape}")

    logger.info(f"Performing k-means clustering with {args.num_clusters} clusters on transformed data...")
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(expr_trans)

    logger.info(f"K-Means completed. Raw cluster labels: min={cluster_labels.min()}, max={cluster_labels.max()}")
    logger.info(f"Unique raw cluster labels found: {np.unique(cluster_labels)}")
    logger.info(f"Number of unique clusters: {len(np.unique(cluster_labels))}")
    
    if len(np.unique(cluster_labels)) == 1:
        logger.warning(f"K-Means resulted in only 1 unique cluster for {os.path.basename(args.input)}. This might indicate uniform data.")
    elif len(np.unique(cluster_labels)) != args.num_clusters:
        logger.warning(f"Expected {args.num_clusters} clusters but got {len(np.unique(cluster_labels))} unique clusters.")

    logger.info("Creating output DataFrame...")
    output_df = fcs_data_df.copy()
    
    output_df['UMAP1'] = umap_embedding[:, 0]
    output_df['UMAP2'] = umap_embedding[:, 1]
    
    cluster_ids = cluster_labels.astype(int) + 1
    output_df['Cluster_ID'] = cluster_ids
    
    logger.info(f"Output DataFrame shape: {output_df.shape}")
    logger.info(f"Output DataFrame columns (last 5): {list(output_df.columns)[-5:]}")
    logger.info(f"UMAP1 range: {output_df['UMAP1'].min():.3f} to {output_df['UMAP1'].max():.3f}")
    logger.info(f"UMAP2 range: {output_df['UMAP2'].min():.3f} to {output_df['UMAP2'].max():.3f}")
    logger.info(f"Cluster_ID unique values: {sorted(output_df['Cluster_ID'].unique())}")
    logger.info(f"Cluster_ID value counts:\n{output_df['Cluster_ID'].value_counts().sort_index()}")
    
    # Check for NaN values
    if output_df['UMAP1'].isna().any() or output_df['UMAP2'].isna().any():
        logger.error("Found NaN values in UMAP coordinates!")
    if output_df['Cluster_ID'].isna().any():
        logger.error("Found NaN values in Cluster_ID!")
    
    # Verify required columns exist
    required_columns = ['UMAP1', 'UMAP2', 'Cluster_ID']
    missing_columns = [col for col in required_columns if col not in output_df.columns]
    if missing_columns:
        logger.error(f"CRITICAL ERROR: Missing required columns in output: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    else:
        logger.info(f"All required columns present: {required_columns}")

    # Data Output
    if args.output:
        logger.info(f"Saving processed data to: {args.output}")
        if args.output.lower().endswith('.fcs'):
            write_fcs_file(output_df, args.output)
        elif args.output.lower().endswith('.csv'):
            logger.info(f"Saving processed data as CSV: {args.output}")
            output_df.to_csv(args.output, index=False)
            logger.info(f"CSV file saved: {args.output}")
            
            try:
                test_df = pd.read_csv(args.output)
                logger.info(f"Verification: CSV file readable with {len(test_df)} rows and {len(test_df.columns)} columns")
                if 'Cluster_ID' in test_df.columns:
                    logger.info(f"Verification: Cluster_ID column present with {test_df['Cluster_ID'].nunique()} unique values")
                else:
                    logger.error("Verification: Cluster_ID column missing from saved CSV!")
            except Exception as e:
                logger.error(f"Verification: Could not read back saved CSV file: {e}")
                
        else:
            logger.warning(f"Output file extension not recognized (neither .csv nor .fcs): {args.output}. Saving as CSV by default.")
            csv_output = args.output + ".csv"
            output_df.to_csv(csv_output, index=False)
            logger.info(f"CSV file saved: {csv_output}")

    if args.plot:
        logger.info(f"Generating UMAP plot to: {args.plot}")

        plot_df = pd.DataFrame({
            'UMAP1': umap_embedding[:, 0],
            'UMAP2': umap_embedding[:, 1],
            'Cluster': cluster_labels + 1
        })

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='UMAP1', y='UMAP2', hue='Cluster', data=plot_df,
                        s=5, alpha=0.7, palette='tab10')
        plt.title(f"UMAP Embedding of {os.path.basename(args.input)}\nClustered into {args.num_clusters} groups")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend(title='Cluster')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        try:
            plt.savefig(args.plot, dpi=300, bbox_inches='tight')
            logger.info(f"UMAP plot generated successfully: {args.plot}")
            logger.info(f"Plot file size: {os.path.getsize(args.plot)} bytes")
        except Exception as e:
            logger.error(f"Failed to save UMAP plot: {e}")
            raise
        finally:
            plt.close()

    logger.info("="*50)
    logger.info("FCS analysis script finished successfully.")
    logger.info("="*50)

if __name__ == "__main__":
    main()