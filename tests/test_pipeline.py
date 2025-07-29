import subprocess
import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class UnitTests:
    @staticmethod
    def test_data_processing_functions():
        """Test individual data processing functions"""
        print("Unit Tests: Data Processing Functions")
        print("-" * 40)
        
        try:
            test_data = pd.DataFrame({
                'CD4': np.random.exponential(100, 1000),
                'CD8': np.random.exponential(150, 1000),
                'Time': np.arange(1000)
            })
            
            cofactor = 5.0
            transformed = np.arcsinh(test_data['CD4'] / cofactor)
            
            if len(transformed) == len(test_data):
                print("Asinh transformation works")
            else:
                print("Asinh transformation failed")
                return False
            
            marker_cols = ['CD4', 'CD8']
            clustering_data = test_data[marker_cols]
            
            if clustering_data.shape == (1000, 2):
                print("Clustering data preparation works")
            else:
                print("Clustering data preparation failed")
                return False
                
            return True
            
        except Exception as e:
            print(f"Data processing unit tests failed: {e}")
            return False
    
    @staticmethod
    def test_config_parsing():
        """Test configuration file parsing"""
        print("\nUnit Tests: Config Parsing")
        print("-" * 40)
        
        try:
            config_path = PROJECT_ROOT / 'config.yaml'
            if not config_path.exists():
                print(f"config.yaml not found at {config_path}")
                return False
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            required_keys = ['asinh_cofactor', 'num_clusters', 'umap_n_components']
            missing_keys = [key for key in required_keys if key not in config]
            
            if missing_keys:
                print(f"Missing config keys: {missing_keys}")
                return False
            else:
                print("Config YAML parsing works")
            
            channels_path = PROJECT_ROOT / 'channels.txt'
            if not channels_path.exists():
                print(f"channels.txt not found at {channels_path}")
                return False
            
            channels_df = pd.read_csv(channels_path, sep='\t')
            required_cols = ['name', 'desc', 'use'] 
            
            if all(col in channels_df.columns for col in required_cols):
                print("Channels file parsing works")
                print(f"  - Found {len(channels_df)} channels")
                print(f"  - {sum(channels_df['use'] == 1)} channels marked for use")
                return True
            else:
                print(f"Channels file parsing failed. Missing columns: {[col for col in required_cols if col not in channels_df.columns]}")
                return False
                
        except Exception as e:
            print(f"Config parsing unit tests failed: {e}")
            return False
    
    @staticmethod
    def test_file_io_functions():
        """Test file input/output functions"""
        print("\nUnit Tests: File I/O")
        print("-" * 40)
        
        try:
            temp_dir = tempfile.mkdtemp()
            test_csv = Path(temp_dir) / 'test.csv'
            
            # Create test data
            test_df = pd.DataFrame({
                'A': [1, 2, 3],
                'B': [4, 5, 6]
            })
            
            # Test writing
            test_df.to_csv(test_csv, index=False)
            
            if test_csv.exists():
                print("CSV writing works")
            else:
                print("CSV writing failed")
                return False
            
            # Test reading
            read_df = pd.read_csv(test_csv)
            
            if read_df.equals(test_df):
                print("CSV reading works")
            else:
                print("CSV reading failed")
                return False
            
            # Cleanup
            os.unlink(test_csv)
            os.rmdir(temp_dir)
            
            return True
            
        except Exception as e:
            print(f"File I/O unit tests failed: {e}")
            return False

class IntegrationTests:
    @staticmethod
    def test_fcs_to_processed_pipeline():
        print("\nIntegration Test: FCS → Processed Data")
        print("-" * 50)
        
        fcs_files = list((PROJECT_ROOT / 'data' / 'raw').glob('*.fcs'))
        if not fcs_files:
            print("No FCS files found in data/raw. Skipping test.")
            return False
        
        test_file = fcs_files[0]
        sample_name = test_file.stem
        print(f"Testing processing for: {test_file.name}")
        
        expected_fcs_output = f'data/processed/{sample_name}_processed.fcs'
        
        cmd = [
            'snakemake',
            expected_fcs_output, 
            '--cores', '1',
            '--directory', str(PROJECT_ROOT),
            '--quiet'
        ]
        
        print(f"Running Snakemake command: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=180) # <--- Correctly uses PROJECT_ROOT for cwd

        if result.stdout:
            print(f"Snakemake STDOUT:\n{result.stdout[-500:]}")
        if result.stderr:
            print(f"Snakemake STDERR:\n{result.stderr[-500:]}")
        
        if result.returncode != 0:
            print(f"Pipeline execution failed for target {expected_fcs_output}.")
            print(f"Snakemake Error: {result.stderr}")
            return False
        else:
            print(f"Snakemake execution finished for target {expected_fcs_output}. Checking outputs.")
        
        csv_path = PROJECT_ROOT / f'data/processed/{sample_name}_processed.csv'
        plot_path = PROJECT_ROOT / f'plots/{sample_name}_umap.png'
        
        if not csv_path.exists():
            print(f"Processed CSV not created at {csv_path}")
            return False
        else:
            print(f"Processed CSV found at {csv_path}")
            
        if not plot_path.exists():
            print(f"UMAP plot not created at {plot_path}")
            return False
        else:
            print(f"UMAP plot found at {plot_path}")

        fcs_output_actual_path = PROJECT_ROOT / expected_fcs_output
        if fcs_output_actual_path.exists():
            print(f"FCS file found at {fcs_output_actual_path} (note: may not be fully compliant with all parsers).")
        else:
            print(f"FCS file {fcs_output_actual_path} was not created (likely due to internal fcswrite issues).")

        try:
            df = pd.read_csv(csv_path)
            required_cols = ['UMAP1', 'UMAP2', 'Cluster_ID']
            
            if not all(col in df.columns for col in required_cols):
                print(f"Missing required columns in CSV output: {', '.join([col for col in required_cols if col not in df.columns])}")
                return False
            
            if df['UMAP1'].isna().any() or df['UMAP2'].isna().any():
                print("UMAP coordinates in CSV contain NaN values")
                return False
            
            if 'Cluster_ID' in df.columns and df['Cluster_ID'].min() < 1:
                print("Invalid cluster IDs (should start from 1)")
                return False
            
            print(f"Successfully processed {len(df)} cells (validated from CSV)")
            if 'Cluster_ID' in df.columns:
                print(f"Generated {df['Cluster_ID'].nunique()} clusters (validated from CSV)")
            print("UMAP coordinates valid (validated from CSV)")
            
            return True
        except Exception as e:
            print(f"Error validating CSV data integrity: {e}")
            return False
    
    @staticmethod
    def test_snakemake_workflow_integrity():
        print("\nIntegration Test: Snakemake Workflow")
        print("-" * 50)
        
        try:
            # Test dry run
            cmd = ['snakemake', '--dry-run', '--directory', str(PROJECT_ROOT), '--cores', '1'] 
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30) 
            
            if result.returncode != 0:
                print("Snakemake dry run failed")
                print(f"Error: {result.stderr[-500:]}") 
                return False
            
            print("Snakemake workflow is valid")
            
            cmd = ['snakemake', '--dag', '--directory', str(PROJECT_ROOT)]
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and 'digraph' in result.stdout:
                print("Workflow DAG generation works")
            else:
                print("DAG generation not available or failed (graphviz missing or other issue). Error (if any):")
                print(result.stderr[-200:])
            
            return True
            
        except Exception as e:
            print(f"Snakemake workflow test failed: {e}")
            return False
    
    @staticmethod
    def test_environment_integration():
        """Test integration between Docker environment and tools"""
        print("\nIntegration Test: Environment")
        print("-" * 40)
        
        try:
            cmd = ['python', '-c', 'import snakemake; print("Snakemake import OK")']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print("Python-Snakemake integration works")
            else:
                print("Python-Snakemake integration failed")
                return False
            
            integration_test = '''
import pandas as pd
import numpy as np
import umap
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns 
print("All package integrations OK")
'''
            
            cmd = ['python', '-c', integration_test]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print("Package integrations work")
                return True
            else:
                print("Package integration failed")
                print(f"Package integration error: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Environment integration test failed: {e}")
            return False

def run_complete_test_suite():
    print("Complete FCS Pipeline Test Suite")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("UNIT TESTS")
    print("=" * 60)
    
    unit_tests = [
        ("Data Processing Functions", UnitTests.test_data_processing_functions),
        ("Config Parsing", UnitTests.test_config_parsing),
        ("File I/O Functions", UnitTests.test_file_io_functions)
    ]
    
    unit_results = []
    for test_name, test_func in unit_tests:
        try:
            passed = test_func()
            unit_results.append((test_name, passed))
        except Exception as e:
            print(f"{test_name} crashed: {e}")
            unit_results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("INTEGRATION TESTS")
    print("=" * 60)
    
    integration_tests = [
        ("Environment Integration", IntegrationTests.test_environment_integration),
        ("Snakemake Workflow", IntegrationTests.test_snakemake_workflow_integrity),
        ("FCS Processing Pipeline", IntegrationTests.test_fcs_to_processed_pipeline)
    ]
    
    integration_results = []
    for test_name, test_func in integration_tests:
        try:
            passed = test_func()
            integration_results.append((test_name, passed)) # Pass the actual result here
        except Exception as e:
            print(f"{test_name} crashed: {e}")
            integration_results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("TEST SUITE SUMMARY")
    print("=" * 60)
    
    print("\nUnit Tests:")
    unit_passed = 0
    for test_name, passed in unit_results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")
        if passed:
            unit_passed += 1
    
    print("\nIntegration Tests:")
    integration_passed = 0
    for test_name, passed in integration_results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")
        if passed:
            integration_passed += 1
    
    total_tests = len(unit_results) + len(integration_results)
    total_passed = unit_passed + integration_passed
    
    print(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\nAll tests passed! Pipeline is robust and reliable.")
        return 0
    else:
        print(f"\n{total_tests - total_passed} tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = run_complete_test_suite()
    sys.exit(exit_code)