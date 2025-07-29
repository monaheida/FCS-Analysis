import os
import glob

configfile: "config.yaml"

FCS_FILES = glob.glob("data/raw/*.fcs")
if not FCS_FILES:
    raise Exception("No FCS files found in data/raw/")

SAMPLES = [os.path.basename(f).replace('.fcs', '') for f in FCS_FILES]

rule all:
    input:
        expand("data/processed/{sample}_processed.fcs", sample=SAMPLES),
        expand("plots/{sample}_umap.png", sample=SAMPLES)

rule process_fcs:
    input:
        fcs="data/raw/{sample}.fcs"
    output:
        processed_fcs="data/processed/{sample}_processed.fcs",
        plot="plots/{sample}_umap.png"
    params:
        channels_arg="-c channels.txt" if os.path.exists("channels.txt") else "",
        asinh_cofactor=config.get("asinh_cofactor", 5.0),
        num_clusters=config.get("num_clusters", 5)
    log:
        "logs/process_{sample}.log"
    shell:
        """
        python scripts/fcs_analysis.py \
            -i {input.fcs} \
            -o {output.processed_fcs} \
            -p {output.plot} \
            {params.channels_arg} \
            --asinh_cofactor {params.asinh_cofactor} \
            --num_clusters {params.num_clusters} \
            2>&1 | tee {log}
        """