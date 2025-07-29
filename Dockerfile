FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    procps \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir \
    pandas>=1.5.0 \
    numpy>=1.21.0 \
    matplotlib>=3.5.0 \
    seaborn>=0.11.0 \
    scikit-learn>=1.1.0 \
    umap-learn>=0.5.3 \
    fcsparser>=0.2.4 \
    snakemake>=7.0.0 \
    pyyaml>=6.0 \
    flowkit>=0.9.0 \
    flowio>=0.9.4 \
    fcswrite

RUN pip install --no-cache-dir uv

RUN useradd --create-home --shell /bin/bash --user-group pipeline

WORKDIR /pipeline

RUN mkdir -p data/raw data/processed plots results logs scripts tests \
    && chown -R pipeline:pipeline /pipeline

COPY --chown=pipeline:pipeline . ./


USER pipeline

ENV PYTHONPATH="/pipeline/scripts:/pipeline"

CMD ["python", "-c", "print('FCS Pipeline ready. Use: python scripts/fcs_analysis.py')"]
