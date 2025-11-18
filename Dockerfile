FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ARG GITHUB_TOKEN

# Clona vfss-data-split
RUN git clone \
    -b CS/medsegdiff \
    https://${GITHUB_TOKEN}:x-oauth-basic@github.com/puc-rio-inca/vfss-data-split.git \
    vfss-data-split

# Instala biblioteca
WORKDIR /workspace/vfss-data-split
RUN pip install --upgrade pip && pip install .

# Clona seu fork do MedSegDiff
WORKDIR /workspace
RUN git clone https://github.com/caioseda/MedSegDiff.git medsegdiff

# Instala dependências do MedSegDiff
WORKDIR /workspace/medsegdiff
RUN pip install -r requirement.txt

# Garante PYTHONPATH
ENV PYTHONPATH="/workspace/medsegdiff:${PYTHONPATH}"

CMD ["bash"]