FROM --platform=linux/amd64 m.daocloud.io/docker.io/pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime AS isles26_final_amd64

ENV PYTHONUNBUFFERED=1 \
    nnUNet_compile=false \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    TORCH_HOME=/tmp/torch \
    CUDA_MODULE_LOADING=LAZY

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user
WORKDIR /opt/app

RUN python3 -m venv --system-site-packages --without-pip /home/user/venv
ENV PATH="/home/user/venv/bin:$PATH"

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user deploy_trainers /opt/app/deploy_trainers
RUN python /opt/app/deploy_trainers/install.py

COPY --chown=user:user app.py inference.py /opt/app/

LABEL org.grand-challenge.api-method="invoke"
EXPOSE 4743
ENTRYPOINT ["python3", "app.py"]
