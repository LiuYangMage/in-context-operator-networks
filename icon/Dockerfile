
FROM ubuntu:22.04 as base

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

RUN apt-get update
RUN apt-get install -y --fix-missing git python3 python3-pip
RUN rm -rf /var/lib/apt/lists/*

USER user
RUN echo 'export PATH=$HOME/.local/bin:$PATH' >> $HOME/.bashrc

RUN export PIP_DEFAULT_TIMEOUT=100

RUN pip install --upgrade pip 

RUN pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html 
RUN pip install numpy scipy absl-py dm-haiku flax einops einshape jupyter matplotlib opt-einsum optax tensorflow pytz tqdm gymnasium gymnax brax pgx jumanji evosax distrax 
RUN pip install torch torchvision torchaudio transformers --index-url https://download.pytorch.org/whl/cu118

WORKDIR /workspace