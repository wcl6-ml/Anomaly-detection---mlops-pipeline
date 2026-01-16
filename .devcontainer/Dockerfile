FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive



#COPY --from=jdxcode/mise /usr/local/bin/mise /usr/local/bin/


# System packages only

RUN apt-get update && apt-get install -y \
    bash curl git ca-certificates build-essential sudo \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://mise.run | MISE_INSTALL_PATH=/usr/local/bin/mise sh 
 
RUN useradd -m -s /bin/bash vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN echo 'eval "$(mise activate bash)"' >> /home/vscode/.bashrc
