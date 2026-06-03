# Image de base officielle Microsoft pour dev containers Python 3.10
# Inclut : git, curl, sudo, zsh, un utilisateur non-root "vscode"
FROM mcr.microsoft.com/devcontainers/python:1-3.10-bullseye

# L'image Microsoft inclut parfois le dépôt Yarn dont la clé GPG expire → apt-get update échoue.
# On le retire avant d'installer les libs OpenCV.
RUN rm -f /etc/apt/sources.list.d/yarn.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Installation de uv (gestionnaire de packages ultra-rapide)
# On l'installe via pip pour que ce soit disponible globalement
RUN pip install --no-cache-dir uv
