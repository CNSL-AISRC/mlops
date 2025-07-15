#! /bin/bash
source ./env_vars.sh

# Install microk8s using snap
sudo snap install microk8s --channel=$K8S_VERSION_STRING/stable --classic

# Enable required addons
sudo microk8s enable dns ingress helm3

# Add user to microk8s group
sudo usermod -a -G microk8s $USER


echo "Please run 'newgrp microk8s' to switch to the microk8s group"
echo "Then run 'microk8s status' to verify the installation"