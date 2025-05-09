#! /bin/bash
source ./env_vars.sh

# Install kubeadm, kubelet and kubectl
sudo apt-get update
sudo apt-get install -y kubelet=$K8S_VERSION_STRING kubeadm=$K8S_VERSION_STRING kubectl=$K8S_VERSION_STRING

# Install containerd
sudo apt-get install -y containerd