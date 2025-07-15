#! /bin/bash

# Configure MicroK8s
sudo microk8s enable dns hostpath-storage metallb:10.64.140.43-10.64.140.49 rbac nvidia

microk8s status