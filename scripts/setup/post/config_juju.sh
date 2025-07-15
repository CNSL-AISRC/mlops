#! /bin/bash

# Configure juju
microk8s config | juju add-k8s my-k8s --client
