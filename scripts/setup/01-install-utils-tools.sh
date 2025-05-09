#! /bin/bash

sudo apt-get update
sudo apt-get install -y socat
sudo apt-get install -y apt-transport-https ca-certificates curl

# Turn off swap
sudo sed -i '/ swap / s/^\(.*\)$/#\1/g' /etc/fstab
sudo swapoff -a

