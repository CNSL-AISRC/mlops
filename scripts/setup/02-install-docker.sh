#! /bin/bash

# Install docker Engine
source ./env_vars.sh
## Run the following command to uninstall all conflicting packages:
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done

# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

## Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

## Install the Docker packages.
sudo apt-get install -y docker-ce=$DOCKER_VERSION_STRING docker-ce-cli=$DOCKER_VERSION_STRING containerd.io docker-buildx-plugin docker-compose-plugin

## Verify that the installation is successful by running the hello-world image:
echo "Adding $USER to the docker group"
sudo usermod -aG docker $USER
# echo "Switching to the docker group"
# newgrp docker
echo "Running hello-world"
sudo docker run hello-world

echo "Please run 'newgrp docker' to switch to the docker group"
echo "Then run 'docker run hello-world' to verify the installation"
echo "Done docker installation"