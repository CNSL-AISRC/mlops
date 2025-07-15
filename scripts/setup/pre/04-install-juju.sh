source ./env_vars.sh

sudo snap install juju --channel=$JUJU_VERSION_STRING/stable

mkdir -p ~/.local/share