#! /bin/bash

sudo apt-get update
sudo apt-get install -y socat
sudo apt-get install -y apt-transport-https ca-certificates curl

# install snap
sudo apt install snapd -y

# ## Instal homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

echo >> $HOME/.zshrc
echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> $HOME/.zshrc
eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"

# install k9s
brew install derailed/k9s/k9s

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh