# RedAgentExperiments
My work in parallel with PokemonRedExperiments. The focus of this repo is to develop a language agent that perform well in difficult long-term tasks.

# Running the Repo

## Env Setup (Mac)
### Do This Once
brew install pyenv
brew install pyenv-virtualenv
pyenv install 3.11
pyenv virtualenv 3.11 red_env
### Do This Every Time
pyenv activate red_env

## Install Dependencies
pip install poetry
poetry install

## Run the Code
python3 run/run_parallel_fast.py

