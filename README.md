# QCMaterial

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://shinaoka.github.io/QCMaterial.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://shinaoka.github.io/QCMaterial.jl/dev)
[![Build Status](https://github.com/sakurairihito/QCMaterialNew/workflows/CI/badge.svg)](https://github.com/sakurairihito/QCMaterialNew/actions)
[![DevNew](https://img.shields.io/badge/docs-dev-blue.svg)](https://sakurairihito.github.io/QCMaterialNew/build/index.html)



## Install packages

```bash
pip install -r requirements.txt
```

## Launch a notebook server
This project is activated automatically when
you launch a server in the project directory.

```bash
$ cd $(PROJECTDIR)
$ jupyter lab
```

## Run tests
```bash
$ cd $(PROJECTDIR)
$ julia --project=@. test/runtests.jl
$ mpirun -np 2 julia --project=@. test/runtests.jl  # Only if MPI is installed on your system
```

## Run a script file depending on QCMaterial
``bash
mpirun -np 2 julia --project=~/.julia/dev/QCMaterial ~/.julia/dev/QCMaterial/samples/run.jl
`` 
