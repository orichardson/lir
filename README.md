This repo contains code and writing for the LIR project. 

# Overview of LIR

_Local Inconsistency Resolution (LIR)_ is a generic recipe that can be used to derive many algorithms in machine learning. At a high level, the idea is very simple: restrict your attention to a small part of your relevant beliefs, calculate their inconsistency in context, and then resolve that inconsistency by changning each parameter in proportion to the control you have in it. For a more detailed mathematical picture, check out the [most recent draft of the paper]() in the`TeX/` folder.

Here is an inscruitable one-line summary to remind those who have seen this before. Given a parametric model $\mathcal M(\theta)$, attention $\varphi$, control $\chi$, one (repeatedly) makes the following update to the parameter settings $\theta$:

$$
  \theta_{\mathrm{new}} \gets \exp_{\theta_{\mathrm{old}}}\Bigg( -\chi \odot \nabla_\theta \mathllap{\Big\langle~}\Big\langle \varphi \odot \mathcal{M}(\theta) \mathllap{\Big\rangle~}\Big\rangle\Bigg)
$$

The approach is based on the theory of [probabilistic dependenency graphs (PDGs)](https://arxiv.org/abs/2012.10800), which provide a [natural way of measuring inconsistency](http://cs.cornell.edu/~oli/files/oli-dissertation.pdf) 
(denoted $\mathllap{\langle}\langle\cdot\mathllap{\rangle}\rangle$),
that [captures and explains many objectives in machine learninng](https://arxiv.org/abs/2202.11862)
as well as graphical models and many other modeling tools in the AI literature. 
The present project (LIR) operationalizes this idea, aiming to augment this explanation of what inconsistency is and how to measure it, with an account of how one goes about resolving it. The result unifies a great deal that is know about learning, inference, and decision making.


# Technical Notes for Contributors

The overall structure is roughly as follows:

```
README.md
TeX/
code/
|-- expts/
|-- pdg/
```

Fragments of math can be found in the `TeX` folder.
Code that aims to test or apply LIR lives in `code/expts/`.  
The folder `code/pdg/` is a git submodule that points to the main [PDG repository](https://github.com/orichardson/pdg). 
Submodules can sometimes be confusing; what you need to know is summarized below. 

## Submodules 

The node `pdg/` in this repository is actually [git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules); think of it a pointer to the `/lir` branch of the (distinct) [`pdg` repository](https://github.com/orichardson/pdg). 

* To start: either clone the repository using `git clone --recurse-submodules git@github.com:orichardson/lir.git` or run `git submodule update --init` after cloning, to integrate the files from the pdg repository to your local filesystem.  
* To update (pull) the submodules: `git submodule update --remote`.  
  To update both this repo and the submodlue, `git pull --recurse-submodules`.
* To push your work on the submodule, commit and push as usual from within the submodule `pdg`. If the change touches anything important and has the possibility of breaking things, do this on a new branch and open a pull request for review. Finally: from this outer repository, run `git add pdg` and commit/push as usual. 

***Detached HEAD?***
Submodules have many conceptual and practical benefits. The drawback: git configuration issues can get nastier.
The most common problem is that is easy to get the submodule into a state that refers to a specific commit but does not track a branch. This situation is called a detached HEAD, and can happen whenever pulling a change that includes a new submodule pointer. The danger is that commits to a detached HEAD can easily be lost. 

If you see a detached head, run `git submodule update --remote`; the project configuration should re-attach the head.


## Index of Important Files (i.e., where to start)

Update this section to index your work!

 * `code/inc-grad.py` --- a suggestion for where to prototype code for taking inconsistency gradients.  
 See [issue #1](https://github.com/orichardson/lir/issues/1) for details of this important preliminary step of the project.

In general: prototypes and general functions that might ideally be integrated into the pdg repository start in `/code`. Once they are stable, we can merge them into the `pdg` submodule. Experiments and applications should go in their appropriate sub-folders. 

