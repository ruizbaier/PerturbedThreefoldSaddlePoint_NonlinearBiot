# PerturbedThreefoldSaddlePoint_NonlinearBiot

This repository contains a FEniCS_{ii} implementation of a fully-mixed poroelasticity model (three-field / multi-block formulation) used in the accompanying manuscript. The code in this folder is intended for convergence tests and demonstration of a mixed finite-element formulation using block assembly with FEniCS and FEniCS_{ii} (xii).

**Variational Problem**

Find $(\boldsymbol{\eta},\boldsymbol{\xi}) \in \mathbf{H}(\mathrm{div};\Omega)\times \mathbb{L}^2(\Omega)$, $p\in L^2(\Omega)$, $(\varphi,\boldsymbol{\sigma}) \in H_{00}^{1/2}(\Gamma)\times \mathbb{H}_\Sigma(\mathbf{div};\Omega)$, and $(\boldsymbol{u},\boldsymbol{\gamma}) \in \mathbf{L}^2(\Omega)\times \mathbb{L}^2_{\mathrm{skew}}(\Omega)$ such that
$$
\int_{\Omega}\kappa^{-1} \,\boldsymbol{\eta}\cdot\boldsymbol{\chi} + \int_{\Omega} p\,\mathrm{div}(\boldsymbol{\chi}) - \langle{\boldsymbol{\chi}\cdot\boldsymbol{n}},{\varphi}\rangle_{\Gamma} = \langle{\boldsymbol{\chi}\cdot\boldsymbol{n}},p_\sigma \rangle_{\Sigma},$$

$$\int_{\Omega}\boldsymbol{\sigma}:\boldsymbol{\rho} - \int_{\Omega}\mathcal{C}\,\boldsymbol{\xi}:\boldsymbol{\rho} + \alpha\int_{\Omega}p\,\mathrm{tr}(\boldsymbol{\rho}) = 0, $$

$$ - \alpha \int_{\Omega}\mathrm{tr}(\boldsymbol{\xi})\, q + \int_{\Omega}\mathrm{div}(\boldsymbol{\eta})\, q - c_{0}\int_{\Omega}p\, q = - \int_{\Omega}g\,q,$$

$$ -\int_{\Omega}\boldsymbol{\xi}:\boldsymbol{\tau} - \int_{\Omega}\boldsymbol{u}\cdot\mathbf{div}(\boldsymbol{\tau}) - \int_{\Omega}\boldsymbol{\gamma}:\boldsymbol{\tau}  = - \langle{\boldsymbol{\tau}\boldsymbol{n}},{\boldsymbol{u}_{\mathrm{D}}}\rangle_{\Gamma},$$

$$\langle{\boldsymbol{\eta}\cdot\boldsymbol{n}},{\psi}\rangle_{\Gamma} =  \langle{g_{\mathrm{N}}},{\psi}\rangle_{\Gamma},$$

$$-\int_{\Omega}\mathbf{div}(\boldsymbol{\sigma}) \cdot \boldsymbol{v}  = \int_{\Omega}\boldsymbol{f}\cdot \boldsymbol{v},$$

$$ - \int_{\Omega} \boldsymbol{\sigma}:\boldsymbol{\delta} = 0,$$

for all $(\boldsymbol{\chi},\boldsymbol{\rho}) \in \mathbf{H}(\mathrm{div};\Omega) \times \mathbb{L}^2(\Omega)$, $q\in L^2(\Omega)$, $(\psi,\boldsymbol{\tau}) \in H_{00}^{1/2}(\Gamma)\times \mathbb{H}_{\Sigma}(\mathbf{div};\Omega)$, and  $(\boldsymbol{v},\boldsymbol{\delta}) \in \mathbf{L}^{2}(\Omega) \times \mathbf{L}^{2}_{\mathrm{skew}}(\Omega)$.

**What this repository provides**
- **`bdgrv_convergence2D_FEniCSii_PEERS_linear.py`**: driver for convergence tests and example runs. 
- **Block-structured formulation**: code uses `block_form`, `ii_assemble`/`block_assemble` and `ii_convert` from `xii` (FEniCS_{ii}) to build multi-block matrices. When possible the code prefers `block_assemble` so block templates (templates for block vectors/matrices) are preserved, which simplifies applying per-block Dirichlet boundary conditions.
- **Boundary conditions**: Dirichlet conditions on the sigma components are assembled as per-block `DirichletBC` objects and passed to `apply_bc` either as a per-block dictionary (e.g. `{5: [bc_sig0], 6: [bc_sig1]}`) or converted to a monolithic map when using a monolithic matrix.
- **Fractional norms**: a helper `fractional_positive_norm_00` is implemented to compute fractional H^s (here used with s=1/2) norms on boundary traces by solving a small generalized eigenproblem on a CG space with homogeneous Dirichlet boundary conditions. The method removes boundary DOFs from assembled numpy matrices directly to reflect H^s_{00} behaviour and compute the induced inner product.

**Implementation notes**
- **FEniCS_{ii} (xii)**: used to build block variational forms and to obtain block matrices/vectors. The code contains logic to detect when the assembled system is monolithic vs block-structured and adapts how boundary conditions are applied.
- **Fractional norm implementation**: see `fractional_positive_norm_00(f, fh, s)` in the linear driver. It builds stiffness `A` and mass `M` matrices on a CG trial/test space, eliminates Dirichlet DOFs and solves the generalized eigenproblem `A u = lambda M u`. The fractional inner product matrix is then assembled via spectral representation and applied to the reduced error vector.

**Quick start (example)**
Assuming you have a FEniCS environment where `xii` (FEniCS_{ii}) is installed and active (`fenicsproject` conda env in the original testing), run the linear driver with:

```bash
# activate your fenics environment (example)
conda activate fenicsproject

python bdgrv_convergence2D_FEniCSii_PEERS_linear.py
```

Notes:
- If your installation lacks command-line tools like `pdftotext` the README will still display equations in LaTeX; renderers such as GitHub or VS Code will show them properly when viewing the file with math support.
- If `apply_bc` raises template/allocation errors when converting between monolithic and block systems, prefer using `block_assemble` so the block templates are preserved and pass per-block BC dictionaries directly.

**Files of interest**
- `bdgrv_convergence2D_FEniCSii_PEERS.py` — full (nonlinear-capable) driver with nonlinear jacobian helpers and richer diagnostics.
- `bdgrv_convergence2D_FEniCSii_PEERS_linear.py` — linear test driver (computes fractional-norm error for the trace variable using `fractional_positive_norm_00`).
- `sympy2fenics.py` — small helper to convert symbolic strings to FEniCS `Expression` objects used in the examples.

**References**

A similar implementation for multiphysics of flow transport in fully mixed form, with Lagrange multipliers was used in 

```bibtex
@article{gnr_jsc23,
	author = {Gatica, Gabriel N and N\'u\~nez, Nicolas and Ruiz-Baier, Ricardo},
	doi = {10.1007/s10915-023-02202-9},
	fjournal = {Journal of {S}cientific {C}omputing},
	journal = {J. Sci. Comput.},
	pages = {e79(1--38)},
	title = {Mixed-primal methods for natural convection driven phase change with {Navier--Stokes--Brinkman} equations},
	volume = {95},
	year = {2023}
}
```


**Requirements & Setup**

Below is a suggested set of packages and an example conda-based setup. The code was developed and tested inside a FEniCS-enabled environment (for example the `fenicsproject` conda environment). Exact package names/versions may vary depending on your platform and package sources; use these as a starting point.

```bash
# Create a conda environment (recommended)
conda create -n fenicsproject -c conda-forge fenics mshr petsc petsc4py mpi4py numpy scipy matplotlib sympy
conda activate fenicsproject

# Install FEniCS_ii (xii) and other Python tools (try pip or follow xii installation instructions)
pip install xii

# Optional: if xii is not published on PyPI, install from source (follow the xii project README)
# git clone <xii-repo-url> && cd xii && pip install -e .