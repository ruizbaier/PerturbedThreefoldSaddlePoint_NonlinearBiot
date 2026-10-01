'''
Parameter-robustness test for the fully mixed Biot formulation (PEERS_{k+1} + RT_{k+1}-DG_k + CG_{k+1} multiplier).
Same discretisation and structure as bdgrv_convergence2D_FEniCSii_PEERS_linear.py, but now for each mesh level
we sweep over
      lambda in {1, 1e3, 1e6, 1e9},   alpha, c0, kappa0 in {1, 1e-3, 1e-6, 1e-9},
with kappa(x,y) = kappa0*exp(x*y), and we record relative errors of all unknowns (k=0 only).

Note: the coupling terms with the Lagrange multiplier are computed on a boundary mesh conforming with the bulk
mesh and then mapped to the (coarser) multiplier mesh; see the comments below. In the convergence scripts the
Trace is taken directly onto the non-matching coarse mesh, which is only correct when p = 0 on Gamma.

Output (folder outputs/robustness/):
  robustness_long.csv       : one row per (parameter combination, mesh level), absolute and relative errors
  robustness_<metric>.txt   : pgfplotstable-friendly files (column 'dofs' + one column per combination), with
                              metric in {total, eta, xi, p, phi, sig, u, gam}; 'total' is the sum of the seven
                              relative errors
  robustness_<metric>.tex   : standalone pgfplots figure (see robustness_plot.py)

Usage: python3 bdgrv_parameter_robustness.py [--xiP2] [lambda_index ...]
  (optional indices restrict the lambda values, e.g. to run the four lambda values in four processes;
   the partial csv files are merged with 'python3 robustness_plot.py <outdir>')
  --xiP2 : use discontinuous P_{k+2} tensors for xi (which contain P_k + B_k) instead of the space of the
           manuscript; results go to outputs/robustness_xiP2/. With the manuscript space the elasticity
           unknowns lock for large lambda; with this enlarged space they do not.
'''
from dolfin import *
import os, sys, itertools, time
from xii import *
import sympy2fenics as sf
from scipy.linalg import eigh
import numpy as np
from petsc4py import PETSc
from block import block_transpose
from xii.assembler.nonconforming_trace_matrix import nonconforming_trace_mat
import robustness_plot as rplot

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = -1

PETScOptions.set("mat_mumps_icntl_14", 1000) # memory estimate
PETScOptions.set("mat_mumps_icntl_13", 1) # accept almost zero pivots
PETScOptions.set("mat_mumps_cntl_1", 1e-8)

xi_enriched = '--xiP2' in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith('--')]
outdir = rplot.OUTDIR + ('_xiP2' if xi_enriched else '')
os.makedirs(outdir, exist_ok=True)

def fractional_positive_norm_00(f, fh, s):
    '''returns (||f-fh||_s, ||f||_s) in H^s norm. We use a CG space with Dirichlet BCs to reflect H^s_{00}'''
    Qelm = fh.function_space().ufl_element()
    mesh = fh.function_space().mesh()

    Qe = FunctionSpace(mesh, Qelm)
    bc = DirichletBC(Qe, Constant(0.0), 'on_boundary')

    # Fractional Laplacian
    p, q = TrialFunction(Qe), TestFunction(Qe)
    a = inner(grad(p), grad(q))*dx
    m = inner(p, q)*dx

    A, M = [assemble(foo).array() for foo in (a, m)]
    # remove rows/cols of boundary dofs (homogeneous Dirichlet BC)
    bc_vals = bc.get_boundary_values()
    n = A.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[list(bc_vals.keys())] = False
    A = A[mask][:, mask]
    M = M[mask][:, mask]

    Lmbda, U = eigh(A, M)
    assert np.all(Lmbda > 0), 'Increase penalty?'

    W = M@U
    # H^s inner product (s>0): e^T M U diag(Lmbda^s) U^T M e  (s=1 gives the H^1_0 seminorm)
    Hs = W@np.diag(Lmbda**s)@W.T
    ex = interpolate(f, Qe).vector().get_local()[mask]
    err_vec = ex - interpolate(fh, Qe).vector().get_local()[mask]
    return np.sqrt(np.inner(err_vec, Hs@err_vec)), np.sqrt(np.inner(ex, Hs@ex))

def block_bcs_to_monolithic_map(BCs_dict, Hh):
    offsets = [0]
    for W in Hh:
        offsets.append(offsets[-1] + W.dim())
    global_map = {}
    for blk_idx, bc_list in BCs_dict.items():
        shift = offsets[blk_idx]
        for bc in (bc_list or []):
            if not isinstance(bc, DirichletBC): continue
            for local_dof, val in bc.get_boundary_values().items():
                global_map[shift + int(local_dof)] = val
    return {0: [global_map]}

def lu_solve(A, x, b):
    '''MUMPS solve with the current options; if the factorisation runs out of workspace (INFOG(1)=-9, which
    may happen with delayed pivots for very large lambda), retry with a larger workspace relaxation'''
    for relax in (1000, 4000, 16000):
        PETScOptions.set("mat_mumps_icntl_14", relax)
        solver = PETScLUSolver(method = "mumps")
        try:
            x.zero(); solver.solve(A, x, b); break
        except RuntimeError:
            if relax == 16000: raise
    PETScOptions.set("mat_mumps_icntl_14", 1000)
    return solver

def solve_checked(A, x, b, tol=1e-8, maxref=5):
    '''LU solve; if the relative residual is larger than tol (e.g. after an inaccurate factorisation with the
    small pivoting threshold set above), refactor with a standard threshold and apply a few steps of iterative
    refinement. Returns the relative residual.'''
    bnorm = max(b.norm('l2'), 1e-300); r = b.copy()
    lu_solve(A, x, b)
    A.mult(x, r); r.axpy(-1.0, b); res = r.norm('l2')/bnorm
    if res > tol:
        PETScOptions.set("mat_mumps_cntl_1", 1e-2)
        solver = lu_solve(A, x, b)
        dx = b.copy()
        for it in range(maxref):
            A.mult(x, r); r.axpy(-1.0, b); res = r.norm('l2')/bnorm
            if res <= tol: break
            solver.solve(A, dx, r); x.axpy(-1.0, dx)
        A.mult(x, r); r.axpy(-1.0, b); res = r.norm('l2')/bnorm
        PETScOptions.set("mat_mumps_cntl_1", 1e-8)
    return res

def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

def tensorify_skew(r):
        return as_tensor((( 0,r),
                          (-r,0)))

# ******* Model parameters (values are assigned inside the sweep) ****** #
ndim = 2
I = Identity(ndim)
e0, e1 = Constant((1, 0)), Constant((0, 1))
mu     = Constant(1.0)
lmbda  = Constant(1.0)
c0     = Constant(1.0)
alpha  = Constant(1.0)
kappa0 = Constant(1.0)

lmbda_vals = rplot.LMBDA_VALS
small_vals = rplot.SMALL_VALS
if args:
    lmbda_vals = [lmbda_vals[int(i)] for i in args]
combos = list(itertools.product(lmbda_vals, small_vals, small_vals, small_vals)) # (lambda, kappa0, alpha, c0)

symmgr = lambda v: sym(grad(v))
skewgr = lambda v: grad(v) - symmgr(v)
curlBub = lambda vec: as_tensor([[vec[0].dx(1), -vec[0].dx(0)], [vec[1].dx(1), -vec[1].dx(0)]])
CTimes = lambda s: 2.*mu*s + lmbda*tr(s)*I
CinvTimes = lambda s: 0.5/mu*(s - lmbda/(2.*mu + ndim*lmbda)*tr(s)*I)

# ******* Exact solutions for error analysis ****** #
# displacement u = curl(psi) + (1/lambda)*u1, so that div(u) = O(1/lambda) and sigma remains bounded as lambda grows
# (nearly incompressible regime). Unlike in the convergence test, p does not vanish on Gamma (so that phi != 0 and
# its relative error makes sense), but it vanishes at the endpoints (0,1), (1,0) of Gamma, as required for
# phi in H^{1/2}_{00}(Gamma)
psi_str = '0.1*sin(pi*x)*sin(pi*y)'
u1_str = '(0.05*cos(1.5*pi*(x+y)),0.05*sin(1.5*pi*(x-y)))'
p_str = 'cos(0.5*pi*x)*cos(0.5*pi*y)'
kappa_str = 'exp(x*y)'

# lowest-order method
k=0; nkmax = 5
set_log_level(40)

names = rplot.NAMES # eta, xi, p, phi, sig, u, gam
results = [] # list of dicts (one per combination and level)
tag = '_'.join(args)
csvfile = os.path.join(outdir, 'robustness_long%s.csv' % ('_part'+tag if tag else ''))

for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)

    # need two meshes / one for the outer boundary
    nps = pow(2,nk+1)+2; npst = pow(2,nk)+1
    mesh = UnitSquareMesh(nps,nps)
    mesht = UnitSquareMesh(npst,npst)

    facet_mesh = MeshFunction('size_t', mesh, ndim-1, 0)
    facet_mesht = MeshFunction('size_t', mesht, ndim-1, 0)

    boundaries = {31: CompiledSubDomain('near(x[0], 0) || near(x[1], 0)'),
                  32: CompiledSubDomain('near(x[0], 1) || near(x[1], 1)')}

    [subb.mark(facet_mesh, tag_) for tag_, subb in boundaries.items()]
    [subb.mark(facet_mesht, tag_) for tag_, subb in boundaries.items()]

    Gamma = 31 # left-bottom: on which we will define the Lagrange multiplier
    Sigma = 32

    # creating sub-boundary meshes only in the needed part: the multiplier lives on the coarse bmesht, whereas
    # the coupling terms <eta.n, psi>_Gamma are computed on bmesh, which is conforming with the bulk mesh
    # (there the trace of RT functions is exact; on the non-matching bmesht fenics_ii would evaluate RT
    # functions at the vertices of Gamma using an arbitrary bulk cell containing them)
    bmesht = EmbeddedMesh(facet_mesht, Gamma)
    bmesh = EmbeddedMesh(facet_mesh, Gamma)

    n = FacetNormal(mesh)
    n_ = OuterNormal(bmesh, [0.5, 0.5])

    h, ht = mesh.hmax(), mesht.hmax()

    # this measure is for the boundary integrals involving the Lagrange multiplier (on the fine boundary mesh)
    dx_ = Measure('dx', domain=bmesh, subdomain_data=bmesh.marking_function)
    ds = Measure('ds', domain=mesh, subdomain_data=facet_mesh)

    # ********* Finite dimensional spaces ********* #
    Heta    = FunctionSpace(mesh, "RT", k+1)
    Hxi_aux = TensorFunctionSpace(mesh, "DG", k+2 if xi_enriched else k)
    Bub     = VectorFunctionSpace(mesh,'B', k + 3)
    Hp      = FunctionSpace(mesh, "DG", k)
    Hphi    = FunctionSpace(bmesht, 'CG', k+1)
    Hphi_f  = FunctionSpace(bmesh, 'CG', k+1)
    Hsig_aux= FunctionSpace(mesh, "RT", k+1)
    Hu      = VectorFunctionSpace(mesh, "DG", k)
    Hgam    = FunctionSpace(mesh, "CG", k+1)

    Hh = [Heta,Hxi_aux,Bub,Hp,Hphi,Hsig_aux,Hsig_aux,Bub,Hu,Hgam]
    ndofs = sum([spa.dim() for spa in Hh])
    print ("....... Total DoFs = ", ndofs)

    # the forms are written with the multiplier in the fine space Hphi_f; after assembly they are mapped
    # to the coarse space Hphi with the interpolation matrix P : Hphi -> Hphi_f (exact, as the meshes are nested)
    Hh_f = Hh[:4] + [Hphi_f] + Hh[5:]
    P = PETScMatrix(nonconforming_trace_mat(Hphi, Hphi_f))
    Pt = block_transpose(P)

    eta,  xi_, btrial_a, p, phi, sig0, sig1, btrial_b, u, gam_ = map(TrialFunction, Hh_f)
    chi, rho_, btest_a, q, psi, tau0, tau1,  btest_b, v, del_ = map(TestFunction, Hh_f)

    T_eta, T_chi = Trace(eta, bmesh), Trace(chi, bmesh)

    gamma = tensorify_skew(gam_); delta = tensorify_skew(del_)
    sig0, sig1 = outer(e0, sig0), outer(e1, sig1)
    tau0, tau1 = outer(e0, tau0), outer(e1, tau1)

    # ******* Instantiating exact solutions and variable coefficients ******* #
    psi_ex  = Expression(str2exp(psi_str), degree=6, domain=mesh)
    u_ex    = as_vector((psi_ex.dx(1), -psi_ex.dx(0))) + 1.0/lmbda*Expression(str2exp(u1_str), degree=6, domain=mesh)
    p_ex    = Expression(str2exp(p_str), degree=6, domain=mesh)
    kappa   = kappa0*Expression(str2exp(kappa_str), degree=6, domain=mesh)
    phi_ex_ = Expression(str2exp(p_str), degree=6, domain=bmesht)

    eta_ex  = kappa*grad(p_ex)
    xi_ex   = symmgr(u_ex)
    sig_ex  = 2.*mu*xi_ex + lmbda*tr(xi_ex)*I - alpha*p_ex*I
    gamma_ex= skewgr(u_ex)

    ff_ex = -div(sig_ex)
    g_ex  = c0*p_ex + alpha*tr(xi_ex)-div(eta_ex)

    p_D   = interpolate(p_ex,Hp)

    # ******* variational forms (parameters enter only through the Constants) ******* #
    a = block_form(Hh_f,2); l = block_form(Hh_f,1)
    a[0][0] = 1.0/kappa*dot(eta,chi)*dx
    a[0][3] = p*div(chi)*dx
    a[0][4] = - dot(T_chi,n_)*phi*dx_(Gamma)

    a[1][1] = inner(CTimes(xi_),rho_)*dx
    a[1][2] = inner(CTimes(curlBub(btrial_a)),rho_)*dx
    a[1][3] = -alpha*p*tr(rho_)*dx
    a[1][5] = -inner(sig0,rho_)*dx
    a[1][6] = -inner(sig1,rho_)*dx
    a[1][7] = -inner(curlBub(btrial_b),rho_)*dx

    a[2][1] = inner(CTimes(xi_),curlBub(btest_a))*dx
    a[2][2] = inner(CTimes(curlBub(btrial_a)),curlBub(btest_a))*dx
    a[2][3] = -alpha*p*tr(curlBub(btest_a))*dx
    a[2][5] = -inner(sig0,curlBub(btest_a))*dx
    a[2][6] = -inner(sig1,curlBub(btest_a))*dx
    a[2][7] = -inner(curlBub(btrial_b),curlBub(btest_a))*dx

    a[3][0] = dot(div(eta),q)*dx
    a[3][1] = - alpha*tr(xi_)*q*dx
    a[3][2] = - alpha*tr(curlBub(btrial_a))*q*dx
    a[3][3] = - c0*p*q*dx

    a[4][0] = - dot(T_eta,n_)*psi*dx_(Gamma)

    a[5][1] = - inner(xi_,tau0)*dx
    a[5][2] = - inner(curlBub(btrial_a),tau0)*dx
    a[5][8] = - dot(u,div(tau0))*dx
    a[5][9] = - inner(gamma,tau0)*dx

    a[6][1] = - inner(xi_,tau1)*dx
    a[6][2] = - inner(curlBub(btrial_a),tau1)*dx
    a[6][8] = - dot(u,div(tau1))*dx
    a[6][9] = - inner(gamma,tau1)*dx

    a[7][1] = - inner(xi_,curlBub(btest_b))*dx
    a[7][2] = - inner(curlBub(btrial_a),curlBub(btest_b))*dx
    a[7][9] = - inner(gamma,curlBub(btest_b))*dx

    a[8][5] = - dot(div(sig0),v)*dx
    a[8][6] = - dot(div(sig1),v)*dx

    a[9][5] = - inner(sig0,delta)*dx
    a[9][6] = - inner(sig1,delta)*dx
    a[9][7] = - inner(curlBub(btrial_b),delta)*dx

    if xi_enriched:
        # the bubble part of xi is already contained in DG_{k+2}: decouple its block, so that it vanishes
        for (i_,j_) in [(1,2),(2,1),(2,3),(2,5),(2,6),(2,7),(3,2),(5,2),(6,2),(7,2)]:
            a[i_][j_] = Constant(0.)*a[i_][j_]

    for (lv, kv, av, cv) in combos:
        tic = time.time()
        lmbda.assign(lv); kappa0.assign(kv); alpha.assign(av); c0.assign(cv)

        # data depending on the parameters
        sig0_ex = project(as_vector((sig_ex[0,0],sig_ex[0,1])),Hsig_aux)
        sig1_ex = project(as_vector((sig_ex[1,0],sig_ex[1,1])),Hsig_aux)
        gNvec = Trace(project(eta_ex, Heta), bmesh)
        u_D   = project(u_ex, Hu)

        bc_sig0 = DirichletBC(Hsig_aux, sig0_ex, facet_mesh, Sigma)
        bc_sig1 = DirichletBC(Hsig_aux, sig1_ex, facet_mesh, Sigma)
        BCs_use = block_bcs_to_monolithic_map({5: [bc_sig0], 6: [bc_sig1]}, Hh)

        l[0]    = dot(chi,n)*p_D*ds(Sigma)
        l[3]    = - g_ex*q*dx
        l[4]    = - dot(gNvec,n_)*psi*dx_(Gamma)
        l[5]    = - dot(tau0*n,u_D)*ds(Gamma)
        l[6]    = - dot(tau1*n,u_D)*ds(Gamma)
        l[7]    = - dot(curlBub(btest_b)*n,u_D)*ds(Gamma)
        l[8]    = dot(ff_ex,v)*dx

        # ******* assembling and solving ******* #
        AA, bb = ii_assemble(a), ii_assemble(l)
        AA[0][4] = AA[0][4]*P; AA[4][0] = Pt*AA[4][0]
        bb[4] = Pt*bb[4]
        A_, b_ = ii_convert(AA), ii_convert(bb)
        A_, b_ = apply_bc(A_, b_, bcs=BCs_use)

        Sol   = ii_Function(Hh)
        algres = solve_checked(A_, Sol.vector(), b_)

        eta_h, xi_h_, ba_h, p_h, phi_h, sig0_h, sig1_h, bb_h, u_h, gam_h = Sol
        xi_h  = xi_h_ + curlBub(ba_h)
        sig_h = as_tensor((sig0_h,sig1_h)) + curlBub(bb_h)
        gamma_h = tensorify_skew(gam_h)

        # ********* absolute errors and norms of the exact solutions ****** #
        L2 = lambda f: pow(assemble(inner(f,f)*dx),0.5)
        err = {'eta': L2(eta_ex-eta_h) + L2(div(eta_ex)-div(eta_h)),
               'xi':  L2(xi_ex-xi_h),
               'p':   L2(p_ex-p_h),
               'sig': L2(sig_ex-sig_h) + L2(div(sig_ex)-div(sig_h)),
               'u':   L2(u_ex-u_h),
               'gam': L2(gamma_ex-gamma_h)}
        nrm = {'eta': L2(eta_ex) + L2(div(eta_ex)),
               'xi':  L2(xi_ex),
               'p':   L2(p_ex),
               'sig': L2(sig_ex) + L2(div(sig_ex)),
               'u':   L2(u_ex),
               'gam': L2(gamma_ex)}
        err['phi'], nrm['phi'] = fractional_positive_norm_00(phi_ex_, phi_h, s=0.5)

        row = {'lmbda': lv, 'kappa0': kv, 'alpha': av, 'c0': cv, 'level': nk, 'dofs': ndofs, 'h': h, 'ht': ht}
        for nm in names:
            row['e_'+nm] = float(err[nm]); row['rel_'+nm] = float(err[nm]/nrm[nm])
        row['rel_total'] = sum(row['rel_'+nm] for nm in names)

        # ********* parameter-weighted norms ****** #
        # cstar = c0 + n alpha^2/(2mu + n lambda) is the effective storage, and varpi = cstar + kappa0 weights the flow
        # unknowns: ||eta||^2 = ||kappa^{-1/2} eta||^2 + varpi^{-1}||div eta||^2, ||p||^2 = varpi ||p||^2,
        # ||phi||^2 = varpi ||phi||_{1/2,00}^2; ||xi||^2 = (C xi, xi), ||sig||^2 = (C^{-1} sig, sig) + (2mu)^{-1}||div sig||^2,
        # ||u||^2 = 2mu ||u||^2, ||gam||^2 = 2mu ||gam||^2. Errors are relative to the norm of the whole solution.
        mv = float(mu); cstar = cv + ndim*av**2/(2.*mv + ndim*lv); varpi = cstar + kv
        def wnorms(e_eta, e_xi, e_p, e_phi, e_sig, e_u, e_gam):
            return {'eta': assemble(1.0/kappa*inner(e_eta,e_eta)*dx) + assemble(div(e_eta)**2*dx)/varpi,
                    'xi':  assemble(inner(CTimes(e_xi),e_xi)*dx),
                    'p':   varpi*assemble(e_p**2*dx),
                    'phi': varpi*e_phi**2,
                    'sig': assemble(inner(CinvTimes(e_sig),e_sig)*dx) + 0.5/mv*assemble(inner(div(e_sig),div(e_sig))*dx),
                    'u':   2.*mv*assemble(inner(e_u,e_u)*dx),
                    'gam': 2.*mv*assemble(inner(e_gam,e_gam)*dx)}
        werr = wnorms(eta_ex-eta_h, xi_ex-xi_h, p_ex-p_h, err['phi'], sig_ex-sig_h, u_ex-u_h, gamma_ex-gamma_h)
        wnrm = wnorms(eta_ex, xi_ex, p_ex, nrm['phi'], sig_ex, u_ex, gamma_ex)
        Nw = np.sqrt(sum(wnrm.values()))
        for nm in names:
            row['w_'+nm] = float(np.sqrt(max(werr[nm],0.))/Nw)
        row['w_total'] = float(np.sqrt(sum(max(werr[nm],0.) for nm in names))/Nw)

        # algebraic residual of the solved system, relative to the rhs (detects loss of accuracy in the solver)
        row['algres'] = algres
        results.append(row)

        print('lam=%.0e kap=%.0e alp=%.0e c0=%.0e | ' % (lv,kv,av,cv) +
              ' '.join('%s=%.2e' % (nm, row['rel_'+nm]) for nm in names) +
              ' | total=%.2e res=%.1e (%.1fs)' % (row['rel_total'], row['algres'], time.time()-tic), flush=True)
        print('      weighted: ' + ' '.join('%s=%.2e' % (nm, row['w_'+nm]) for nm in names) +
              ' | total=%.2e' % row['w_total'], flush=True)

    rplot.write_long_csv(results, csvfile) # save after every level

# ********* pgfplots data + figures (only when the full sweep was done in this process) ****** #
if not tag:
    rplot.write_tables_and_figures(results, outdir)
