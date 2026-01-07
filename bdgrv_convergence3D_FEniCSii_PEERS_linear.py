from dolfin import *
import os
from xii import *
import sympy2fenics as sf
from scipy.linalg import eigh
import numpy as np
from petsc4py import PETSc


parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4

PETScOptions.set("mat_mumps_icntl_14", 1000) # memory estimate
PETScOptions.set("mat_mumps_icntl_13", 1) # accept almost zero pivots
PETScOptions.set("mat_mumps_cntl_1", 1e-8)

fileO = XDMFFile("outputs/out-3D-convergence-bulk.xdmf")
fileO.parameters["functions_share_mesh"] = True
fileO.parameters['rewrite_function_mesh'] = True
fileO.parameters["flush_output"] = True

fileOM = XDMFFile("outputs/out-3D-convergence-phi.xdmf")
fileOM.parameters["rewrite_function_mesh"] = True
fileOM.parameters["flush_output"] = True


def fproject(expr, V):
    ''' explicit projection function (needed because default projection is done with UMFPACK, which crashes for large systems)'''
    u = TrialFunction(V)
    v = TestFunction(V)

    A = assemble(inner(u, v)*dx)

    solver = PETScKrylovSolver('cg')
    solver.set_operator(A)

    b = assemble(inner(expr, v)*dx)
    uh = Function(V)
    solver.solve(uh.vector(), b)

    return uh
    
def fractional_positive_norm_00(f, fh, s):
    '''||f-fh||_s in H^s norm. We use a CG space with Dirichlet BCs to reflect H^s_{00}'''
    Q = fh.function_space()
    Qelm = Q.ufl_element()
    mesh = Q.mesh()

    Qe = FunctionSpace(mesh, Qelm)
    bc = DirichletBC(Qe, Constant(0.0), 'on_boundary')
    
    # Fractional Laplacian 
    p, q = TrialFunction(Qe), TestFunction(Qe)
    a = inner(grad(p), grad(q))*dx
    m = inner(p, q)*dx

    A, M = [assemble(foo).array() for foo in (a, m)]
    # Apply homogeneous Dirichlet BC by removing rows/cols corresponding
    # to boundary dofs from the assembled numpy matrices (no blocks here).
    bc_vals = bc.get_boundary_values()
    if bc_vals:
        n = A.shape[0]
        mask = np.ones(n, dtype=bool)
        mask[list(bc_vals.keys())] = False
        A = A[mask][:, mask]
        M = M[mask][:, mask]

    Lmbda, U = eigh(A, M)
    assert np.all(Lmbda > 0), 'Increase penalty?'

    W = M@U
    # Fractional inner product is induced by
    Hs = W@np.diag(Lmbda**s)@W.T
    # Compute coefficient of the error vector and restrict to free dofs
    error = interpolate(fh, Qe).vector()
    error.axpy(-1, fproject(f, Qe).vector())
    error = error.get_local()
    if bc_vals:
        error = error[mask]

    norm = np.inner(error, Hs@error)
    return np.sqrt(norm)    
# end fractional_positive_norm_00

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

def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

def tensorify_skew3D(r):
        return as_tensor(((    0,  r[0], r[1]),
                          (-r[0],     0, r[2]),
                          (-r[1], -r[2],    0)))

# ******* Model parameters ****** #    
ndim = 3
I = Identity(ndim)
e0, e1, e2 = Constant((1, 0, 0)), Constant((0, 1, 0)), Constant((0, 0, 1))
mu  = Constant(1.0)
lmbda = Constant(1.0)
c0    = Constant(0.1)
alpha = Constant(0.1)
uinf = Constant(0.1)

symmgr = lambda v: sym(grad(v))
skewgr = lambda v: grad(v) - symmgr(v)

curlTen = lambda ten: as_tensor([[ten[0,2].dx(1)-ten[0,1].dx(2),ten[0,0].dx(2)-ten[0,2].dx(0),ten[0,1].dx(0)-ten[0,0].dx(1)],
                                 [ten[1,2].dx(1)-ten[1,1].dx(2),ten[1,0].dx(2)-ten[1,2].dx(0),ten[1,1].dx(0)-ten[1,0].dx(1)],
                                 [ten[2,2].dx(1)-ten[2,1].dx(2),ten[2,0].dx(2)-ten[2,2].dx(0),ten[2,1].dx(0)-ten[2,0].dx(1)]])

CTimes = lambda s: 2.*mu*s + lmbda*tr(s)*I

# ******* Exact solutions for error analysis ****** #
u_str = '(uinf*(sin(x)*cos(y)*cos(z)+x*x*0.5/lmbda),uinf*(-2*cos(x)*sin(y)*cos(z)+y*y*0.5/lmbda),uinf*(cos(x)*cos(y)*sin(z)+z*z*0.5/lmbda))'
p_str = 'sin(pi*x)*sin(pi*y)*sin(pi*z)'
kappa_str = 'exp(-x*y*z)'

# polynomial degree
k=0; nkmax = 5
set_log_level(40)

hh = []; hht = []; nn = []; eu = []; ru = []
esig = []; rsig = []; ep = []; rp = []
exi = []; rxi = []; egam = []; rgam = []
eeta = []; reta = []; ephi = []; rphi = []

reta.append(0.); rsig.append(0.0)
ru.append(0.0); rgam.append(0.)
rxi.append(0.0); rphi.append(0.0) 
rp.append(0.)

for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)

    # need two meshes / one for the outer boundary
    nps = pow(2,nk+1); npst = pow(2,nk)+1
    mesh = UnitCubeMesh(nps,nps,nps)
    mesht = UnitCubeMesh(npst,npst,npst)

    facet_mesh = MeshFunction('size_t', mesh, ndim-1, 0)
    facet_mesht = MeshFunction('size_t', mesht, ndim-1, 0)

    boundaries = {31: CompiledSubDomain('near(x[0], 0) || near(x[1], 0) || near(x[2],0)'),
                  32: CompiledSubDomain('near(x[0], 1) || near(x[1], 1) || near(x[2],1)')}

    [subb.mark(facet_mesh, tag) for tag, subb in boundaries.items()]
    [subb.mark(facet_mesht, tag) for tag, subb in boundaries.items()]

    Gamma = 31 # left-bottom-front: on which we will define the Lagrange multiplier 
    Sigma = 32

    # creating sub-boundary mesh only in the needed part
    bmesht = EmbeddedMesh(facet_mesht, Gamma)
    bmesht_subd = bmesht.marking_function
    
    n = FacetNormal(mesh)
    n_ = OuterNormal(bmesht, [0.5, 0.5, 0.5])
    
    hh.append(mesh.hmax())
    hht.append(mesht.hmax())

    # this measure is for the boundary integrals
    dx_ = Measure('dx', domain=bmesht, subdomain_data=bmesht_subd)

    # this is for the boundary integrals that do not require the Lagrange multiplier
    ds = Measure('ds', domain=mesh, subdomain_data=facet_mesh)
    
    # ********* Finite dimensional spaces ********* #
    Heta    = FunctionSpace(mesh, "RT", k+1)
    Hxi_aux = TensorFunctionSpace(mesh, "DG", k)
    Bub     = TensorFunctionSpace(mesh,'B', k + 4)
    Hp      = FunctionSpace(mesh, "DG", k)
    Hphi    = FunctionSpace(bmesht, 'CG', k+1) 
    Hsig_aux= FunctionSpace(mesh, "RT", k+1)
    Hu      = VectorFunctionSpace(mesh, "DG", k)
    Hgam    = VectorFunctionSpace(mesh, "CG", k+1, dim=3)
    
    Hh = [Heta,Hxi_aux,Bub,Hp,Hphi,Hsig_aux,Hsig_aux,Hsig_aux,Bub,Hu,Hgam]
    ndofs = sum([spa.dim() for spa in Hh])
    print ("....... Total DoFs = ", ndofs)
    nn.append(ndofs)
    
    eta,  xi_, btrial_a, p, phi, sig0, sig1, sig2, btrial_b, u, gam_ = map(TrialFunction, Hh) 
    chi, rho_, btest_a, q, psi, tau0, tau1, tau2, btest_b, v, del_ = map(TestFunction, Hh)

    # this is important to interpolate integrals
    T_eta, T_chi = Trace(eta, bmesht), Trace(chi, bmesht)

    gamma = tensorify_skew3D(gam_); delta = tensorify_skew3D(del_)

    # ****** Tensorification of sig, tau, xi, rho ******* #
    '''
    maybe to try simply doing
    sig = as_tensor((sig0,sig1)) + curlB(btrial)
    tau = as_tensor((tau0,tau1)) + curlB(btest)
    ? 
    it seems not to work because it mixes the blocks of the system matrix. OK for after solving
    '''
    sig0, sig1, sig2 = outer(e0, sig0), outer(e1, sig1), outer(e2,sig2)
    tau0, tau1, tau2 = outer(e0, tau0), outer(e1, tau1), outer(e2,tau2)
    
    # ******* Instantiating exact solutions and variable coefficients ******* #

    # primal unknowns                  
    u_ex    = Expression(str2exp(u_str), lmbda=lmbda, uinf=uinf, degree=6, domain=mesh)
    p_ex    = Expression(str2exp(p_str), degree=6, domain=mesh)
    kappa   = Expression(str2exp(kappa_str), degree=6, domain=mesh)
    # the exact phi coincides with p on the boundary:
    phi_ex_ = Expression(str2exp(p_str), degree=6, domain=bmesht)

    # mixed variables
    eta_ex  = kappa*grad(p_ex)
    xi_ex   = symmgr(u_ex)

    sig_ex  = 2.*mu*xi_ex + lmbda*tr(xi_ex)*I - alpha*p_ex*I
    sig0_ex = fproject(as_vector((sig_ex[0,0],sig_ex[0,1],sig_ex[0,2])),Hsig_aux)
    sig1_ex = fproject(as_vector((sig_ex[1,0],sig_ex[1,1],sig_ex[1,2])),Hsig_aux)
    sig2_ex = fproject(as_vector((sig_ex[2,0],sig_ex[2,1],sig_ex[2,2])),Hsig_aux)
    
    gamma_ex= skewgr(u_ex)

    ff_ex = -div(sig_ex)
    g_ex  = c0*p_ex + alpha*tr(xi_ex)-div(eta_ex)

    p_D   = interpolate(p_ex,Hp)
    u_D   = interpolate(u_ex,Hu) 

    gNvec = Trace(fproject(eta_ex, Heta), bmesht) 

    # ******* essential BCs for sigma: ok just with the RT since the bubbles vanish on the boundary ******* #
    # apply Dirichlet BCs on the marked facets in `Sigma` 
    bc_sig0 = DirichletBC(Hsig_aux, sig0_ex, facet_mesh, Sigma)
    bc_sig1 = DirichletBC(Hsig_aux, sig1_ex, facet_mesh, Sigma)
    bc_sig2 = DirichletBC(Hsig_aux, sig2_ex, facet_mesh, Sigma)
    BCs_dict = {5: [bc_sig0], 6: [bc_sig1], 7: [bc_sig2]} # 5,6,7 are the positions in the product space 
    BCs_use = block_bcs_to_monolithic_map(BCs_dict, Hh)

    # ******* variational forms ******* eta,  xi_, btrial_a, p, phi, sig0, sig1, sig2, btrial_b, u, gamma
    ''' this has to be entered row-wise chi, rho_, btesta, q, psi, tau0, tau1, tau2, btestb, v, delta'''
    a = block_form(Hh,2); l = block_form(Hh,1)
    a[0][0] = 1.0/kappa*dot(eta,chi)*dx
    a[0][3] = p*div(chi)*dx
    a[0][4] = - dot(T_chi,n_)*phi*dx_(Gamma)

    a[1][1] = inner(CTimes(xi_),rho_)*dx
    a[1][2] = inner(CTimes(curlTen(btrial_a)),rho_)*dx
    a[1][3] = -alpha*p*tr(rho_)*dx
    a[1][5] = -inner(sig0,rho_)*dx
    a[1][6] = -inner(sig1,rho_)*dx
    a[1][7] = -inner(sig2,rho_)*dx
    a[1][8] = -inner(curlTen(btrial_b),rho_)*dx

    a[2][1] = inner(CTimes(xi_),curlTen(btest_a))*dx
    a[2][2] = inner(CTimes(curlTen(btrial_a)),curlTen(btest_a))*dx
    a[2][3] = -alpha*p*tr(curlTen(btest_a))*dx
    a[2][5] = -inner(sig0,curlTen(btest_a))*dx
    a[2][6] = -inner(sig1,curlTen(btest_a))*dx
    a[2][7] = -inner(sig2,curlTen(btest_a))*dx
    a[2][8] = -inner(curlTen(btrial_b),curlTen(btest_a))*dx

    a[3][0] = dot(div(eta),q)*dx
    a[3][1] = - alpha*tr(xi_)*q*dx
    a[3][2] = - alpha*tr(curlTen(btrial_a))*q*dx
    a[3][3] = - c0*p*q*dx

    a[4][0] = - dot(T_eta,n_)*psi*dx_(Gamma)

    a[5][1] = - inner(xi_,tau0)*dx
    a[5][2] = - inner(curlTen(btrial_a),tau0)*dx
    a[5][9] = - dot(u,div(tau0))*dx
    a[5][10] = - inner(gamma,tau0)*dx   

    a[6][1] = - inner(xi_,tau1)*dx
    a[6][2] = - inner(curlTen(btrial_a),tau1)*dx        
    a[6][9] = - dot(u,div(tau1))*dx
    a[6][10] = - inner(gamma,tau1)*dx

    a[7][1] = - inner(xi_,tau2)*dx
    a[7][2] = - inner(curlTen(btrial_a),tau2)*dx        
    a[7][9] = - dot(u,div(tau2))*dx
    a[7][10] = - inner(gamma,tau2)*dx

    a[8][1] = - inner(xi_,curlTen(btest_b))*dx
    a[8][2] = - inner(curlTen(btrial_a),curlTen(btest_b))*dx
    a[8][9] = - dot(u,div(curlTen(btest_b)))*dx
    a[8][10] = - inner(gamma,curlTen(btest_b))*dx

    a[9][5] = - dot(div(sig0),v)*dx
    a[9][6] = - dot(div(sig1),v)*dx
    a[9][7] = - dot(div(sig2),v)*dx
    a[9][8] = - dot(div(curlTen(btrial_b)),v)*dx

    a[10][5] = - inner(sig0,delta)*dx
    a[10][6] = - inner(sig1,delta)*dx
    a[10][7] = - inner(sig2,delta)*dx
    a[10][8] = - inner(curlTen(btrial_b),delta)*dx

    l[0]    = dot(chi,n)*p_D*ds(Sigma)
    l[3]    = - g_ex*q*dx
    l[4]    = - dot(gNvec,n_)*psi*dx_(Gamma)
    l[5]    = - dot(tau0*n,u_D)*ds(Gamma)
    l[6]    = - dot(tau1*n,u_D)*ds(Gamma)
    l[7]    = - dot(tau2*n,u_D)*ds(Gamma)
    l[8]    = - dot(curlTen(btest_b)*n,u_D)*ds(Gamma)
    l[9]    = dot(ff_ex,v)*dx

    # ******* assembling forms ******* #
    A_, b_ = (ii_convert(ii_assemble(x)) for x in (a,l))
    A_, b_ = apply_bc(A_, b_, bcs=BCs_use)

    # ******* solving linear system ******* #
    solver = PETScLUSolver(method = "mumps")
    Sol   = ii_Function(Hh)
    solver.solve(A_, Sol.vector(), b_)

    # ******* extracting solutions ******* #
    eta,  xi_, btrial_a, p, phi, sig0, sig1, sig2, btrial_b, u, gam_ = Sol
    xi = xi_ + curlTen(btrial_a)
    sig = as_tensor((sig0,sig1,sig2)) + curlTen(btrial_b)
    gamma = tensorify_skew3D(gam_)

    Th = TensorFunctionSpace(mesh,'DG', k)
    sigh = fproject(sig,Th)
    gamh = fproject(gamma,Th)
    xih = fproject(xi,Th)
    
    # ******* checking momentum and mass balance ****** #
    mome_ = fproject(div(sig) + ff_ex,Hu)
    mome = norm(mome_.vector(),'linf')
    print("momentum loss = ", mome)
     
    mass_ = fproject(c0*p + alpha*tr(xi) - div(eta) - g_ex,Hp)
    mass = norm(mass_.vector(),'linf')
    print("mass loss = ", mass)

    # ******* saving solutions to file ******* #
    # we need two files since we have two different meshes
    u.rename("u", "u")
    p.rename("p", "u")
    gamh.rename("gam", "gam")
    sigh.rename("sig", "sig")
    phi.rename("phi", "phi")
    xih.rename("xi", "xi")
    sigh.rename("sig", "sig")
    gamh.rename("gam", "gam")
    eta.rename("eta", "eta")
    p.rename("p", "p")

    fileO.write(u, nk*1.0)
    fileOM.write(phi, nk*1.0)
    fileO.write(xih, nk*1.0)
    fileO.write(sigh, nk*1.0)
    fileO.write(gamh, nk*1.0)
    fileO.write(eta, nk*1.0)
    fileO.write(p, nk*1.0)

    # ********* Computing errors ****** #

    E_eta = pow(assemble((eta_ex-eta)**2*dx),0.5) \
            + pow(assemble((div(eta_ex)-div(eta))**2*dx),0.5)
    E_xi  = pow(assemble((xi-xi_ex)**2*dx),0.5)
    E_sig = pow(assemble((sig_ex-sig)**2*dx),0.5) \
            + pow(assemble((div(sig_ex)-div(sig))**2*dx),0.5)
    E_u   = pow(assemble((u-u_ex)**2*dx),0.5)
    E_gam = pow(assemble((gamma-gamma_ex)**2*dx),0.5)
    E_p   = pow(assemble((p-p_ex)**2*dx),0.5)

    eeta.append(float(E_eta)); esig.append(float(E_sig))
    eu.append(float(E_u)); egam.append(float(E_gam))
    exi.append(float(E_xi)); ep.append(float(E_p))    
    ephi.append(fractional_positive_norm_00(phi_ex_, phi, s=0.5))
    
    if(nk>0):
        reta.append(ln(eeta[nk]/eeta[nk-1])/ln(hh[nk]/hh[nk-1]))
        rxi.append(ln(exi[nk]/exi[nk-1])/ln(hh[nk]/hh[nk-1]))
        rsig.append(ln(esig[nk]/esig[nk-1])/ln(hh[nk]/hh[nk-1]))
        rp.append(ln(ep[nk]/ep[nk-1])/ln(hh[nk]/hh[nk-1]))
        rphi.append(ln(ephi[nk]/ephi[nk-1])/ln(hht[nk]/hht[nk-1]))
        ru.append(ln(eu[nk]/eu[nk-1])/ln(hh[nk]/hh[nk-1]))
        rgam.append(ln(egam[nk]/egam[nk-1])/ln(hh[nk]/hh[nk-1]))
        
        
# ********* Generating error history ****** #
print('====================================================================================================================================================')
print('  DoFs     h      ht     e(eta)   r(eta)    e(xi)   r(xi)     e(p)     r(p)     e(phi)  r(phi)    e(sig)  r(sig)     e(u)    r(u)    e(gam)   r(gam)')
print('====================================================================================================================================================')
for nk in range(nkmax):
    print('{:6d} & {:.3f} & {:.3f} & {:1.2e} & {:.2f} & {:1.2e} & {:.2f} & {:1.2e} & {:.2f} & {:1.2e} & {:.2f} & {:1.2e} & {:.2f} & {:1.2e} & {:.2f} & {:1.2e} & {:.2f}'.format(nn[nk], hh[nk], hht[nk], eeta[nk], reta[nk], exi[nk], rxi[nk],  ep[nk], rp[nk], ephi[nk], rphi[nk], esig[nk], rsig[nk], eu[nk], ru[nk], egam[nk], rgam[nk]))
print('====================================================================================================================================================')



'''
result for k=0:

  2068 & 0.866 & 0.866 & 5.63e+00 & 0.00 & 4.83e-02 & 0.00 & 2.28e-01 & 0.00 & 3.09e+01 & 0.00 & 2.28e-01 & 0.00 & 2.38e-02 & 0.00 & 6.14e-02 & 0.00
 15772 & 0.433 & 0.577 & 3.09e+00 & 0.86 & 2.83e-02 & 0.77 & 1.14e-01 & 0.99 & 2.33e-02 & 17.73 & 1.26e-01 & 0.86 & 1.16e-02 & 1.04 & 2.48e-02 & 1.31
123622 & 0.217 & 0.346 & 1.59e+00 & 0.96 & 1.59e-02 & 0.83 & 5.85e-02 & 0.97 & 4.04e-03 & 3.43 & 6.67e-02 & 0.92 & 5.77e-03 & 1.01 & 1.08e-02 & 1.20
979618 & 0.108 & 0.192 & 7.98e-01 & 0.99 & 8.98e-03 & 0.82 & 2.95e-02 & 0.99 & 1.23e-03 & 2.02 & 3.54e-02 & 0.91 & 2.88e-03 & 1.00 & 5.09e-03 & 1.09
7801018 & 0.054 & 0.102 & 4.00e-01 & 1.00 & 5.27e-03 & 0.77 & 1.48e-02 & 1.00 & 3.63e-04 & 1.92 & 1.92e-02 & 0.88 & 1.44e-03 & 1.00 & 2.51e-03 & 1.02

'''
