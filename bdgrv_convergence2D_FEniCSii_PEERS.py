from dolfin import *
from xii import *
import sympy2fenics as sf
from scipy.linalg import eigh
import numpy as np

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4

def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

def tensorify_skew(r):
        return as_tensor((( 0,r[0]),
                          (-r[0],0)))

# ******* Model parameters ****** #    
ndim = 2
I = Identity(ndim)
e0, e1 = Constant((1, 0)), Constant((0, 1))
mu  = Constant(1.0)
lmbda = Constant(1.0)
c0    = Constant(0.1)
alpha = Constant(0.25)


symmgr = lambda v: sym(grad(v))
skewgr = lambda v: grad(v) - symmgr(v)
curlBub = lambda vec: as_tensor([[vec[0].dx(1), -vec[0].dx(0)], [vec[1].dx(1), -vec[1].dx(0)]])
CinvTimes = lambda s: 0.5/mu * s - lmbda/(2.*mu*(ndim*lmbda+2.*mu))*tr(s)*I

def fractional_norm(f, fh, s, degree_rise=1, gamma=10):
    '''||f-fh||_s in H^s DG norm with penalry gamma'''
    # consider a higher order DG space for the error
    Q = fh.function_space()
    Qelm = Q.ufl_element()
    mesh = Q.mesh()

    Qe = FunctionSpace(mesh, Qelm.reconstruct(degree=Qelm.degree()+degree_rise))
    # penalty
    gamma = Constant(gamma*Qe.ufl_element().degree())
    
    # Fractional Laplacian there
    p, q = TrialFunction(Qe), TestFunction(Qe)
    # full H1 inner product to get something invertible, but if BCs are of concern we should modify
    n = FacetNormal(mesh)
    hE = CellDiameter(mesh)

    a = (inner(grad(p), grad(q))*dx + inner(p, q)*dx
         - inner(avg(grad(p)), jump(q, n))*dS
         - inner(avg(grad(q)), jump(p, n))*dS
         + (avg(gamma)/avg(hE))*inner(jump(p, n), jump(q, n))*dS)

    m = inner(p, q)*dx

    A, M = [assemble(foo).array() for foo in (a, m)]
    
    Lmbda, U = eigh(A, M)
    assert np.all(Lmbda > 0), 'Increase penalty?'

    W = M@U
    # Fractional inner product is induced by
    Hs = W@np.diag(Lmbda**s)@W.T
    # Compute coefficient of the error vector 
    error = interpolate(fh, Qe).vector()
    error.axpy(-1, project(f, Qe).vector())
    error = error.get_local()

    norm = np.inner(error, Hs@error)
    return np.sqrt(norm)    



# ******* Exact solutions for error analysis ****** #
u_str = '(0.05*cos(1.5*pi*(x+y)),0.05*sin(1.5*pi*(x-y)))'
p_str = 'sin(pi*x)*sin(pi*y)'
kappa_str = 'exp(-x*y)'

# polynomial degree
l=0
set_log_level(40)

nkmax = 6; k = 0

hh = []; hht = []; nn = [];
eu = []; ru = [];
esig = []; rsig = []; ep = []; rp = [];
exi = []; rxi = []; egam = []; rgam = [];
eeta = []; reta = []; ephi = []; rphi = []; it = []

reta.append(0.); rsig.append(0.0); 
ru.append(0.0); rgam.append(0.)
rxi.append(0.0); rphi.append(0.0); 
rp.append(0.)

for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)

    # need two meshes / one for the outer boundary
    nps = pow(2,nk+1)+2; npst = pow(2,nk)+1
    mesh = UnitSquareMesh(nps,nps)
    mesht = UnitSquareMesh(npst,npst)

    facet_mesh = MeshFunction('size_t', mesh, 1, 0)
    facet_mesht = MeshFunction('size_t', mesht, 1, 0)

    subdomains = {1: CompiledSubDomain('near(x[0], 0)'),
                  2: CompiledSubDomain('near(x[0], 1)'),
                  3: CompiledSubDomain('near(x[1], 0)'),
                  4: CompiledSubDomain('near(x[1], 1)')}

    [subdomain.mark(facet_mesh, tag) for tag, subdomain in subdomains.items()]
    [subdomain.mark(facet_mesht, tag) for tag, subdomain in subdomains.items()]

    Gamma = (1,2) # on which we will define the Lagrange multiplier 
    Sigma = (3,4)

    # creating sub-boundary mesh only in the needed part
    bmesht = EmbeddedMesh(facet_mesht, Gamma)
    bmesht_subd = bmesht.marking_function
    
    n = FacetNormal(mesh)
    n_ = OuterNormal(bmesht, [0.5, 0.5]) #Constant((-1, 0)) # only the LM boundary?
    
    hh.append(mesh.hmax())
    hht.append(mesht.hmax())

    # this measure is for the boundary integrals
    dx_ = Measure('dx', domain=bmesht, subdomain_data=bmesht_subd)

    # this is for the boundary integrals that do not require the Lagrange multiplier
    ds = Measure('ds', domain=mesh, subdomain_data=facet_mesh)
    
    # ********* Finite dimensional spaces ********* #
    Heta    = FunctionSpace(mesh, "RT", k+1)
    Hxi_aux = VectorFunctionSpace(mesh, "DG", k)
    Bub     = VectorFunctionSpace(mesh,'B', k + 3)
    Hp      = FunctionSpace(mesh, "DG", k)
    Hphi    = FunctionSpace(bmesht, 'CG', k+1) # we do need continuity in 2D, right?
    Hsig_aux= FunctionSpace(mesh, "RT", k+1)
    Hu      = VectorFunctionSpace(mesh, "DG", k)
    Hgam    = VectorFunctionSpace(mesh, "CG", k+1) 

    
    Hh = [Heta,Hxi_aux,Hxi_aux,Bub,Hp,Hphi,Hsig_aux,Hsig_aux,Bub,Hu,Hgam]

    print ("....... Total DoFs = ", Heta.dim() + Hxi_aux.dim() + Hxi_aux.dim() + Bub.dim() + Hp.dim() + Hphi.dim() + Hsig_aux.dim() + Hsig_aux.dim() + Bub.dim()  + Hu.dim() + Hgam.dim() )

    Hu_aux = VectorFunctionSpace(mesh, "CG", 1)

    nn.append(Heta.dim() + Hxi_aux.dim() + Hxi_aux.dim() + Bub.dim() + Hp.dim() + Hphi.dim() + Hsig_aux.dim() + Hsig_aux.dim() + Bub.dim()  + Hu.dim() + Hgam.dim())

    Sol =  ii_Function(Hh)
    
    eta,  xi0,  xi1, btrial_a, p, phi, sig0, sig1, btrial_b, u, gam_ = Sol 
    chi, rho0, rho1,  btest_a, q, psi, tau0, tau1,  btest_b, v, del_ = map(TestFunction, Hh)

    # this is important to interpolate integrals
    Tn_eta, Tn_chi = Trace(dot(eta,n), bmesht), Trace(dot(chi,n), bmesht)

    gamma = tensorify_skew(gam_); delta = tensorify_skew(del_)

    # ****** Tensorification of sig, tau, xi, rho ******* #
    '''
    maybe to try simply doing
    sig = as_tensor((sig0,sig1)) + curlB(btrial)
    tau = as_tensor((tau0,tau1)) + curlB(btest)
    ? it seems not to work because it mixes the blocks of the system matrix. OK for after solving
    for the moment we are doing the same for xi since we need to sum it bubbles. 
    But perhaps we can do this differently?
    '''
    sig0, sig1 = outer(e0, sig0), outer(e1, sig1)
    tau0, tau1 = outer(e0, tau0), outer(e1, tau1)
    xi0, xi1 = outer(e0, xi0), outer(e1, xi1)
    rho0, rho1 = outer(e0, rho0), outer(e1, rho1)
    
    # ******* Instantiating exact solutions and variable coefficients ******* #

    # primal unknowns                  
    u_ex    = Expression(str2exp(u_str), degree=6, domain=mesh)
    p_ex    = Expression(str2exp(p_str), degree=6, domain=mesh)
    kappa   = Expression(str2exp(kappa_str), degree=6, domain=mesh)

    # note that the exact phi coincides with p on the boundary:
    phi_ex_ = Expression(str2exp(p_str), degree=6, domain=bmesht)

    # mixed variables
    eta_ex  = kappa*grad(p_ex)
    xi_ex   = symmgr(u_ex)
    xi0_ex = as_vector((xi_ex[0,0],xi_ex[0,1]))
    xi1_ex = as_vector((xi_ex[1,0],xi_ex[1,1]))

    sig_ex  = 2.*mu*xi_ex + lmbda*tr(xi_ex)*I - alpha*p_ex*I
    sig0_ex = as_vector((sig_ex[0,0],sig_ex[0,1]))
    sig1_ex = as_vector((sig_ex[1,0],sig_ex[1,1]))
    
    gamma_ex= skewgr(u_ex)

    ff_ex = project(-div(sig_ex),Hu)
    g_ex  = project(c0*p_ex + alpha*tr(xi_ex)-div(eta_ex),Hp)

    u_D     = interpolate(u_ex,Hu_aux)


    gN = project(dot(eta_ex,n_),Hphi) # not sure if this will work? check older code
    
    # ******* variational forms ******* #
    ''' this has to be entered row-wise chi, rho0, rho1, btesta, q, psi, tau0, tau1, btestb, v, del'''

    FF = [kappa^(-1)*dot(eta,chi)*dx + p*div(chi)*dx - Tn_chi*phi*dx_(Gamma), 
            inner(sig0+sig1+curlBub(btrial_b),rho0)*dx \
            + inner(CinvTimes(xi0+xi1+curlBub(btrial_a)),rho0)*dx - alpha*p*tr(rho0)*dx,
            inner(sig0+sig1+curlBub(btrial_b),rho1)*dx \
            + inner(CinvTimes(xi0+xi1+curlBub(btrial_a)),rho1)*dx - alpha*p*tr(rho1)*dx,
            inner(sig0+sig1+curlBub(btrial_b),curlBub(btest_a))*dx \
            + inner(CinvTimes(xi0+xi1+curlBub(btrial_a)),curlBub(btest_a))*dx - alpha*p*tr(curlBub(btest_a))*dx,
            - alpha*tr(xi0+xi1+curlBub(btrial_a))*q*dx + dot(div(eta),q)*dx - c0*p*q*dx + g_ex*q*dx,
            Tn_eta*psi*dx_(Gamma)- gN*psi*dx_(Gamma),    
            - inner(xi0+xi1+curlBub(btrial_a),tau0)*dx - dot(u,div(tau0))*dx - inner(gamma,tau0)*dx + dot(tau0*n,u_D)*ds(Gamma), 
            - inner(xi0+xi1+curlBub(btrial_a),tau1)*dx - dot(u,div(tau1))*dx - inner(gamma,tau1)*dx + dot(tau1*n,u_D)*ds(Gamma), 
            - inner(xi0+xi1+curlBub(btrial_a),curlBub(btest_b))*dx - dot(u,div(curlBub(btest_b)))*dx - inner(gamma,curlBub(btest_b))*dx + dot(curlBub(btest_b)*n,u_D)*ds(Gamma), 
            - dot(div(sig0+sig1+curlBub(btrial_b)),v)*dx - dot(ff_ex,v)*dx, 
            - inner(sig0+sig1+curlBub(btrial_b),delta)*dx]

    '''old one:
    FF = [lam*mu(phi)*inner(tt,ss)*dx - inner(sig0+sig1+curlBub(bsol),ss)*dx - inner(outer(u,u),ss)*dx,
          inner(tt,tau0)*dx + inner(gamma,tau0)*dx + dot(u,div(tau0))*dx + tr(tau0)*a_trial*dx - dot(tau0*n,u_D)*ds,
          inner(tt,tau1)*dx + inner(gamma,tau1)*dx + dot(u,div(tau1))*dx + tr(tau1)*a_trial*dx - dot(tau1*n,u_D)*ds,
          inner(tt,curlBub(btest))*dx + inner(gamma,curlBub(btest))*dx + dot(u,div(curlBub(btest)))*dx + tr(curlBub(btest))*a_trial*dx - dot(curlBub(btest)*n,u_D)*ds,
          dot(div(sig0+sig1+curlBub(bsol)),v)*dx - eta(phi)*dot(u,v)*dx + dot(f(phi)*g+ff_ex,v)*dx,
          inner(sig0+sig1+curlBub(bsol),delta)*dx,
          kappa*rho*dot(grad(phi),grad(psi))*dx + dot(u,grad(phi+s(phi)))*psi*dx + chi*Tpsi*dx_(lm_tags) - mm_ex*psi*dx + dot(flux_ex,n)*psi*ds(Sigma),
          (Tphi-Tphi_ex)*xi*dx_(Gamma),
          (tr(sig0+sig1+curlBub(bsol)+outer(u,u))-trs_ex) * a_test * dx]
    '''
    JJ   = block_jacobian(FF, Sol)

    # ******* nonlinear solver *********** #

    solver = PETScLUSolver(method = "mumps")
    dSol   = ii_Function(Hh)
    
    tol = 1.0E-6; maxiter = 11; eps = 1.0; niter = 0; niters = 0
    while eps > tol and niter < maxiter:
        niter += 1
        A_, b_ = (ii_convert(ii_assemble(x)) for x in (JJ, FF))
        niters = solver.solve(A_, dSol.vector(), b_)
        eps = sqrt(sum(x.norm('l2')**2 for x in dSol.vectors()))

        print('|dSol| = %g |A|= %g |b| = %g | niters %d' % (
            eps, A_.norm('linf'), b_.norm('l2'), niter
        ))

        for i in range(len(Hh)):
            Sol[i].vector().axpy(-1, dSol[i].vector())

    it.append(niter)
    tt_, sig0, sig1, bsolh, u, gam_, phi, chi, a_trial = Sol
    tt=tensorify_symm_zerotr(tt_)
    sig = as_tensor((sig0,sig1)) + curlBub(bsolh)
    gamma = tensorify_skew(gam_)
    p = project(-1./ndim*tr(sig + outer(u,u)), Ph)

    Th = TensorFunctionSpace(mesh,'DG',l)
    sigh = project(sig,Th)
    gamh = project(gamma,Th)
    th = project(tt,Th)


    mome_ = project(eta(phi)*u - div(sigh) - f(phi)*g - eta(phi_ex)*u_ex + div(sig_ex) + f(phi_ex)*g,Hu)
    mome = norm(mome_.vector(),'linf')

    print("mome = ", mome)
    

    '''

    th.rename("t", "")
    u.rename("u", "")
    p.rename("p", "")
    gamh.rename("gam", "")
    sigh.rename("sig", "")
    phi.rename("phi", "")
    chi.rename("chi", "")

    File('outputs/out-ex01-fenicsiibabuska_u.pvd') << u
    File('outputs/out-ex01-fenicsiibabuska_phi.pvd') << phi
    File('outputs/out-ex01-fenicsiibabuska_chi.pvd') << chi
    File('outputs/out-ex01-fenicsiibabuska_sig.pvd') << sigh
    File('outputs/out-ex01-fenicsiibabuska_gam.pvd') << gamh
    File('outputs/out-ex01-fenicsiibabuska_t.pvd') << th
    File('outputs/out-ex01-fenicsiibabuska_p.pvd') << p

    '''


    # ********* Computing errors in weighted norms ****** #
    varrho = Constant(4./3.)
    E_t   = pow(assemble((tt-tt_ex)**2*dx),0.5)
    E_sig = pow(assemble((sig_ex-sig)**2*dx),0.5) \
            + pow(assemble(((div(sig_ex)-div(sig))**2)**(0.5*varrho)*dx),1./varrho)
    E_u   = pow(assemble(dot(u-u_ex,u-u_ex)**2*dx),0.25)
    E_gam = pow(assemble((gamma-gamma_ex)**2*dx),0.5)
    E_p   = pow(assemble((p-p_ex)**2*dx),0.5)

    et.append(float(E_t)); esig.append(float(E_sig));
    eu.append(float(E_u)); egam.append(float(E_gam));
    ephi.append(errornorm(phi_ex, phi, 'H1'));
    echi.append(fractional_norm(chi_ex, chi, s=-0.5));
    ep.append(float(E_p))    
    
    if(nk>0):
        rt.append(ln(et[nk]/et[nk-1])/ln(hh[nk]/hh[nk-1]))
        rsig.append(ln(esig[nk]/esig[nk-1])/ln(hh[nk]/hh[nk-1]))
        ru.append(ln(eu[nk]/eu[nk-1])/ln(hh[nk]/hh[nk-1]))
        rgam.append(ln(egam[nk]/egam[nk-1])/ln(hh[nk]/hh[nk-1]))
        rphi.append(ln(ephi[nk]/ephi[nk-1])/ln(hh[nk]/hh[nk-1]))
        rchi.append(ln(echi[nk]/echi[nk-1])/ln(hht[nk]/hht[nk-1]))
        rp.append(ln(ep[nk]/ep[nk-1])/ln(hh[nk]/hh[nk-1]))
        

# ********* Generating error history ****** #
print('=======================================================================')
print('  DoFs     h    ht   e(t)   r(t)   e(sig) r(sig)  e(u)  r(u)  e(gam)  r(gam)  e(phi) r(phi) e(chi) r(chi)  e(p)   r(p)  it')
print('=======================================================================')
for nk in range(nkmax):
    print('{:6d} & {:.3f} & {:.3f} & {:1.2e} & {:.2f} & {:1.2e} & {:.2f} & {:1.2e} & {:.2f} & {:1.2e} & {:.2f} & {:1.2e} & {:.2f} & {:1.2e} & {:.2f} & {:1.2e} & {:.2f} & {:2d}'.format(nn[nk], hh[nk], hht[nk], et[nk], rt[nk], esig[nk], rsig[nk], eu[nk], ru[nk], egam[nk], rgam[nk], ephi[nk], rphi[nk], echi[nk], rchi[nk], ep[nk], rp[nk], it[nk]))
print('=======================================================================')