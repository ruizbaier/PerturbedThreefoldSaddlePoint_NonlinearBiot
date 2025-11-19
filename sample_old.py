from dolfin import *
from xii import *
import ufl
import sympy2fenics as sf

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4

def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

def tensorify_skew(r):
        return as_tensor((( 0,r[0]),
                          (-r[0],0)))

def tensorify_symm_zerotr(r):   
        return as_tensor(((r[0],r[1]),
                          (r[1],-r[0])))


from scipy.linalg import eigh
import numpy as np

def fractional_norm(f, fh, s, degree_rise=1, gamma=10):
    '''||f-fh||_s in H^s DG norm with penalry gamma'''
    # Try to do something similar as `errornorm`, i.e. consider a higher
    # order DG space for the error
    Q = fh.function_space()
    Qelm = Q.ufl_element()
    mesh = Q.mesh()

    Qe = FunctionSpace(mesh, Qelm.reconstruct(degree=Qelm.degree()+degree_rise))
    # I don't know about this penalty
    gamma = Constant(gamma*Qe.ufl_element().degree())
    
    # Fractional Laplacian there
    p, q = TrialFunction(Qe), TestFunction(Qe)
    # NOTE: I base in on helmholtz to get something invertible, maybe there
    # should be bcs
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

symmgr = lambda v: sym(grad(v))
skewgr = lambda v: grad(v) - symmgr(v)
mu     = lambda phi: mu0*exp(-phi)
s      = lambda phi: 0.5*(1 + ufl.tanh((phir-phi)/rad)) 
eta    = lambda phi: eta1 + eta2*(1 + ufl.tanh((phir-phi)/rad))    
f      = lambda phi: phi*(1-phi) 


curlBub = lambda vec: as_tensor([[vec[0].dx(1), -vec[0].dx(0)], [vec[1].dx(1), -vec[1].dx(0)]])


ndim = 2
e0, e1 = Constant((1, 0)), Constant((0, 1))
g   = Constant((0,1))
mu0 = Constant(25.)
phir = Constant(0.25)
rad  = Constant(0.5)
eta1 = Constant(0.25)
eta2 = Constant(0.5)
Re = Constant(1000.)
lam = Constant(1./Re)
rho = Constant(1.)
kappa = Constant(0.1)

C0 = Constant(1.)

phiHot = Constant(1)
phiCold = Constant(0)


# ******* Exact solutions for error analysis ****** #
'''
u = (g'h',-g"h)
p = h"'g + g"h' + 0.5*(g')**2 *(h*h"-(h')**2)

g  = 0.2*x**5 - 0.5*x**4 + 1./3*x**3
g' = x**4 - 2*x**3 + x**2 
g" = 4*x**3 - 6*x**2 + 2*x

h  = y**4 - y**2 
h' = 4*y**3 - 2*y
h" = 12*y**2 - 2
h"'= 24*y
'''
u_str = '(C0*(x**4 - 2*x**3 + x**2)*(4*y**3 - 2*y),-C0*(4*x**3 -6*x**2 + 2*x)*(y**4-y**2))' 
p_str = 'C0/Re*(24*y*(1./5*x**5-0.5*x**4+1./3*x**3)+(4*x**3-6*x**2+2*x)*(4*y**3-2*y))+0.5*C0**2*(x**4 - 2*x**3 + x**2)**2*((y**4-y**2)*(12*y**2 - 2)-(4*y**3 - 2*y)**2)'
phi_str = 'y+cos(pi*x)*y*(1-y)'

# ******* Model parameters ****** #
# polynomial degree
l=0
set_log_level(40)

nkmax = 6

hh = []; hht = []; nn = []; et = []; rt = []; eu = []; ru = [];
esig = []; rsig = []; ep = []; rp = [];
echi = []; rchi = []; egam = []; rgam = [];
ephi = []; rphi = []; it = []

rt.append(0.); rsig.append(0.0); 
ru.append(0.0); rgam.append(0.)
rphi.append(0.0); rchi.append(0.0); 
rp.append(0.)

for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)
    
    nps = pow(2,nk+1)+2; npst = pow(2,nk)+1
    mesh = UnitSquareMesh(nps,nps)#,'crossed')
    mesht = UnitSquareMesh(npst,npst)#,'crossed')
    #bmesht = BoundaryMesh(mesht, 'exterior')

    facet_f1 = MeshFunction('size_t', mesh, 1, 0)
    facet_ft = MeshFunction('size_t', mesht, 1, 0)

    subdomains = {1: CompiledSubDomain('near(x[0], 0)'),
                  2: CompiledSubDomain('near(x[0], 1)'),
                  3: CompiledSubDomain('near(x[1], 0)'),
                  4: CompiledSubDomain('near(x[1], 1)')}

    [subdomain.mark(facet_f1, tag) for tag, subdomain in subdomains.items()]
    [subdomain.mark(facet_ft, tag) for tag, subdomain in subdomains.items()]

    lm_tags = (1,2)
    nta = (3,4)
    
    bmesht = EmbeddedMesh(facet_ft, lm_tags)
    bmesht_subd = bmesht.marking_function
    
    n = FacetNormal(mesh)
    n_ = OuterNormal(bmesht, [0.5, 0.5]) #Constant((-1, 0)) # only the LM boundary?
    
    hh.append(mesh.hmax())
    hht.append(mesht.hmax())

    #dx_ = Measure('dx', domain=bmesht)
    dx_ = Measure('dx', domain=bmesht, subdomain_data=bmesht_subd)
    ds = Measure('ds', domain=mesh, subdomain_data=facet_f1)
    
    # ********* Finite dimensional spaces ********* #
    Ht  = VectorFunctionSpace(mesh, "DG", l+ndim, dim = 2)
    RTl = FunctionSpace(mesh, "RT", l+1)
    Bub = VectorFunctionSpace(mesh,'B', l + 3)

    
    Hu = VectorFunctionSpace(mesh, "DG", l)
    Hg = VectorFunctionSpace(mesh, "CG", l+1, dim = 1)
    Hphi = FunctionSpace(mesh, 'CG', l+1)
    Hchi = FunctionSpace(bmesht, 'DG',l)
    H0 = FunctionSpace(mesh, 'R',0)
    Hh = [Ht,RTl,RTl, Bub,Hu,Hg,Hphi,Hchi,H0]
    Ph = FunctionSpace(mesh, 'DG', l)

    Hu_aux = VectorFunctionSpace(mesh, "CG", 1)

    print ("....... Total DoFs = ", Ht.dim() + RTl.dim() + RTl.dim() + Bub.dim()  + Hu.dim() + Hg.dim() + Hphi.dim() + Hchi.dim() + H0.dim())

    nn.append(Ht.dim() + RTl.dim() + RTl.dim() + Bub.dim() + Hu.dim() + Hg.dim() + Hphi.dim() + Hchi.dim() + H0.dim())

    Sol =  ii_Function(Hh)
    
    tt_, sig0, sig1, bsol, u, gam_, phi, chi, a_trial = Sol 
    ss_, tau0, tau1, btest, v, del_, psi,  xi, a_test  = map(TestFunction, Hh)

    Tphi, Tpsi = Trace(phi, bmesht), Trace(psi, bmesht)

    tt=tensorify_symm_zerotr(tt_); ss = tensorify_symm_zerotr(ss_)
    gamma = tensorify_skew(gam_); delta = tensorify_skew(del_)
    
    sig0, sig1 = outer(e0, sig0), outer(e1, sig1)
    tau0, tau1 = outer(e0, tau0), outer(e1, tau1)
                      
    u_ex    = Expression(str2exp(u_str), degree=6,C0 = C0, domain=mesh)
    p_ex    = Expression(str2exp(p_str), degree=6,C0 = C0, Re = Re, domain=mesh)
    phi_ex  = Expression(str2exp(phi_str), degree=6, domain=mesh)
    phi_ex_ = Expression(str2exp(phi_str), degree=6, domain=bmesht)
    
    tt_ex   = symmgr(u_ex)
    gamma_ex= skewgr(u_ex)
    sig_ex  = lam*mu(phi_ex)*tt_ex - outer(u_ex,u_ex) - p_ex*Identity(ndim)

    ff_ex   = project(eta(phi_ex)*u_ex - div(sig_ex) - f(phi_ex)*g,Hu)
    mm_ex   = project(-rho*div(kappa*grad(phi_ex)) + dot(u_ex, grad(phi_ex + s(phi_ex))),Hphi)
    u_D     = interpolate(u_ex,Hu_aux)
    Tphi_ex = interpolate(phi_ex_,Hchi)
    trs_ex  = project(tr(sig_ex+outer(u_ex,u_ex)),Ph)
    chi_ex  = -kappa*rho*dot(grad(phi_ex_),n_)
    flux_ex = project(-kappa*rho*grad(phi_ex),Hu)
    

    FF = [lam*mu(phi)*inner(tt,ss)*dx - inner(sig0+sig1+curlBub(bsol),ss)*dx - inner(outer(u,u),ss)*dx,
          inner(tt,tau0)*dx + inner(gamma,tau0)*dx + dot(u,div(tau0))*dx + tr(tau0)*a_trial*dx - dot(tau0*n,u_D)*ds,
          inner(tt,tau1)*dx + inner(gamma,tau1)*dx + dot(u,div(tau1))*dx + tr(tau1)*a_trial*dx - dot(tau1*n,u_D)*ds,
          inner(tt,curlBub(btest))*dx + inner(gamma,curlBub(btest))*dx + dot(u,div(curlBub(btest)))*dx + tr(curlBub(btest))*a_trial*dx - dot(curlBub(btest)*n,u_D)*ds,
          dot(div(sig0+sig1+curlBub(bsol)),v)*dx - eta(phi)*dot(u,v)*dx + dot(f(phi)*g+ff_ex,v)*dx,
          inner(sig0+sig1+curlBub(bsol),delta)*dx,
          kappa*rho*dot(grad(phi),grad(psi))*dx + dot(u,grad(phi+s(phi)))*psi*dx + chi*Tpsi*dx_(lm_tags) - mm_ex*psi*dx + dot(flux_ex,n)*psi*ds(nta),
          (Tphi-Tphi_ex)*xi*dx_(lm_tags),
          (tr(sig0+sig1+curlBub(bsol)+outer(u,u))-trs_ex) * a_test * dx]

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





'''

CHECK that the sols in the paper coincide with the xx_str. The xx_str are well approximated. 





24*coordsY*(1/5*coordsX^5-1/4*coordsX^4+1/3*coordsX^3)+(4*coordsX^3-3*coordsX^2+2*coordsX)*(4*coordsY^3-2*coordsY)+0.5*(coordsX^4 - coordsX^3 + coordsX^2)^2*((coordsY^4-coordsY^2)*(12*coordsY^2 - 2)-(4*coordsY^3 - 2*coordsY)^2)


(coordsX^4 - coordsX^3 + coordsX^2)*(4*coordsY^3 - 2*coordsY)*iHat-(4*coordsX^3 -3*coordsX^2 + 2*coordsX)*(coordsY^4-coordsY^2)*jHat

'''
