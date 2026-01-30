folder = "."   # <-- change this
from sys import stderr
from numpy import zeros, arange, isscalar,diag, dot,eye, ix_, ones, r_, pi, flatnonzero as find
from scipy.sparse import csr_matrix as sparse
from numpy.linalg import solve
import cloudpickle
import pickle
import pyomo.environ as pyo
import os, glob
import pandas as pd
import numpy as np
BUS_TYPE = 1
REF = 3
BUS_I = 0
F_BUS = 1
T_BUS = 2
BR_X = 4
TAP = 9
SHIFT = 10
BR_STATUS = 12
files = sorted(glob.glob(os.path.join(folder, "*.xlsx")))
# print("Found", len(files), "xlsx files")
# for f in files[:10]:
#     print(os.path.basename(f))

# multivariate normal
def create_scenario_multivariate(case, model, N, sigma_scaling = 0.03):
    buses = case['bus'].values
    baseMVA = case['baseMVA'][0].values
    # buses_cols = {col:num for num,col in enumerate(case.bus.columns.values)}
    base_demand = case['bus'].Pd.values / baseMVA
    # base_demand = buses[:,buses_cols['PD']] / case.baseMVA
    sigma = (sigma_scaling * base_demand)
    mean = np.zeros(len(base_demand))
    covariance_matrix = np.diag(sigma**2)
    omega = np.random.multivariate_normal(mean, covariance_matrix, N)
    return omega

# uniform
def create_scenario_uniform(case, model, percentage, N):
    buses = case.bus.values
    buses_cols = {col:num for num,col in enumerate(case['bus'].columns.values)}
    base_demand = buses[:,buses_cols['PD']] / case.baseMVA
    demand_test = {}
    for i in range(N):
        demand_list = []
        for d in base_demand:
            dev = np.random.uniform(-percentage,percentage)
            demand_list.append(d * dev)
        demand_test[i] = demand_list
    return demand_test

def update_injection_constraints(case, model, omega_bound):
    omega = case['omega']
    omega.LB = omega_bound
    omega.UB = omega_bound
    model.update()  # Update the model to reflect these changes
                
def makeBdc(baseMVA, bus, branch):
    """Builds the B matrices and phase shift injections for DC power flow.

    Returns the B matrices and phase shift injection vectors needed for a
    DC power flow.
    The bus real power injections are related to bus voltage angles by::
        P = Bbus * Va + PBusinj
    The real power flows at the from end the lines are related to the bus
    voltage angles by::
        Pf = Bf * Va + Pfinj
    Does appropriate conversions to p.u.
    @see: L{dcpf}
    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    @author: Ray Zimmerman (PSERC Cornell)
    """
    ## constants
    nb = bus.shape[0]          ## number of buses
    nl = branch.shape[0]       ## number of lines

    ## check that bus numbers are equal to indices to bus (one set of bus nums)
    if any(bus[:, BUS_I]-1 != list(range(nb))):
        stderr.write('makeBdc: buses must be numbered consecutively in '
                     'bus matrix\n')

    ## for each branch, compute the elements of the branch B matrix and the phase
    ## shift "quiescent" injections, where
    ##
    ##      | Pf |   | Bff  Bft |   | Vaf |   | Pfinj |
    ##      |    | = |          | * |     | + |       |
    ##      | Pt |   | Btf  Btt |   | Vat |   | Ptinj |
    ##
    stat = branch[:, BR_STATUS]               ## ones at in-service branches
    b = stat / branch[:, BR_X]                ## series susceptance
    tap = ones(nl)                            ## default tap ratio = 1
    i = find(branch[:, TAP])               ## indices of non-zero tap ratios
    tap[i] = branch[i, TAP]                   ## assign non-zero tap ratios
    b = b / tap

    ## build connection matrix Cft = Cf - Ct for line and from - to buses
    f = branch[:, F_BUS] -1                           ## list of "from" buses
    t = branch[:, T_BUS] -1                          ## list of "to" buses
    i = r_[range(nl), range(nl)]                   ## double set of row indices
    ## connection matrix
    Cft = sparse((r_[ones(nl), -ones(nl)], (i, r_[f, t])), (nl, nb))

    ## build Bf such that Bf * Va is the vector of real branch powers injected
    ## at each branch's "from" bus
    Bf = sparse((r_[b, -b], (i, r_[f, t])), shape = (nl, nb))## = spdiags(b, 0, nl, nl) * Cft

    ## build Bbus
    Bbus = Cft.T * Bf

    ## build phase shift injection vectors
    Pfinj = b * (-branch[:, SHIFT] * pi / 180)  ## injected at the from bus ...
    # Ptinj = -Pfinj                            ## and extracted at the to bus
    Pbusinj = Cft.T * Pfinj                ## Pbusinj = Cf * Pfinj + Ct * Ptinj

    return Bbus, Bf, Pbusinj, Pfinj

def makePTDF(baseMVA, bus, branch, slack=None):
    """Builds the DC PTDF matrix for a given choice of slack.

    Returns the DC PTDF matrix for a given choice of slack. The matrix is
    C{nbr x nb}, where C{nbr} is the number of branches and C{nb} is the
    number of buses. The C{slack} can be a scalar (single slack bus) or an
    C{nb x 1} column vector of weights specifying the proportion of the
    slack taken up at each bus. If the C{slack} is not specified the
    reference bus is used by default.

    For convenience, C{slack} can also be an C{nb x nb} matrix, where each
    column specifies how the slack should be handled for injections
    at that bus.

    @see: L{makeLODF}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    ## use reference bus for slack by default
    if slack is None:
        slack = find(bus[:, BUS_TYPE] == REF)
        slack = slack[0]

    ## set the slack bus to be used to compute initial PTDF
    if isscalar(slack):
        slack_bus = slack
    else:
        slack_bus = 0      ## use bus 1 for temp slack bus

    nb = bus.shape[0]
    nbr = branch.shape[0]
    noref = arange(1, nb)      ## use bus 1 for voltage angle reference
    noslack = find(arange(nb) != slack_bus)

    ## check that bus numbers are equal to indices to bus (one set of bus numbers)
    if any(bus[:, BUS_I]-1 != arange(nb)):
        stderr.write('makePTDF: buses must be numbered consecutively')

    ## compute PTDF for single slack_bus
    Bbus, Bf, _, _ = makeBdc(baseMVA, bus, branch)
    Bbus, Bf = Bbus.todense(), Bf.todense()
    H = zeros((nbr, nb))
    H[:, noslack] = solve( Bbus[ix_(noslack, noref)].T, Bf[:, noref].T ).T
    #             = Bf[:, noref] * inv(Bbus[ix_(noslack, noref)])

    ## distribute slack, if requested
    if not isscalar(slack):
        if len(slack.shape) == 1:  ## slack is a vector of weights
            slack = slack / sum(slack)   ## normalize weights

            ## conceptually, we want to do ...
            ##    H = H * (eye(nb, nb) - slack * ones((1, nb)))
            ## ... we just do it more efficiently
            v = dot(H, slack)
            for k in range(nb):
                H[:, k] = H[:, k] - v
        else:
            H = dot(H, slack)

    return H                

def makeLODF(branch, PTDF):
    nl, nb = PTDF.shape
    f = branch[:, F_BUS].astype(int)-1
    t = branch[:, T_BUS].astype(int)-1
    Cft = sparse((r_[ones(nl), -ones(nl)],
                      (r_[f, t], r_[arange(nl), arange(nl)])), (nb, nl))

    H = PTDF * Cft
    h = diag(H, 0)
    LODF = H / (ones((nl, nl)) - ones((nl, 1)) * h.T)
    LODF = LODF - diag(diag(LODF)) - eye(nl, nl)
    # if there's a radial outage
    LODF[np.isinf(LODF)] = 0.0
    LODF[np.isnan(LODF)] = 0.0
    np.fill_diagonal(LODF, -1)

    return LODF

def islanding_line_indices(branch, PTDF, tol=1e-9):
    # replicate the pieces in makeLODF up to h
    nl, nb = PTDF.shape
    f = branch[:, F_BUS].astype(int) - 1
    t = branch[:, T_BUS].astype(int) - 1

    # sparse incidence Cft (nb x nl)
    from numpy import arange, r_, ones
    Cft = sparse(
        (r_[ones(nl), -ones(nl)],
         (r_[f, t], r_[arange(nl), arange(nl)])),
        shape=(nb, nl)
    )

    H = PTDF @ Cft          # (nl x nl)
    h = np.diag(H)          # length nl
    denom = 1.0 - h         # per-line denominator in LODF formula

    # islanding iff 1 - h[k] ~ 0
    mask_islanding = np.isclose(denom, 0.0, atol=tol)
    idx_islanding = np.where(mask_islanding)[0]
    return idx_islanding

def load_case_from_excel(file):
    # === Initialization ===
    # case_name = 'pglib_opf_case60_c'
    # case_path = f'M:\OneDrive - KFUPM\ISE 712 PhD. Dissertation\code\excel_outputs\{case_name}.xlsx'
    case = pd.read_excel(file, sheet_name=['baseMVA','bus','gen','gencost','branch'])
    bus_to_idx = {bus: i+1 for i, bus in enumerate(case['bus'].bus_i.values)}
    # bus_idx = [bus_to_idx[bus] for bus in case['bus'].bus_i.values]
    case['bus'].bus_i = case['bus'].bus_i.replace(bus_to_idx) # rename the bus for making PTDF
    case['gen'].bus_i = case['gen'].bus_i.replace(bus_to_idx)
    baseMVA = case['baseMVA'][0].values
    # remove generators and costgen with pmax and pmin equal to zero
    zero_gen_idx = []
    for num,i in enumerate(case['gen'].Pmax.values/ baseMVA):
        if (i == 0 and (case['gen'].Pmin.values / baseMVA)[num] == 0) or (case['gen'].Pmin.values / baseMVA)[num] < 0:
            zero_gen_idx.append(num)
    case['gen'].drop(index=zero_gen_idx, inplace=True) # drop
    case['gencost'].drop(index=zero_gen_idx, inplace=True) # drop
    # pmax = case['gen'].Pmax.values / baseMVA
    # pmin = case['gen'].Pmin.values / baseMVA
    case['branch'].bus_i = case['branch'].bus_i.replace(bus_to_idx)
    case['branch'].bus_j = case['branch'].bus_j.replace(bus_to_idx)
    case['K'] = makePTDF(baseMVA, case['bus'].values, case['branch'].values, slack=None)
    case['L'] = makeLODF(case['branch'].values, case['K']) # makeLODF(branch, PTDF)

    # case['L'] = makeLODF(case['branch'].values, case['K']) # makeLODF(branch, PTDF)
    nbus = case['bus'].shape[0]
    ngen = case['gen'].shape[0]
    # nbranch = case['branch'].shape[0]
    case['B'] = np.zeros((nbus,ngen))
    for gen,bus in enumerate(case['gen'].bus_i.values): #case['gen'].bus_i
        case['B'][int(bus)-1][gen] = 1    
    case['gamma'] = 0.05 # An exact and scalable problem decomposition for security-constrained optimal power flow: We performed simulations for various values of γ, β1, and β2. Our results indicate that varying the parameters may impact the perfor￾mance of the CCGA. However, the dominance of the CCGA over other methods (EF and BDDC) was a constant, despite the parameterization. We have reported results for β1 = 5, β2 = 1.2, and γi = 0.05 for all i ∈ 𝒢.
    # d = case['bus'].Pd.values/baseMVA

    # case['Kg'] = [0]
    case['Kg'] = [i for i in range(ngen)]
    case['Ke'] = islanding_line_indices(case['branch'].values, case['K'], tol=1e-9)
    case['M_eta'] = 1500 # Self-Supervised Learning for Large-Scale Preventive Security Constrained DC Optimal Power Flow
    case['rho'] = 1
    return case
    # print(case)

def P1_formulation(case):
    """
    Build Pyomo model for Q1 based on provided `case` dict.
    Returns: ConcreteModel
    """
    # read shapes
    nbus = case['bus'].shape[0]
    ngen = case['gen'].shape[0]
    nbranch = case['branch'].shape[0]
    Kg = list(case['Kg']) if 'Kg' in case else list(range(ngen))
    Ke = list(case['Ke']) if 'Ke' in case else []

    K = np.array(case['K'])               # shape (nbranch, nbus)
    B = np.array(case['B'])               # shape (nbus, ngen)
    L  = np.array(case['L'])
    baseMVA = float(case['baseMVA'].loc[0,0])   # scalar
    d = np.array(case['bus'].Pd.values / baseMVA).flatten()  # length nbus
    pmax = np.array(case['gen'].Pmax.values / baseMVA).flatten()
    pmin = np.array(case['gen'].Pmin.values / baseMVA).flatten()
    limit = np.array(case['branch'].rateA.values / baseMVA).flatten()
    M_eta = float(case.get('M_eta', 0.0))
    rho = case['rho']
    # gencost linear coeffs for objective (using same pattern as your code)
    gencost = np.array(case['gencost'][['c2', 'c1', 'c0']].values)
    # Build linear objective coefficients vector (as in your code)
    obj_coeff = (gencost[:, 0] * baseMVA**2 + gencost[:, 1] * baseMVA + gencost[:, 2]).flatten()

    model = pyo.ConcreteModel(name="model A")

    model.G = pyo.RangeSet(0, ngen-1)
    model.N = pyo.RangeSet(0, nbus-1)
    model.BR = pyo.RangeSet(0, nbranch-1)
    model.Kg = pyo.Set(initialize=Kg)
    model.Ke = pyo.Set(initialize=Ke)
    model.gk_index = pyo.Set(initialize=[(k,i) for k in Kg for i in model.G])
    model.xk_index = pyo.Set(initialize=[(k,i) for k in Kg for i in model.G if i!=k])
    model.eta_ke_index = pyo.Set(initialize=[(k,b) for k in Ke for b in model.BR])
    model.eta_kg_index = pyo.Set(initialize=[(k,b) for k in Kg for b in model.BR])
    
    model.g = pyo.Var(model.G, domain=pyo.NonNegativeReals)                       # generator outputs
    model.gk = pyo.Var(model.gk_index, domain=pyo.NonNegativeReals)
    model.eta0 = pyo.Var(model.BR, domain=pyo.NonNegativeReals)                    # slack on branch limits (base)
    model.eta_ke = pyo.Var(model.eta_ke_index, domain=pyo.NonNegativeReals)
    model.eta_kg = pyo.Var(model.eta_kg_index, domain=pyo.NonNegativeReals)
    model.xk = pyo.Var(model.xk_index, domain=pyo.Binary)
    model.zk = pyo.Var(model.G, domain = pyo.NonNegativeReals, bounds=(0,1))

    model.omega = pyo.Param(model.N, initialize=0.0, mutable=True)  
    model.M_eta = pyo.Param(initialize=M_eta, mutable=True)  
    
    model.obj = pyo.ObjectiveList()
    # (17)
    model.obj.add( 
    expr= 
        sum(obj_coeff[i] * model.g[i] for i in model.G)
        + model.M_eta * ( sum(model.eta0[b] for b in model.BR)
                          + sum(model.eta_ke[k,b] for (k,b) in model.eta_ke_index)
                          + sum(model.eta_kg[k,b] for (k,b) in model.eta_kg_index) )
        # + 0.5 * sum(model.y[i]*(model.g[i] - model.xi[i]) for i in model.G)
        # + model.rho * sum( (model.g[i] - model.xi[i])**2 for i in model.G )
        ,
    sense=pyo.minimize
)    
    
    model.balance = pyo.ConstraintList() 
    # (2)
    model.balance.add(
        sum(model.g[i] for i in model.G) == float(np.sum(d)) + sum(model.omega[n] for n in model.N)
    )

    def inj_expr(model, n):
        return d[n] + (model.omega[n] if n in model.N else 0.0) - sum(B[n,gg] * model.g[gg] for gg in model.G)
    model.inj = pyo.Expression(model.N, rule=inj_expr)
    
    def flow_expr(model, b):
        return sum(K[b,n] * model.inj[n] for n in model.N)
    model.flow = pyo.Expression(model.BR, rule=flow_expr) 
    
    model.branch_upper = pyo.ConstraintList()
    model.branch_lower = pyo.ConstraintList()
    # (3)
    for b in model.BR: 
        model.branch_upper.add(
            model.flow[b] <= float(limit[b]) + model.eta0[b]
        )
        model.branch_lower.add(
            model.flow[b] >= -float(limit[b]) - model.eta0[b]  
        )
    
    model.gen_ub = pyo.ConstraintList()
    model.gen_lb = pyo.ConstraintList() 
    # (4)
    for i in model.G: 
        model.gen_ub.add(
            model.g[i] <= float(pmax[i])
        )
        model.gen_lb.add(
            model.g[i] >= float(pmin[i])
        )

    model.balance_k = pyo.ConstraintList()
    # (5)
    for k in model.Kg: 
        model.balance_k.add(
            sum(model.gk[k,i] for i in model.G)
                          == float(np.sum(d)) + sum(model.omega[n] for n in model.N)
        )    

    def inj_kg_expr(model, k, n):
        return d[n] + model.omega[n] - sum(B[n,gg] * model.gk[k,gg] for gg in model.G)
    model.inj_kg = pyo.Expression(model.Kg, model.N, rule=inj_kg_expr)
    def flow_expr_kg(model, k, b):
        return sum(K[b,n] * model.inj_kg[k,n] for n in model.N)
    model.flow_kg = pyo.Expression(model.Kg, model.BR, rule=flow_expr_kg) 
    model.fk_upper = pyo.ConstraintList()
    model.fk_lower = pyo.ConstraintList()
    # (6)
    for k in model.Kg: 
        for b in model.BR:
            model.fk_upper.add(
                model.flow_kg[k,b] <= float(limit[b]) + model.eta_kg[k,b]
            )
            model.fk_lower.add(
                model.flow_kg[k,b] >= -float(limit[b]) - model.eta_kg[k,b] 
            )
    model.gk_ub = pyo.ConstraintList()  
    model.gk_lb = pyo.ConstraintList()
    model.gk_fx = pyo.ConstraintList()
    # (7)
    for k in model.Kg: 
        for i in model.G:
            if i != k:
                model.gk_ub.add(
                    model.gk[k,i] <= float(pmax[i])
                )
                model.gk_lb.add(
                    model.gk[k,i] >= float(pmin[i])
                )
            # (8)
            else:
                model.gk_fx.add( 
                    model.gk[k,i] == 0.0
                )          

    model.c1 = pyo.ConstraintList()
    model.c2 = pyo.ConstraintList()
    model.c3 = pyo.ConstraintList()
    model.c4 = pyo.ConstraintList()
    gamma = case['gamma']
    gcap = pmax - pmin
    for k in model.Kg:
        for i in model.G:
            if i != k:
                # (9)
                model.c1.add( 
                    model.gk[k,i] - model.g[i] - model.zk[k]*gamma*gcap[i] <= pmax[i]*(1-model.xk[k,i])
                )
                # (10)
                model.c2.add( 
                    -pmax[i]*(1-model.xk[k,i]) <= model.gk[k,i] - model.g[i] - model.zk[k]*gamma*gcap[i]
                )
                # (11)
                model.c3.add( 
                    model.g[i] + model.zk[k]*gamma*gcap[i] >= pmax[i]*(1-model.xk[k,i])
                )
                # (12)
                model.c4.add( 
                    model.gk[k,i] >= pmax[i] * (1 - model.xk[k,i])
                )

    # (13)
    def flow_expr_ke(model, k, b):
        return model.flow[k] if k != b else 0
    model.flow_ke = pyo.Expression(model.Ke, model.BR, rule=flow_expr_ke) 
    model.line_upper = pyo.ConstraintList()
    model.line_lower = pyo.ConstraintList()
    for k in model.Ke:
        for b in model.BR:
            model.line_upper.add(
                model.flow[b] + model.flow_ke[k,b]*L[b,k] <= float(limit[b]) + model.eta_ke[k,b]
            )
            model.line_lower.add(
                model.flow[b] + model.flow_ke[k,b]*L[b,k] >= -float(limit[b]) - model.eta_ke[k,b]
            )

    return model

for file in files:
    case = load_case_from_excel(file)
    model_P1 = P1_formulation(case)
    with open(f'{file[-19:-5]}_model.pkl', mode='wb') as file:
        cloudpickle.dump(model_P1, file)    