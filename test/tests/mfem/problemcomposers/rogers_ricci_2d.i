# 2D Rogers-Ricci plasma turbulence model through a custom
# problem operator plugged in by a custom problem composer.
#
# T, omega and n are evolved implicitly in time by the operator; phi is solved for from omega at
# the start of each stage and so is declared as an AuxVariable rather than a Variable.

S0n = 0.03    # source amplitude
rs = 20.0     # source radius
Ls = 0.5      # source edge width
xc = 50.0     # domain centre, i.e. half the 100.0 side length of rogers-ricci-quad.mesh
yc = 50.0

[Mesh]
  type = MFEMMesh
  file = ../mesh/rogers-ricci-quad.mesh
[]

[Problem]
  type = MFEMProblem
[]

[FESpaces]
  [H1FESpace]
    type = MFEMScalarFESpace
    fec_type = H1
    fec_order = FIRST
  []
[]

[Variables]
  [T]
    type = MFEMVariable
    fespace = H1FESpace
  []
  [omega]
    type = MFEMVariable
    fespace = H1FESpace
  []
  [n]
    type = MFEMVariable
    fespace = H1FESpace
  []
[]

[AuxVariables]
  [phi]
    type = MFEMVariable
    fespace = H1FESpace
  []
[]

[Functions]
  [source]
    type = ParsedFunction
    expression = '0.5 * ${S0n} * (1.0 - tanh((sqrt((x - ${xc})^2 + (y - ${yc})^2) - ${rs}) / ${Ls}))'
  []
[]

[QuadratureFunctions]
  [phi_qf]
    type = MFEMScalarQuadratureFunction
    coefficient = phi
    order = 5
    updates = NONLINEAR
  []
  [source_qf]
    type = MFEMScalarQuadratureFunction
    coefficient = source
    order = 5
    updates = NONE
  []
[]

[ICs]
  [T_ic]
    type = MFEMScalarIC
    variable = T
    coefficient = 1e-4
  []
  [omega_ic]
    type = MFEMScalarIC
    variable = omega
    coefficient = 0.0
  []
  [n_ic]
    type = MFEMScalarIC
    variable = n
    coefficient = 1e-4
  []
  [phi_ic]
    type = MFEMScalarIC
    variable = phi
    coefficient = 0.03
  []
[]

[Solvers]
  # The linear driver here serves the electrostatic potential solve. The Newton step is solved by
  # a block GMRES built inside the problem operator, since a block preconditioner cannot be
  # expressed in this block.
  [boomeramg]
    type = MFEMHypreBoomerAMG
    print_level = 0
  []
  [phi_solver]
    type = MFEMCGSolver
    preconditioner = boomeramg
    l_tol = 1e-12
    l_max_its = 2000
    print_level = 0
  []
  [newton]
    type = MFEMNewtonNonlinearSolver
    line_search = backtracking
    rel_tol = 1e-8
    abs_tol = 0
    max_its = 15
    print_level = 1
    use_initial_guess = false
  []
[]

[ProblemComposer]
  [rogers_ricci]
    type = RogersRicci2DProblemComposer
    variables = 'T omega n'
    potential_variable = phi
    potential_coefficient = phi_qf
    source_coefficient = source_qf
    eps2 = 1e-4
    Binv = 40.0
    Lambda = 3.0
    num_active = 2
    l_tol = 1e-10
    l_max_its = 200
    krylov_dim = 50
  []
[]

[Executioner]
  type = MFEMTransient
  device = cpu
  assembly_level = legacy
  dt = 2.4e-3
  start_time = 0.0
  end_time = 100.0
[]

[Outputs]
  [ParaViewDataCollection]
    type = MFEMParaViewDataCollection
    file_base = OutputData/RogersRicci2D
    vtk_format = ASCII
  []
[]