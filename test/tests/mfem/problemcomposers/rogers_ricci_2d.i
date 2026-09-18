# 2D Rogers-Ricci drift-turbulence model, solved through a custom problem operator.
#
# Three transported fields (temperature T, vorticity omega, density n) share a weak form
#
#   M x_dot + K x + (F_x, v) = 0
#
# where K combines E x B advection (a Poisson bracket) with streamline-upwind
# stabilisation, both entering through the drift velocity v_d = R grad(phi). The
# electrostatic potential phi is a fourth block unknown obeying grad^2 phi = omega, so all
# four fields are advanced by a single implicit solve per step.
#
# The block assembly has no equivalent in the Kernels/BCs machinery, so it lives in
# RogersRicci2DProblemOperator. Everything else is set up from this file.

[Mesh]
  type = MFEMFileMesh
  file = ../mesh/rogers_ricci_64x64.mesh
  uniform_refine = 2
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
  [phi]
    type = MFEMVariable
    fespace = H1FESpace
  []
[]

[Functions]
  [Sn]
    type = ParsedFunction
    expression = '0.5 * S_0n * (1 - tanh((sqrt((x - xL / 2)^2 + (y - yL / 2)^2) - r_s) / L_s))'
    symbol_names = 'S_0n xL yL r_s L_s'
    symbol_values = '0.03 100.0 100.0 20.0 0.5'
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

[ProblemComposers]
  [rogers_ricci]
    type = RogersRicci2DProblemComposer
    temperature = T
    vorticity = omega
    density = n
    potential = phi
    source = Sn
    lambda = 3.0
    b_inv = 40.0
    eps_squared = 1e-4
  []
[]

[Solvers]
  [superlu]
    type = MFEMSuperLU
  []
[]

[Executioner]
  type = MFEMTransient
  device = cpu
  dt = 0.12
  start_time = 0.0
  end_time = 120
[]

[Outputs]
  [ParaViewDataCollection]
    type = MFEMParaViewDataCollection
    file_base = OutputData/RogersRicci2D_R2
    vtk_format = ASCII
  []
[]
