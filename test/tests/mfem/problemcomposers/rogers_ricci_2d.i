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
  # Particle source S_n. In the normalised equations this also supplies S_T.
  [Sn]
    type = ParsedFunction
    expression = 'S_0n * (1 - tanh(((sqrt((x - xL / 2)^2 + (y - yL / 2)^2) / r_unit - 20) / 0.5) / b_att))'
    symbol_names = 'S_0n xL yL r_unit b_att'
    symbol_values = '0.03 1.0 1.0 1e-2 5.0'
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
  # phi is uniform and omega is zero at t = 0, so the Poisson constraint
  # L phi + M omega = 0 holds initially and the first step starts consistent.
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
  dt = 1.5e-3
  start_time = 0.0
  end_time = 0.75
[]

[Postprocessors]
  [T_l2]
    type = MFEML2Error
    variable = T
    function = 0
  []
  [omega_l2]
    type = MFEML2Error
    variable = omega
    function = 0
  []
  [n_l2]
    type = MFEML2Error
    variable = n
    function = 0
  []
  [phi_l2]
    type = MFEML2Error
    variable = phi
    function = 0
  []
[]

[Outputs]
  [CSV]
    type = CSV
    execute_on = 'timestep_end'
    file_base = OutputData/rogers_ricci_2d
  []
[]
