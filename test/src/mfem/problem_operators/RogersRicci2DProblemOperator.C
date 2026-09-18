//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#ifdef MOOSE_MFEM_ENABLED

#include "RogersRicci2DProblemOperator.h"
#include "ScaleIntegrator.h"

#include <array>

namespace
{
/// Prefix keeping these coefficients clear of user-declared ones, whose names come from
/// input-file object names and so never begin with underscores.
const std::string rr_prefix = "__rogers_ricci_";

/// Suffixes identifying each block variable, in RogersRicciVarIdx order.
const std::array<std::string, RR_NUM_VARS> rr_var_suffix = {"T", "omega", "n", "phi"};

std::string
coefName(const std::string & suffix)
{
  return rr_prefix + suffix;
}

std::string
varCoefName(int i)
{
  return rr_prefix + "var_" + rr_var_suffix[i];
}

std::string
reactionCoefName(int i)
{
  return rr_prefix + "F_" + rr_var_suffix[i];
}

std::string
dPhiFCoefName(int i)
{
  return rr_prefix + "DphiF_" + rr_var_suffix[i];
}

std::string
rotGradCoefName(int i)
{
  return rr_prefix + "rotgrad_" + rr_var_suffix[i];
}
}

RogersRicci2DProblemOperator::RogersRicci2DProblemOperator(
    MFEMProblem & problem,
    const std::vector<VariableName> & variable_names,
    const MFEMScalarCoefficientName & source_name,
    mfem::real_t b_inv,
    mfem::real_t lambda,
    mfem::real_t eps_squared)
  : Moose::MFEM::TimeDependentProblemOperator(problem),
    _variable_names(variable_names),
    _source_name(source_name),
    _b_inv(b_inv),
    _lambda(lambda),
    _eps_squared(eps_squared)
{
}

RogersRicci2DProblemOperator::~RogersRicci2DProblemOperator() { deleteBlocks(); }

void
RogersRicci2DProblemOperator::SetGridFunctions()
{
  _trial_var_names.assign(_variable_names.begin(), _variable_names.end());
  _test_var_names = _trial_var_names;
  Moose::MFEM::TimeDependentProblemOperator::SetGridFunctions();
}

void
RogersRicci2DProblemOperator::Init(mfem::BlockVector & X)
{
  Moose::MFEM::TimeDependentProblemOperator::Init(X);

  _state = &X;
  _fespace = _trial_variables[RR_T]->ParFESpace();

  auto & pmesh = *_problem_data.pmesh;
  // GetTypicalElementTransformation is the empty-local-mesh-safe way to sample an element.
  _element_size = pmesh.GetElementSize(pmesh.GetTypicalElementTransformation());

  // phi is Dirichlet on the whole boundary, pinned at its initial value. The transported
  // variables are pure Neumann and need no essential DoFs.
  mfem::Array<int> ess_bdr(pmesh.bdr_attributes.Max());
  ess_bdr = 1;
  _fespace->GetEssentialTrueDofs(ess_bdr, _phi_ess_tdofs);

  declareCoefficients();

  // Backward Euler in slope form: ImplicitSolve returns k and the solver forms
  // x <- x + dt k.
  _problem_data.ode_solver = std::make_unique<mfem::BackwardEulerSolver>();
  _problem_data.ode_solver->Init(*this);
  SetTime(_problem.time());
}

void
RogersRicci2DProblemOperator::declareCoefficients()
{
  auto & coefs = _problem_data.coefficients;

  // Scalar coefficients tracking each block variable's grid function.
  for (const auto i : make_range(RR_NUM_VARS))
    coefs.declareScalar<mfem::GridFunctionCoefficient>(varCoefName(i), _trial_variables[i]);

  auto & t_coef = coefs.getScalarCoefficient(varCoefName(RR_T));
  auto & n_coef = coefs.getScalarCoefficient(varCoefName(RR_N));
  auto & phi_coef = coefs.getScalarCoefficient(varCoefName(RR_PHI));
  auto & source = coefs.getScalarCoefficient(_source_name);

  // Drift velocity v_d = R grad(phi), so that v_d . grad(u) is the Poisson bracket
  // [phi, u], and the streamline-upwind diffusivity (h/2)|a| vhat (x) vhat built from it.
  // The transported variables are advected at a = b_inv * v_d (the ConvectionIntegrator in
  // formSystem() carries the b_inv factor), so the diffusivity is scaled by |a| rather than
  // by |v_d|.
  mfem::DenseMatrix rotation({{0.0, -1.0}, {1.0, 0.0}});
  auto & rot =
      coefs.declareMatrix<mfem::MatrixConstantCoefficient>(coefName("rotation"), rotation);
  auto & grad_phi = coefs.declareVector<mfem::GradientGridFunctionCoefficient>(
      coefName("grad_phi"), _trial_variables[RR_PHI]);
  auto & drift =
      coefs.declareVector<mfem::MatrixVectorProductCoefficient>(coefName("vd"), rot, grad_phi);
  auto & drift_hat = coefs.declareVector<mfem::NormalizedVectorCoefficient>(
      coefName("vd_hat"), drift, _eps_squared);
  auto & drift_scaled = coefs.declareVector<mfem::ScalarVectorProductCoefficient>(
      coefName("vd_scaled"), _b_inv * _element_size / 2.0, drift_hat);
  coefs.declareMatrix<mfem::OuterProductCoefficient>(coefName("suw"), drift, drift_scaled);

  // exp(Lambda - phi/sqrt(T^2 + eps2)), shared by every reaction term and its derivatives.
  auto & thermal_exp = coefs.declareScalar<mfem::TransformedCoefficient>(
      coefName("thermal_exp"),
      &t_coef,
      &phi_coef,
      [this](mfem::real_t T, mfem::real_t phi)
      { return std::exp(_lambda - phi / std::sqrt(T * T + _eps_squared)); });

  // Reaction terms. S_T equals S_n in the normalised equations, so both use the same source.
  auto & ft_no_source = coefs.declareScalar<mfem::TransformedCoefficient>(
      coefName("ft_no_source"),
      &t_coef,
      &thermal_exp,
      [](mfem::real_t T, mfem::real_t e) { return (T / 36.) * (1.71 * e - 0.71); });
  coefs.declareScalar<mfem::TransformedCoefficient>(
      reactionCoefName(RR_T),
      &ft_no_source,
      &source,
      [](mfem::real_t f, mfem::real_t s) { return f - s; });

  coefs.declareScalar<mfem::TransformedCoefficient>(
      reactionCoefName(RR_OMEGA),
      &thermal_exp,
      [](mfem::real_t e) { return (1. / 24.) * (e - 1.); });

  auto & fn_no_source = coefs.declareScalar<mfem::TransformedCoefficient>(
      coefName("fn_no_source"),
      &n_coef,
      &thermal_exp,
      [](mfem::real_t n, mfem::real_t e) { return (n / 24.) * e; });
  coefs.declareScalar<mfem::TransformedCoefficient>(
      reactionCoefName(RR_N),
      &fn_no_source,
      &source,
      [](mfem::real_t f, mfem::real_t s) { return f - s; });

  // Gateaux derivatives of the reaction terms. With Ttil = sqrt(T^2 + eps2) and
  // E = Lambda - phi/Ttil, the exact derivatives are dE/dT = phi T / Ttil^3 and
  // dE/dphi = -1/Ttil.
  //
  // dt is folded into these coefficients, so they enter as plain mass terms.
  coefs.declareScalar<mfem::TransformedCoefficient>(
      coefName("DT_FT"),
      &t_coef,
      &phi_coef,
      [this](mfem::real_t T, mfem::real_t phi)
      {
        const mfem::real_t t_reg = std::sqrt(T * T + _eps_squared);
        const mfem::real_t e = std::exp(_lambda - phi / t_reg);
        return (_dt / 36.) * (1.71 * e - 0.71) +
               _dt * (1.71 / 36.) * e * (phi / t_reg) * (T * T) / (t_reg * t_reg);
      });

  auto & dt_fw = coefs.declareScalar<mfem::TransformedCoefficient>(
      coefName("DT_Fw"),
      &t_coef,
      &phi_coef,
      [this](mfem::real_t T, mfem::real_t phi)
      {
        const mfem::real_t t_reg = std::sqrt(T * T + _eps_squared);
        return (_dt / 24.) * std::exp(_lambda - phi / t_reg) * phi * T /
               (t_reg * t_reg * t_reg);
      });

  coefs.declareScalar<mfem::TransformedCoefficient>(
      coefName("Dn_Fn"),
      &thermal_exp,
      [this](mfem::real_t e) { return (_dt / 24.) * e; });

  coefs.declareScalar<mfem::TransformedCoefficient>(
      coefName("DT_Fn"), &n_coef, &dt_fw, [](mfem::real_t n, mfem::real_t d) { return n * d; });

  // Derivatives with respect to phi.
  coefs.declareScalar<mfem::TransformedCoefficient>(
      dPhiFCoefName(RR_T),
      &t_coef,
      &phi_coef,
      [this](mfem::real_t T, mfem::real_t phi)
      {
        const mfem::real_t t_reg = std::sqrt(T * T + _eps_squared);
        return -_dt * (1.71 / 36.) * (T / t_reg) * std::exp(_lambda - phi / t_reg);
      });

  auto & dphi_fw = coefs.declareScalar<mfem::TransformedCoefficient>(
      dPhiFCoefName(RR_OMEGA),
      &t_coef,
      &phi_coef,
      [this](mfem::real_t T, mfem::real_t phi)
      {
        const mfem::real_t t_reg = std::sqrt(T * T + _eps_squared);
        return -(_dt / 24.) * std::exp(_lambda - phi / t_reg) / t_reg;
      });

  coefs.declareScalar<mfem::TransformedCoefficient>(
      dPhiFCoefName(RR_N), &n_coef, &dphi_fw, [](mfem::real_t n, mfem::real_t d)
      { return n * d; });

  // Poisson-bracket part of the derivative of K with respect to phi. The advection term
  // contributes -(1/B)((R grad phi) . grad x, v); using
  // (R grad psi) . grad x = -(R grad x) . grad psi, its derivative in direction psi is
  // +(1/B)((R grad x) . grad psi, v), a ConvectionIntegrator acting on the trial function
  // psi = delta phi with velocity R grad x.
  //
  // The streamline-upwind contribution to this derivative is deliberately lagged:
  // differentiating (h/2)|v_d| vhat (x) vhat through the normalisation gives a
  // fourth-order tensor that would need a bespoke integrator. Lagging the linearisation of
  // an O(h) artificial diffusion term leaves a mild CFL-like restriction but cannot move
  // the fixed points, because the residual itself is exact.
  for (const auto i : make_range(RR_NUM_TRANSPORTED))
  {
    auto & grad = coefs.declareVector<mfem::GradientGridFunctionCoefficient>(
        rr_prefix + "grad_" + rr_var_suffix[i], _trial_variables[i]);
    coefs.declareVector<mfem::MatrixVectorProductCoefficient>(rotGradCoefName(i), rot, grad);
  }
}

void
RogersRicci2DProblemOperator::deleteBlocks()
{
  for (const auto i : make_range(_h_blocks.NumRows()))
    for (const auto j : make_range(_h_blocks.NumCols()))
      delete _h_blocks(i, j);
  _h_blocks.DeleteAll();
}

void
RogersRicci2DProblemOperator::formSystem(mfem::real_t dt)
{
  // The reaction-term lambdas read _dt, so setting it here updates every Gateaux
  // coefficient without rebuilding any of them.
  _dt = dt;

  auto & coefs = _problem_data.coefficients;
  auto & drift = coefs.getVectorCoefficient(coefName("vd"));
  auto & suw = coefs.getMatrixCoefficient(coefName("suw"));
  auto & state = *_state;

  deleteBlocks();
  _h_blocks.SetSize(RR_NUM_VARS, RR_NUM_VARS);
  // HypreParMatrixFromBlocks treats a null entry as a zero block, so only the non-zero
  // blocks below need filling.
  for (const auto i : make_range(RR_NUM_VARS))
    for (const auto j : make_range(RR_NUM_VARS))
      _h_blocks(i, j) = nullptr;

  for (const auto i : make_range(RR_NUM_TRANSPORTED))
  {
    mfem::ParBilinearForm k(_fespace);
    mfem::ParBilinearForm m(_fespace);
    mfem::ParLinearForm b(_fespace);

    // K = E x B advection (the Poisson bracket) + streamline-upwind stabilisation.
    auto * k_int = new mfem::SumIntegrator;
    k_int->AddIntegrator(new mfem::ConvectionIntegrator(drift, -_b_inv));
    k_int->AddIntegrator(new mfem::DiffusionIntegrator(suw));
    // ScaleIntegrator does not take ownership here: k owns k_int exclusively.
    auto * k_dt_int = new Moose::MFEM::ScaleIntegrator(k_int, dt, false);

    k.AddDomainIntegrator(k_int);
    k.Assemble();
    k.Finalize();

    auto * m_int = new mfem::SumIntegrator;
    m_int->AddIntegrator(new mfem::MassIntegrator); // Time derivative term
    if (i == RR_T)
      m_int->AddIntegrator(
          new mfem::MassIntegrator(coefs.getScalarCoefficient(coefName("DT_FT"))));
    else if (i == RR_N)
      m_int->AddIntegrator(
          new mfem::MassIntegrator(coefs.getScalarCoefficient(coefName("Dn_Fn"))));
    // RR_OMEGA has no diagonal Gateaux term.
    m_int->AddIntegrator(k_dt_int);

    m.AddDomainIntegrator(m_int);
    m.Assemble();
    m.Finalize();

    // RHS = -K x - (F_i, v). ParallelAssemble sums local contributions across ranks
    // (P^T b) and TrueAddMult applies the true-DoF action P^T K P.
    b.AddDomainIntegrator(
        new mfem::DomainLFIntegrator(coefs.getScalarCoefficient(reactionCoefName(i))));
    b.Assemble();
    std::unique_ptr<mfem::HypreParVector> b_true(b.ParallelAssemble());
    *b_true *= -1.0;
    k.TrueAddMult(state.GetBlock(i), *b_true, -1.0);
    _true_rhs.GetBlock(i) = *b_true;

    _h_blocks(i, i) = m.ParallelAssemble();

    // Coupling of this transported variable to phi.
    mfem::ParBilinearForm m_i_phi(_fespace);
    m_i_phi.AddDomainIntegrator(
        new mfem::MassIntegrator(coefs.getScalarCoefficient(dPhiFCoefName(i))));
    m_i_phi.AddDomainIntegrator(new mfem::ConvectionIntegrator(
        coefs.getVectorCoefficient(rotGradCoefName(i)), dt * _b_inv));
    m_i_phi.Assemble();
    m_i_phi.Finalize();
    _h_blocks(i, RR_PHI) = m_i_phi.ParallelAssemble();
  }

  {
    mfem::ParBilinearForm m_omega_t(_fespace);
    m_omega_t.AddDomainIntegrator(
        new mfem::MassIntegrator(coefs.getScalarCoefficient(coefName("DT_Fw"))));
    m_omega_t.Assemble();
    m_omega_t.Finalize();
    _h_blocks(RR_OMEGA, RR_T) = m_omega_t.ParallelAssemble();
  }

  {
    mfem::ParBilinearForm m_n_t(_fespace);
    m_n_t.AddDomainIntegrator(
        new mfem::MassIntegrator(coefs.getScalarCoefficient(coefName("DT_Fn"))));
    m_n_t.Assemble();
    m_n_t.Finalize();
    _h_blocks(RR_N, RR_T) = m_n_t.ParallelAssemble();
  }

  // phi block row. Requiring the discrete Poisson equation to hold at the updated state,
  //   L phi_new + M omega_new = 0, with phi_new = phi + dt k_phi, omega_new = omega + dt k_omega
  // gives
  //   dt L k_phi + dt M k_omega = -(L phi + M omega).
  // The row is linear, so the constraint is satisfied exactly at every step.
  {
    mfem::ParBilinearForm l_phi(_fespace);
    l_phi.AddDomainIntegrator(new mfem::DiffusionIntegrator);
    l_phi.Assemble();
    l_phi.Finalize();

    mfem::ParBilinearForm m_phi(_fespace);
    m_phi.AddDomainIntegrator(new mfem::MassIntegrator);
    m_phi.Assemble();
    m_phi.Finalize();

    mfem::Vector b_phi(_fespace->GetTrueVSize());
    b_phi = 0.0;
    l_phi.TrueAddMult(state.GetBlock(RR_PHI), b_phi, -1.0);
    m_phi.TrueAddMult(state.GetBlock(RR_OMEGA), b_phi, -1.0);
    // Dirichlet DoFs are pinned: k_phi = 0 there, so phi keeps its boundary values.
    b_phi.SetSubVector(_phi_ess_tdofs, 0.0);
    _true_rhs.GetBlock(RR_PHI) = b_phi;

    // EliminateRowsCols zeroes the essential rows and columns and leaves 1 on the
    // diagonal, which with a zero right-hand side enforces k_phi = 0 there.
    auto * a_phi_phi = l_phi.ParallelAssemble();
    *a_phi_phi *= dt;
    delete a_phi_phi->EliminateRowsCols(_phi_ess_tdofs);
    _h_blocks(RR_PHI, RR_PHI) = a_phi_phi;

    // Only the rows are eliminated here: the essential rows must read 1 * k_phi = 0.
    auto * a_phi_omega = m_phi.ParallelAssemble();
    *a_phi_omega *= dt;
    a_phi_omega->EliminateRows(_phi_ess_tdofs);
    _h_blocks(RR_PHI, RR_OMEGA) = a_phi_omega;
  }

  _system_operator.reset(mfem::HypreParMatrixFromBlocks(_h_blocks));
}

void
RogersRicci2DProblemOperator::ImplicitSolve(const mfem::real_t dt,
                                            const mfem::Vector &,
                                            mfem::Vector & k)
{
  _problem_data.coefficients.setTime(GetTime());
  formSystem(dt);

  // A HypreParMatrix has no GetGradient, so the linear operator is supplied explicitly
  // rather than through the two-argument overload.
  SolveWithOperator(*_system_operator, *_system_operator, _true_rhs, _true_x);

  k.MakeRef(_true_x, 0);
}

void
RogersRicci2DProblemOperator::Solve()
{
  auto & dt = _problem.dt();
  auto & gfs = _problem_data.gridfunctions;
  auto & tdm = _problem_data.time_derivative_map;

  // Snapshot the state so the time derivatives can be differenced after the step.
  for (const auto & trial_var_name : _trial_var_names)
    gfs.GetRef(tdm.getTimeDerivativeName(trial_var_name)) = gfs.GetRef(trial_var_name);

  _problem_data.ode_solver->Step(*_trial_true_vector, _problem.time(), dt);
  SetTrialVariablesFromTrueVectors();

  for (const auto & trial_var_name : _trial_var_names)
    (gfs.GetRef(tdm.getTimeDerivativeName(trial_var_name)) -= gfs.GetRef(trial_var_name)) /= -dt;
}

#endif