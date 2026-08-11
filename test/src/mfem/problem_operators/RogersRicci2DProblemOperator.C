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

namespace
{

/// Krylov solver for the block stage Jacobian, preconditioned by one AMG per variable

class BlockNewtonLinearSolver : public mfem::IterativeSolver
{
public:
  BlockNewtonLinearSolver(MPI_Comm comm, mfem::real_t rtol, int max_its, int kdim);

  void SetOperator(const mfem::Operator & op) override;
  void Mult(const mfem::Vector & b, mfem::Vector & x) const override;

private:
  mutable mfem::GMRESSolver _gmres;
  std::unique_ptr<mfem::BlockDiagonalPreconditioner> _bdp;
  std::vector<std::unique_ptr<mfem::HypreBoomerAMG>> _amgs;
};

BlockNewtonLinearSolver::BlockNewtonLinearSolver(MPI_Comm comm,
                                                 mfem::real_t rtol,
                                                 int max_its,
                                                 int kdim)
  : mfem::IterativeSolver(comm), _gmres(comm)
{
  SetRelTol(rtol);
  SetAbsTol(0.0);
  SetMaxIter(max_its);
  _gmres.SetKDim(kdim);
  _gmres.SetPrintLevel(0);

  _amgs.resize(NUM_VARS);
  for (const auto s : make_range(NUM_VARS))
  {
    _amgs[s] = std::make_unique<mfem::HypreBoomerAMG>();
    _amgs[s]->SetPrintLevel(0);
  }
}

void
BlockNewtonLinearSolver::SetOperator(const mfem::Operator & op)
{
  const auto * const bop = dynamic_cast<const mfem::BlockOperator *>(&op);
  MFEM_VERIFY(bop != nullptr,
              "BlockNewtonLinearSolver: expected a BlockOperator from "
              "RogersRicciStageOperator::GetGradient().");

  for (const auto s : make_range(NUM_VARS))
    _amgs[s]->SetOperator(dynamic_cast<const mfem::HypreParMatrix &>(bop->GetBlock(s, s)));

  if (!_bdp)
  {
    _bdp = std::make_unique<mfem::BlockDiagonalPreconditioner>(bop->RowOffsets());
    for (const auto s : make_range(NUM_VARS))
      _bdp->SetDiagonalBlock(s, _amgs[s].get());
    _gmres.SetPreconditioner(*_bdp);
  }

  _gmres.SetOperator(*bop);
  height = bop->Height();
  width = bop->Width();
}

void
BlockNewtonLinearSolver::Mult(const mfem::Vector & b, mfem::Vector & x) const
{
  // Pin Newton's residual (input) and direction (output) to device memory.
  const_cast<mfem::Vector &>(b).UseDevice(true);
  x.UseDevice(true);

  // Propagate any adaptive tolerance the Newton solver has set since the last call.
  _gmres.SetRelTol(rel_tol);
  _gmres.SetAbsTol(abs_tol);
  _gmres.SetMaxIter(max_iter);
  _gmres.Mult(b, x);
}
} // namespace

RogersRicci2DProblemOperator::RogersRicci2DProblemOperator(MFEMProblem & problem,
                                                           const InputParameters & parameters)
  : Moose::MFEM::TimeDependentProblemOperator(problem),
    _phi_var_name(parameters.get<VariableName>("potential_variable")),
    _potential_coef_name(parameters.get<MFEMScalarCoefficientName>("potential_coefficient")),
    _source_coef_name(parameters.get<MFEMScalarCoefficientName>("source_coefficient")),
    _eps2(parameters.get<mfem::real_t>("eps2")),
    _Binv(parameters.get<mfem::real_t>("Binv")),
    _Lambda(parameters.get<mfem::real_t>("Lambda")),
    _num_active(parameters.get<int>("num_active")),
    _lin_rtol(parameters.get<mfem::real_t>("l_tol")),
    _lin_max_its(parameters.get<int>("l_max_its")),
    _kdim(parameters.get<int>("krylov_dim")),
    // Rotation by pi/2, so that v_d = R grad(phi) is the ExB drift direction in 2D.
    _rotmat({{0.0, -1.0}, {1.0, 0.0}})
{
  // The executioner calls SetGridFunctions() before Init(), so the trial variable names must be
  // known by the end of construction. They are plain names taken from the input, so no problem
  // data, which does not exist yet at this point, needs to be touched to resolve them.
  const auto & variables = parameters.get<std::vector<VariableName>>("variables");
  _trial_var_names.assign(variables.begin(), variables.end());
  _test_var_names = _trial_var_names;
}

void
RogersRicci2DProblemOperator::Init(mfem::BlockVector & X)
{
  // Copies the initial conditions out of the trial gridfunctions into X
  Moose::MFEM::ProblemOperatorBase::Init(X);

  _phi = _problem_data.gridfunctions.Get(_phi_var_name);

  auto & pmesh = _problem.mesh().getMFEMParMesh();
  _h = pmesh.GetElementSize(pmesh.GetTypicalElementTransformation());

  buildCoefficients();
  buildMassMatrix();
  buildKForm();
  buildPhiSolver();
  buildNonlinearForm();
  buildNewtonLinearSolver();

  _stage_op =
      std::make_unique<RogersRicciStageOperator>(*_F_form, _block_true_offsets_trial, _num_active);

  // Give K a matrix consistent with the initial omega. Every stage reassembles it, but the stage
  // operator reads it in SetParameters() before that first reassembly has happened.
  updatePhi();
  reassembleK();

  auto & ode_solver = _problem_data.ode_solver;
  ode_solver = std::make_unique<mfem::BackwardEulerSolver>();
  ode_solver->Init(*this);
  SetTime(_problem.time());
}

void
RogersRicci2DProblemOperator::buildCoefficients()
{
  _grad_rotate = std::make_unique<mfem::MatrixConstantCoefficient>(_rotmat);

  // v_d = R grad(phi). The gradient coefficient is declared automatically by the phi MFEMVariable.
  _vd = std::make_unique<mfem::MatrixVectorProductCoefficient>(
      *_grad_rotate, _problem_data.coefficients.getVectorCoefficient(_phi_var_name + "_grad"));

  // Drift velocity v_E = v_d / B.
  _v_E = std::make_unique<mfem::ScalarVectorProductCoefficient>(_Binv, *_vd);

  // SUW stabilisation tensor h / (2 sqrt(|v_E|^2 + eps^2)) * (v_E outer v_E), which adds
  // diffusion along the drift direction only.
  _v_E_sq = std::make_unique<mfem::InnerProductCoefficient>(*_v_E, *_v_E);
  _suw_scalar = std::make_unique<mfem::TransformedCoefficient>(
      _v_E_sq.get(),
      [h = _h, eps2 = _eps2](mfem::real_t v_sq) { return h / (2.0 * std::sqrt(v_sq + eps2)); });
  _v_E_outer = std::make_unique<mfem::OuterProductCoefficient>(*_v_E, *_v_E);
  _SUW_matcoef = std::make_unique<mfem::ScalarMatrixProductCoefficient>(*_suw_scalar, *_v_E_outer);
}

void
RogersRicci2DProblemOperator::buildMassMatrix()
{
  _M_form = std::make_unique<mfem::ParBilinearForm>(_trial_variables.at(T_IDX)->ParFESpace());
  _M_form->AddDomainIntegrator(new mfem::MassIntegrator);
  _M_form->Assemble();
  _M_form->Finalize();
  _M_mat.reset(_M_form->ParallelAssemble());
}

void
RogersRicci2DProblemOperator::buildKForm()
{
  // K = -(1/B) (v_d . grad u, v) + (c v_E outer v_E : grad u outer grad v), i.e. advection by the
  // drift velocity plus SUW stabilisation
  _K_form = std::make_unique<mfem::ParBilinearForm>(_trial_variables.at(T_IDX)->ParFESpace());
  _K_form->AddDomainIntegrator(new mfem::ConvectionIntegrator(*_vd, -_Binv));
  _K_form->AddDomainIntegrator(new mfem::DiffusionIntegrator(*_SUW_matcoef));
}

void
RogersRicci2DProblemOperator::buildPhiSolver()
{
  auto * const fes = _phi->ParFESpace();

  mfem::Array<int> ess_bdr(_problem.mesh().getMFEMParMesh().bdr_attributes.Max());
  ess_bdr = 1;
  fes->GetEssentialTrueDofs(ess_bdr, _ess_tdof_list_phi);

  _a_phi = std::make_unique<mfem::ParBilinearForm>(fes);
  _a_phi->AddDomainIntegrator(new mfem::DiffusionIntegrator);
  _a_phi->Assemble();
  _a_phi->Finalize();

  _neg_omega_coef = std::make_unique<mfem::ProductCoefficient>(
      -1.0, _problem_data.coefficients.getScalarCoefficient(_trial_var_names.at(OMEGA_IDX)));
  _b_phi = std::make_unique<mfem::ParLinearForm>(fes);
  _b_phi->AddDomainIntegrator(new mfem::DomainLFIntegrator(*_neg_omega_coef));

  if (!_problem_data.jacobian_solver)
    mooseError("RogersRicci2DProblemOperator requires a linear solver to be configured in the "
               "[Solvers] block, which it uses for the electrostatic potential solve.");

  // The potential operator does not change with time, so it is formed and handed to the solver
  // once here rather than on every stage, which would rebuild the AMG hierarchy each time.
  _a_phi->FormSystemMatrix(_ess_tdof_list_phi, _A_phi);
  _problem_data.jacobian_solver->SetOperator(*_A_phi);
}

void
RogersRicci2DProblemOperator::buildNonlinearForm()
{
  mfem::Array<mfem::ParFiniteElementSpace *> fes(NUM_VARS);
  for (const auto i : make_range(NUM_VARS))
    fes[i] = _trial_variables.at(i)->ParFESpace();

  _F_form = std::make_unique<mfem::ParBlockNonlinearForm>(fes);
  _F_form->AddDomainIntegrator(new RogersRicciNLFIntegrator(
      _problem_data.coefficients.getScalarCoefficient(_potential_coef_name),
      _problem_data.coefficients.getScalarCoefficient(_source_coef_name),
      _num_active,
      _Lambda,
      _eps2));
}

void
RogersRicci2DProblemOperator::buildNewtonLinearSolver()
{
  if (!_problem_data.nonlinear_solver)
    mooseError("RogersRicci2DProblemOperator requires a nonlinear solver to be configured in the "
               "[Solvers] block, which it uses to solve each implicit stage.");

  _newton_lin_solver =
      std::make_unique<BlockNewtonLinearSolver>(_problem.getComm(), _lin_rtol, _lin_max_its, _kdim);
  _problem_data.nonlinear_solver->SetLinearSolver(*_newton_lin_solver);
}

void
RogersRicci2DProblemOperator::updatePhi()
{
  _b_phi->Assemble();

  // The operator returned here is discarded: it is time-invariant and the solver already holds an
  // equivalent matrix from buildPhiSolver(). Only the eliminated right hand side B is wanted.
  mfem::OperatorHandle A_unused;
  mfem::Vector X, B;
  _a_phi->FormLinearSystem(_ess_tdof_list_phi, *_phi, *_b_phi, A_unused, X, B);

  _problem_data.jacobian_solver->Mult(B, X);
  _a_phi->RecoverFEMSolution(X, *_b_phi, *_phi);
}

void
RogersRicci2DProblemOperator::reassembleK()
{
  _K_form->Update();
  _K_form->Assemble();
  _K_form->Finalize();
  _K_mat.reset(_K_form->ParallelAssemble());
}

void
RogersRicci2DProblemOperator::Solve()
{
  auto & dt = _problem.dt();
  _problem_data.ode_solver->Step(*_trial_true_vector, _problem.time(), dt);

  // Synchronise the gridfunctions with the updated true DoF data for output and postprocessing.
  SetTrialVariablesFromTrueVectors();
}

void
RogersRicci2DProblemOperator::ImplicitSolve(const mfem::real_t gamma,
                                            const mfem::Vector & u_pred,
                                            mfem::Vector & k)
{
  _problem_data.coefficients.setTime(GetTime());

  mfem::BlockVector u_pred_blk(const_cast<mfem::Vector &>(u_pred), _block_true_offsets_trial);
  u_pred_blk.UseDevice(true);
  for (const auto s : make_range(NUM_VARS))
    u_pred_blk.GetBlock(s).UseDevice(true);
  u_pred_blk.SyncToBlocks();

  _trial_variables.at(OMEGA_IDX)->SetFromTrueDofs(u_pred_blk.GetBlock(OMEGA_IDX));
  updatePhi();
  reassembleK();

  _problem_data.coefficients.markSolutionChanged();

  _stage_op->SetParameters(gamma, u_pred_blk, *_M_mat, *_K_mat);

  auto & nonlinear_solver = *_problem_data.nonlinear_solver;
  nonlinear_solver.SetOperator(*_stage_op);

  k = 0.0;
  const mfem::Vector zero;
  nonlinear_solver.Mult(zero, k);

  if (!static_cast<mfem::IterativeSolver &>(nonlinear_solver.GetSolver()).GetConverged())
    mooseError("RogersRicci2DProblemOperator: the Newton solve for the implicit stage failed to "
               "converge (gamma = ",
               gamma,
               ").");
}

RogersRicciStageOperator::RogersRicciStageOperator(mfem::ParBlockNonlinearForm & F_form,
                                                   const mfem::Array<int> & block_true_offsets,
                                                   int num_active)
  : mfem::Operator(block_true_offsets.Last()),
    _F_form(F_form),
    _block_true_offsets(block_true_offsets),
    _num_active(num_active),
    _z(block_true_offsets),
    _tmp_block(block_true_offsets[1] - block_true_offsets[0]),
    _Jac_block(block_true_offsets)
{
  _z.UseDevice(true);
  _tmp_block.UseDevice(true);
}

void
RogersRicciStageOperator::SetParameters(mfem::real_t gamma,
                                        const mfem::BlockVector & u_pred,
                                        mfem::HypreParMatrix & M,
                                        mfem::HypreParMatrix & K)
{
  _gamma = gamma;
  _u_pred = &u_pred;
  _M = &M;
  _K = &K;
  _M_plus_gK.reset(Add(1.0, M, _gamma, K));
}

void
RogersRicciStageOperator::Mult(const mfem::Vector & k_vec, mfem::Vector & R_vec) const
{
  MFEM_VERIFY(_u_pred != nullptr,
              "RogersRicciStageOperator: SetParameters() not called before Mult().");

  const_cast<mfem::Vector &>(k_vec).UseDevice(true);
  R_vec.UseDevice(true);

  // z = u_pred + gamma * k
  add(*_u_pred, _gamma, k_vec, _z);
  _z.SyncToBlocks();

  // R = F(z), evaluated on host: RogersRicciNLFIntegrator is host-only.
  _z.HostRead();
  R_vec.HostWrite();
  _F_form.Mult(_z, R_vec);
  R_vec.ReadWrite();

  // R += M k + K z
  mfem::BlockVector k_blk(const_cast<mfem::Vector &>(k_vec), _block_true_offsets);
  mfem::BlockVector R_blk(R_vec, _block_true_offsets);
  k_blk.UseDevice(true);
  R_blk.UseDevice(true);
  for (const auto s : make_range(NUM_VARS))
  {
    k_blk.GetBlock(s).UseDevice(true);
    R_blk.GetBlock(s).UseDevice(true);
  }
  k_blk.SyncToBlocks();
  R_blk.SyncToBlocks();

  for (const auto s : make_range(NUM_VARS))
  {
    _M->Mult(k_blk.GetBlock(s), _tmp_block);
    R_blk.GetBlock(s) += _tmp_block;
    _K->Mult(_z.GetBlock(s), _tmp_block);
    R_blk.GetBlock(s) += _tmp_block;
  }

  R_blk.SyncFromBlocks();
}

mfem::Operator &
RogersRicciStageOperator::GetGradient(const mfem::Vector & k_vec) const
{
  MFEM_VERIFY(_u_pred != nullptr,
              "RogersRicciStageOperator: SetParameters() not called before GetGradient().");

  // z = u_pred + gamma * k
  add(*_u_pred, _gamma, k_vec, _z);
  _z.SyncToBlocks();
  _z.HostRead();

  mfem::BlockOperator & Fgrad = _F_form.GetGradient(_z);
  auto Fblock = [&](int i, int j) -> mfem::HypreParMatrix &
  { return dynamic_cast<mfem::HypreParMatrix &>(Fgrad.GetBlock(i, j)); };

  _diag_T.reset(Add(1.0, *_M_plus_gK, _gamma, Fblock(T_IDX, T_IDX)));
  _Jac_block.SetDiagonalBlock(T_IDX, _diag_T.get());

  _Jac_block.SetDiagonalBlock(OMEGA_IDX, (_num_active >= 2) ? _M_plus_gK.get() : _M);

  // n row: M + gamma * K + gamma * dF_n/dn when active; bare M when frozen.
  if (_num_active >= 3)
  {
    _diag_n.reset(Add(1.0, *_M_plus_gK, _gamma, Fblock(N_IDX, N_IDX)));
    _Jac_block.SetDiagonalBlock(N_IDX, _diag_n.get());
  }
  else
  {
    _diag_n.reset();
    _Jac_block.SetDiagonalBlock(N_IDX, _M);
  }

  // Off-diagonals: gamma * dF_omega/dT and gamma * dF_n/dT.
  if (_num_active >= 2)
  {
    _gFwT.reset(Add(0.0, Fblock(OMEGA_IDX, T_IDX), _gamma, Fblock(OMEGA_IDX, T_IDX)));
    _Jac_block.SetBlock(OMEGA_IDX, T_IDX, _gFwT.get());
  }
  if (_num_active >= 3)
  {
    _gFnT.reset(Add(0.0, Fblock(N_IDX, T_IDX), _gamma, Fblock(N_IDX, T_IDX)));
    _Jac_block.SetBlock(N_IDX, T_IDX, _gFnT.get());
  }

  return _Jac_block;
}

void
RogersRicciNLFIntegrator::AssembleElementVector(const mfem::Array<const mfem::FiniteElement *> & el,
                                                mfem::ElementTransformation & Tr,
                                                const mfem::Array<const mfem::Vector *> & elfun,
                                                const mfem::Array<mfem::Vector *> & elvec)
{
  MFEM_VERIFY(el.Size() == NUM_VARS,
              "RogersRicciNLFIntegrator expects exactly " << NUM_VARS << " FE spaces.");
  const int dof = el[0]->GetDof();

  mfem::Vector shape(dof);
  for (const auto s : make_range(NUM_VARS))
  {
    elvec[s]->SetSize(dof);
    *elvec[s] = 0.0;
  }

  const mfem::IntegrationRule & ir =
      mfem::IntRules.Get(el[0]->GetGeomType(), 2 * el[0]->GetOrder() + 3);

  for (const auto q : make_range(ir.GetNPoints()))
  {
    const mfem::IntegrationPoint & ip = ir.IntPoint(q);
    Tr.SetIntPoint(&ip);
    el[0]->CalcShape(ip, shape);

    const mfem::real_t T = shape * (*elfun[T_IDX]);
    const mfem::real_t n = shape * (*elfun[N_IDX]);

    const mfem::real_t phi = _phi->Eval(Tr, ip);
    const mfem::real_t Sn = _Sn->Eval(Tr, ip);

    const mfem::real_t Treg = std::sqrt(T * T + _eps2);
    const mfem::real_t A = _Lambda - phi / Treg;
    const mfem::real_t eA = std::exp(A);

    const mfem::real_t F_T = (T / 36.0) * (1.71 * eA - 0.71) - Sn;
    const mfem::real_t F_w = (eA - 1.0) / 24.0;
    const mfem::real_t F_n = (n / 24.0) * eA - Sn;

    const mfem::real_t w = ip.weight * Tr.Weight();

    elvec[T_IDX]->Add(w * F_T, shape);
    if (_num_active >= 2)
    {
      elvec[OMEGA_IDX]->Add(w * F_w, shape);
    }
    if (_num_active >= 3)
    {
      elvec[N_IDX]->Add(w * F_n, shape);
    }
  }
}

void
RogersRicciNLFIntegrator::AssembleElementGrad(const mfem::Array<const mfem::FiniteElement *> & el,
                                              mfem::ElementTransformation & Tr,
                                              const mfem::Array<const mfem::Vector *> & elfun,
                                              const mfem::Array2D<mfem::DenseMatrix *> & elmats)
{
  MFEM_VERIFY(el.Size() == NUM_VARS,
              "RogersRicciNLFIntegrator expects exactly " << NUM_VARS << " FE spaces.");
  const int dof = el[0]->GetDof();

  mfem::Vector shape(dof);
  for (const auto s : make_range(NUM_VARS))
  {
    for (const auto t : make_range(NUM_VARS))
    {
      elmats(s, t)->SetSize(dof, dof);
      *elmats(s, t) = 0.0;
    }
  }

  const mfem::IntegrationRule & ir =
      mfem::IntRules.Get(el[0]->GetGeomType(), 2 * el[0]->GetOrder() + 3);

  for (const auto q : make_range(ir.GetNPoints()))
  {
    const mfem::IntegrationPoint & ip = ir.IntPoint(q);
    Tr.SetIntPoint(&ip);
    el[0]->CalcShape(ip, shape);

    const mfem::real_t T = shape * (*elfun[T_IDX]);
    const mfem::real_t n = shape * (*elfun[N_IDX]);

    const mfem::real_t phi = _phi->Eval(Tr, ip);
    const mfem::real_t Treg = std::sqrt(T * T + _eps2);
    const mfem::real_t A = _Lambda - phi / Treg;
    const mfem::real_t eA = std::exp(A);

    // d(1/Treg)/dT = -T/Treg^3   ->   dA/dT = phi*T/Treg^3.
    const mfem::real_t Treg3 = Treg * Treg * Treg;
    const mfem::real_t deA_dT = eA * phi * T / Treg3;

    // dF_T/dT = (1.71 eA - 0.71)/36 + (T/36)*1.71*dE/dT
    const mfem::real_t dFT_dT = (1.71 * eA - 0.71) / 36.0 + (T * 1.71 / 36.0) * deA_dT;
    const mfem::real_t dFw_dT = deA_dT / 24.0;
    const mfem::real_t dFn_dT = n * deA_dT / 24.0;
    const mfem::real_t dFn_dn = eA / 24.0;

    const mfem::real_t w = ip.weight * Tr.Weight();

    mfem::AddMult_a_VVt(w * dFT_dT, shape, *elmats(T_IDX, T_IDX));
    if (_num_active >= 2)
    {
      mfem::AddMult_a_VVt(w * dFw_dT, shape, *elmats(OMEGA_IDX, T_IDX));
    }
    if (_num_active >= 3)
    {
      mfem::AddMult_a_VVt(w * dFn_dT, shape, *elmats(N_IDX, T_IDX));
      mfem::AddMult_a_VVt(w * dFn_dn, shape, *elmats(N_IDX, N_IDX));
    }
  }
}

#endif