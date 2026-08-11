//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#ifdef MOOSE_MFEM_ENABLED

#include "MFEMNewtonNonlinearSolver.h"
#include "MFEMProblem.h"

registerMooseObject("MooseApp", MFEMNewtonNonlinearSolver);

namespace
{
/**
 * Newton solver optionally damping its step with a backtracking Armijo line search. Damping is
 * needed for residuals with a stiff exponential dependence on the state, where a full Newton step
 * can land on an iterate whose residual is orders of magnitude larger than the previous one's,
 * so that the undamped iteration diverges instead of converging.
 *
 * mfem::NewtonSolver exposes the step length only through the protected virtual
 * ComputeScalingFactor(), which it calls from inside its own iteration loop, so damping cannot be
 * injected without deriving from it.
 */
class DampedNewtonSolver : public mfem::NewtonSolver
{
public:
  DampedNewtonSolver(MPI_Comm comm, bool line_search)
    : mfem::NewtonSolver(comm), _line_search(line_search)
  {
  }

  /**
   * Return 1 if the line search is disabled, else alpha in (0, 1] satisfying the Armijo
   * sufficient decrease condition ||F(x - alpha c) - b|| <= (1 - _sigma * alpha) ||F(x) - b||,
   * halving alpha until it holds.
   */
  mfem::real_t ComputeScalingFactor(const mfem::Vector & x, const mfem::Vector & b) const override;

private:
  /// Whether to damp the Newton step; when false the full step is taken.
  const bool _line_search;
  /// Armijo sufficient decrease constant
  static constexpr mfem::real_t _sigma = 1e-4;
  /// Step halvings attempted before the shortest step tried is accepted
  static constexpr int _max_tries = 20;
  mutable mfem::Vector _x_trial, _r_trial;
};

mfem::real_t
DampedNewtonSolver::ComputeScalingFactor(const mfem::Vector & x, const mfem::Vector & b) const
{
  if (!_line_search)
    return 1.0;

  // r holds the residual at x, evaluated by Mult() before this is called.
  const mfem::real_t norm0 = Norm(r);
  if (norm0 == 0.0)
    return 1.0;

  _x_trial.SetSize(x.Size());
  _x_trial.UseDevice(true);
  _r_trial.SetSize(r.Size());
  _r_trial.UseDevice(true);

  mfem::real_t alpha = 1.0;
  for (const auto i : make_range(_max_tries))
  {
    if (i > 0)
      alpha *= 0.5;

    add(x, -alpha, c, _x_trial);
    oper->Mult(_x_trial, _r_trial);
    if (b.Size() == _r_trial.Size())
      _r_trial -= b;

    if (Norm(_r_trial) <= (1.0 - _sigma * alpha) * norm0)
      return alpha;
  }

  return alpha;
}
} // namespace

InputParameters
MFEMNewtonNonlinearSolver::validParams()
{
  InputParameters params = Moose::MFEM::NonlinearSolverBase::validParams();
  params.addClassDescription("MFEM native nonlinear solver using Newton's method.");
  MooseEnum line_search("none backtracking", "none");
  params.addParam<MooseEnum>(
      "line_search",
      line_search,
      "Damping applied to the Newton step. 'none' takes the full step; 'backtracking' halves the "
      "step until it satisfies an Armijo sufficient decrease condition, which is needed for "
      "residuals stiff enough that the undamped iteration diverges.");
  return params;
}

MFEMNewtonNonlinearSolver::MFEMNewtonNonlinearSolver(const InputParameters & parameters)
  : Moose::MFEM::NonlinearSolverBase(parameters)
{
  ConstructSolver();
}

void
MFEMNewtonNonlinearSolver::ConstructSolver()
{
  auto solver = std::make_unique<DampedNewtonSolver>(
      getMFEMProblem().getComm(), getParam<MooseEnum>("line_search") == "backtracking");
  solver->iterative_mode = getParam<bool>("use_initial_guess");
  solver->SetRelTol(getParam<mfem::real_t>("rel_tol"));
  solver->SetAbsTol(getParam<mfem::real_t>("abs_tol"));
  solver->SetMaxIter(getParam<unsigned int>("max_its"));
  solver->SetPrintLevel(getParam<unsigned int>("print_level"));
  _solver = std::move(solver);
}

void
MFEMNewtonNonlinearSolver::SetLinearSolver(mfem::Solver & solver)
{
  static_cast<mfem::NewtonSolver &>(GetSolver()).SetSolver(solver);
}
#endif
