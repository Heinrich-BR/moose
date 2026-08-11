//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#ifdef MOOSE_MFEM_ENABLED

#include "RogersRicci2DProblemComposer.h"

namespace Moose::MFEM
{
registerMooseObject("MooseApp", RogersRicci2DProblemComposer);

InputParameters
RogersRicci2DProblemComposer::validParams()
{
  InputParameters params = ProblemComposerBase::validParams();
  params.addClassDescription("Builds the problem operator for the 2D Rogers-Ricci plasma "
                             "turbulence model");

  params.addRequiredParam<std::vector<VariableName>>(
      "variables",
      "The three variables evolved in time, in the order 'T omega n'. Their ordering fixes their "
      "position in the state block vector.");
  params.addRequiredParam<VariableName>(
      "potential_variable",
      "Variable holding the electrostatic potential phi, solved for at each stage rather than "
      "evolved in time.");
  params.addRequiredParam<MFEMScalarCoefficientName>(
      "potential_coefficient",
      "Coefficient the reaction terms sample the electrostatic potential from. Either the "
      "potential variable itself or a quadrature function caching it.");
  params.addRequiredParam<MFEMScalarCoefficientName>(
      "source_coefficient", "Coefficient supplying the particle and heat source S_n.");

  params.addParam<mfem::real_t>(
      "eps2",
      1e-4,
      "Regularisation eps^2 used in sqrt(T^2 + eps^2), guarding the division by T in the sheath "
      "factor exp(Lambda - phi/T).");
  params.addParam<mfem::real_t>("Binv", 40.0, "Inverse of the magnetic field strength.");
  params.addParam<mfem::real_t>("Lambda", 3.0, "Sheath potential parameter.");
  params.addRangeCheckedParam<int>(
      "num_active",
      NUM_VARS,
      "num_active >= 1 & num_active <= 3",
      "How many variables to evolve in time: 1 = T only, 2 = T and omega, 3 = the full system. "
      "The remaining variables are held at their initial values.");

  params.addParam<mfem::real_t>(
      "l_tol", 1e-10, "Relative tolerance for the block GMRES solving each Newton step.");
  params.addParam<int>("l_max_its", 200, "Maximum block GMRES iterations per Newton step.");
  params.addParam<int>("krylov_dim", 50, "Krylov subspace dimension for the block GMRES.");

  return params;
}

RogersRicci2DProblemComposer::RogersRicci2DProblemComposer(const InputParameters & parameters)
  : ProblemComposerBase(parameters)
{
  if (getParam<std::vector<VariableName>>("variables").size() != NUM_VARS)
    paramError("variables", "Exactly ", NUM_VARS, " variables must be supplied, as 'T omega n'.");
}

std::shared_ptr<ProblemOperatorBase>
RogersRicci2DProblemComposer::createProblemOperator(MFEMProblem & _mfem_problem)
{
  return std::make_shared<RogersRicci2DProblemOperator>(_mfem_problem, parameters());
}
};

#endif