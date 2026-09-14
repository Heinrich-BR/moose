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

registerMooseObject("MooseApp", RogersRicci2DProblemComposer);

InputParameters
RogersRicci2DProblemComposer::validParams()
{
  InputParameters params = MFEMProblemComposer::validParams();
  params.addClassDescription("Composes the problem operator for the 2D Rogers-Ricci "
                             "drift-turbulence model.");
  params.addRequiredParam<VariableName>("temperature", "The temperature variable, T.");
  params.addRequiredParam<VariableName>("vorticity", "The vorticity variable, omega.");
  params.addRequiredParam<VariableName>("density", "The density variable, n.");
  params.addRequiredParam<VariableName>("potential",
                                        "The electrostatic potential variable, phi.");
  params.addRequiredParam<MFEMScalarCoefficientName>(
      "source",
      "Particle source S_n. In the normalised equations this also supplies the "
      "temperature source S_T.");
  params.addParam<mfem::real_t>(
      "b_inv", 40.0, "Prefactor on the Poisson brackets, 1/B in normalised units.");
  params.addParam<mfem::real_t>(
      "lambda", 3.0, "Constant in the exponent of the sheath-loss terms.");
  params.addParam<mfem::real_t>(
      "eps_squared",
      1e-12,
      "Regularisation keeping sqrt(T^2 + eps_squared) bounded away from zero.");
  return params;
}

RogersRicci2DProblemComposer::RogersRicci2DProblemComposer(const InputParameters & parameters)
  : MFEMProblemComposer(parameters)
{
}

std::shared_ptr<Moose::MFEM::ProblemOperatorBase>
RogersRicci2DProblemComposer::createProblemOperator(MFEMProblem & mfem_problem)
{
  // Ordered to match RogersRicciVarIdx, which fixes the block ordering of the system.
  const std::vector<VariableName> variable_names{getParam<VariableName>("temperature"),
                                                 getParam<VariableName>("vorticity"),
                                                 getParam<VariableName>("density"),
                                                 getParam<VariableName>("potential")};

  return std::make_shared<RogersRicci2DProblemOperator>(
      mfem_problem,
      variable_names,
      getParam<MFEMScalarCoefficientName>("source"),
      getParam<mfem::real_t>("b_inv"),
      getParam<mfem::real_t>("lambda"),
      getParam<mfem::real_t>("eps_squared"));
}

#endif
