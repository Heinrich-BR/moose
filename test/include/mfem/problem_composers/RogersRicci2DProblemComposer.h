//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#ifdef MOOSE_MFEM_ENABLED

#pragma once

#include "MFEMProblemComposer.h"
#include "RogersRicci2DProblemOperator.h"

/**
 * Builds the problem operator for the 2D Rogers-Ricci drift-turbulence model, collecting
 * the four block variables, the particle source and the physical parameters from the
 * input file.
 */
class RogersRicci2DProblemComposer : public MFEMProblemComposer
{
public:
  static InputParameters validParams();

  RogersRicci2DProblemComposer(const InputParameters & parameters);

  /// Returns a pointer to a freshly minted operator.
  std::shared_ptr<Moose::MFEM::ProblemOperatorBase>
  createProblemOperator(MFEMProblem & mfem_problem) override;
};

#endif
