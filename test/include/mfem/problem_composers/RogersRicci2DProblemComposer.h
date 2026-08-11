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

#include "MFEMProblem.h"
#include "ProblemComposerBase.h"
#include "ProblemOperatorBase.h"
#include "RogersRicci2DProblemOperator.h"

namespace Moose::MFEM
{
/**
 * Custom 2D Rogers Ricci builder required to build MFEM Problem Operators
 * used by the executioner
 */
class RogersRicci2DProblemComposer : public ProblemComposerBase
{
public:
  static InputParameters validParams();

  RogersRicci2DProblemComposer(const InputParameters & parameters);

  ~RogersRicci2DProblemComposer() = default;

  /// Returns a pointer to a freshly minted operator.
  std::shared_ptr<ProblemOperatorBase> createProblemOperator(MFEMProblem & _mfem_problem) override;
};
};

#endif
