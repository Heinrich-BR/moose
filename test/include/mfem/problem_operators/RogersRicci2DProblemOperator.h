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

#include "TimeDependentProblemOperator.h"

/// Block ordering of the Rogers-Ricci system.
enum RogersRicciVarIdx : int
{
  RR_T = 0,
  RR_OMEGA = 1,
  RR_N = 2,
  RR_PHI = 3
};

/// Total number of block unknowns.
static constexpr int RR_NUM_VARS = 4;

/// Number of fields carrying a time derivative. The transported variables occupy the
/// leading block rows; RR_PHI is the algebraic one.
static constexpr int RR_NUM_TRANSPORTED = 3;

/**
 * Time-dependent problem operator for the 2D Rogers-Ricci drift-turbulence model.
 *
 * Three transported fields (temperature T, vorticity omega, density n) obey
 *
 *     M x_dot + K x + (F_x, v) = 0
 *
 * with a shared K built from E x B advection (a Poisson bracket) plus streamline-upwind
 * stabilisation, both depending on the solution only through the drift velocity
 * v_d = R grad(phi). The electrostatic potential phi is a fourth block unknown obeying the
 * algebraic Poisson constraint grad^2 phi = omega, which enters as a fourth block row.
 *
 * Everything MOOSE already provides is taken from the input file: the mesh, the FE space,
 * the four variables and their initial conditions, the source coefficient, the linear
 * solver and the outputs. This class supplies only the block assembly, which has no
 * equivalent in the existing kernel and equation-system machinery.
 *
 * The coefficients are owned by the problem's CoefficientManager rather than by this
 * class. They are declared once during Init() and retrieved by name when assembling. That
 * is sound because every MFEM coefficient involved holds pointers to its inputs rather
 * than copies, so the whole graph tracks the grid functions as the solution advances; the
 * time step reaches the reaction terms through the _dt member, which the lambdas read.
 */
class RogersRicci2DProblemOperator : public Moose::MFEM::TimeDependentProblemOperator
{
public:
  RogersRicci2DProblemOperator(MFEMProblem & problem,
                               const std::vector<VariableName> & variable_names,
                               const MFEMScalarCoefficientName & source_name,
                               mfem::real_t b_inv,
                               mfem::real_t lambda,
                               mfem::real_t eps_squared);

  ~RogersRicci2DProblemOperator() override;

  virtual void SetGridFunctions() override;
  virtual void Init(mfem::BlockVector & X) override;
  virtual void Solve() override;
  virtual void
  ImplicitSolve(const mfem::real_t dt, const mfem::Vector & x, mfem::Vector & k) override;

protected:
  /// Declare every solution-dependent coefficient in the problem's CoefficientManager.
  /// Called once from Init(); the coefficients then track the grid functions on their own.
  void declareCoefficients();

  /// Assemble the block system operator and right-hand side for a step of size dt.
  void formSystem(mfem::real_t dt);

  /// Release the HypreParMatrix blocks owned by _h_blocks.
  void deleteBlocks();

private:
  /// Names of the four block variables, in RogersRicciVarIdx order.
  const std::vector<VariableName> _variable_names;

  /// Name of the particle source S_n, which in the normalised equations also supplies S_T.
  const MFEMScalarCoefficientName _source_name;

  /// Prefactor on the Poisson brackets, 1/B in normalised units.
  const mfem::real_t _b_inv;

  /// Constant in the exponent of the sheath-loss terms.
  const mfem::real_t _lambda;

  /// Regularisation keeping sqrt(T^2 + eps2) bounded away from zero.
  const mfem::real_t _eps_squared;

  /// Step size of the step currently being taken. Read by the reaction-term lambdas, which
  /// bake dt into the Gateaux derivative coefficients.
  mfem::real_t _dt{0.0};

  /// Shared H1 space of all four variables, cached at Init().
  mfem::ParFiniteElementSpace * _fespace{nullptr};

  /// Representative element size, used by the streamline-upwind term. Sampling a single
  /// element is only rank-independent because this mesh is uniform.
  mfem::real_t _element_size{0.0};

  /// Essential true DoFs of phi. The transported variables have pure Neumann conditions,
  /// so they need no list.
  mfem::Array<int> _phi_ess_tdofs;

  /// Block-typed handle on the true-DoF state vector. The base class stores the same
  /// vector as a plain mfem::Vector, which cannot be indexed by block.
  mfem::BlockVector * _state{nullptr};

  /// Assembled blocks and the operator built from them.
  mfem::Array2D<const mfem::HypreParMatrix *> _h_blocks;
  std::unique_ptr<mfem::HypreParMatrix> _system_operator;
};

#endif