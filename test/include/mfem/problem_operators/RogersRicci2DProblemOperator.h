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
#include "ProblemOperatorBase.h"
#include "TimeDependentProblemOperator.h"

#include <memory>
#include <string>
#include <vector>

/// Ordering of the evolved variables within the state block vector.
enum VarIdx : int
{
  T_IDX = 0,
  OMEGA_IDX = 1,
  N_IDX = 2
};
constexpr int NUM_VARS = 3;

/**
 * Custom BlockNonlinearFormIntegrator implementing the reaction terms F_T, F_omega and F_n of the
 * Rogers-Ricci model. All three couple to the electrostatic potential phi through the sheath
 * factor exp(Lambda - phi / sqrt(T^2 + eps^2)), which is what makes the system stiff.
 */
class RogersRicciNLFIntegrator : public mfem::BlockNonlinearFormIntegrator
{
public:
  RogersRicciNLFIntegrator(mfem::Coefficient & phi,
                           mfem::Coefficient & Sn,
                           int num_active = NUM_VARS,
                           mfem::real_t Lambda = 3.0,
                           mfem::real_t eps2 = 1e-4)
    : _phi(&phi), _Sn(&Sn), _Lambda(Lambda), _eps2(eps2), _num_active(num_active)
  {
    MFEM_VERIFY(num_active >= 1 && num_active <= NUM_VARS,
                "RogersRicciNLFIntegrator: num_active out of range.");
  }

  // Residual contribution to each block:
  //   elvec[T_IDX]     += integral of F_T(T, phi) * shape
  //   elvec[OMEGA_IDX] += integral of F_omega(T, phi) * shape
  //   elvec[N_IDX]     += integral of F_n(n, T, phi) * shape
  void AssembleElementVector(const mfem::Array<const mfem::FiniteElement *> & el,
                             mfem::ElementTransformation & Tr,
                             const mfem::Array<const mfem::Vector *> & elfun,
                             const mfem::Array<mfem::Vector *> & elvec) override;

  // Jacobian (state-derivative of the residual).  Non-zero blocks:
  //   (T,T)     dF_T/dT
  //   (omega,T) dF_omega/dT
  //   (n,T)     dF_n/dT
  //   (n,n)     dF_n/dn
  void AssembleElementGrad(const mfem::Array<const mfem::FiniteElement *> & el,
                           mfem::ElementTransformation & Tr,
                           const mfem::Array<const mfem::Vector *> & elfun,
                           const mfem::Array2D<mfem::DenseMatrix *> & elmats) override;

private:
  mfem::Coefficient * _phi;
  mfem::Coefficient * _Sn;
  mfem::real_t _Lambda;
  mfem::real_t _eps2;
  int _num_active;
};

/**
 * Residual of the implicit stage equation solved once per time step. Writing the stage state as
 * z = u_pred + gamma * k, the residual in the stage slope k is
 *
 *   R(k) = M k + K z + F(z),
 *
 * with M the mass matrix, K the combined advection and SUW stabilisation operator, and F the
 * reaction terms contributed by RogersRicciNLFIntegrator. M and K are held fixed over the stage:
 * K depends on the state only through phi, which is frozen at its predicted value.
 */
class RogersRicciStageOperator : public mfem::Operator
{
public:
  RogersRicciStageOperator(mfem::ParBlockNonlinearForm & F_form,
                           const mfem::Array<int> & block_true_offsets,
                           int num_active);

  void SetParameters(mfem::real_t gamma,
                     const mfem::BlockVector & u_pred,
                     mfem::HypreParMatrix & M,
                     mfem::HypreParMatrix & K);

  void Mult(const mfem::Vector & k_vec, mfem::Vector & R_vec) const override;
  mfem::Operator & GetGradient(const mfem::Vector & k_vec) const override;

private:
  mfem::ParBlockNonlinearForm & _F_form;
  const mfem::Array<int> & _block_true_offsets;
  const int _num_active;

  mfem::real_t _gamma{0.0};
  const mfem::BlockVector * _u_pred{nullptr};
  mfem::HypreParMatrix * _M{nullptr};
  mfem::HypreParMatrix * _K{nullptr};

  /// Stage state z = u_pred + gamma * k
  mutable mfem::BlockVector _z;
  mutable mfem::Vector _tmp_block;

  /// M + gamma * K
  std::unique_ptr<mfem::HypreParMatrix> _M_plus_gK;
  mutable std::unique_ptr<mfem::HypreParMatrix> _diag_T, _diag_n, _gFwT, _gFnT;
  mutable mfem::BlockOperator _Jac_block;
};

class RogersRicci2DProblemOperator : public Moose::MFEM::TimeDependentProblemOperator
{
public:
  RogersRicci2DProblemOperator(MFEMProblem & problem, const InputParameters & parameters);

  ~RogersRicci2DProblemOperator() = default;

  virtual void Init(mfem::BlockVector & X) override;
  virtual void Solve() override;
  virtual void
  ImplicitSolve(const mfem::real_t gamma, const mfem::Vector & u_pred, mfem::Vector & k) override;

private:
  void buildCoefficients();
  void buildMassMatrix();
  void buildKForm();
  void buildPhiSolver();
  void buildNonlinearForm();
  void buildNewtonLinearSolver();

  /// Solve -div(grad phi) = -omega
  void updatePhi();
  void reassembleK();

  const std::string _phi_var_name;
  const std::string _potential_coef_name;
  const std::string _source_coef_name;

  const mfem::real_t _eps2;
  const mfem::real_t _Binv;
  const mfem::real_t _Lambda;
  /// Number of variables evolved in time: 1 = T, 2 = T and omega, 3 = the full system.
  const int _num_active;

  /// Controls for the block GMRES solving each Newton step.
  const mfem::real_t _lin_rtol;
  const int _lin_max_its;
  const int _kdim;

  /// Element size
  mfem::real_t _h{0.0};

  mfem::ParGridFunction * _phi{nullptr};

  mfem::DenseMatrix _rotmat;
  std::unique_ptr<mfem::MatrixConstantCoefficient> _grad_rotate;
  std::unique_ptr<mfem::MatrixVectorProductCoefficient> _vd;
  std::unique_ptr<mfem::ScalarVectorProductCoefficient> _v_E;
  std::unique_ptr<mfem::InnerProductCoefficient> _v_E_sq;
  std::unique_ptr<mfem::TransformedCoefficient> _suw_scalar;
  std::unique_ptr<mfem::OuterProductCoefficient> _v_E_outer;
  std::unique_ptr<mfem::ScalarMatrixProductCoefficient> _SUW_matcoef;

  std::unique_ptr<mfem::ParBilinearForm> _M_form;
  std::unique_ptr<mfem::HypreParMatrix> _M_mat;
  std::unique_ptr<mfem::ParBilinearForm> _K_form;
  std::unique_ptr<mfem::HypreParMatrix> _K_mat;

  std::unique_ptr<mfem::ParBilinearForm> _a_phi;
  std::unique_ptr<mfem::ParLinearForm> _b_phi;
  std::unique_ptr<mfem::ProductCoefficient> _neg_omega_coef;
  mfem::Array<int> _ess_tdof_list_phi;
  mfem::OperatorHandle _A_phi;

  std::unique_ptr<mfem::ParBlockNonlinearForm> _F_form;
  std::unique_ptr<RogersRicciStageOperator> _stage_op;

  std::unique_ptr<mfem::Solver> _newton_lin_solver;
};

#endif