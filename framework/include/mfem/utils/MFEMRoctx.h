//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#ifdef MOOSE_MFEM_ENABLED

#ifdef MOOSE_HAVE_ROCTX

#include <roctx.h>
#include <cstdio>

namespace Moose::MFEM
{
class RoctxRange
{
public:
  explicit RoctxRange(const char * name) { roctxRangePushA(name); }
  ~RoctxRange() { roctxRangePop(); }
  RoctxRange(const RoctxRange &) = delete;
  RoctxRange & operator=(const RoctxRange &) = delete;
};
}

#define MOOSE_MFEM_ROCTX_CONCAT_INNER(a, b) a##b
#define MOOSE_MFEM_ROCTX_CONCAT(a, b) MOOSE_MFEM_ROCTX_CONCAT_INNER(a, b)
#define MOOSE_MFEM_ROCTX_RANGE(name)                                                               \
  ::Moose::MFEM::RoctxRange MOOSE_MFEM_ROCTX_CONCAT(_moose_mfem_roctx_, __LINE__)(name)
#define MOOSE_MFEM_ROCTX_MARK(name) ::roctxMarkA(name)
#define MOOSE_MFEM_ROCTX_MARKF(fmt, ...)                                                           \
  do                                                                                               \
  {                                                                                                \
    char _moose_mfem_roctx_buf[128];                                                               \
    std::snprintf(_moose_mfem_roctx_buf, sizeof(_moose_mfem_roctx_buf), fmt, __VA_ARGS__);         \
    ::roctxMarkA(_moose_mfem_roctx_buf);                                                           \
  } while (0)

#else // MOOSE_HAVE_ROCTX undefined: compile to no-ops

#define MOOSE_MFEM_ROCTX_RANGE(name) ((void)0)
#define MOOSE_MFEM_ROCTX_MARK(name) ((void)0)
#define MOOSE_MFEM_ROCTX_MARKF(fmt, ...) ((void)0)

#endif // MOOSE_HAVE_ROCTX

#endif // MOOSE_MFEM_ENABLED
