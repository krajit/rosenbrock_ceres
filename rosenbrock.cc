// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/ceres.h"
#include "glog/logging.h"
#include "ceres/autodiff_first_order_function.h"


class RosenbrockFunctor
{
public:
  explicit RosenbrockFunctor(){}
  template <typename T>
  bool operator()(const T *const x, T *cost) const
  {
    *cost = (1.0 - x[0]) * (1.0 - x[0]) + 100.0 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]);;
    return true;
  }
};


int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);

  double parameters[2] = {10, 10};

  ceres::GradientProblemSolver::Options options;
  options.minimizer_progress_to_stdout = true;

  ceres::GradientProblemSolver::Summary summary;
  
   ceres::FirstOrderFunction* function =
     new ceres::AutoDiffFirstOrderFunction<RosenbrockFunctor, 2>(
         new RosenbrockFunctor());

  ceres::GradientProblem problem(function);
  ceres::Solve(options, problem, parameters, &summary);

  std::cout << summary.FullReport() << "\n";
  std::cout << "Initial x: " << -1.2 << " y: " << 1.0 << "\n";
  std::cout << "Final   x: " << parameters[0]
            << " y: " << parameters[1] << "\n";

   return 0;
}
