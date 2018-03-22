/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    testImuFactor.cpp
 * @brief   Unit test for ImuFactor
 * @author  Luca Carlone
 * @author  Frank Dellaert
 * @author  Richard Roberts
 * @author  Stephen Williams
 */

#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/ScenarioRunner.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/linear/Sampler.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>

#include <boost/bind.hpp>
#include <list>

#include <gtsam/navigation/ImuBias.h>
#include <gtsam/inference/Symbol.h>
#include "wave/gtsam/preintg_imu_factor.hpp"
#include "wave/wave_test.hpp"
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <wave/gtsam/pose_prior.hpp>
#include <wave/gtsam/twist_prior.hpp>
#include <wave/gtsam/bias_prior.hpp>
#include <wave/gtsam/pose_vel_bias.hpp>

namespace wave {

using namespace std;
using namespace gtsam;

// Convenience for named keys
using symbol_shorthand::X;
using symbol_shorthand::V;
using symbol_shorthand::B;

static const Vector3 kZero = Z_3x1;
typedef imuBias::ConstantBias Bias;
static const Bias kZeroBiasHat, kZeroBias;

static const Vector3 kZeroOmegaCoriolis(0, 0, 0);
static const Vector3 kNonZeroOmegaCoriolis(0, 0.1, 0.1);

static const double kGravity = 10;
static const Vector3 kGravityAlongNavZDown(0, 0, kGravity);

// Realistic MEMS white noise characteristics. Angular and velocity random walk
// expressed in degrees respectively m/s per sqrt(hr).
auto radians = [](double t) { return t * M_PI / 180; };
static const double kGyroSigma = radians(0.5) / 60;  // 0.5 degree ARW
static const double kAccelSigma = 0.1 / 60;          // 10 cm VRW
namespace testing {
// Create default parameters with Z-down and above noise parameters
static boost::shared_ptr<PreintegrationParams> Params() {
  auto p = PreintegrationParams::MakeSharedD(kGravity);
  p->gyroscopeCovariance = kGyroSigma * kGyroSigma * I_3x3;
  p->accelerometerCovariance = kAccelSigma * kAccelSigma * I_3x3;
  p->integrationCovariance = 0.0001 * I_3x3;
  return p;
}
}

namespace common {

/**
 * Linearize a nonlinear factor using numerical differentiation
 * The benefit of this method is that it does not need to know what types are
 * involved to evaluate the factor. If all the machinery of gtsam is working
 * correctly, we should get the correct numerical derivatives out the other side.
 * NOTE(frank): factors that have non vector-space measurements use between or LocalCoordinates
 * to evaluate the error, and their derivatives will only be correct for near-zero errors.
 * This is fixable but expensive, and does not matter in practice as most factors will sit near
 * zero errors anyway. However, it means that below will only be exact for the correct measurement.
 */
JacobianFactor linearizeNumerically(const NoiseModelFactor& factor,
    const Values& values, double delta = 1e-5) {

  // We will fill a vector of key/Jacobians pairs (a map would sort)
  std::vector<std::pair<Key, Matrix> > jacobians;

  // Get size
  const Eigen::VectorXd e = factor.whitenedError(values);
  const size_t rows = e.size();

  // Loop over all variables
  const double one_over_2delta = 1.0 / (2.0 * delta);
  VectorValues dX = values.zeroVectors();
  for(Key key: factor) {
    // Compute central differences using the values struct.
    const size_t cols = dX.dim(key);
    Matrix J = Matrix::Zero(rows, cols);
    for (size_t col = 0; col < cols; ++col) {
      Eigen::VectorXd dx = Eigen::VectorXd::Zero(cols);
      dx[col] = delta;
      dX[key] = dx;
      Values eval_values = values.retract(dX);
      const Eigen::VectorXd left = factor.whitenedError(eval_values);
      dx[col] = -delta;
      dX[key] = dx;
      eval_values = values.retract(dX);
      const Eigen::VectorXd right = factor.whitenedError(eval_values);
      J.col(col) = (left - right) * one_over_2delta;
    }
    jacobians.push_back(std::make_pair(key,J));
  }

  // Next step...return JacobianFactor
  return JacobianFactor(jacobians, -e);
}

namespace internal {
// CPPUnitLite-style test for linearization of a factor
bool testFactorJacobians(
    const NoiseModelFactor& factor, const gtsam::Values& values, double delta,
    double tolerance) {

  // Create expected value by numerical differentiation
  JacobianFactor expected = linearizeNumerically(factor, values, delta);

  // Create actual value by linearize
  boost::shared_ptr<JacobianFactor> actual = //
      boost::dynamic_pointer_cast<JacobianFactor>(factor.linearize(values));
  if (!actual) return false;

  // Check cast result and then equality
  bool equal = assert_equal(expected, *actual, tolerance);

  // if not equal, test individual jacobians:
  if (!equal) {
    for (size_t i = 0; i < actual->size(); i++) {
      bool i_good = assert_equal((Matrix) (expected.getA(expected.begin() + i)),
          (Matrix) (actual->getA(actual->begin() + i)), tolerance);
      if (!i_good) {
        std::cout << "Mismatch in Jacobian " << i+1 << " (base 1), as shown above" << std::endl;
      }
    }
  }

  return equal;
}
}

/// \brief Check the Jacobians produced by a factor against finite differences.
/// \param factor The factor to test.
/// \param values Values filled in for testing the Jacobians.
/// \param numerical_derivative_step The step to use when computing the numerical derivative Jacobians
/// \param tolerance The numerical tolerance to use when comparing Jacobians.
#define EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, numerical_derivative_step, tolerance) \
    { EXPECT_TRUE(common::internal::testFactorJacobians(factor, values, numerical_derivative_step, tolerance)); }

} // namespace gtsam

namespace {
// Auxiliary functions to test evaluate error in ImuFactor
/* ************************************************************************* */
Rot3 evaluateRotationError(const PreintgIMUFactor& factor, const Pose3& pose_i,
    const Vector3& vel_i, const Pose3& pose_j, const Vector3& vel_j,
    const Bias& bias) {


    wave::PoseVelBias state_i, state_j;
    state_i.pose = pose_i;
    state_i.vel.block(3, 0, 3, 1) = vel_i;
    state_j.pose = pose_j;
    state_j.vel.block(3, 0, 3, 1) = vel_j;
    
  return Rot3::Expmap(
      factor.evaluateError(state_i, state_j, bias).head(3));

}

} // namespace

namespace common {
static const Pose3 x1(Rot3::RzRyRx(M_PI / 12.0, M_PI / 6.0, M_PI / 4.0),
    Point3(5.0, 1.0, 0));
static const Vector3 v1(Vector3(0.5, 0.0, 0.0));
static const NavState state1(x1, v1);


// Measurements
static const double w = M_PI / 100;
static const Vector3 measuredOmega(w, 0, 0);
static const Vector3 measuredAcc = x1.rotation().unrotate(
    -kGravityAlongNavZDown);
static const double deltaT = 1.0;

static const Pose3 x2(Rot3::RzRyRx(M_PI / 12.0 + w, M_PI / 6.0, M_PI / 4.0),
    Point3(5.5, 1.0, 0));
static const Vector3 v2(Vector3(0.5, 0.0, 0.0));
static const NavState state2(x2, v2);
} // namespace common

TEST(preintg_imu, PreintegrationBaseMethods) {
  using namespace common;
  auto p = testing::Params();
  p->omegaCoriolis = Vector3(0.02, 0.03, 0.04);
  p->use2ndOrderCoriolis = true;

  PreintegratedImuMeasurements pim(p, kZeroBiasHat);
  pim.integrateMeasurement(measuredAcc, measuredOmega, deltaT);
  pim.integrateMeasurement(measuredAcc, measuredOmega, deltaT);

  // biasCorrectedDelta
  Matrix96 actualH;
  pim.biasCorrectedDelta(kZeroBias, actualH);
  Matrix expectedH = numericalDerivative11<Vector9, Bias>(
      boost::bind(&PreintegrationBase::biasCorrectedDelta, pim, _1,
          boost::none), kZeroBias);
  EXPECT_TRUE(assert_equal(expectedH, actualH));

  Matrix9 aH1;
  Matrix96 aH2;
  NavState predictedState = pim.predict(state1, kZeroBias, aH1, aH2);
  Matrix eH1 = numericalDerivative11<NavState, NavState>(
      boost::bind(&PreintegrationBase::predict, pim, _1, kZeroBias, boost::none,
          boost::none), state1);
  EXPECT_TRUE(assert_equal(eH1, aH1));
  Matrix eH2 = numericalDerivative11<NavState, Bias>(
      boost::bind(&PreintegrationBase::predict, pim, state1, _1, boost::none,
          boost::none), kZeroBias);
  EXPECT_TRUE(assert_equal(eH2, aH2));
}


/* ************************************************************************* */
TEST(preintg_imu, MultipleMeasurements) {
  using namespace common;

  PreintegratedImuMeasurements expected(testing::Params(), kZeroBiasHat);
  expected.integrateMeasurement(measuredAcc, measuredOmega, deltaT);
  expected.integrateMeasurement(measuredAcc, measuredOmega, deltaT);

  Matrix32 acc,gyro;
  Matrix12 dts;
  acc << measuredAcc, measuredAcc;
  gyro << measuredOmega, measuredOmega;
  dts << deltaT, deltaT;
  PreintegratedImuMeasurements actual(testing::Params(), kZeroBiasHat);
  actual.integrateMeasurements(acc,gyro,dts);

  EXPECT_TRUE(assert_equal(expected,actual));
}

/* ************************************************************************* */
TEST(preintg_imu, ErrorAndJacobians) {
  using namespace common;
  PreintegratedImuMeasurements pim(testing::Params());

  pim.integrateMeasurement(measuredAcc, measuredOmega, deltaT);
  EXPECT_TRUE(assert_equal(state2, pim.predict(state1, kZeroBias)));

  wave::PoseVelBias state_i, state_j;
  state_i.pose = x1;
  state_i.vel.block(3, 0, 3, 1) = v1;
  state_j.pose = x2;
  state_j.vel.block(3, 0, 3, 1) = v2;

  // Create factor
  PreintgIMUFactor factor(X(1), X(2), B(1), pim);

  // Expected error
  Vector expectedError(9);
  expectedError << 0, 0, 0, 0, 0, 0, 0, 0, 0;
  EXPECT_TRUE(
      assert_equal(expectedError,
          factor.evaluateError(state_i,state_j, kZeroBias)));



  Values values;
  values.insert(X(1), state_i);
  values.insert(X(2), state_j);
  values.insert(B(1), kZeroBias);
  EXPECT_TRUE(assert_equal(expectedError, factor.unwhitenedError(values)));

  // Make sure linearization is correct
  double diffDelta = 1e-7;
  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, diffDelta, 1e-3);

  // Actual Jacobians
  Matrix H1a, H2a, H3a, H4a, H5a;
  (void) factor.evaluateError(state_i, state_j, kZeroBias, H1a, H2a, H3a);


  Matrix6 H1a_Pose,H2a_Pose;
  H1a_Pose.block<6,6>(0,0).noalias() = H1a.block<6,6>(0,0);
  H2a_Pose.block<6,6>(0,0).noalias() = H2a.block<6,6>(0,0);
  // Make sure rotation part is correct when error is interpreted as axis-angle
  // Jacobians are around zero, so the rotation part is the same as:
  Matrix H1Rot3 = numericalDerivative11<Rot3, Pose3>(
      boost::bind(&evaluateRotationError, factor, _1, v1, x2, v2, kZeroBias),
      x1);

  EXPECT_TRUE(assert_equal(H1Rot3, H1a_Pose.topRows(3)));

  Matrix H3Rot3 = numericalDerivative11<Rot3, Pose3>(
      boost::bind(&evaluateRotationError, factor, x1, v1, _1, v2, kZeroBias),
      x2);
  EXPECT_TRUE(assert_equal(H3Rot3, H2a_Pose.topRows(3)));

  // Evaluate error with wrong values
  Vector3 v2_wrong = v2 + Vector3(0.1, 0.1, 0.1);

  state_j.vel.block(3, 0, 3, 1) = v2_wrong;

  values.update(X(2), state_j);

  expectedError << 0, 0, 0, 0, 0, 0, -0.0724744871, -0.040715657, -0.151952901;
  EXPECT_TRUE(
      assert_equal(expectedError,
          factor.evaluateError(state_i,state_j, kZeroBias), 1e-2));
  EXPECT_TRUE(assert_equal(expectedError, factor.unwhitenedError(values), 1e-2));

  // Make sure the whitening is done correctly
  Matrix cov = pim.preintMeasCov();
  Matrix R = RtR(cov.inverse());
  Vector whitened = R * expectedError;
  EXPECT_TRUE(assert_equal(0.5 * whitened.squaredNorm(), factor.error(values), 1e-4));

  // Make sure linearization is correct
  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, diffDelta, 1e-3);
}

TEST(preintg_imu, ErrorAndJacobianWithBiases) {
  using common::x1;
  using common::v1;
  using common::v2;
  Bias bias(Vector3(0.2, 0, 0), Vector3(0.1, 0, 0.3)); // Biases (acc, rot)
  Pose3 x2(Rot3::Expmap(Vector3(0, 0, M_PI / 10.0 + M_PI / 10.0)),
      Point3(5.5, 1.0, -50.0));

  // Measurements
  Vector3 measuredOmega;
  measuredOmega << 0, 0, M_PI / 10.0 + 0.3;
  Vector3 measuredAcc = x1.rotation().unrotate(-kGravityAlongNavZDown)
      + Vector3(0.2, 0.0, 0.0);
  double deltaT = 1.0;

  auto p = testing::Params();
  p->omegaCoriolis = kNonZeroOmegaCoriolis;

  Bias biasHat(Vector3(0.2, 0.0, 0.0), Vector3(0.0, 0.0, 0.1));
  PreintegratedImuMeasurements pim(p, biasHat);
  pim.integrateMeasurement(measuredAcc, measuredOmega, deltaT);

  // Make sure of biasCorrectedDelta
  Matrix96 actualH;
  pim.biasCorrectedDelta(bias, actualH);
  Matrix expectedH = numericalDerivative11<Vector9, Bias>(
      boost::bind(&PreintegrationBase::biasCorrectedDelta, pim, _1,
          boost::none), bias);
  EXPECT_TRUE(assert_equal(expectedH, actualH));


  wave::PoseVelBias state_i, state_j;
  state_i.pose = x1;
  state_i.vel.block(3, 0, 3, 1) = v1;
  state_j.pose = x2;
  state_j.vel.block(3, 0, 3, 1) = v2;

  // Create factor
  PreintgIMUFactor factor(X(1), X(2), B(1), pim);

  Values values;
  values.insert(X(1), state_i);
  values.insert(X(2), state_j);
  values.insert(B(1), bias);

  // Make sure linearization is correct
  double diffDelta = 1e-7;
  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, diffDelta, 1e-3);
}

TEST(preintg_imu, ErrorAndJacobianWith2ndOrderCoriolis) {
  using common::x1;
  using common::v1;
  using common::v2;
  Bias bias(Vector3(0.2, 0, 0), Vector3(0.1, 0, 0.3)); // Biases (acc, rot)
  Pose3 x2(Rot3::Expmap(Vector3(0, 0, M_PI / 10.0 + M_PI / 10.0)),
      Point3(5.5, 1.0, -50.0));

  // Measurements
  Vector3 measuredOmega;
  measuredOmega << 0, 0, M_PI / 10.0 + 0.3;
  Vector3 measuredAcc = x1.rotation().unrotate(-kGravityAlongNavZDown)
      + Vector3(0.2, 0.0, 0.0);
  double deltaT = 1.0;

  auto p = testing::Params();
  p->omegaCoriolis = kNonZeroOmegaCoriolis;
  p->use2ndOrderCoriolis = true;

  PreintegratedImuMeasurements pim(p,
      Bias(Vector3(0.2, 0.0, 0.0), Vector3(0.0, 0.0, 0.1)));
  pim.integrateMeasurement(measuredAcc, measuredOmega, deltaT);


  wave::PoseVelBias state_i, state_j;
  state_i.pose = x1;
  state_i.vel.block(3, 0, 3, 1) = v1;
  state_j.pose = x2;
  state_j.vel.block(3, 0, 3, 1) = v2;

  // Create factor
  PreintgIMUFactor factor(X(1), X(2), B(1), pim);

  Values values;
  values.insert(X(1), state_i);
  values.insert(X(2), state_j);
  values.insert(B(1), bias);

  // Make sure linearization is correct
  double diffDelta = 1e-7;
  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, diffDelta, 1e-3);
}

TEST(preintg_imu, PartialDerivative_wrt_Bias) {
  // Linearization point
  Vector3 biasOmega(0, 0, 0); // Current estimate of rotation rate bias

  // Measurements
  Vector3 measuredOmega(0.1, 0, 0);
  double deltaT = 0.5;

  auto evaluateRotation = [=](const Vector3 biasOmega) {
    return Rot3::Expmap((measuredOmega - biasOmega) * deltaT);
  };

  // Compute numerical derivatives
  Matrix expectedDelRdelBiasOmega =
      numericalDerivative11<Rot3, Vector3>(evaluateRotation, biasOmega);

  const Matrix3 Jr =
      Rot3::ExpmapDerivative((measuredOmega - biasOmega) * deltaT);

  Matrix3 actualdelRdelBiasOmega = -Jr * deltaT; // the delta bias appears with the minus sign

  // Compare Jacobians
  EXPECT_TRUE(assert_equal(expectedDelRdelBiasOmega, actualdelRdelBiasOmega, 1e-9));
}


Vector3 correctedAcc(const PreintegratedImuMeasurements& pim,
    const Vector3& measuredAcc, const Vector3& measuredOmega) {
  Vector3 correctedAcc = pim.biasHat().correctAccelerometer(measuredAcc);
  Vector3 correctedOmega = pim.biasHat().correctGyroscope(measuredOmega);
  return pim.correctMeasurementsBySensorPose(correctedAcc, correctedOmega).first;
}

TEST(preintg_imu, ErrorWithBiasesAndSensorBodyDisplacement) {
  const Rot3 nRb = Rot3::Expmap(Vector3(0, 0, M_PI / 4.0));
  const Point3 p1(5.0, 1.0, -50.0);
  const Vector3 v1(0.5, 0.0, 0.0);

  const Vector3 a = nRb * Vector3(0.2, 0.0, 0.0);
  const AcceleratingScenario scenario(nRb, p1, v1, a,
      Vector3(0, 0, M_PI / 10.0 + 0.3));

  auto p = testing::Params();
  p->body_P_sensor = Pose3(Rot3::Expmap(Vector3(0, M_PI / 2, 0)),
      Point3(0.1, 0.05, 0.01));
  p->omegaCoriolis = kNonZeroOmegaCoriolis;

  Bias biasHat(Vector3(0.2, 0.0, 0.0), Vector3(0.0, 0.0, 0.0));

  const double T = 3.0; // seconds
  ScenarioRunner runner(&scenario, p, T / 10);

  Pose3 x1(nRb, p1);

  // Measurements
  Vector3 measuredOmega = runner.actualAngularVelocity(0);
  Vector3 measuredAcc = runner.actualSpecificForce(0);

  // Get mean prediction from "ground truth" measurements
  const Vector3 accNoiseVar2(0.01, 0.02, 0.03);
  const Vector3 omegaNoiseVar2(0.03, 0.01, 0.02);
  PreintegratedImuMeasurements pim(p, biasHat);

  // Check updatedDeltaXij derivatives
  Matrix3 D_correctedAcc_measuredOmega = Z_3x3;
  pim.correctMeasurementsBySensorPose(measuredAcc, measuredOmega,
      boost::none, D_correctedAcc_measuredOmega, boost::none);
  Matrix3 expectedD = numericalDerivative11<Vector3, Vector3>(
      boost::bind(correctedAcc, pim, measuredAcc, _1), measuredOmega, 1e-6);
  EXPECT_TRUE(assert_equal(expectedD, D_correctedAcc_measuredOmega, 1e-5));

  double dt = 0.1;


  Bias bias(Vector3(0.2, 0, 0), Vector3(0, 0, 0.3)); // Biases (acc, rot)

  // integrate at least twice to get position information
  // otherwise factor cov noise from preint_cov is not positive definite
  pim.integrateMeasurement(measuredAcc, measuredOmega, dt);
  pim.integrateMeasurement(measuredAcc, measuredOmega, dt);

  // Create factor
  PreintgIMUFactor factor(X(1), X(2), B(1), pim);

  Pose3 x2(Rot3::Expmap(Vector3(0, 0, M_PI / 4.0 + M_PI / 10.0)),
      Point3(5.5, 1.0, -50.0));
  Vector3 v2(Vector3(0.5, 0.0, 0.0));

  wave::PoseVelBias state_i, state_j;
  state_i.pose = x1;
  state_i.vel.block(3, 0, 3, 1) = v1;
  state_j.pose = x2;
  state_j.vel.block(3, 0, 3, 1) = v2;


  Values values;
  values.insert(X(1), state_i);
  values.insert(X(2), state_j);
  values.insert(B(1), bias);

  // Make sure linearization is correct
  double diffDelta = 1e-8;
  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, diffDelta, 1e-3);
}

/* ************************************************************************* */
TEST(preintg_imu, bodyPSensorWithBias) {
  using noiseModel::Diagonal;
  typedef Bias Bias;

  int numFactors = 80;
  Vector6 noiseBetweenBiasSigma;

  noiseBetweenBiasSigma << Vector3(2.0e-5, 2.0e-5, 2.0e-5), Vector3(3.0e-6,
      3.0e-6, 3.0e-6);
  SharedDiagonal biasNoiseModel = Diagonal::Sigmas(noiseBetweenBiasSigma);

  // Measurements
  // Sensor frame is z-down
  // Gyroscope measurement is the angular velocity of sensor w.r.t nav frame in sensor frame
  Vector3 measuredOmega(0, 0.01, 0);
  // Acc measurement is acceleration of sensor in the sensor frame, when stationary,
  // table exerts an equal and opposite force w.r.t gravity
  Vector3 measuredAcc(0, 0, -kGravity);

  auto p = testing::Params();
  p->n_gravity = Vector3(0, 0, -kGravity);
  p->body_P_sensor = Pose3(Rot3::Ypr(0, 0, M_PI), Point3(0,0,0));
  p->accelerometerCovariance = 1e-7 * I_3x3;
  p->gyroscopeCovariance = 1e-8 * I_3x3;
  p->integrationCovariance = 1e-9 * I_3x3;
  double deltaT = 0.005;

  //   Specify noise values on priors
  Vector6 priorNoisePoseSigmas(
      (Vector(6) << 0.001, 0.001, 0.001, 0.01, 0.01, 0.01).finished());
  Vector6 priorNoiseVelSigmas((Vector(6) << 0.0, 0.0, 0.0 ,0.1, 0.1, 0.1).finished());
  Vector6 priorNoiseBiasSigmas(
      (Vector(6) << 0.1, 0.1, 0.1, 0.5e-1, 0.5e-1, 0.5e-1).finished());
  SharedDiagonal priorNoisePose = Diagonal::Sigmas(priorNoisePoseSigmas);
  SharedDiagonal priorNoiseVel = Diagonal::Sigmas(priorNoiseVelSigmas);
  SharedDiagonal priorNoiseBias = Diagonal::Sigmas(priorNoiseBiasSigmas);
  Vector6 zeroVel((Vector(6)<<0.0 ,0.0, 0.0, 0.0,0.0, 0.0).finished());

  // Create a factor graph with priors on initial pose, vlocity and bias
  NonlinearFactorGraph graph;
  Values values;

  wave::PoseVelBias state_0;

  state_0.pose = Pose3();
  state_0.vel = zeroVel;


  wave::PosePrior<wave::PoseVelBias> priorPose(X(0), Pose3(), priorNoisePose);
  graph.add(priorPose);

  wave::TwistPrior<wave::PoseVelBias> priorVel(X(0), zeroVel, priorNoiseVel);
  graph.add(priorVel);
  values.insert(X(0), state_0);

  // The key to this test is that we specify the bias, in the sensor frame, as known a priori
  // We also create factors below that encode our assumption that this bias is constant over time
  // In theory, after optimization, we should recover that same bias estimate
  Bias priorBias(Vector3(0, 0, 0), Vector3(0, 0.01, 0)); // Biases (acc, rot)
  PriorFactor<Bias> priorBiasFactor(B(0), priorBias, priorNoiseBias);
  graph.add(priorBiasFactor);
  values.insert(B(0), priorBias);

  wave::PoseVelBias state_i;

  state_i.pose = Pose3();
  state_i.vel = zeroVel;
  // Now add IMU factors and bias noise models
  Bias zeroBias(Vector3(0, 0, 0), Vector3(0, 0, 0));
  for (int i = 1; i < numFactors; i++) {
    PreintegratedImuMeasurements pim = PreintegratedImuMeasurements(p,
        priorBias);
    for (int j = 0; j < 200; ++j)
      pim.integrateMeasurement(measuredAcc, measuredOmega, deltaT);

    // Create factors
    graph.add(PreintgIMUFactor(X(i - 1), X(i), B(i - 1), pim));
    graph.add(BetweenFactor<Bias>(B(i - 1), B(i), zeroBias, biasNoiseModel));

    values.insert(X(i), state_i);
    values.insert(B(i), priorBias);
  }

  // Finally, optimize, and get bias at last time step
  Values results = LevenbergMarquardtOptimizer(graph, values).optimize();
  Bias biasActual = results.at<Bias>(B(numFactors - 1));

  // And compare it with expected value (our prior)
  Bias biasExpected(Vector3(0, 0, 0), Vector3(0, 0.01, 0));
  EXPECT_TRUE(assert_equal(biasExpected, biasActual, 1e-3));

}

TEST(preintg_imu, PredictPositionAndVelocity) {
  Bias bias(Vector3(0, 0, 0), Vector3(0, 0, 0)); // Biases (acc, rot)

  // Measurements
  Vector3 measuredOmega;
  measuredOmega << 0, 0, 0; // M_PI/10.0+0.3;
  Vector3 measuredAcc;
  measuredAcc << 0, 1, -kGravity;
  double deltaT = 0.001;

  PreintegratedImuMeasurements pim(testing::Params(),
      Bias(Vector3(0.2, 0.0, 0.0), Vector3(0.0, 0.0, 0.0)));

  for (int i = 0; i < 1000; ++i)
    pim.integrateMeasurement(measuredAcc, measuredOmega, deltaT);

  // Create factor
  PreintgIMUFactor factor(X(1), X(2),B(1), pim);

  // Predict
  Pose3 x1;
  Vector3 v1(0, 0.0, 0.0);
  wave::PoseVelBias state_i;

  state_i.pose = x1;
  state_i.vel.block<3,1>(3,0) = v1;

  NavState actual = pim.predict(NavState(x1, v1), bias);
  NavState expected(Rot3(), Point3(0, 0.5, 0), Vector3(0, 1, 0));
  EXPECT_TRUE(assert_equal(expected, actual));
}



}
