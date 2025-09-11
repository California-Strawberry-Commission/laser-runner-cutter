#include "runner_cutter_control/calibration/point_correspondences.hpp"

#include <iostream>
#include <unsupported/Eigen/NonLinearOptimization>

PointCorrespondences::PointCorrespondences()
    : cameraToLaserTransform_{Eigen::MatrixXd::Zero(4, 3)} {}

std::size_t PointCorrespondences::size() const { return laserCoords_.size(); }

void PointCorrespondences::add(const LaserCoord& laserCoord,
                               const PixelCoord& cameraPixelCoord,
                               const Position& cameraPosition) {
  laserCoords_.push_back(laserCoord);
  cameraPixelCoords_.push_back(cameraPixelCoord);
  cameraPositions_.push_back(cameraPosition);
  updateLaserBounds();
}

void PointCorrespondences::clear() {
  laserCoords_.clear();
  cameraPixelCoords_.clear();
  cameraPositions_.clear();
  cameraToLaserTransform_ = Eigen::MatrixXd::Zero(4, 3);
  updateLaserBounds();
}

void PointCorrespondences::updateTransformLinearLeastSquares() {
  if (cameraPositions_.empty() || laserCoords_.empty() ||
      cameraPositions_.size() != laserCoords_.size()) {
    return;
  }

  // Homogeneous camera positions
  Eigen::MatrixXd cameraMat{cameraPositions_.size(), 4};
  for (size_t i = 0; i < cameraPositions_.size(); ++i) {
    auto [x, y, z]{cameraPositions_[i]};
    cameraMat(i, 0) = x;
    cameraMat(i, 1) = y;
    cameraMat(i, 2) = z;
    cameraMat(i, 3) = 1.0;
  }

  // Homogeneous laser coords
  Eigen::MatrixXd laserMat{laserCoords_.size(), 3};
  for (size_t i = 0; i < laserCoords_.size(); ++i) {
    laserMat(i, 0) = laserCoords_[i].x;
    laserMat(i, 1) = laserCoords_[i].y;
    laserMat(i, 2) = 1.0;
  }

  // ColPivHouseholderQR provided the best accuracy for speed
  cameraToLaserTransform_ = cameraMat.colPivHouseholderQr().solve(laserMat);
}

struct TransformResidual {
  enum {
    InputsAtCompileTime = Eigen::Dynamic,
    ValuesAtCompileTime = Eigen::Dynamic
  };

  typedef double Scalar;
  typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
  typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
  typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime>
      JacobianType;

  TransformResidual(const Eigen::MatrixXd& cameraPositions,
                    const Eigen::MatrixXd& laserCoords)
      : cameraPositions{cameraPositions}, laserCoords{laserCoords} {}

  // Calculate residuals
  int operator()(const Eigen::VectorXd& params,
                 Eigen::VectorXd& residuals) const {
    // params is the flattened transform matrix, so reshape back to 4x3
    Eigen::Map<const Eigen::MatrixXd> transform{params.data(), 4, 3};

    Eigen::MatrixXd transformed{cameraPositions * transform};
    // Normalize by the third (homogeneous) coordinate to get (x, y) coordinates
    Eigen::MatrixXd homogeneousTransformed{transformed.array().colwise() /
                                           transformed.col(2).array()};

    // Compute the residuals as the difference between transformed coords and
    // laser coords, flattened to a vector
    for (int i = 0; i < homogeneousTransformed.rows(); ++i) {
      residuals[i * 2] =
          homogeneousTransformed(i, 0) - laserCoords(i, 0);  // x residual
      residuals[i * 2 + 1] =
          homogeneousTransformed(i, 1) - laserCoords(i, 1);  // y residual
    }

    return 0;
  }

  // Compute the Jacobian of the errors
  int df(const Eigen::VectorXd& params, Eigen::MatrixXd& jacobian) const {
    // The Jacobian is computed by differentiating the residual function.
    // Each row in the Jacobian corresponds to the partial derivative of one
    // residual with respect to each parameter. In this case, the rows are
    // interleaved x-residuals and y-residuals for each point correspondence.

    double epsilon{1e-5};
    for (int i = 0; i < params.size(); i++) {
      Eigen::VectorXd xPlus{params};
      xPlus(i) += epsilon;
      Eigen::VectorXd xMinus{params};
      xMinus(i) -= epsilon;

      Eigen::VectorXd fvecPlus{values()};
      operator()(xPlus, fvecPlus);

      Eigen::VectorXd fvecMinus{values()};
      operator()(xMinus, fvecMinus);

      Eigen::VectorXd fvecDiff{values()};
      fvecDiff = (fvecPlus - fvecMinus) / (2.0 * epsilon);

      jacobian.block(0, i, values(), 1) = fvecDiff;
    }

    return 0;
  }

  int values() const {
    // For each coord, there are 2 residuals (x and y)
    return laserCoords.rows() * 2;
  }

  int inputs() const {
    // 4x3 transform matrix
    return 12;
  }

  const Eigen::MatrixXd& cameraPositions;
  const Eigen::MatrixXd& laserCoords;
};

void PointCorrespondences::updateTransformNonlinearLeastSquares() {
  if (cameraPositions_.empty() || laserCoords_.empty() ||
      cameraPositions_.size() != laserCoords_.size()) {
    return;
  }

  // Homogeneous camera positions
  Eigen::MatrixXd cameraMat{cameraPositions_.size(), 4};
  for (size_t i = 0; i < cameraPositions_.size(); ++i) {
    auto [x, y, z]{cameraPositions_[i]};
    cameraMat(i, 0) = x;
    cameraMat(i, 1) = y;
    cameraMat(i, 2) = z;
    cameraMat(i, 3) = 1.0;
  }

  // Homogeneous laser coords
  Eigen::MatrixXd laserMat{laserCoords_.size(), 3};
  for (size_t i = 0; i < laserCoords_.size(); ++i) {
    laserMat(i, 0) = laserCoords_[i].x;
    laserMat(i, 1) = laserCoords_[i].y;
    laserMat(i, 2) = 1.0;
  }

  // We are attempting to estimate 12 parameters (the 4x3
  // cameraToLaserTransform_ matrix). Use cameraToLaserTransform_ as the initial
  // guess
  Eigen::VectorXd params{Eigen::Map<Eigen::VectorXd>{
      cameraToLaserTransform_.data(), cameraToLaserTransform_.size()}};

  // Define the Levenberg-Marquardt solver
  TransformResidual functor{cameraMat, laserMat};
  Eigen::LevenbergMarquardt<TransformResidual, double> levenbergMarquardt{
      functor};

  // Solve the problem
  levenbergMarquardt.minimize(params);

  // Update the cameraToLaserTransform matrix
  cameraToLaserTransform_ =
      Eigen::Map<const Eigen::MatrixXd>{params.data(), 4, 3};
}

void PointCorrespondences::updateCameraPixelToLaserCoordJacobian() {
  auto numSamples{size()};
  if (numSamples < 2 || cameraPixelCoords_.size() != laserCoords_.size()) {
    // Mismatched or insufficient data for Jacobian estimation
    return;
  }

  // Since we typically only need small-step deltas, use an affine model -
  // for a camera pixel coordinate (u, v) and laser coordinate (x, y):
  // x = a0*u + a1*v + a2
  // y = b0*u + b1*v + b2
  // Vectorizing the above,
  // [u v 1] * A = [x y], where A = [[a0 b0], [a1 b1], [a2 b2]]
  Eigen::MatrixXd P_camera{numSamples, 3};
  Eigen::MatrixXd P_laser{numSamples, 2};
  for (std::size_t i = 0; i < numSamples; ++i) {
    P_camera(i, 0) = static_cast<double>(cameraPixelCoords_[i].u);
    P_camera(i, 1) = static_cast<double>(cameraPixelCoords_[i].v);
    P_camera(i, 2) = 1.0;
    P_laser(i, 0) = static_cast<double>(laserCoords_[i].x);
    P_laser(i, 1) = static_cast<double>(laserCoords_[i].y);
  }

  // Solve for A using least squares
  Eigen::Matrix<double, 3, 2> A{P_camera.colPivHouseholderQr().solve(P_laser)};

  // For deltas (and thus the Jacobian), we only care about a0, a1, b0, and b1
  // J = [[a0 a1], [b0 b1]]
  cameraToLaserJacobian_ = A.topRows<2>().transpose();
}

float PointCorrespondences::getReprojectionError() const {
  // Homogeneous camera positions
  Eigen::MatrixXd cameraMat{cameraPositions_.size(), 4};
  for (size_t i = 0; i < cameraPositions_.size(); ++i) {
    auto [x, y, z]{cameraPositions_[i]};
    cameraMat(i, 0) = x;
    cameraMat(i, 1) = y;
    cameraMat(i, 2) = z;
    cameraMat(i, 3) = 1.0;
  }

  Eigen::MatrixXd transformed{cameraMat * cameraToLaserTransform_};
  // Normalize by the third (homogeneous) coordinate to get (x, y) coordinates
  Eigen::MatrixXd homogeneousTransformed{transformed.array().colwise() /
                                         transformed.col(2).array()};

  float error{0.0f};
  for (int i = 0; i < homogeneousTransformed.rows(); ++i) {
    float dx{static_cast<float>(homogeneousTransformed(i, 0)) -
             laserCoords_[i].x};
    float dy{static_cast<float>(homogeneousTransformed(i, 1)) -
             laserCoords_[i].y};
    error += std::sqrt(dx * dx + dy * dy);
  }
  return error / laserCoords_.size();
}

void PointCorrespondences::serialize(std::ostream& os) const {
  size_t laserCoordsSize{laserCoords_.size()};
  os.write(reinterpret_cast<const char*>(&laserCoordsSize),
           sizeof(laserCoordsSize));
  os.write(reinterpret_cast<const char*>(laserCoords_.data()),
           laserCoordsSize * sizeof(laserCoords_[0]));

  size_t cameraPixelCoordsSize{cameraPixelCoords_.size()};
  os.write(reinterpret_cast<const char*>(&cameraPixelCoordsSize),
           sizeof(cameraPixelCoordsSize));
  os.write(reinterpret_cast<const char*>(cameraPixelCoords_.data()),
           cameraPixelCoordsSize * sizeof(cameraPixelCoords_[0]));

  size_t cameraPositionsSize{cameraPositions_.size()};
  os.write(reinterpret_cast<const char*>(&cameraPositionsSize),
           sizeof(cameraPositionsSize));
  os.write(reinterpret_cast<const char*>(cameraPositions_.data()),
           cameraPositionsSize * sizeof(cameraPositions_[0]));
}

void PointCorrespondences::deserialize(std::istream& is) {
  size_t laserCoordsSize;
  is.read(reinterpret_cast<char*>(&laserCoordsSize), sizeof(laserCoordsSize));
  laserCoords_.clear();
  laserCoords_.resize(laserCoordsSize);
  is.read(reinterpret_cast<char*>(laserCoords_.data()),
          laserCoordsSize * sizeof(laserCoords_[0]));

  size_t cameraPixelCoordsSize;
  is.read(reinterpret_cast<char*>(&cameraPixelCoordsSize),
          sizeof(cameraPixelCoordsSize));
  cameraPixelCoords_.clear();
  cameraPixelCoords_.resize(cameraPixelCoordsSize);
  is.read(reinterpret_cast<char*>(cameraPixelCoords_.data()),
          cameraPixelCoordsSize * sizeof(cameraPixelCoords_[0]));

  size_t cameraPositionsSize;
  is.read(reinterpret_cast<char*>(&cameraPositionsSize),
          sizeof(cameraPositionsSize));
  cameraPositions_.clear();
  cameraPositions_.resize(cameraPositionsSize);
  is.read(reinterpret_cast<char*>(cameraPositions_.data()),
          cameraPositionsSize * sizeof(cameraPositions_[0]));

  updateLaserBounds();
  // Use linear least squares for an initial estimate, then refine using
  // nonlinear least squares
  updateTransformLinearLeastSquares();
  updateTransformNonlinearLeastSquares();
  updateCameraPixelToLaserCoordJacobian();
}

void PointCorrespondences::updateLaserBounds() {
  if (cameraPixelCoords_.empty()) {
    laserBounds_ = {0, 0, 0, 0};
    return;
  };

  auto [minX, maxX]{std::minmax_element(
      cameraPixelCoords_.begin(), cameraPixelCoords_.end(),
      [](const auto& a, const auto& b) { return a.u < b.u; })};
  auto [minY, maxY]{std::minmax_element(
      cameraPixelCoords_.begin(), cameraPixelCoords_.end(),
      [](const auto& a, const auto& b) { return a.v < b.v; })};

  laserBounds_ = {minX->u, minY->v, maxX->u - minX->u, maxY->v - minY->v};
}
