#include "runner_cutter_control/calibration/point_correspondences.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <unsupported/Eigen/NonLinearOptimization>

namespace {

constexpr double PLANARITY_RATIO_THRESHOLD{0.05};
// A correspondence is an outlier if its laser coord error exceeds this multiple
// of the median error
constexpr double OUTLIER_MEDIAN_MULTIPLIER{4.0};
// A correspondence is never an outlier if its laser coord error is less than
// this
constexpr double MIN_OUTLIER_THRESHOLD{2e-3};
constexpr int MAX_OUTLIER_ITERATIONS{3};
// Residual assigned to a position that projects behind the scanner
constexpr double INVALID_PROJECTION_RESIDUAL{10.0};

bool isNearlyPlanar(const Eigen::MatrixXd& positions) {
  Eigen::MatrixXd centered{positions.rowwise() - positions.colwise().mean()};
  Eigen::JacobiSVD<Eigen::MatrixXd> svd{centered};
  Eigen::VectorXd singularValues{svd.singularValues()};
  if (singularValues[0] <= 0.0) {
    return true;
  }
  return singularValues[2] / singularValues[0] < PLANARITY_RATIO_THRESHOLD;
}

// Transform for Hartley normalization - moves the centroid of the points to the
// origin and scales them to a mean distance of sqrt(dim) from it.
Eigen::MatrixXd normalizingTransform(const Eigen::MatrixXd& points) {
  auto dim{points.cols()};
  Eigen::RowVectorXd centroid{points.colwise().mean()};
  double meanDistance{(points.rowwise() - centroid).rowwise().norm().mean()};
  double scale{meanDistance > 0.0
                   ? std::sqrt(static_cast<double>(dim)) / meanDistance
                   : 1.0};
  Eigen::MatrixXd transform{Eigen::MatrixXd::Identity(dim + 1, dim + 1)};
  transform.topLeftCorner(dim, dim) *= scale;
  transform.topRightCorner(dim, 1) = -scale * centroid.transpose();
  return transform;
}

// Initial estimate from a pinhole approximation of the scanner. Solve for the
// 3x4 projection matrix with a normalized DLT, then decompose it into
// intrinsics (which map to scale/offset) and pose. Requires non-planar
// positions.
std::optional<GalvoModel> initialEstimatePinhole(
    const Eigen::MatrixXd& positions, const Eigen::MatrixXd& laserCoords) {
  auto numPoints{positions.rows()};
  Eigen::MatrixXd positionsTransform{normalizingTransform(positions)};
  Eigen::MatrixXd laserTransform{normalizingTransform(laserCoords)};

  // Each correspondence gives two rows of A * p = 0, where p is the flattened
  // (row-major) projection matrix
  Eigen::MatrixXd A{Eigen::MatrixXd::Zero(2 * numPoints, 12)};
  for (Eigen::Index i = 0; i < numPoints; ++i) {
    Eigen::Vector4d X{positionsTransform *
                      positions.row(i).transpose().homogeneous()};
    Eigen::Vector3d x{laserTransform *
                      laserCoords.row(i).transpose().homogeneous()};
    A.block<1, 4>(2 * i, 0) = X.transpose();
    A.block<1, 4>(2 * i, 8) = -x[0] * X.transpose();
    A.block<1, 4>(2 * i + 1, 4) = X.transpose();
    A.block<1, 4>(2 * i + 1, 8) = -x[1] * X.transpose();
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd{A, Eigen::ComputeThinV};
  Eigen::VectorXd p{svd.matrixV().col(11)};
  Eigen::Matrix<double, 3, 4> normalizedP{
      Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>{p.data()}};
  Eigen::Matrix<double, 3, 4> P{laserTransform.inverse() * normalizedP *
                                positionsTransform};

  // Choose the sign of P so that the positions are in front of the scanner
  Eigen::MatrixXd homogeneousPositions{positions.rowwise().homogeneous()};
  if ((homogeneousPositions * P.row(2).transpose()).sum() < 0.0) {
    P = -P;
  }

  // If the laser x axis is mirrored relative to a right-handed frame, flip it
  // so that the decomposition yields a proper rotation, and flip it back in
  // the intrinsics afterwards
  Eigen::Matrix3d flip{Eigen::Matrix3d::Identity()};
  if (P.leftCols<3>().determinant() < 0.0) {
    flip(0, 0) = -1.0;
  }
  Eigen::Matrix<double, 3, 4> flippedP{flip * P};

  // RQ decomposition of M = K * R via QR of (J * M)^T, where J reverses rows
  Eigen::Matrix3d J{Eigen::Matrix3d::Identity().rowwise().reverse()};
  Eigen::HouseholderQR<Eigen::Matrix3d> qr{
      (J * flippedP.leftCols<3>()).transpose()};
  Eigen::Matrix3d Q{qr.householderQ()};
  Eigen::Matrix3d U{qr.matrixQR().triangularView<Eigen::Upper>()};
  Eigen::Matrix3d K{J * U.transpose() * J};
  Eigen::Matrix3d R{J * Q.transpose()};
  // Make the diagonal of K positive
  Eigen::Matrix3d D{K.diagonal().cwiseSign().asDiagonal()};
  K = K * D;
  R = D * R;

  Eigen::Vector3d translation{K.inverse() * flippedP.col(3)};
  K = flip * K / K(2, 2);

  GalvoModel model;
  Eigen::AngleAxisd angleAxis{R};
  model.rotation = angleAxis.angle() * angleAxis.axis();
  model.translation = translation;
  model.scale = {K(0, 0), K(1, 1)};
  model.offset = {K(0, 2), K(1, 2)};
  model.mirrorDistance = 0.0;
  if (!model.getParams().allFinite()) {
    return std::nullopt;
  }
  return model;
}

// Initial estimate assuming the scanner is at the camera origin with the same
// orientation, fitting only scale and offset. Used when the positions are
// nearly planar (so the DLT is degenerate), and as a fallback otherwise.
std::optional<GalvoModel> initialEstimateColocated(
    const Eigen::MatrixXd& positions, const Eigen::MatrixXd& laserCoords,
    bool xMirrorFirst) {
  auto numPoints{positions.rows()};
  // With unit scale and zero offset, the model projects to mirror angles
  GalvoModel model;
  model.xMirrorFirst = xMirrorFirst;
  model.scale = {1.0, 1.0};
  Eigen::MatrixXd angles{numPoints, 2};
  for (Eigen::Index i = 0; i < numPoints; ++i) {
    auto anglesOpt{model.project(Eigen::Vector3d{positions.row(i)})};
    if (!anglesOpt) {
      return std::nullopt;
    }
    angles.row(i) = anglesOpt->transpose();
  }

  // Per axis: laser coord = scale * angle + offset
  for (int axis = 0; axis < 2; ++axis) {
    Eigen::MatrixXd design{numPoints, 2};
    design.col(0) = angles.col(axis);
    design.col(1).setOnes();
    Eigen::Vector2d solution{
        design.colPivHouseholderQr().solve(laserCoords.col(axis))};
    model.scale[axis] = solution[0];
    model.offset[axis] = solution[1];
  }
  if (!model.getParams().allFinite()) {
    return std::nullopt;
  }
  return model;
}

// Laser coord residuals of the galvo model, for Levenberg-Marquardt
struct GalvoModelResidual {
  enum {
    InputsAtCompileTime = Eigen::Dynamic,
    ValuesAtCompileTime = Eigen::Dynamic
  };

  typedef double Scalar;
  typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
  typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
  typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime>
      JacobianType;

  GalvoModelResidual(const Eigen::MatrixXd& positions,
                     const Eigen::MatrixXd& laserCoords, bool xMirrorFirst)
      : positions{positions},
        laserCoords{laserCoords},
        xMirrorFirst{xMirrorFirst} {}

  int operator()(const Eigen::VectorXd& params,
                 Eigen::VectorXd& residuals) const {
    GalvoModel model;
    model.xMirrorFirst = xMirrorFirst;
    model.setParams(params);
    for (Eigen::Index i = 0; i < positions.rows(); ++i) {
      auto laserCoordOpt{model.project(Eigen::Vector3d{positions.row(i)})};
      if (laserCoordOpt) {
        residuals.segment<2>(2 * i) =
            *laserCoordOpt - laserCoords.row(i).transpose();
      } else {
        residuals.segment<2>(2 * i).setConstant(INVALID_PROJECTION_RESIDUAL);
      }
    }
    return 0;
  }

  // Central-difference Jacobian. The step is relative to each parameter's
  // magnitude, as parameters range from radians to millimeters.
  int df(const Eigen::VectorXd& params, Eigen::MatrixXd& jacobian) const {
    Eigen::VectorXd residualsPlus{values()};
    Eigen::VectorXd residualsMinus{values()};
    for (Eigen::Index i = 0; i < params.size(); ++i) {
      double step{1e-6 * std::max(1.0, std::abs(params[i]))};
      Eigen::VectorXd paramsPlus{params};
      paramsPlus[i] += step;
      Eigen::VectorXd paramsMinus{params};
      paramsMinus[i] -= step;
      operator()(paramsPlus, residualsPlus);
      operator()(paramsMinus, residualsMinus);
      jacobian.col(i) = (residualsPlus - residualsMinus) / (2.0 * step);
    }
    return 0;
  }

  int values() const { return static_cast<int>(laserCoords.rows()) * 2; }

  int inputs() const { return GalvoModel::NUM_PARAMS; }

  const Eigen::MatrixXd& positions;
  const Eigen::MatrixXd& laserCoords;
  bool xMirrorFirst;
};

std::optional<GalvoModel> fitModel(const Eigen::MatrixXd& positions,
                                   const Eigen::MatrixXd& laserCoords,
                                   bool nearlyPlanar) {
  // Levenberg–Marquardt only finds the nearest good solution to where it
  // starts, so we need a good initial estimate. We try both pinhole (treats the
  // projector as a pinhole camera, but fails when positions are coplanar) and
  // colocated (assumes the projector sits at the camera and faces the same way;
  // fits only scale and offset, but works even when positions are coplanar).
  std::optional<GalvoModel> pinholeOpt;
  if (!nearlyPlanar) {
    pinholeOpt = initialEstimatePinhole(positions, laserCoords);
  }

  std::vector<GalvoModel> initialModels;
  // Try both mirror orders. We will eventually select the best fitting model.
  for (bool xMirrorFirst : {true, false}) {
    if (pinholeOpt) {
      initialModels.push_back(*pinholeOpt);
      initialModels.back().xMirrorFirst = xMirrorFirst;
    }
    auto colocatedOpt{
        initialEstimateColocated(positions, laserCoords, xMirrorFirst)};
    if (colocatedOpt) {
      initialModels.push_back(*colocatedOpt);
    }
  }

  // For each initial model, refine using Levenberg-Marquardt, and select the
  // best one
  std::optional<GalvoModel> bestModel;
  double bestCost{std::numeric_limits<double>::max()};
  for (const auto& initialModel : initialModels) {
    GalvoModelResidual functor{positions, laserCoords,
                               initialModel.xMirrorFirst};
    Eigen::LevenbergMarquardt<GalvoModelResidual, double> levenbergMarquardt{
        functor};
    levenbergMarquardt.parameters.maxfev = 1000;
    Eigen::VectorXd params{initialModel.getParams()};
    levenbergMarquardt.minimize(params);

    GalvoModel refinedModel{initialModel};
    refinedModel.setParams(params);
    Eigen::VectorXd residuals{functor.values()};
    functor(params, residuals);

    double cost{residuals.squaredNorm()};
    if (!std::isfinite(cost) || !refinedModel.getParams().allFinite()) {
      continue;
    }

    if (!bestModel || cost < bestCost) {
      bestModel = refinedModel;
      bestCost = cost;
    }
  }

  return bestModel;
}

Eigen::MatrixXd selectRows(const Eigen::MatrixXd& matrix,
                           const std::vector<Eigen::Index>& rows) {
  Eigen::MatrixXd selected{static_cast<Eigen::Index>(rows.size()),
                           matrix.cols()};
  for (std::size_t i = 0; i < rows.size(); ++i) {
    selected.row(i) = matrix.row(rows[i]);
  }
  return selected;
}

double median(std::vector<double> values) {
  auto middle{values.begin() + values.size() / 2};
  std::nth_element(values.begin(), middle, values.end());
  return *middle;
}

}  // namespace

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
  model_ = {};
  hasModel_ = false;
  fitStats_ = {};
  cameraToLaserJacobian_.setZero();
  updateLaserBounds();
}

void PointCorrespondences::updateModel() {
  updateLaserBounds();
  updateCameraPixelToLaserCoordJacobian();
  fitGalvoModel();
}

void PointCorrespondences::fitGalvoModel() {
  model_ = {};
  hasModel_ = false;
  fitStats_ = {};
  if (cameraPositions_.size() != laserCoords_.size() ||
      cameraPositions_.size() < MIN_CORRESPONDENCES) {
    return;
  }

  auto numPoints{static_cast<Eigen::Index>(cameraPositions_.size())};
  Eigen::MatrixXd positions{numPoints, 3};
  Eigen::MatrixXd laserCoords{numPoints, 2};
  for (Eigen::Index i = 0; i < numPoints; ++i) {
    auto [x, y, z]{cameraPositions_[i]};
    positions.row(i) << x, y, z;
    laserCoords.row(i) << laserCoords_[i].x, laserCoords_[i].y;
  }
  bool nearlyPlanar{isNearlyPlanar(positions)};

  // Iteratively fit, then drop correspondences with large errors and refit,
  // until the set of inliers no longer changes
  std::vector<Eigen::Index> inliers(numPoints);
  std::iota(inliers.begin(), inliers.end(), 0);
  std::optional<GalvoModel> model;
  std::vector<Eigen::Index> modelInliers;
  for (int iteration = 0; iteration < MAX_OUTLIER_ITERATIONS; ++iteration) {
    auto fitOpt{fitModel(selectRows(positions, inliers),
                         selectRows(laserCoords, inliers), nearlyPlanar)};
    if (!fitOpt) {
      break;
    }
    model = fitOpt;
    modelInliers = inliers;

    std::vector<double> errors(numPoints);
    for (Eigen::Index i = 0; i < numPoints; ++i) {
      auto laserCoordOpt{model->project(Eigen::Vector3d{positions.row(i)})};
      errors[i] = laserCoordOpt
                      ? (*laserCoordOpt - laserCoords.row(i).transpose()).norm()
                      : std::numeric_limits<double>::infinity();
    }
    std::vector<double> inlierErrors;
    for (auto i : inliers) {
      inlierErrors.push_back(errors[i]);
    }
    double threshold{std::max(MIN_OUTLIER_THRESHOLD, OUTLIER_MEDIAN_MULTIPLIER *
                                                         median(inlierErrors))};

    std::vector<Eigen::Index> newInliers;
    for (Eigen::Index i = 0; i < numPoints; ++i) {
      if (errors[i] <= threshold) {
        newInliers.push_back(i);
      }
    }
    if (newInliers == inliers || newInliers.size() < MIN_CORRESPONDENCES) {
      break;
    }
    inliers = std::move(newInliers);
  }

  if (!model) {
    return;
  }

  model_ = *model;
  hasModel_ = true;

  fitStats_.numInliers = modelInliers.size();
  fitStats_.nearlyPlanar = nearlyPlanar;
  fitStats_.minDepth = std::numeric_limits<float>::max();
  fitStats_.maxDepth = std::numeric_limits<float>::lowest();
  double laserErrorSum{0.0};
  double positionErrorSum{0.0};
  for (auto i : modelInliers) {
    Eigen::Vector3d position{positions.row(i)};
    Eigen::Vector2d laserCoord{laserCoords.row(i)};
    auto projectedOpt{model_.project(position)};
    double laserError{projectedOpt ? (*projectedOpt - laserCoord).norm()
                                   : std::numeric_limits<double>::infinity()};
    double positionError{model_.distanceToBeam(laserCoord, position)};
    laserErrorSum += laserError;
    positionErrorSum += positionError;
    fitStats_.maxLaserError =
        std::max(fitStats_.maxLaserError, static_cast<float>(laserError));
    fitStats_.maxPositionError =
        std::max(fitStats_.maxPositionError, static_cast<float>(positionError));
    fitStats_.minDepth =
        std::min(fitStats_.minDepth, static_cast<float>(position.z()));
    fitStats_.maxDepth =
        std::max(fitStats_.maxDepth, static_cast<float>(position.z()));
  }
  fitStats_.meanLaserError =
      static_cast<float>(laserErrorSum / modelInliers.size());
  fitStats_.meanPositionError =
      static_cast<float>(positionErrorSum / modelInliers.size());
}

std::optional<LaserCoord> PointCorrespondences::project(
    const Position& cameraPosition) const {
  if (!hasModel_) {
    return std::nullopt;
  }
  return model_.project(cameraPosition);
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

  updateModel();
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
