#include <gtest/gtest.h>

#include <sstream>

#include "runner_cutter_control/calibration/point_correspondences.hpp"

namespace {

// An arbitrary affine mapping from a 3D position to laser coord
LaserCoord projectToLaser(const Position& position) {
  return {2.0f * position.x + -0.3f * position.y + 10.0f,
          0.1f * position.x + 1.5f * position.y + -5.0f};
}

// Used to exercise camera <-> laser transform tests.
// Adds 5 arbitrary, non-degenerate 3D positions using the affine transformation
// projectToLaser.
// The reprojection error is expected to be 0.
void addPositionToLaserCorrespondences(PointCorrespondences& correspondences) {
  const std::vector<Position> positions{{0.0f, 0.0f, 0.0f},
                                        {100.0f, 0.0f, 10.0f},
                                        {0.0f, 100.0f, -10.0f},
                                        {100.0f, 100.0f, 5.0f},
                                        {50.0f, 25.0f, 0.0f}};
  for (std::size_t i = 0; i < positions.size(); ++i) {
    // Note that the camera pixel coords are not used when calculating the
    // camera to laser transform and thus are just placeholders here.
    correspondences.add(projectToLaser(positions[i]), {10, 20}, positions[i]);
  }
}

// Used to exercise camera pixel <-> laser coord Jacobian tests.
void addCameraPixelToLaserCorrespondences(
    PointCorrespondences& correspondences) {
  const std::vector<PixelCoord> pixels{
      {0, 0}, {10, 0}, {0, 10}, {10, 10}, {5, 3}};
  for (const auto& pixel : pixels) {
    LaserCoord laser{2.0f * pixel.u + 3.0f * pixel.v + 1.0f,
                     0.5f * pixel.u - 1.0f * pixel.v + 2.0f};
    // Note that the 3D positions are not used when calculating the Jacobian and
    // thus are just placeholders here.
    correspondences.add(laser, pixel, {0.0f, 0.0f, 0.0f});
  }
}

}  // namespace

TEST(PointCorrespondencesTest, InitialStateIsEmpty) {
  PointCorrespondences correspondences;

  EXPECT_EQ(correspondences.size(), 0);

  PixelRect bounds{correspondences.getLaserBounds()};
  EXPECT_EQ(bounds.u, 0);
  EXPECT_EQ(bounds.v, 0);
  EXPECT_EQ(bounds.width, 0);
  EXPECT_EQ(bounds.height, 0);

  Eigen::MatrixXd transform{correspondences.getCameraToLaserTransform()};
  EXPECT_TRUE(transform.isApproxToConstant(0.0));
  ASSERT_EQ(transform.rows(), 4);
  ASSERT_EQ(transform.cols(), 3);

  EXPECT_NEAR(correspondences.getReprojectionError(), 0.0f, 1e-3f);
}

TEST(PointCorrespondencesTest, AddIncreasesSizeAndUpdatesLaserBounds) {
  PointCorrespondences correspondences;

  correspondences.add({0.0f, 0.0f}, PixelCoord{10, 100}, {0.0f, 0.0f, 0.0f});
  EXPECT_EQ(correspondences.size(), 1);
  PixelRect bounds{correspondences.getLaserBounds()};
  EXPECT_EQ(bounds.u, 10);
  EXPECT_EQ(bounds.v, 100);
  EXPECT_EQ(bounds.width, 0);
  EXPECT_EQ(bounds.height, 0);

  correspondences.add({1.0f, 1.0f}, PixelCoord{50, 40}, {1.0f, 1.0f, 1.0f});
  EXPECT_EQ(correspondences.size(), 2);
  bounds = correspondences.getLaserBounds();
  EXPECT_EQ(bounds.u, 10);
  EXPECT_EQ(bounds.v, 40);
  EXPECT_EQ(bounds.width, 40);
  EXPECT_EQ(bounds.height, 60);
}

TEST(PointCorrespondencesTest, ClearResetsState) {
  PointCorrespondences correspondences;
  addPositionToLaserCorrespondences(correspondences);
  correspondences.updateTransformLinearLeastSquares();

  correspondences.clear();

  EXPECT_EQ(correspondences.size(), 0);
  PixelRect bounds{correspondences.getLaserBounds()};
  EXPECT_EQ(bounds.u, 0);
  EXPECT_EQ(bounds.v, 0);
  EXPECT_EQ(bounds.width, 0);
  EXPECT_EQ(bounds.height, 0);
  EXPECT_TRUE(
      correspondences.getCameraToLaserTransform().isApproxToConstant(0.0));
}

TEST(PointCorrespondencesTest, UpdateTransformLinearLeastSquaresFitsExactData) {
  PointCorrespondences correspondences;
  addPositionToLaserCorrespondences(correspondences);

  correspondences.updateTransformLinearLeastSquares();

  EXPECT_NEAR(correspondences.getReprojectionError(), 0.0f, 1e-3f);
}

TEST(PointCorrespondencesTest,
     UpdateTransformNonlinearLeastSquaresRefinesExactData) {
  PointCorrespondences correspondences;
  addPositionToLaserCorrespondences(correspondences);

  correspondences.updateTransformLinearLeastSquares();
  correspondences.updateTransformNonlinearLeastSquares();

  EXPECT_NEAR(correspondences.getReprojectionError(), 0.0f, 1e-3f);
}

TEST(PointCorrespondencesTest, EmptyCorrespondencesDoNotUpdateTransform) {
  PointCorrespondences correspondences;

  correspondences.updateTransformLinearLeastSquares();
  correspondences.updateTransformNonlinearLeastSquares();

  EXPECT_TRUE(
      correspondences.getCameraToLaserTransform().isApproxToConstant(0.0));
}

TEST(PointCorrespondencesTest, UpdateCameraPixelToLaserCoordJacobian) {
  PointCorrespondences correspondences;
  addCameraPixelToLaserCorrespondences(correspondences);

  correspondences.updateCameraPixelToLaserCoordJacobian();

  Eigen::Matrix2d jacobian{
      correspondences.getCameraPixelToLaserCoordJacobian()};
  EXPECT_NEAR(jacobian(0, 0), 2.0, 1e-3);
  EXPECT_NEAR(jacobian(0, 1), 3.0, 1e-3);
  EXPECT_NEAR(jacobian(1, 0), 0.5, 1e-3);
  EXPECT_NEAR(jacobian(1, 1), -1.0, 1e-3);
}

TEST(PointCorrespondencesTest, SerializeDeserializeRoundTrip) {
  PointCorrespondences original;
  addPositionToLaserCorrespondences(original);
  original.updateTransformLinearLeastSquares();
  original.updateTransformNonlinearLeastSquares();
  original.updateCameraPixelToLaserCoordJacobian();

  std::stringstream stream;
  original.serialize(stream);

  PointCorrespondences restored;
  restored.deserialize(stream);

  EXPECT_EQ(restored.size(), original.size());

  PixelRect originalBounds{original.getLaserBounds()};
  PixelRect restoredBounds{restored.getLaserBounds()};
  EXPECT_EQ(restoredBounds.u, originalBounds.u);
  EXPECT_EQ(restoredBounds.v, originalBounds.v);
  EXPECT_EQ(restoredBounds.width, originalBounds.width);
  EXPECT_EQ(restoredBounds.height, originalBounds.height);

  // deserialize() recomputes the transform and Jacobian from the restored
  // point correspondences, so they should match what fitting the original
  // data produced.
  EXPECT_TRUE(restored.getCameraToLaserTransform().isApprox(
      original.getCameraToLaserTransform(), 1e-2));
  EXPECT_NEAR(restored.getReprojectionError(), 0.0f, 1e-3f);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
