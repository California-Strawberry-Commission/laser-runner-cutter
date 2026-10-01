#include <gtest/gtest.h>

#include <random>
#include <sstream>

#include "runner_cutter_control/calibration/point_correspondences.hpp"

namespace {

GalvoModel makeTestModel(bool xMirrorFirst = true) {
  GalvoModel model;
  model.rotation = {0.02, -0.03, 0.01};
  model.translation = {60.0, -40.0, 10.0};
  model.scale = {-1.0 / 0.698, 1.0 / 0.698};
  model.offset = {0.5, 0.5};
  model.mirrorDistance = 12.0;
  model.xMirrorFirst = xMirrorFirst;
  return model;
}

// Camera-space position where the beam for a laser coord hits the plane
// z = depth
Eigen::Vector3d beamAtDepth(const GalvoModel& model,
                            const Eigen::Vector2d& laserCoord, double depth) {
  GalvoModel::Ray ray{model.backproject(laserCoord)};
  double distance{(depth - ray.origin.z()) / ray.direction.z()};
  return ray.origin + distance * ray.direction;
}

// Adds correspondences for a grid of laser coords hitting the plane z = depth,
// with optional Gaussian noise on the positions
void addGridAtDepth(PointCorrespondences& correspondences,
                    const GalvoModel& model, double depth, int gridSize = 7,
                    double noiseStdDev = 0.0, unsigned int seed = 0) {
  std::mt19937 rng{seed};
  std::normal_distribution<double> noise{0.0, noiseStdDev};
  for (int i = 0; i < gridSize; ++i) {
    for (int j = 0; j < gridSize; ++j) {
      Eigen::Vector2d laserCoord{0.1 + 0.8 * i / (gridSize - 1),
                                 0.1 + 0.8 * j / (gridSize - 1)};
      Eigen::Vector3d position{beamAtDepth(model, laserCoord, depth)};
      if (noiseStdDev > 0.0) {
        position += Eigen::Vector3d{noise(rng), noise(rng), noise(rng)};
      }
      // Note that the camera pixel coords are not used when fitting the model
      // and thus are just placeholders here.
      correspondences.add(
          {static_cast<float>(laserCoord.x()),
           static_cast<float>(laserCoord.y())},
          {10, 20},
          {static_cast<float>(position.x()), static_cast<float>(position.y()),
           static_cast<float>(position.z())});
    }
  }
}

void addGridsAcrossDepths(PointCorrespondences& correspondences,
                          const GalvoModel& model, double noiseStdDev = 0.0) {
  unsigned int seed{0};
  for (double depth : {500.0, 1000.0, 1500.0}) {
    addGridAtDepth(correspondences, model, depth, 7, noiseStdDev, seed++);
  }
}

// Aims the laser with the fitted model at positions across 500-1500mm (off the
// calibration grid), and returns the max distance by which the true beam
// misses them.
double maxTargetingError(const PointCorrespondences& correspondences,
                         const GalvoModel& model) {
  double maxError{0.0};
  for (double depth : {500.0, 750.0, 1000.0, 1250.0, 1500.0}) {
    for (double x = 0.15; x < 0.9; x += 0.1) {
      for (double y = 0.15; y < 0.9; y += 0.1) {
        Eigen::Vector3d target{beamAtDepth(model, {x, y}, depth)};
        auto laserCoordOpt{correspondences.project(
            {static_cast<float>(target.x()), static_cast<float>(target.y()),
             static_cast<float>(target.z())})};
        if (!laserCoordOpt) {
          return std::numeric_limits<double>::infinity();
        }
        Eigen::Vector2d laserCoord{laserCoordOpt->x, laserCoordOpt->y};
        maxError = std::max(maxError, model.distanceToBeam(laserCoord, target));
      }
    }
  }
  return maxError;
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

TEST(GalvoModelTest, BackprojectInvertsProject) {
  for (bool xMirrorFirst : {true, false}) {
    GalvoModel model{makeTestModel(xMirrorFirst)};
    for (double depth : {500.0, 1500.0}) {
      for (Eigen::Vector2d laserCoord :
           {Eigen::Vector2d{0.1, 0.9}, Eigen::Vector2d{0.5, 0.5},
            Eigen::Vector2d{0.9, 0.2}}) {
        Eigen::Vector3d position{beamAtDepth(model, laserCoord, depth)};
        auto projectedOpt{model.project(position)};
        ASSERT_TRUE(projectedOpt);
        EXPECT_TRUE(projectedOpt->isApprox(laserCoord, 1e-9));
        EXPECT_NEAR(model.distanceToBeam(laserCoord, position), 0.0, 1e-9);
      }
    }
  }
}

TEST(GalvoModelTest, PositionBehindScannerDoesNotProject) {
  GalvoModel model{makeTestModel()};
  EXPECT_FALSE(model.project(Eigen::Vector3d{0.0, 0.0, -500.0}));
}

TEST(PointCorrespondencesTest, InitialStateIsEmpty) {
  PointCorrespondences correspondences;

  EXPECT_EQ(correspondences.size(), 0);

  PixelRect bounds{correspondences.getLaserBounds()};
  EXPECT_EQ(bounds.u, 0);
  EXPECT_EQ(bounds.v, 0);
  EXPECT_EQ(bounds.width, 0);
  EXPECT_EQ(bounds.height, 0);

  EXPECT_FALSE(correspondences.hasModel());
  EXPECT_FALSE(correspondences.project({0.0f, 0.0f, 1000.0f}));
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
  addGridsAcrossDepths(correspondences, makeTestModel());
  correspondences.updateModel();
  ASSERT_TRUE(correspondences.hasModel());

  correspondences.clear();

  EXPECT_EQ(correspondences.size(), 0);
  PixelRect bounds{correspondences.getLaserBounds()};
  EXPECT_EQ(bounds.u, 0);
  EXPECT_EQ(bounds.v, 0);
  EXPECT_EQ(bounds.width, 0);
  EXPECT_EQ(bounds.height, 0);
  EXPECT_FALSE(correspondences.hasModel());
}

TEST(PointCorrespondencesTest, TooFewCorrespondencesDoNotFitModel) {
  PointCorrespondences correspondences;
  addGridAtDepth(correspondences, makeTestModel(), 1000.0, 2);
  ASSERT_LT(correspondences.size(), PointCorrespondences::MIN_CORRESPONDENCES);

  correspondences.updateModel();

  EXPECT_FALSE(correspondences.hasModel());
}

TEST(PointCorrespondencesTest, FitsExactDataAcrossDepths) {
  GalvoModel model{makeTestModel()};
  PointCorrespondences correspondences;
  addGridsAcrossDepths(correspondences, model);

  correspondences.updateModel();

  ASSERT_TRUE(correspondences.hasModel());
  auto stats{correspondences.getFitStats()};
  EXPECT_EQ(stats.numInliers, correspondences.size());
  EXPECT_FALSE(stats.nearlyPlanar);
  EXPECT_NEAR(stats.minDepth, 500.0f, 1.0f);
  EXPECT_NEAR(stats.maxDepth, 1500.0f, 1.0f);
  EXPECT_LT(stats.meanLaserError, 1e-5f);
  EXPECT_LT(stats.maxPositionError, 0.05f);
  EXPECT_TRUE(correspondences.getModel().xMirrorFirst);
  EXPECT_NEAR(correspondences.getModel().mirrorDistance, 12.0, 0.5);
  EXPECT_LT(maxTargetingError(correspondences, model), 0.1);
}

TEST(PointCorrespondencesTest, RecoversMirrorOrder) {
  GalvoModel model{makeTestModel(false)};
  PointCorrespondences correspondences;
  addGridsAcrossDepths(correspondences, model);

  correspondences.updateModel();

  ASSERT_TRUE(correspondences.hasModel());
  EXPECT_FALSE(correspondences.getModel().xMirrorFirst);
  EXPECT_LT(maxTargetingError(correspondences, model), 0.1);
}

TEST(PointCorrespondencesTest, FitsRotatedScanner) {
  // Scanner rotated (90, -90, and 180 degrees) about its optical axis
  // relative to the camera
  for (double rollDegrees : {90.0, -90.0, 180.0}) {
    SCOPED_TRACE("roll = " + std::to_string(rollDegrees) + " degrees");
    GalvoModel model{makeTestModel()};
    Eigen::AngleAxisd roll{rollDegrees * M_PI / 180.0,
                           Eigen::Vector3d::UnitZ()};
    Eigen::AngleAxisd mountingError{model.rotation.norm(),
                                    model.rotation.normalized()};
    Eigen::AngleAxisd rotation{roll * mountingError};
    model.rotation = rotation.angle() * rotation.axis();

    PointCorrespondences correspondences;
    addGridsAcrossDepths(correspondences, model);

    correspondences.updateModel();

    ASSERT_TRUE(correspondences.hasModel());
    EXPECT_EQ(correspondences.getFitStats().numInliers, correspondences.size());
    EXPECT_TRUE(correspondences.getModel().xMirrorFirst);
    EXPECT_NEAR(correspondences.getModel().mirrorDistance, 12.0, 0.5);
    EXPECT_LT(maxTargetingError(correspondences, model), 0.1);
  }
}

TEST(PointCorrespondencesTest, FitsNoisyDataAcrossDepths) {
  GalvoModel model{makeTestModel()};
  PointCorrespondences correspondences;
  // 1mm standard deviation of noise on each axis of each position
  addGridsAcrossDepths(correspondences, model, 1.0);

  correspondences.updateModel();

  ASSERT_TRUE(correspondences.hasModel());
  EXPECT_LT(correspondences.getFitStats().meanPositionError, 2.0f);
  EXPECT_LT(maxTargetingError(correspondences, model), 1.5);
}

TEST(PointCorrespondencesTest, SingleDepthIsFlaggedAsNearlyPlanar) {
  GalvoModel model{makeTestModel()};
  PointCorrespondences correspondences;
  addGridAtDepth(correspondences, model, 1000.0);

  correspondences.updateModel();

  ASSERT_TRUE(correspondences.hasModel());
  EXPECT_TRUE(correspondences.getFitStats().nearlyPlanar);
  // The calibration plane itself is still fit well
  EXPECT_LT(correspondences.getFitStats().maxPositionError, 0.5f);
}

TEST(PointCorrespondencesTest, RejectsOutliers) {
  GalvoModel model{makeTestModel()};
  PointCorrespondences correspondences;
  addGridsAcrossDepths(correspondences, model);
  // A false laser detection far from where the laser actually was
  Eigen::Vector3d position{beamAtDepth(model, {0.5, 0.5}, 1000.0)};
  correspondences.add(
      {0.3f, 0.7f}, {10, 20},
      {static_cast<float>(position.x()), static_cast<float>(position.y()),
       static_cast<float>(position.z())});

  correspondences.updateModel();

  ASSERT_TRUE(correspondences.hasModel());
  EXPECT_EQ(correspondences.getFitStats().numInliers,
            correspondences.size() - 1);
  EXPECT_LT(maxTargetingError(correspondences, model), 0.1);
}

TEST(PointCorrespondencesTest, UpdateCameraPixelToLaserCoordJacobian) {
  PointCorrespondences correspondences;
  addCameraPixelToLaserCorrespondences(correspondences);

  correspondences.updateModel();

  Eigen::Matrix2d jacobian{
      correspondences.getCameraPixelToLaserCoordJacobian()};
  EXPECT_NEAR(jacobian(0, 0), 2.0, 1e-3);
  EXPECT_NEAR(jacobian(0, 1), 3.0, 1e-3);
  EXPECT_NEAR(jacobian(1, 0), 0.5, 1e-3);
  EXPECT_NEAR(jacobian(1, 1), -1.0, 1e-3);
}

TEST(PointCorrespondencesTest, SerializeDeserializeRoundTrip) {
  PointCorrespondences original;
  addGridsAcrossDepths(original, makeTestModel());
  original.updateModel();

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

  // deserialize() refits the model from the restored point correspondences,
  // so it should match what fitting the original data produced.
  ASSERT_TRUE(restored.hasModel());
  EXPECT_TRUE(restored.getModel().getParams().isApprox(
      original.getModel().getParams(), 1e-6));
  EXPECT_EQ(restored.getModel().xMirrorFirst, original.getModel().xMirrorFirst);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
