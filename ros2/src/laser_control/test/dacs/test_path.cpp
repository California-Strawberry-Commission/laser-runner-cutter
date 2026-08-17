#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "laser_control/dacs/path.hpp"

namespace {

double nowSec() {
  return std::chrono::duration<double>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

}  // namespace

TEST(PathTest, NoWaypointIsNotRenderable) {
  Path path{1};

  EXPECT_FALSE(path.getCurrentPoint().has_value());
}

TEST(PathTest, SingleWaypointIsNotRenderableBeforeNorAfterTime) {
  Path path{1};
  path.addWaypoint(Point{1.0f, 1.0f}, nowSec() + 0.05);

  EXPECT_FALSE(path.getCurrentPoint().has_value());

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  EXPECT_FALSE(path.getCurrentPoint().has_value());
}

TEST(PathTest, AccumulatesWaypointsAndTraversesThemInOrder) {
  Path path{1};
  double start{nowSec()};
  path.addWaypoint(Point{1.0f, 0.0f}, start + 0.05);
  path.addWaypoint(Point{1.0f, 1.0f}, start + 0.1);
  path.addWaypoint(Point{1.0f, 2.0f}, start + 0.15);

  auto beforeFirstWaypoint{path.getCurrentPoint()};
  EXPECT_FALSE(beforeFirstWaypoint.has_value());

  std::this_thread::sleep_for(std::chrono::milliseconds(75));
  auto betweenFirstAndSecond{path.getCurrentPoint()};

  // Should be past the first waypoint and partway to the second
  ASSERT_TRUE(betweenFirstAndSecond.has_value());
  EXPECT_FLOAT_EQ(betweenFirstAndSecond->x, 1.0f);
  EXPECT_GT(betweenFirstAndSecond->y, 0.0f);
  EXPECT_LT(betweenFirstAndSecond->y, 1.0f);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  auto betweenSecondAndThird{path.getCurrentPoint()};

  // Should be past the second waypoint and partway to the third
  ASSERT_TRUE(betweenSecondAndThird.has_value());
  EXPECT_FLOAT_EQ(betweenSecondAndThird->x, 1.0f);
  EXPECT_GT(betweenSecondAndThird->y, 1.0f);
  EXPECT_LT(betweenSecondAndThird->y, 2.0f);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  auto afterLastWaypoint{path.getCurrentPoint()};
  EXPECT_FALSE(afterLastWaypoint.has_value());
}

TEST(PathTest, OutOfOrderWaypointIsIgnored) {
  Path path{1};
  double start{nowSec()};
  path.addWaypoint(Point{1.0f, 1.0f}, start + 0.1);
  // A waypoint with a timestamp earlier than what's already queued should be
  // dropped
  path.addWaypoint(Point{0.5f, 0.5f}, start + 0.05);
  path.addWaypoint(Point{2.0f, 2.0f}, start + 0.2);

  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  auto point{path.getCurrentPoint()};

  ASSERT_TRUE(point.has_value());
  EXPECT_GT(point->x, 1.0f);
  EXPECT_LT(point->x, 2.0f);
  EXPECT_GT(point->y, 1.0f);
  EXPECT_LT(point->y, 2.0f);
}

TEST(PathTest, NonPositiveTimestampFiresImmediately) {
  Path path{1};

  path.addWaypoint(Point{1.0f, 1.0f}, /*timestampSec=*/0.0);

  auto point{path.getCurrentPoint()};
  ASSERT_TRUE(point.has_value());
  EXPECT_FLOAT_EQ(point->x, 1.0f);
  EXPECT_FLOAT_EQ(point->y, 1.0f);
}

TEST(PathTest, FireImmediatelyDiscardsQueuedWaypoints) {
  Path path{1};

  path.addWaypoint(Point{0.5f, 0.5f}, nowSec() + 10.0);
  path.addWaypoint(Point{1.0f, 1.0f}, /*timestampSec=*/0.0);

  auto point{path.getCurrentPoint()};
  ASSERT_TRUE(point.has_value());
  EXPECT_FLOAT_EQ(point->x, 1.0f);
  EXPECT_FLOAT_EQ(point->y, 1.0f);
}

TEST(PathTest, SetPointHoldsUntilSuperseded) {
  Path path{1};

  path.setPoint(Point{1.0f, 1.0f});

  auto heldPoint{path.getCurrentPoint()};
  ASSERT_TRUE(heldPoint.has_value());
  EXPECT_FLOAT_EQ(heldPoint->x, 1.0f);
  EXPECT_FLOAT_EQ(heldPoint->y, 1.0f);

  // The held point should keep rendering on subsequent calls
  auto stillHeldPoint{path.getCurrentPoint()};
  ASSERT_TRUE(stillHeldPoint.has_value());
  EXPECT_FLOAT_EQ(stillHeldPoint->x, 1.0f);
  EXPECT_FLOAT_EQ(stillHeldPoint->y, 1.0f);

  // Calling setPoint again should now hold to the new point
  path.setPoint(Point{2.0f, 2.0f});
  auto newHeldPoint{path.getCurrentPoint()};
  ASSERT_TRUE(newHeldPoint.has_value());
  EXPECT_FLOAT_EQ(newHeldPoint->x, 2.0f);
  EXPECT_FLOAT_EQ(newHeldPoint->y, 2.0f);

  path.addWaypoint(Point{3.0f, 3.0f}, nowSec() + 0.05);

  // Calling addWaypoint should stop holding, but only once the waypoint is
  // reached
  auto stillNewHeldPoint{path.getCurrentPoint()};
  ASSERT_TRUE(stillNewHeldPoint.has_value());
  EXPECT_FLOAT_EQ(stillNewHeldPoint->x, 2.0f);
  EXPECT_FLOAT_EQ(stillNewHeldPoint->y, 2.0f);

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_FALSE(path.getCurrentPoint().has_value());
}

TEST(PathTest, SetPointDiscardsQueuedWaypoints) {
  Path path{1};

  path.addWaypoint(Point{0.5f, 0.5f}, nowSec() + 10.0);
  path.setPoint(Point{1.0f, 1.0f});

  auto point{path.getCurrentPoint()};
  ASSERT_TRUE(point.has_value());
  EXPECT_FLOAT_EQ(point->x, 1.0f);
  EXPECT_FLOAT_EQ(point->y, 1.0f);
}
