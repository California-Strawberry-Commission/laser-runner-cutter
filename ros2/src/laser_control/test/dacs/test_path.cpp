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

TEST(PathTest, SingleWaypointIsNotRenderableBeforeTimeThenHeldAfter) {
  Path path{1};
  path.addWaypoint(Point{1.0f, 1.0f}, nowSec() + 0.05);

  EXPECT_FALSE(path.getCurrentPoint().has_value());

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  auto point{path.getCurrentPoint()};
  ASSERT_TRUE(point.has_value());
  EXPECT_FLOAT_EQ(point->x, 1.0f);
  EXPECT_FLOAT_EQ(point->y, 1.0f);

  // The last waypoint should keep rendering on subsequent calls
  auto stillHeldPoint{path.getCurrentPoint()};
  ASSERT_TRUE(stillHeldPoint.has_value());
  EXPECT_FLOAT_EQ(stillHeldPoint->x, 1.0f);
  EXPECT_FLOAT_EQ(stillHeldPoint->y, 1.0f);
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

  // Should hold at the last waypoint
  ASSERT_TRUE(afterLastWaypoint.has_value());
  EXPECT_FLOAT_EQ(afterLastWaypoint->x, 1.0f);
  EXPECT_FLOAT_EQ(afterLastWaypoint->y, 2.0f);
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

TEST(PathTest, NewWaypointAfterPathEndsInterpolatesFromHeldPointNotStaleTime) {
  Path path{1};
  double start{nowSec()};
  path.addWaypoint(Point{1.0f, 1.0f}, start + 0.05);
  path.addWaypoint(Point{2.0f, 2.0f}, start + 0.1);

  // Wait until both waypoints pass and the path is holding at the last one
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  auto heldPoint{path.getCurrentPoint()};
  ASSERT_TRUE(heldPoint.has_value());
  EXPECT_FLOAT_EQ(heldPoint->x, 2.0f);
  EXPECT_FLOAT_EQ(heldPoint->y, 2.0f);

  // Queue a new waypoint well after the hold began. The current point should
  // still be very close to the second waypoint.
  path.addWaypoint(Point{3.0f, 3.0f}, nowSec() + 0.2);
  auto justQueued{path.getCurrentPoint()};
  ASSERT_TRUE(justQueued.has_value());
  EXPECT_GE(justQueued->x, 2.0f);
  EXPECT_LT(justQueued->x, 2.1f);
  EXPECT_GE(justQueued->y, 2.0f);
  EXPECT_LT(justQueued->y, 2.1f);

  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  auto afterNewWaypoint{path.getCurrentPoint()};
  ASSERT_TRUE(afterNewWaypoint.has_value());
  EXPECT_FLOAT_EQ(afterNewWaypoint->x, 3.0f);
  EXPECT_FLOAT_EQ(afterNewWaypoint->y, 3.0f);
}

TEST(PathTest, NonPositiveTimestampFiresImmediately) {
  Path path{1};

  path.addWaypoint(Point{1.0f, 1.0f}, /*timestampSec=*/0.0);

  auto point{path.getCurrentPoint()};
  ASSERT_TRUE(point.has_value());
  EXPECT_FLOAT_EQ(point->x, 1.0f);
  EXPECT_FLOAT_EQ(point->y, 1.0f);
}

TEST(PathTest, PastTimestampFiresImmediatelyAndHoldsUntilSuperseded) {
  Path path{1};

  path.addWaypoint(Point{1.0f, 1.0f}, nowSec() - 5.0);

  auto heldPoint{path.getCurrentPoint()};
  ASSERT_TRUE(heldPoint.has_value());
  EXPECT_FLOAT_EQ(heldPoint->x, 1.0f);
  EXPECT_FLOAT_EQ(heldPoint->y, 1.0f);

  // The held point should keep rendering on subsequent calls
  auto stillHeldPoint{path.getCurrentPoint()};
  ASSERT_TRUE(stillHeldPoint.has_value());
  EXPECT_FLOAT_EQ(stillHeldPoint->x, 1.0f);
  EXPECT_FLOAT_EQ(stillHeldPoint->y, 1.0f);

  // Another past-timestamp waypoint should move to and hold the new point
  path.addWaypoint(Point{2.0f, 2.0f}, nowSec() - 1.0);
  auto newHeldPoint{path.getCurrentPoint()};
  ASSERT_TRUE(newHeldPoint.has_value());
  EXPECT_FLOAT_EQ(newHeldPoint->x, 2.0f);
  EXPECT_FLOAT_EQ(newHeldPoint->y, 2.0f);
}

TEST(PathTest, PastTimestampDiscardsQueuedWaypoints) {
  Path path{1};

  path.addWaypoint(Point{0.5f, 0.5f}, nowSec() + 10.0);
  path.addWaypoint(Point{1.0f, 1.0f}, nowSec() - 1.0);

  auto point{path.getCurrentPoint()};
  ASSERT_TRUE(point.has_value());
  EXPECT_FLOAT_EQ(point->x, 1.0f);
  EXPECT_FLOAT_EQ(point->y, 1.0f);
}
