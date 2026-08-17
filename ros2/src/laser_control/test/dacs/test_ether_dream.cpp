#include <gtest/gtest.h>

#include "laser_control/dacs/ether_dream.hpp"

namespace {

bool isBlank(const etherdream_point& p) {
  return p.r == 0 && p.g == 0 && p.b == 0 && p.i == 0;
}

}  // namespace

class EtherDreamGetFrameTest : public ::testing::Test {
 protected:
  std::vector<etherdream_point> getFrame(EtherDream& etherDream, int fps,
                                         int pps, float transitionDurationMs) {
    return etherDream.getFrame(fps, pps, transitionDurationMs);
  }
};

TEST_F(EtherDreamGetFrameTest, NoPathsProducesBlankFrameOfExpectedSize) {
  EtherDream etherDream;

  auto frame{getFrame(etherDream, /*fps=*/1, /*pps=*/10,
                      /*transitionDurationMs=*/0.0f)};

  ASSERT_EQ(frame.size(), 10u);
  for (const auto& p : frame) {
    EXPECT_EQ(p.x, 0);
    EXPECT_EQ(p.y, 0);
    EXPECT_TRUE(isBlank(p));
  }
}

TEST_F(EtherDreamGetFrameTest, SinglePathProducesDenormalizedPositionAndColor) {
  EtherDream etherDream;
  etherDream.setColor(1.0f, 0.5f, 0.25f, 0.1f);
  etherDream.addWaypoint(1, Point{0.0f, 0.0f}, /*timestampSec=*/0.0);

  auto frame{getFrame(etherDream, /*fps=*/1, /*pps=*/10,
                      /*transitionDurationMs=*/0.0f)};

  ASSERT_EQ(frame.size(), 10u);
  for (const auto& p : frame) {
    EXPECT_EQ(p.x, -32768);  // round(65535 * 0.0 + (-32768))
    EXPECT_EQ(p.y, -32768);
    EXPECT_EQ(p.r, 65535);  // round(1.0 * 65535)
    EXPECT_EQ(p.g, 32768);  // round(0.5 * 65535)
    EXPECT_EQ(p.b, 16384);  // round(0.25 * 65535)
    EXPECT_EQ(p.i, 6554);   // round(0.1 * 65535)
  }
}

TEST_F(EtherDreamGetFrameTest, LaxelsPerPointIsClampedToAtLeastOne) {
  EtherDream etherDream;
  // ppf / numPoints = (pps / fps) / numPoints = (100 / 100) / 5 = 0.2, which
  // rounds to 0 and should be clamped to 1
  for (uint32_t id = 0; id < 5; ++id) {
    etherDream.addWaypoint(id, Point{0.0f, 0.0f}, /*timestampSec=*/0.0);
  }

  auto frame{getFrame(etherDream, /*fps=*/100, /*pps=*/100,
                      /*transitionDurationMs=*/0.0f)};

  EXPECT_EQ(frame.size(), 5u);
}

TEST_F(EtherDreamGetFrameTest, TransitionLaxelsAreColorBlankedButKeepPosition) {
  EtherDream etherDream;
  etherDream.setColor(1.0f, 1.0f, 1.0f, 1.0f);
  Point pointA{0.0f, 0.0f};
  Point pointB{1.0f, 1.0f};
  etherDream.addWaypoint(1, pointA, /*timestampSec=*/0.0);
  etherDream.addWaypoint(2, pointB, /*timestampSec=*/0.0);

  // pps=100 -> 10ms per laxel. transitionDurationMs=30 -> 3 transition
  // laxels per point out of 50 laxels per point (100 pps / 1 fps / 2 points)
  auto frame{getFrame(etherDream, /*fps=*/1, /*pps=*/100,
                      /*transitionDurationMs=*/30.0f)};

  ASSERT_EQ(frame.size(), 100u);
  const size_t laxelsPerPoint{50};
  const size_t laxelsPerTransition{3};

  for (size_t block = 0; block < 2; ++block) {
    size_t base{block * laxelsPerPoint};
    int16_t blockX{frame[base].x};
    int16_t blockY{frame[base].y};
    bool isPointA{blockX == -32768 && blockY == -32768};
    bool isPointB{blockX == 32767 && blockY == 32767};
    EXPECT_TRUE(isPointA || isPointB);

    for (size_t i = 0; i < laxelsPerPoint; ++i) {
      const auto& p{frame[base + i]};
      EXPECT_EQ(p.x, blockX);
      EXPECT_EQ(p.y, blockY);
      if (i < laxelsPerTransition) {
        EXPECT_TRUE(isBlank(p)) << "block=" << block << " laxel=" << i;
      } else {
        EXPECT_FALSE(isBlank(p)) << "block=" << block << " laxel=" << i;
        EXPECT_EQ(p.r, 65535);
        EXPECT_EQ(p.g, 65535);
        EXPECT_EQ(p.b, 65535);
        EXPECT_EQ(p.i, 65535);
      }
    }
  }
}
