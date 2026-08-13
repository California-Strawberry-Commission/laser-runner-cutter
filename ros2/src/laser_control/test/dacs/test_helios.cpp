#include <gtest/gtest.h>

#include "laser_control/dacs/helios.hpp"

namespace {

bool isBlank(const HeliosPoint& p) {
  return p.r == 0 && p.g == 0 && p.b == 0 && p.i == 0;
}

}  // namespace

class HeliosGetFrameTest : public ::testing::Test {
 protected:
  std::vector<HeliosPoint> getFrame(Helios& helios, int fps, int pps,
                                    float transitionDurationMs) {
    return helios.getFrame(fps, pps, transitionDurationMs);
  }
};

TEST_F(HeliosGetFrameTest, NoPathsProducesBlankFrameOfExpectedSize) {
  Helios helios;

  auto frame{
      getFrame(helios, /*fps=*/1, /*pps=*/10, /*transitionDurationMs=*/0.0f)};

  // pps / fps = 10 laxels for the blank point
  ASSERT_EQ(frame.size(), 10u);

  for (const auto& p : frame) {
    EXPECT_EQ(p.x, 0);
    EXPECT_EQ(p.y, 0);
    EXPECT_TRUE(isBlank(p));
  }
}

TEST_F(HeliosGetFrameTest, SinglePathProducesDenormalizedPositionAndColor) {
  Helios helios;
  helios.setColor(1.0f, 0.5f, 0.25f, 0.1f);
  helios.setPath(1, Point{0.5f, 0.5f}, /*durationMs=*/0.0f);

  auto frame{getFrame(helios, /*fps=*/1, /*pps=*/10,
                      /*transitionDurationMs=*/0.0f)};

  ASSERT_EQ(frame.size(), 10u);
  for (const auto& p : frame) {
    EXPECT_EQ(p.x, 2048);  // round(0.5 * 4095)
    EXPECT_EQ(p.y, 2048);
    EXPECT_EQ(p.r, 255);  // round(1.0 * 255)
    EXPECT_EQ(p.g, 128);  // round(0.5 * 255)
    EXPECT_EQ(p.b, 64);   // round(0.25 * 255)
    EXPECT_EQ(p.i, 26);   // round(0.1 * 255)
  }
}

TEST_F(HeliosGetFrameTest, LaxelsPerPointIsClampedToAtLeastOne) {
  Helios helios;
  // ppf / numPoints = (pps / fps) / numPoints = (100 / 100) / 5 = 0.2, which
  // rounds to 0 and should be clamped to 1
  for (uint32_t id = 0; id < 5; ++id) {
    helios.setPath(id, Point{0.0f, 0.0f}, /*durationMs=*/0.0f);
  }

  auto frame{getFrame(helios, /*fps=*/100, /*pps=*/100,
                      /*transitionDurationMs=*/0.0f)};

  EXPECT_EQ(frame.size(), 5u);
}

TEST_F(HeliosGetFrameTest, TransitionLaxelsAreBlankButKeepPosition) {
  Helios helios;
  helios.setColor(1.0f, 1.0f, 1.0f, 1.0f);
  Point pointA{0.0f, 0.0f};
  Point pointB{1.0f, 1.0f};
  helios.setPath(1, pointA, /*durationMs=*/0.0f);
  helios.setPath(2, pointB, /*durationMs=*/0.0f);

  // pps=100 -> 10ms per laxel. transitionDurationMs=30 -> 3 transition
  // laxels per point out of 50 laxels per point (100 pps / 1 fps / 2 points)
  auto frame{getFrame(helios, /*fps=*/1, /*pps=*/100,
                      /*transitionDurationMs=*/30.0f)};

  ASSERT_EQ(frame.size(), 100u);
  const size_t laxelsPerPoint{50};
  const size_t laxelsPerTransition{3};

  for (size_t block = 0; block < 2; ++block) {
    size_t base{block * laxelsPerPoint};
    // All laxels in the block share the same denormalized position
    uint16_t blockX{frame[base].x};
    uint16_t blockY{frame[base].y};
    bool isPointA{blockX == 0 && blockY == 0};
    bool isPointB{blockX == 4095 && blockY == 4095};
    EXPECT_TRUE(isPointA || isPointB);

    for (size_t i = 0; i < laxelsPerPoint; ++i) {
      const auto& p{frame[base + i]};
      EXPECT_EQ(p.x, blockX);
      EXPECT_EQ(p.y, blockY);
      if (i < laxelsPerTransition) {
        EXPECT_TRUE(isBlank(p)) << "block=" << block << " laxel=" << i;
      } else {
        EXPECT_FALSE(isBlank(p)) << "block=" << block << " laxel=" << i;
        EXPECT_EQ(p.r, 255);
        EXPECT_EQ(p.g, 255);
        EXPECT_EQ(p.b, 255);
        EXPECT_EQ(p.i, 255);
      }
    }
  }
}
