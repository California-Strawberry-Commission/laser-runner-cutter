#include <gtest/gtest.h>

#include "runner_cutter_control/prediction/kalman_filter_predictor.hpp"

namespace {

constexpr Position ZERO_POSITION{0.0f, 0.0f, 0.0f};

void expectPositionEq(const Position& actual, const Position& expected) {
  EXPECT_FLOAT_EQ(actual.x, expected.x);
  EXPECT_FLOAT_EQ(actual.y, expected.y);
  EXPECT_FLOAT_EQ(actual.z, expected.z);
}

}  // namespace

TEST(KalmanFilterPredictorTest, PredictBeforeAnyMeasurementReturnsZero) {
  KalmanFilterPredictor predictor;

  expectPositionEq(predictor.predict(0.0), ZERO_POSITION);
  expectPositionEq(predictor.predict(10.0), ZERO_POSITION);
}

TEST(KalmanFilterPredictorTest, SingleMeasurementPrediction) {
  KalmanFilterPredictor predictor;
  Position position{5.0f, 6.0f, 7.0f};
  predictor.add(2.0, position, 1.0f);

  // A prediction at the single measurement timestamp should return the
  // measurement
  expectPositionEq(predictor.predict(2.0), position);

  // A prediction for before the single measurement timestamp should return the
  // measurement
  expectPositionEq(predictor.predict(0.0), position);

  // A prediction for after the single measurement timestamp should return the
  // measurement
  expectPositionEq(predictor.predict(50.0), position);
}

TEST(KalmanFilterPredictorTest, OutOfOrderMeasurementIsIgnored) {
  KalmanFilterPredictor predictor;
  Position positionA{5.0f, 5.0f, 5.0f};
  Position positionB{100.0f, 100.0f, 100.0f};

  EXPECT_TRUE(predictor.add(2.0, positionA, 1.0f));
  // Attempting to add a measurement with a timestamp earlier than or equal to
  // the last added should be rejected
  EXPECT_FALSE(predictor.add(1.0, positionB, 1.0f));
  EXPECT_FALSE(predictor.add(2.0, positionB, 1.0f));

  EXPECT_EQ(predictor.getHistory().size(), 1u);
  EXPECT_DOUBLE_EQ(predictor.getLastTimestampSec(), 2.0);

  // Both a past and a future prediction should reflect only positionA
  expectPositionEq(predictor.predict(1.0), positionA);
  expectPositionEq(predictor.predict(3.0), positionA);
}

TEST(KalmanFilterPredictorTest, ResetClearsStateAndHistory) {
  KalmanFilterPredictor predictor;
  predictor.add(5.0, {1.0f, 2.0f, 3.0f}, 1.0f);

  predictor.reset();

  EXPECT_TRUE(predictor.getHistory().empty());
  EXPECT_DOUBLE_EQ(predictor.getLastTimestampSec(), 0.0);
  expectPositionEq(predictor.predict(0.0), ZERO_POSITION);
  expectPositionEq(predictor.predict(100.0), ZERO_POSITION);

  Position position{7.0f, 8.0f, 9.0f};
  predictor.add(10.0, position, 1.0f);
  expectPositionEq(predictor.predict(20.0), position);
}

TEST(KalmanFilterPredictorTest, ConstantVelocityMeasurementsPrediction) {
  KalmanFilterPredictor predictor;

  // Feed 20 seconds of constant-velocity measurements
  float velocityX{50.0f};
  float velocityY{-30.0f};
  float velocityZ{20.0f};
  int numMeasurements{20};
  for (int i = 0; i < numMeasurements; ++i) {
    Position position{velocityX * i, velocityY * i, velocityZ * i};
    predictor.add(static_cast<double>(i), position, 1.0f);
  }

  Position lastMeasuredPosition{velocityX * (numMeasurements - 1),
                                velocityY * (numMeasurements - 1),
                                velocityZ * (numMeasurements - 1)};

  // The predicted position in the future should be close to the expected
  // position at constant velocity.
  double futureSecs{5.0};
  Position futurePos{
      predictor.predict(predictor.getLastTimestampSec() + futureSecs)};
  EXPECT_NEAR(futurePos.x, lastMeasuredPosition.x + velocityX * futureSecs,
              1.0f);
  EXPECT_NEAR(futurePos.y, lastMeasuredPosition.y + velocityY * futureSecs,
              1.0f);
  EXPECT_NEAR(futurePos.z, lastMeasuredPosition.z + velocityZ * futureSecs,
              1.0f);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
