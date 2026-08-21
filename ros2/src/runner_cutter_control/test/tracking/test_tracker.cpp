#include <gtest/gtest.h>

#include <stdexcept>

#include "runner_cutter_control/tracking/tracker.hpp"

namespace {

constexpr PixelCoord PIXEL_COORD_1{10, 20};
constexpr PixelCoord PIXEL_COORD_2{30, 40};
constexpr Position POSITION_1{1.0f, 2.0f, 3.0f};
constexpr Position POSITION_2{4.0f, 5.0f, 6.0f};

void expectPixelCoordEq(const PixelCoord& actual, const PixelCoord& expected) {
  EXPECT_EQ(actual.u, expected.u);
  EXPECT_EQ(actual.v, expected.v);
}

void expectPositionEq(const Position& actual, const Position& expected) {
  EXPECT_FLOAT_EQ(actual.x, expected.x);
  EXPECT_FLOAT_EQ(actual.y, expected.y);
  EXPECT_FLOAT_EQ(actual.z, expected.z);
}

size_t getCount(const std::unordered_map<Track::State, size_t>& countsByState,
                Track::State state) {
  auto it{countsByState.find(state)};
  return it != countsByState.end() ? it->second : 0;
}

}  // namespace

TEST(TrackerTest, EmptyTrackerHasNoTracks) {
  Tracker tracker;

  EXPECT_FALSE(tracker.hasTrackWithState(Track::State::PENDING));
  EXPECT_FALSE(tracker.hasTrackWithState(Track::State::ACTIVE));
  EXPECT_TRUE(tracker.getTracks().empty());
  EXPECT_TRUE(tracker.getTracksWithState(Track::State::PENDING).empty());
  EXPECT_FALSE(tracker.getTrack(1).has_value());
  EXPECT_FALSE(tracker.activateNextPendingTrack().has_value());
  EXPECT_TRUE(tracker.getCountsByState().empty());
}

TEST(TrackerTest, AddTrackCreatesNewPendingTrack) {
  Tracker tracker;

  auto track{
      tracker.addOrUpdateTrack(1, PIXEL_COORD_1, POSITION_1, 100.0, 0.9f)};

  ASSERT_NE(track, nullptr);
  EXPECT_EQ(track->getId(), 1u);
  expectPixelCoordEq(track->getPixel(), PIXEL_COORD_1);
  expectPositionEq(track->getPosition(), POSITION_1);
  EXPECT_DOUBLE_EQ(track->getTimestampSecs(), 100.0);
  EXPECT_EQ(track->getState(), Track::State::PENDING);

  EXPECT_TRUE(tracker.hasTrackWithState(Track::State::PENDING));
  EXPECT_EQ(tracker.getTracksWithState(Track::State::PENDING).size(), 1u);

  auto trackOpt{tracker.getTrack(1)};
  ASSERT_TRUE(trackOpt.has_value());
  EXPECT_EQ(*trackOpt, track);

  // addOrUpdateTrack() should have fed the measurement into the track's
  // predictor
  EXPECT_EQ(track->getPredictor().getHistory().size(), 1u);
  EXPECT_DOUBLE_EQ(track->getPredictor().getLastTimestampSec(), 100.0);
}

TEST(TrackerTest, AddTrackWithExistingIdUpdatesInPlace) {
  Tracker tracker;
  auto original{tracker.addOrUpdateTrack(1, PIXEL_COORD_1, POSITION_1, 100.0)};

  auto updated{tracker.addOrUpdateTrack(1, PIXEL_COORD_2, POSITION_2, 200.0)};

  EXPECT_EQ(updated, original);
  expectPixelCoordEq(updated->getPixel(), PIXEL_COORD_2);
  expectPositionEq(updated->getPosition(), POSITION_2);
  EXPECT_DOUBLE_EQ(updated->getTimestampSecs(), 200.0);
  EXPECT_EQ(updated->getState(), Track::State::PENDING);

  // Both calls to addOrUpdateTrack() should have recorded a predictor
  // measurement
  EXPECT_EQ(updated->getPredictor().getHistory().size(), 2u);

  EXPECT_EQ(tracker.getTracks().size(), 1u);
}

TEST(TrackerTest, AddTrackOnActiveTrackRemainsActive) {
  Tracker tracker;
  tracker.addOrUpdateTrack(1, PIXEL_COORD_1, POSITION_1, 100.0);
  tracker.activateNextPendingTrack();  // transitions track 1 to ACTIVE

  tracker.addOrUpdateTrack(1, PIXEL_COORD_2, POSITION_2, 200.0);

  auto trackOpt{tracker.getTrack(1)};
  ASSERT_TRUE(trackOpt.has_value());
  EXPECT_EQ((*trackOpt)->getState(), Track::State::ACTIVE);
  EXPECT_FALSE(tracker.activateNextPendingTrack().has_value());
}

TEST(TrackerTest, AddTrackThrowsForZeroId) {
  Tracker tracker;
  EXPECT_THROW(tracker.addOrUpdateTrack(0, PIXEL_COORD_1, POSITION_1, 0.0),
               std::invalid_argument);
}

TEST(TrackerTest, ActivateNextPendingTrackReturnsFifoOrder) {
  Tracker tracker;
  tracker.addOrUpdateTrack(1, PIXEL_COORD_1, POSITION_1, 100.0);
  tracker.addOrUpdateTrack(2, PIXEL_COORD_1, POSITION_1, 200.0);
  tracker.addOrUpdateTrack(3, PIXEL_COORD_1, POSITION_1, 300.0);

  auto first{tracker.activateNextPendingTrack()};
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ((*first)->getId(), 1u);
  EXPECT_EQ((*first)->getState(), Track::State::ACTIVE);
  EXPECT_TRUE(tracker.hasTrackWithState(Track::State::PENDING));

  auto second{tracker.activateNextPendingTrack()};
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ((*second)->getId(), 2u);
  EXPECT_EQ((*second)->getState(), Track::State::ACTIVE);
  EXPECT_TRUE(tracker.hasTrackWithState(Track::State::PENDING));

  auto third{tracker.activateNextPendingTrack()};
  ASSERT_TRUE(third.has_value());
  EXPECT_EQ((*third)->getId(), 3u);
  EXPECT_EQ((*third)->getState(), Track::State::ACTIVE);
  EXPECT_FALSE(tracker.activateNextPendingTrack().has_value());
}

TEST(TrackerTest, ProcessTrackTransitionsStateAndUpdatesPendingQueue) {
  Tracker tracker;
  tracker.addOrUpdateTrack(1, PIXEL_COORD_1, POSITION_1, 100.0);
  tracker.addOrUpdateTrack(2, PIXEL_COORD_1, POSITION_1, 200.0);

  EXPECT_TRUE(tracker.processTrack(1, Track::State::ACTIVE));

  EXPECT_TRUE(tracker.hasTrackWithState(Track::State::ACTIVE));
  auto activeTracks{tracker.getTracksWithState(Track::State::ACTIVE)};
  ASSERT_EQ(activeTracks.size(), 1u);
  EXPECT_EQ(activeTracks[0]->getId(), 1u);

  // Track 1 should have been removed from the pending queue, so the next
  // pending track should be track 2.
  auto nextPending{tracker.activateNextPendingTrack()};
  ASSERT_TRUE(nextPending.has_value());
  EXPECT_EQ((*nextPending)->getId(), 2u);
}

TEST(TrackerTest, ProcessTrackBackToPendingReAddsToQueue) {
  Tracker tracker;
  tracker.addOrUpdateTrack(1, PIXEL_COORD_1, POSITION_1, 100.0);
  tracker.activateNextPendingTrack();  // transitions track 1 to ACTIVE

  EXPECT_TRUE(tracker.processTrack(1, Track::State::PENDING));

  auto nextPending{tracker.activateNextPendingTrack()};
  ASSERT_TRUE(nextPending.has_value());
  EXPECT_EQ((*nextPending)->getId(), 1u);
  EXPECT_EQ((*nextPending)->getState(), Track::State::ACTIVE);
}

TEST(TrackerTest, ProcessTrackToSameStateIsNoOp) {
  Tracker tracker;
  tracker.addOrUpdateTrack(1, PIXEL_COORD_1, POSITION_1, 100.0);

  // Track 1 is already PENDING, so this should report no transition.
  EXPECT_FALSE(tracker.processTrack(1, Track::State::PENDING));

  // Track 1 should still only appear once in the pending queue
  ASSERT_TRUE(tracker.activateNextPendingTrack().has_value());
  EXPECT_FALSE(tracker.activateNextPendingTrack().has_value());
}

TEST(TrackerTest, ProcessTrackOnUnknownIdIsNoOp) {
  Tracker tracker;

  EXPECT_FALSE(tracker.processTrack(999, Track::State::ACTIVE));

  EXPECT_TRUE(tracker.getTracks().empty());
}

TEST(TrackerTest, ProcessTrackToCompletedResetsPredictor) {
  Tracker tracker;
  auto track{tracker.addOrUpdateTrack(1, PIXEL_COORD_1, POSITION_1, 100.0)};
  ASSERT_EQ(track->getPredictor().getHistory().size(), 1u);

  EXPECT_TRUE(tracker.processTrack(1, Track::State::COMPLETED));

  EXPECT_EQ(track->getState(), Track::State::COMPLETED);
  EXPECT_TRUE(track->getPredictor().getHistory().empty());
}

TEST(TrackerTest, ProcessTrackToFailedResetsPredictor) {
  Tracker tracker;
  auto track{tracker.addOrUpdateTrack(1, PIXEL_COORD_1, POSITION_1, 100.0)};
  ASSERT_EQ(track->getPredictor().getHistory().size(), 1u);

  EXPECT_TRUE(tracker.processTrack(1, Track::State::FAILED));

  EXPECT_EQ(track->getState(), Track::State::FAILED);
  EXPECT_TRUE(track->getPredictor().getHistory().empty());
}

TEST(TrackerTest, ClearRemovesAllTracks) {
  Tracker tracker;
  tracker.addOrUpdateTrack(1, PIXEL_COORD_1, POSITION_1, 100.0);
  tracker.addOrUpdateTrack(2, PIXEL_COORD_1, POSITION_1, 200.0);
  tracker.activateNextPendingTrack();

  tracker.clear();

  EXPECT_TRUE(tracker.getTracks().empty());
  EXPECT_FALSE(tracker.hasTrackWithState(Track::State::PENDING));
  EXPECT_FALSE(tracker.hasTrackWithState(Track::State::ACTIVE));
  EXPECT_FALSE(tracker.activateNextPendingTrack().has_value());
  EXPECT_TRUE(tracker.getCountsByState().empty());
}

TEST(TrackerTest, GetCountsByState) {
  Tracker tracker;
  tracker.addOrUpdateTrack(1, PIXEL_COORD_1, POSITION_1, 100.0);
  tracker.addOrUpdateTrack(2, PIXEL_COORD_1, POSITION_1, 200.0);
  tracker.addOrUpdateTrack(3, PIXEL_COORD_1, POSITION_1, 300.0);
  tracker.addOrUpdateTrack(4, PIXEL_COORD_1, POSITION_1, 400.0);
  tracker.addOrUpdateTrack(5, PIXEL_COORD_1, POSITION_1, 500.0);
  tracker.activateNextPendingTrack();  // track 1 -> ACTIVE
  tracker.activateNextPendingTrack();  // track 2 -> ACTIVE
  tracker.processTrack(3, Track::State::FAILED);
  tracker.processTrack(4, Track::State::COMPLETED);

  auto countsByState{tracker.getCountsByState()};

  EXPECT_EQ(getCount(countsByState, Track::State::ACTIVE), 2u);
  EXPECT_EQ(getCount(countsByState, Track::State::FAILED), 1u);
  EXPECT_EQ(getCount(countsByState, Track::State::PENDING), 1u);
  EXPECT_EQ(getCount(countsByState, Track::State::COMPLETED), 1u);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
