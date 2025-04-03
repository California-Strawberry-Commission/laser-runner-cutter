#include <chrono>
#include <iostream>

#include "runner_cutter_control_cpp/calibration/point_correspondences.hpp"

int main() {
  PointCorrespondences correspondences;
  correspondences.add({0.0, 0.0}, {569, 1411}, {-112.25, 122.0, 422.0});
  correspondences.add({0.0, 0.25}, {569, 1256}, {-112.25, 82.75, 425.5});
  correspondences.add({0.0, 0.5}, {568, 1099}, {-113.25, 45.0, 432.25});
  correspondences.add({0.0, 0.75}, {564, 937}, {-114.25, 3.75, 433.75});
  correspondences.add({0.0, 1.0}, {557, 764}, {-114.5, -38.75, 427.75});
  correspondences.add({0.25, 0.0}, {721, 1433}, {-72.75, 124.0, 412.5});
  correspondences.add({0.25, 0.25}, {720, 1275}, {-73.5, 85.5, 421.75});
  correspondences.add({0.25, 0.5}, {720, 1110}, {-73.5, 45.5, 424.25});
  correspondences.add({0.25, 0.75}, {718, 944}, {-73.25, 5.25, 424.25});
  correspondences.add({0.25, 1.0}, {716, 766}, {-73.5, -37.75, 420.75});
  correspondences.add({0.5, 0.0}, {877, 1454}, {-34.5, 124.25, 401.25});
  correspondences.add({0.5, 0.25}, {877, 1289}, {-34.75, 86.0, 410.25});
  correspondences.add({0.5, 0.5}, {878, 1124}, {-34.25, 47.75, 417.25});
  correspondences.add({0.5, 0.75}, {878, 950}, {-34.0, 6.0, 414.0});
  correspondences.add({0.5, 1.0}, {883, 767}, {-33.0, -36.75, 412.0});
  correspondences.add({0.75, 0.0}, {1045, 1470}, {4.75, 125.25, 392.25});
  correspondences.add({0.75, 0.25}, {1045, 1306}, {4.75, 87.75, 400.25});
  correspondences.add({0.75, 0.5}, {1046, 1134}, {4.75, 47.75, 404.25});
  correspondences.add({0.75, 0.75}, {1054, 955}, {6.25, 6.5, 405.75});
  correspondences.add({0.75, 1.0}, {1061, 770}, {7.75, -35.0, 402.5});
  correspondences.add({1.0, 0.0}, {1222, 1487}, {44.5, 127.0, 384.25});
  correspondences.add({1.0, 0.5}, {1234, 1144}, {46.5, 49.75, 395.0});

  auto start = std::chrono::high_resolution_clock::now();
  correspondences.updateTransformLinearLeastSquares();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "updateTransformLinearLeastSquares took " << elapsed.count()
            << " ms" << std::endl;

  double error = correspondences.getReprojectionError();
  std::cout << "Reprojection error: " << error << std::endl;

  start = std::chrono::high_resolution_clock::now();
  correspondences.updateTransformNonlinearLeastSquares();
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "updateTransformNonlinearLeastSquares took " << elapsed.count()
            << " ms" << std::endl;

  error = correspondences.getReprojectionError();
  std::cout << "Reprojection error: " << error << std::endl;

  return 0;
}