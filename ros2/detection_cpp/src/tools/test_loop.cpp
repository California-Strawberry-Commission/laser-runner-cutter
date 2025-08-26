#include "detection_cpp/tools/test_loop.hpp"

#define CONFIDENCE_THRESHOLD 0.5f
#define NMS_THRESHOLD 0.45f

#define TRACKER_FPS 1
#define TRACKER_BUFFER_SIZE 1

bool sendResults = false;

std::string overlays_dir = "/tmp/overlays/";
std::string laptop_address = "paul@100.111.178.61:/home/paul/Desktop/testing/";

int main(int argc, char const *argv[]) {
  std::string inputDir, engineFile;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-i" && i + 1 < argc) {
      inputDir = argv[++i];
    } else if (arg == "-e" && i + 1 < argc) {
      engineFile = argv[++i];
    }
  }
  if (inputDir.empty() || engineFile.empty()) {
    spdlog::error(
        "Missing required arguments. Usage: {} -i <inputDir> -e <engineFile>",
        argv[0]);
    return -1;
  }

  std::ifstream file(engineFile, std::ios::binary);
  file.seekg(0, file.end);
  std::size_t fsize = file.tellg();
  file.seekg(0, file.beg);
  std::vector<char> engineData(fsize);
  file.read(engineData.data(), fsize);
  file.close();

  auto startSetup = std::chrono::high_resolution_clock::now();

  BYTETracker tracker(TRACKER_FPS, TRACKER_BUFFER_SIZE);

  int deviceNum = 0;
  cudaSetDevice(deviceNum);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, deviceNum);
  spdlog::info("CUDA device set to: {}.", deviceProp.name);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  nvinfer1::IRuntime *mRuntime{nvinfer1::createInferRuntime(logger)};
  nvinfer1::ICudaEngine *mEngine = mRuntime->deserializeCudaEngine(engineData.data(), fsize);
  nvinfer1::IExecutionContext *mContext = mEngine->createExecutionContext();

  char const *input_name = "images";
  char const *output0_name = "output0";
  char const *output1_name = "output1";

  // nvinfer1::Dims input_dims = mEngine->getTensorShape(input_name);
  nvinfer1::Dims output0_dims = mEngine->getTensorShape(output0_name);
  nvinfer1::Dims output1_dims = mEngine->getTensorShape(output1_name);

  nvinfer1::DataType inputType = mEngine->getTensorDataType(input_name);
  bool isHalfPrecision = (inputType == nvinfer1::DataType::kHALF);
  spdlog::info("Model uses {} precision", isHalfPrecision ? "FP16" : "FP32");

  size_t dtype_size = isHalfPrecision ? sizeof(uint16_t) : sizeof(float);
  size_t output0_size = output0_dims.d[0] * output0_dims.d[1] * output0_dims.d[2] * dtype_size;
  size_t output1_size = output1_dims.d[0] * output1_dims.d[1] * output1_dims.d[2] * output1_dims.d[3] * dtype_size;
  void *inputMem, *outputMem0, *outputMem1;
  cudaMalloc(&outputMem0, output0_size);
  cudaMalloc(&outputMem1, output1_size);

  cv::Mat output0(37, 16128, CV_32FC1);
  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  std::vector<Object> objects;

  std::filesystem::create_directories(overlays_dir);

  auto endSetup = std::chrono::high_resolution_clock::now();
  auto setupDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSetup - startSetup).count();
  spdlog::info("Total time for setup: {} ms", setupDuration);

  std::cout << "\n" << std::endl;

  std::vector<std::pair<cv::Mat, std::string>> images;
  for (const auto &entry : std::filesystem::directory_iterator(inputDir)) {
    if (entry.is_regular_file()) {
      auto ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
        cv::Mat img{cv::imread(entry.path().string())};
        if (!img.empty()) {
          images.push_back(std::pair(img, entry.path().filename().string()));
        }
      }
    }
  }

  // sort in an attempt to fix non-video format
  std::sort(images.begin(), images.end(), [](const auto& a, const auto& b) {
    return a.second < b.second;
  });

  std::vector<int64_t> prepTimes, infTimes, postTimes, totalTimes;
  
  for (const auto& [img, filename] : images) {
    for (int iter = 1; iter < 11; iter++) {
      spdlog::info("=====Processing Image: {} | ({}/10)=====", filename, iter);

      cv::cuda::GpuMat gpuImg;
      gpuImg.upload(img);
      spdlog::info("Loaded RGB image: {} || size = {} x {}, channels = {}", filename, img.cols, img.rows, img.channels());
      auto startTime = std::chrono::high_resolution_clock::now();

      cv::cuda::cvtColor(gpuImg, gpuImg, cv::COLOR_BGR2RGB);
      cv::cuda::resize(gpuImg, gpuImg, cv::Size(1024, 768), 0.0, 0.0, cv::INTER_LINEAR);
      gpuImg.convertTo(gpuImg, CV_32FC1, 1.0 / 255.0);

      std::vector<cv::cuda::GpuMat> channels(3);
      cv::cuda::split(gpuImg, channels);
      cv::cuda::GpuMat nchw(channels[0].rows * 3, channels[0].cols, channels[0].type());
      for (int c = 0; c < 3; c++) {
        cv::cuda::GpuMat roi(nchw, cv::Rect(0, c * channels[0].rows, channels[0].cols, channels[0].rows));
        channels[c].copyTo(roi);
      }

      auto endPreprocessTime = std::chrono::high_resolution_clock::now();
      auto preprocessDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPreprocessTime - startTime).count();
      prepTimes.push_back(preprocessDuration);
      spdlog::info("Preprocessing completed successfully in {} ms", preprocessDuration);

      if (isHalfPrecision) {
        int numElems = nchw.rows * nchw.cols;
        __half* inputFp16;
        cudaMalloc((void **)&inputFp16, numElems * sizeof(__half));
        convertFp32ToFp16(nchw, inputFp16, numElems);
        inputMem = inputFp16;
      } else {
        inputMem = nchw.ptr<void>();
      }

      cudaMemset(outputMem0, 0, output0_size);
      cudaMemset(outputMem1, 0, output1_size);

      mContext->setTensorAddress(input_name, inputMem);
      mContext->setTensorAddress(output0_name, outputMem0);
      mContext->setTensorAddress(output1_name, outputMem1);

      auto endInferenceSetupTime = std::chrono::high_resolution_clock::now();
      auto inferenceSetupDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endInferenceSetupTime - endPreprocessTime).count();
      spdlog::info("Inference Setup completed successfully in {} ms", inferenceSetupDuration);

      // bool success = mContext->executeV2(bindings);
      bool success = mContext->enqueueV3(stream);
      if (!success) {
        spdlog::error("TensorRT inference failed for image {}", filename);
        continue;
      }
      cudaStreamSynchronize(stream);

      auto endInferenceTime = std::chrono::high_resolution_clock::now();
      auto inferenceDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endInferenceTime - endInferenceSetupTime).count();
      infTimes.push_back(inferenceDuration);
      spdlog::info("Inferencing completed successfully in {} ms", inferenceDuration);

      // cudaMemcpy(output0.data, outputMem0, output0_size, cudaMemcpyDeviceToHost);
      if (isHalfPrecision) {
        cv::Mat half_output0(37, 16128, CV_16FC1);
        cudaMemcpy(half_output0.data, outputMem0, output0_size, cudaMemcpyDeviceToHost);
        half_output0.convertTo(output0, CV_32FC1);
      } else {
        cudaMemcpy(output0.data, outputMem0, output0_size, cudaMemcpyDeviceToHost);
      }

      boxes.clear();
      confidences.clear();
      objects.clear();

      for (int i = 0; i < 16128; ++i) {
        float confidence = output0.at<float>(4, i);
        if (confidence > CONFIDENCE_THRESHOLD) {
          float x_center = output0.at<float>(0, i);
          float y_center = output0.at<float>(1, i);
          float width = output0.at<float>(2, i);
          float height = output0.at<float>(3, i);

          int x = static_cast<int>(x_center - width / 2) * 2;
          int y = static_cast<int>(y_center - height / 2) * 2;
          int w = static_cast<int>(width) * 2;
          int h = static_cast<int>(height) * 2;

          Object obj;
          obj.rect = cv::Rect(x, y, w, h);        // x, y, w, h
          obj.prob = confidence;                  // confidence
          obj.label = 0;                          // optional (class index)

          boxes.push_back(obj.rect);
          confidences.push_back(obj.prob);
          objects.push_back(obj);
        }
      }

      std::vector<int> nms_indices;
      cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, nms_indices);

      std::vector<Object> nms_objects;
      for (int idx : nms_indices) {
        nms_objects.push_back(objects[idx]);
      }
      std::vector<STrack> tracked = tracker.update(nms_objects);

      auto endPostprocessingTime = std::chrono::high_resolution_clock::now();
      auto postprocessingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPostprocessingTime - endInferenceTime).count();
      postTimes.push_back(postprocessingDuration);
      spdlog::info("Post-Processing completed successfully in {} ms", postprocessingDuration);

      for (const auto& track : tracked) {
        if (!track.is_activated) continue;
        int baseLine = 0;
        std::string label = cv::format("ID: %d | Conf: %.2f", track.track_id, track.score);
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::Rect box(track.tlwh[0], track.tlwh[1], track.tlwh[2], track.tlwh[3]);
        int top = std::max(box.y, labelSize.height);

        cv::rectangle(img, box, cv::Scalar(0, 255, 0), 2);
        cv::rectangle(img, cv::Point(box.x, top - labelSize.height), cv::Point(box.x + labelSize.width, top + baseLine), cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(img, label, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,0,0), 2);
      }

      auto endOverlayingTime = std::chrono::high_resolution_clock::now();
      auto overlayingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endOverlayingTime - endPostprocessingTime).count();
      spdlog::info("Overlaying completed successfully in {} ms", overlayingDuration);

      if (sendResults) {
        std::string img_filename = overlays_dir + filename;
        cv::imwrite(img_filename, img);

        auto endDumpTime = std::chrono::high_resolution_clock::now();
        auto dumpDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDumpTime - endPostprocessingTime).count();
        spdlog::info("Dumping completed successfully in {} ms", dumpDuration);
      }

      if (isHalfPrecision) {
        cudaFree(inputMem);
      }

      auto endTime = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
      totalTimes.push_back(duration);
      spdlog::info("===Total time taken: {} ms===", duration);
    }
    std::cout << "\n" << std::endl;
  }

  auto avg = [](const std::vector<int64_t>& v) -> double {
    if (v.empty()) return 0.0;
    int64_t sum = std::accumulate(v.begin(), v.end(), int64_t(0));
    return static_cast<double>(sum) / v.size();
  };

  spdlog::info("Average Preprocessing Time: {:.2f} ms", avg(prepTimes));
  spdlog::info("Average Inference Time: {:.2f} ms", avg(infTimes));
  spdlog::info("Average Postprocessing Time: {:.2f} ms", avg(postTimes));
  spdlog::info("Average Total Time: {:.2f} ms", avg(totalTimes));
  std::cout << std::endl;


  if (sendResults) {
    system(("rsync -avz --progress " + overlays_dir + " " + laptop_address).c_str());
  }

  std::filesystem::remove_all(overlays_dir);

  cudaFree(outputMem0);
  cudaFree(outputMem1);

  delete mContext;
  delete mEngine;
  delete mRuntime;

  cudaStreamDestroy(stream);

  return 0;
}
