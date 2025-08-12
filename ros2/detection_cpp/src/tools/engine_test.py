# import tensorrt as trt
# TRT_LOGGER = trt.Logger(trt.Logger.INFO)
# with open("/home/csc-jetson2/Documents/laser-runner-cutter/ros2/camera_control/models/RunnerSegYoloV8l.engine.new.two", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
#     engine = runtime.deserialize_cuda_engine(f.read())
#     print("Success!" if engine else "Failed to load engine")


import tensorrt as trt
engine = trt.Runtime(trt.Logger()).deserialize_cuda_engine(open("/home/csc-jetson2/Documents/laser-runner-cutter/ros2/camera_control/models/RunnerSegYoloV8l.trtexec.engine", "rb").read())
print(engine.device_memory_size) # just to trigger load
