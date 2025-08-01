import tensorrt as trt
import threading
import time
import sys
import os

onnx_path = "/home/csc-jetson2/Documents/laser-runner-cutter/ros2/camera_control/models/RunnerSegYoloV8l.onnx"      # change as needed
engine_path = "/home/csc-jetson2/Documents/laser-runner-cutter/ros2/camera_control/models/RunnerSegYoloV8l.engine.new.two"  # output engine

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def spinner(stop_event):
    spinner_chars = ['|', '/', '-', '\\']
    i = 0
    while not stop_event.is_set():
        print(f"\rBuilding engine... {spinner_chars[i % len(spinner_chars)]}", end='', flush=True)
        time.sleep(0.1)
        i += 1
    print("\rEngine build completed!   ")

def build_engine(onnx_file_path, engine_file_path):
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)   # 1 GB
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    stop_event = threading.Event()
    t = threading.Thread(target=spinner, args=(stop_event,))
    t.start()

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("ERROR: Failed to build serialized engine")
        return None

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if engine is None:
        print("ERROR: Failed to deserialize engine")
        return None

    stop_event.set()
    t.join()

    if engine is None:
        print("ERROR: Failed to build the engine")
        return None

    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())
    print(f"Engine saved to {engine_file_path}")

    return engine

if __name__ == "__main__":
    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX file '{onnx_path}' not found.")
        sys.exit(1)

    build_engine(onnx_path, engine_path)
