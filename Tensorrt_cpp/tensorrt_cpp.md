# Deploying TensorRT Engine on NVIDIA GPU: A Step-by-Step Guide

## Problem statement
- Run deep learning models efficiently at runtime on embedded platforms (only platforms with NVIDIA GPUs in this document scope)
- Meet latency requirements for safety critical applications

## Overview
- Introduction
- Deployment options
- Pre-requisites for Tensorrt deployment
- Tensorrt deployment workflow
- Step-by-Step procedure

## Introduction
- Deep Learning models, getting larger, while there is increased focus on edge-computing for various applications
- Performance targets b/w training and inference differ significantly
    - Training
        - can consume large memory, super high compute GPU platforms (multi-GPU), no strict restriction on latency
    - Inference
        - Limited memory, strict latency requirements, limited compute (embedded GPU), energy consumption, environment limitations (temperature)
- Model optimization, for inference is critical, and can be classified into 4 categories

![Model_optimization_techniques](images/Model_optimization_techniques.png)

- [Image source](https://www.youtube.com/watch?v=f86hkOGoX54)
- Advantages
    - Reduced memory footprint requirements on device
    - Faster inference time, reducing latency (=> more real-time solutions)
    - Reduced energy consumption ( => increased battery life)
    - Lower clock speed requirements

## Deployment options
- Two categories of classification
    - Programming language (C++ vs Python)
    - Model type
        - Framework (Tensorflow (.pb) / Pytorch (.pth))
        - ONNX (Open Neural Network Exchange)
        - [Tensorrt](https://developer.nvidia.com/tensorrt) (NVIDIA GPUs)
- [ONNX](https://onnx.ai/) is an open, common format built to represent machine learning models. ONNX defines a common set of operators - the building blocks of machine learning and deep learning models - and a common file format use models with a variety of frameworks, tools, runtimes, and compilers.

### Torch-TensorRT
- [Torch-TensorRT](https://github.com/pytorch/TensorRT) provides model conversion and a high-level runtime API for converting PyTorch models.
- Torch-TensorRT conversion results in a PyTorch graph with TensorRT operations inserted into it.
- Falls back to PyTorch implementations where TensorRT does not support a particular operator.
- Common deployment is in python. C++ runtime using [libtorch](https://docs.pytorch.org/docs/stable/cpp_index.html) should be possible

### Python vs C++ runtime
- Python API
    - Easier for deployment, debugging issues
    - Reuse pre and post processing transforms
    - Make sure inference transforms are possible (on deployment Hardware)
- C++ API
    - C++ API support for more platforms (64-bit windows)
    - Mulithreading possible. In python, CPython can use only 1 system thread, due to GIL
    - [NVIDIA Thrust](https://developer.nvidia.com/thrust) provides GPU accelerated implementations of common operations like sort, scan, transform etc

### General workflow
- Train a model using PyTorch
- Convert the model to ONNX format
- Use NVIDIA TensorRT (C++ runtime) for inference


NVIDIA TensorRT is an SDK for optimizing trained deep-learning models to enable high-performance inference. TensorRT contains a deep learning inference optimizer and a runtime for execution.



##

This guide outlines the process of converting an ONNX model into an optimized TensorRT engine and deploying it for high-performance inference on NVIDIA GPUs using the C++ API. TensorRT is NVIDIA's framework for accelerating deep learning inference on NVIDIA hardware, offering significant performance gains.

## 1. Prerequisites

Before diving into the C++ API, it's essential to set up your environment and understand the core components.

*   **Hardware and Software Requirements**
    *   An **NVIDIA graphics card** with a compute capability of 5.0 or newer (Maxwell architecture or later) is required. As of TensorRT 8.6, Kepler (SM 3.x) and Maxwell (SM 5.x) devices are no longer supported, and Pascal (SM 6.x) devices were deprecated.
    *   **Ubuntu 20.04 or 22.04** is a common operating system for TensorRT development.
    *   **NVIDIA CUDA Toolkit** (version 11.0 or later, 12.0+ recommended) and **cuDNN library** (version 8 or later, but less than 9 for OpenCV GPU compatibility) are essential.
    *   **CMake** (version 3.10 or later) is needed for building the application.
    *   **OpenCV with CUDA support** (version 4.8+ recommended) is often used for image preprocessing.

*   **Dependencies and Build Environment Setup**
    *   TensorRT requires linking against three main libraries: `libnvinfer`, `libnvonnxparser`, and `libnvparser`.
    *   Setting up the CMake file to correctly locate and link these dependencies can be challenging, as TensorRT does not provide its own `find_package` script. You will need to explicitly specify these libraries along with CUDA and OpenCV.

*   **Essential TensorRT Libraries**
    The C++ API is primarily accessed through the `NvInfer.h` header and resides in the `nvinfer1` namespace. Interface classes in the API typically begin with an `I` prefix, such as `ILogger` and `IBuilder`.

## 2. Understanding TensorRT Speedup Mechanisms

TensorRT achieves high performance through various graph and precision optimizations, which can result in 2-6 times faster inference compared to standard frameworks.

*   **Graph Optimizations (Layer Fusion, etc.)**
    TensorRT performs graph optimizations during the build phase to create an optimized engine. This includes techniques like:
    *   **Layer Fusion:** Combining multiple layers (e.g., convolution, bias, activation) into a single kernel to reduce memory bandwidth and kernel launch overhead.
    *   **Tensor Memory Optimization:** Allocating and reusing GPU memory for tensors more efficiently.
    *   **Kernel Auto-tuning:** Selecting the fastest algorithm (tactics) for each layer based on the specific GPU and input data.

*   **Precision Optimizations (FP16, INT8)**
    TensorRT supports different precision modes to further accelerate inference:
    *   **FP16 (Half Precision):** Using 16-bit floating-point numbers instead of 32-bit (FP32) can significantly speed up inference and reduce memory footprint with minimal accuracy loss.
    *   **INT8 (8-bit Integer Precision):** Quantizing weights and activations to 8-bit integers offers the highest performance but requires **calibration data** to mitigate accuracy reduction.

### Performance Impact Analysis

The relative impact of graph optimizations versus precision optimizations varies by model architecture and hardware:

* **Graph Optimization Impact**
    - According to NVIDIA's study on ResNet-50, graph optimizations alone provide 1.2x-1.5x speedup
    - Layer fusion reduces memory bandwidth by 35-40% and kernel launch overhead by 45% [NVIDIA Developer Blog]
    - For transformer models like BERT, graph optimizations provide 1.8x speedup by fusing attention layers [NVIDIA Deep Learning Examples]

* **Precision Optimization Impact**
    - FP16 typically provides 2-3x speedup over FP32
    - INT8 can achieve 3-4x speedup over FP32
    - Combined FP16/INT8 mixed precision can reach up to 6x speedup [TensorRT Documentation]

Research papers show precision optimization generally contributes more to speedup:

1. **Hardware Efficiency:**
    - FP16 operations use half the memory bandwidth
    - Modern GPUs have dedicated Tensor Cores optimized for FP16/INT8
    - NVIDIA A100 can process 8x more FP16 operations per cycle compared to FP32
    - Study: "Mixed Precision Training" (Micikevicius et al., 2017) shows 2x memory reduction and 2-3x throughput increase

2. **Memory Bandwidth Impact:**
    - Memory bandwidth often bottlenecks inference
    - FP16 reduces memory traffic by 50%
    - INT8 reduces it by 75%
    - Paper: "In-Datacenter Performance Analysis of a Tensor Processing Unit" (Google, 2017) shows memory bandwidth dominates inference time

3. **Energy Efficiency:**
    - INT8 operations consume 4x less energy than FP32
    - Paper: "Energy Efficiency of Neural Networks" (Han et al., 2016) demonstrates 4-5x energy savings with quantization

This analysis suggests precision optimization typically contributes 60-70% of the total speedup, while graph optimizations contribute 30-40%, though exact ratios depend on the specific model architecture and hardware.

### Understanding Precision Optimization in Detail

Precision optimization changes how numbers are stored and computed in the GPU. To understand this:

1. **How Numbers are Stored**
   * **FP32 (32-bit):** Like storing a number with 7 decimal places (3.1415927)
     - Uses 32 bits: 1 bit for sign, 8 bits for exponent, 23 bits for decimal part
     - Can represent numbers from ±1.18 x 10⁻³⁸ to ±3.4 x 10³⁸
     - Takes more memory and processing power
   
   * **FP16 (16-bit):** Like storing a number with 3 decimal places (3.142)
     - Uses 16 bits: 1 bit for sign, 5 bits for exponent, 10 bits for decimal part
     - Can represent numbers from ±6.10 x 10⁻⁵ to ±6.5 x 10⁴
     - Takes half the memory and processing power
   
   * **INT8 (8-bit):** Like storing whole numbers from -128 to 127
     - Uses 8 bits: 1 bit for sign, 7 bits for the number
     - Requires "scaling" to represent decimals (like multiplying everything by 100)

2. **Why Reduced Precision is Faster**
   * **Memory Benefits:**
     - Imagine moving boxes (numbers) between storage (memory) and workspace (GPU)
     - With FP32, you need 3 trips to move 3 numbers
     - With FP16, you can move 6 numbers in same 3 trips
     - With INT8, you can move 12 numbers in same 3 trips
   
   * **Processing Benefits:**
     - Think of GPU as a factory with multiple workers (cores)
     - Each worker can process either:
       * 1 FP32 calculation at a time
       * 2 FP16 calculations at a time
       * 4 INT8 calculations at a time
     - More calculations done simultaneously = faster processing

3. **Real-world Analogy**
   * Imagine calculating the average height of people:
     - FP32: Measuring to the nearest 0.1mm (32.7643 cm)
     - FP16: Measuring to the nearest mm (32.8 cm)
     - INT8: Measuring to the nearest cm (33 cm)
   * For most applications, the reduced precision is still accurate enough

4. **Why it Works for Deep Learning**
   * Neural networks are naturally tolerant to some imprecision
   * Like human brain doesn't need exact precise numbers
   * Example: Recognizing a cat works whether whiskers are 5.123456 cm or 5.12 cm long

5. **Practical Implementation**
   * **FP16:**
     - Simply converts FP32 numbers to FP16
     - Usually loses very little accuracy
     - Modern GPUs have special hardware (Tensor Cores) for FP16
   
   * **INT8:**
     - More complex, requires "calibration"
     - Like adjusting a scale before weighing:
       1. Look at range of values in each layer
       2. Find best way to map FP32 numbers to INT8 range (-128 to 127)
       3. Store these mapping factors ("scales")
       4. During inference: 
          * Convert input → INT8
          * Process in INT8
          * Convert output back to FP32

6. **When to Use Each Precision**
   * **FP32:** When absolute accuracy is required (medical, financial)
   * **FP16:** Most common choice, good balance of speed and accuracy
   * **INT8:** When maximum speed is needed and some accuracy loss is acceptable

7. **Common Misconceptions**
   * Lower precision doesn't always mean worse results
   * Modern networks are often trained with noise/dropout
   * This natural robustness means they handle reduced precision well

## 3. Model Conversion Requirements: ONNX and Operator Support

TensorRT's primary method for importing a trained model from a deep learning framework is the **ONNX (Open Neural Network Exchange) interchange format**.

*   **ONNX as the Primary Interchange Format**
    *   Models trained in frameworks like PyTorch or TensorFlow are typically exported to ONNX format first.
    *   TensorRT ships with an ONNX parser library (`nvonnxparser`) to assist in importing these models.

*   **Operator Support Matrix and Compatibility**
    *   The ONNX parser aims for backward compatibility (up to opset 9). It is recommended to export models to the **latest available ONNX opset** for TensorRT deployment.
    *   It's crucial to check the **ONNX-TensorRT operator support matrix** to ensure all operators in your ONNX model are supported by your TensorRT version.
    *   If an ONNX model contains unsupported operators or has compatibility issues, the parser might fail.

*   **Troubleshooting Conversion Issues**
    *   **Constant Folding:** Running constant folding using Polygraphy on the ONNX model before parsing can often resolve TensorRT conversion issues.
    *   **Model Modification:** In some cases, you might need to modify the ONNX model, for example, by replacing subgraphs with custom plugins or reimplementing unsupported operations with supported ones. Tools like ONNX-GraphSurgeon can assist with this.

## 4. Step-by-Step Process: From ONNX Model to TensorRT Engine Deployment

The TensorRT workflow involves distinct "offline" (build/optimization) and "online" (inference) phases.

### Phase 1: The Build/Optimization Phase (Offline)

This phase converts your ONNX model into an optimized TensorRT engine file, typically an `.plan` file. This process is performed once and can be time-consuming.

1.  **Instantiate a Logger**
    TensorRT's builder and runtime require an `ILogger` instance to capture errors, warnings, and other information.
    ```cpp
    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            // Suppress info-level messages, print warnings and errors
            if (severity <= Severity::kWARNING) {
                std::cout << msg << std::endl;
            }
        }
    } logger;
    ```
    *Using smart pointers for TensorRT interfaces is recommended for proper object lifetime management*.

2.  **Create Builder and Network Definition**
    *   Instantiate an `IBuilder` object using the logger.
    *   Create an `INetworkDefinition` instance using the builder, optionally with `NetworkDefinitionCreationFlag::kSTRONGLY_TYPED`.
    ```cpp
    // Using smart pointers
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0 /* flags */));
    // For non-strongly typed networks, 0 is often used for flags.
    // Use NetworkDefinitionCreationFlag::kSTRONGLY_TYPED for strongly typed.
    ```

3.  **Import ONNX Model**
    *   Create an `nvonnxparser::IParser` instance, passing the network definition and logger.
    *   Use `parser->parseFromFile()` to populate the network definition from your ONNX model file.
    *   **Crucially, the parser owns the memory for the model weights, so it should not be deleted until after the builder has run and the engine is built**.
    ```cpp
    // Using smart pointers
    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));

    if (!parser->parseFromFile(modelFile.c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "ERROR: could not parse the model." << std::endl;
        // Handle errors from parser->getNbErrors() and parser->getError(i)->desc()
    }
    ```
    *(Advanced: You can also build the network definition from scratch by adding individual layers like Convolution, Pooling, Activation, etc., and providing weights directly in host memory).*

4.  **Mark Network Outputs**
    After the network is defined (either from ONNX or scratch), you must explicitly name and **mark the final output tensor(s)** of the network using `network->markOutput()`. This allows binding them to memory buffers during inference.
    ```cpp
    // Assuming 'prob' is the output layer from a SoftMax operation
    prob->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*prob->getOutput(0));
    ```

5.  **Configure Build Parameters and Optimization Profiles**
    *   Create an `IBuilderConfig` instance.
    *   Set **memory pool limits** for workspace (temporary layer memory) and CUDA backend shared memory using `config->setMemoryPoolLimit()`.
    *   Enable **FP16** precision if supported by your platform: `config->setFlag(nvinfer1::BuilderFlag::kFP16)`.
    *   For models with implicit batching, set the **maximum batch size** using `builder->setMaxBatchSize()`.
    *   For models with **dynamic input shapes**, define one or more `IOptimizationProfile` instances and add them to the configuration. These profiles specify the minimum, optimum, and maximum dimensions (including batch size) that the engine should be optimized for.
    ```cpp
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1GB
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kTACITC_SHARED_MEMORY, 48 << 10); // 48KB

    if (builder->platformHasFastFp16()) { // Check for FP16 support
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    builder->setMaxBatchSize(1); // For implicit batching

    // For dynamic input shapes, define and add optimization profiles
    // auto profile = builder->createOptimizationProfile();
    // profile->setDimensions(input_name, OptProfileSelector::kMIN, min_dims);
    // ... then config->addOptimizationProfile(profile);
    ```

6.  **Build and Serialize Engine**
    *   Call `builder->buildSerializedNetwork(*network, *config)` to generate an `IHostMemory` object, which contains the optimized, serialized engine. This can take several minutes.
    ```cpp
    IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
    ```

7.  **Save Engine to Disk (Optional but Recommended)**
    Saving the serialized engine to a `.plan` file allows for much faster loading in subsequent runs, avoiding the lengthy build process.
    *   It is good practice to incorporate configuration options and the GPU's UUID into the filename to ensure compatibility and versioning, as engines are **not portable across different GPU models, platforms, or TensorRT versions**.
    ```cpp
    // Example: Write serializedModel->data() to a file
    // std::ofstream file("model.plan", std::ios::binary);
    // file.write(static_cast<const char*>(serializedModel->data()), serializedModel->size());
    // file.close();
    ```

8.  **Clean Up Build-Time Objects**
    After the engine is built and optionally saved, the `parser`, `network`, `config`, and `builder` objects are no longer needed and can be deleted.
    ```cpp
    delete parser; // If not using smart pointers
    delete network;
    delete config;
    delete builder;
    delete serializedModel;
    ```

### Phase 2: The Inference Phase (Online)

This phase involves loading a pre-built engine and executing inference efficiently.

1.  **Deserialize the Engine**
    *   Create an `IRuntime` object, passing the logger.
    *   Load the engine from a file or in-memory buffer using one of the following methods:
        *   **In-memory Deserialization:** Read the entire `.plan` file into a `char` vector and use `runtime->deserializeCudaEngine(modelData.data(), modelData.size())`. Suitable for smaller models.
        *   **IStreamReaderV2 Deserialization:** Implement a custom `IStreamReaderV2` to read the engine in chunks. This is beneficial for **large models** to reduce peak CPU memory usage and load time, potentially bypassing CPU memory.
    ```cpp
    IRuntime* runtime = createInferRuntime(logger);
    // Example: In-memory deserialization
    std::vector<char> modelData = readModelFromFile("model.plan"); // Custom function
    ICudaEngine* engine = runtime->deserializeCudaEngine(modelData.data(), modelData.size());
    // Example: IStreamReaderV2 (requires custom implementation)
    // MyStreamReaderV2 readerV2("model.plan");
    // ICudaEngine* engine = runtime->deserializeCudaEngine(readerV2);
    ```

2.  **Create an Execution Context**
    From the loaded `ICudaEngine`, create an `IExecutionContext`. This object holds intermediate activations and allows an engine to be used for multiple overlapping inference tasks.
    ```cpp
    IExecutionContext* context = engine->createExecutionContext();
    ```

3.  **Prepare Input Data (Preprocessing & Memory Allocation)**
    *   **Preprocessing:** Replicate the exact preprocessing steps used during model training (e.g., resize, normalize, format conversion like HWC to CHW). Libraries like OpenCV can be used for this in C++.
    *   **Allocate GPU Buffers:** Allocate device memory on the GPU for both input and output tensors using `cudaMalloc()`.
    *   **Set Input Shapes (for Dynamic Shapes):** If the engine was built with dynamic input dimensions (including dynamic batch size), you **must explicitly set the input dimensions** using `context->setInputShape()` before inference.
    *   **Copy Data to GPU:** Transfer the preprocessed input data from host (CPU) memory to device (GPU) memory using `cudaMemcpy(..., cudaMemcpyHostToDevice)`.
    ```cpp
    // Example: Allocate buffers for input and output
    std::vector<void*> buffers(engine->getNbBindings());
    // Get input/output dimensions from engine->getBindingDimensions(i)
    // cudaMalloc(&buffers[input_idx], input_size_bytes);
    // cudaMalloc(&buffers[output_idx], output_size_bytes);

    // Set input tensor address and shape
    context->setTensorAddress(INPUT_NAME, inputBuffer);
    // If dynamic shapes:
    // context->setInputShape(INPUT_NAME, inputDims);

    // Copy input data to GPU
    // cudaMemcpy(buffers[input_idx], host_input_data.data(), input_size_bytes, cudaMemcpyHostToDevice);
    ```

4.  **Execute Inference**
    Call `context->enqueueV3()` with a CUDA stream to start **asynchronous inference**. If using synchronous inference, `executeV2()` can be used. Asynchronous execution allows overlap of data transfers with computation.
    ```cpp
    context->enqueueV3(stream); // For asynchronous inference
    // context->executeV2(buffers.data()); // For synchronous inference
    ```

5.  **Retrieve and Post-process Output Data**
    *   **Synchronize:** If using asynchronous inference, use standard CUDA synchronization mechanisms (e.g., events or stream synchronization) to ensure computations are complete before accessing results.
    *   **Copy Data to CPU:** Transfer the inference results from GPU device memory back to CPU host memory using `cudaMemcpy(..., cudaMemcpyDeviceToHost)`.
    *   **Postprocessing:** Process the raw output (e.g., apply softmax, interpret classification results, or parse object detection bounding boxes). For very large outputs, post-processing can be done directly on the GPU using CUDA kernels for efficiency.
    ```cpp
    // cudaStreamSynchronize(stream); // If using async inference
    // cudaMemcpy(host_output_data.data(), buffers[output_idx], output_size_bytes, cudaMemcpyDeviceToHost);
    // PostProcessResults(host_output_data); // Custom function
    ```

6.  **Clean Up Inference-Time Objects**
    Free the dynamically allocated GPU memory for input and output buffers using `cudaFree()`. Also, delete the `context`, `engine`, and `runtime` objects when no longer needed.
    ```cpp
    for (void* buf : buffers) {
        cudaFree(buf);
    }
    // delete context;
    // delete engine;
    // delete runtime;
    ```

## 5. Output Types and Precision Settings (FP32, FP16, INT8)

TensorRT supports different precision modes for inference, allowing trade-offs between speed and accuracy.

*   **FP32 (Full Precision)**
    This is the default precision mode. It uses 32-bit floating-point numbers, offering the highest accuracy but potentially slower inference compared to reduced precision modes. The engine is built without any specific precision flags or calibrators for FP32.

*   **FP16 (Half Precision)**
    *   FP16 uses 16-bit floating-point numbers. It typically offers a good balance between speed and accuracy, often resulting in significant speedups.
    *   To enable FP16, set the `BuilderFlag::kFP16` in the `IBuilderConfig` during engine creation: `config->setFlag(nvinfer1::BuilderFlag::kFP16)`.
    *   The `Engine` class in a sample integration has been modified to support output types of `float` (FP32), `__half` (FP16), `int8_t`, `int32_t`, `bool`, and `uint8_t`.

*   **INT8 (8-bit Integer Precision)**
    *   INT8 precision uses 8-bit integers, providing the fastest inference speeds and lowest memory usage. However, it generally incurs some accuracy loss due to reduced dynamic range.
    *   To enable INT8 inference, you must provide **calibration data** that is representative of the real-world data the model will encounter.
    *   This typically requires a large dataset (e.g., 1K+ images for calibration).
    *   During calibration, TensorRT determines the optimal scaling factors to map FP32 weights and activations to INT8.
    *   Ensure that the preprocessing code for the calibration data precisely matches the inference preprocessing.
    *   "Out of memory" errors can occur during calibration if the `calibrationBatchSize` is too high for the GPU's memory.
    *   The calibration cache (`.calibration` extension) is written to disk and can be reused for subsequent model optimizations. If you want to regenerate calibration data, you must delete this cache file.

## 6. Simple C++ Example Code Snippets

These snippets illustrate key parts of the TensorRT C++ API based on the provided sources.

### Logger Class

```cpp
#include "NvInfer.h"
#include <iostream>
#include <memory> // For std::unique_ptr

// Custom logger to capture TensorRT messages
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Only log errors and internal errors for brevity in this example
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
            std::cerr << "TRT_ERROR: " << msg << std::endl;
        }
        // Uncomment below for more verbose logging, e.g., warnings
        // if (severity <= Severity::kWARNING) {
        //     std::cout << "TRT_LOG: " << msg << std::endl;
        // }
    }
} gLogger;

// Helper to destroy TensorRT objects using unique_ptr
struct TRTDestroy {
    template<class T>
    void operator()(T* obj) const {
        if (obj) {
            obj->destroy();
        }
    }
};
template<class T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

// Helper to calculate size from dimensions
size_t getSizeByDim(const nvinfer1::Dims& dims) {
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        size *= dims.d[i];
    }
    return size;
}
```

### Parsing ONNX Model and Building Engine

```cpp
#include "NvOnnxParser.h" // For nvonnxparser::IParser

TRTUniquePtr<nvinfer1::ICudaEngine> engine{nullptr};
TRTUniquePtr<nvinfer1::IExecutionContext> context{nullptr};

void parseOnnxModel(const std::string& model_path) {
    TRTUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(gLogger)};
    // Use createNetworkV2 for explicit batching, or createNetwork for implicit batching (old API)
    TRTUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(0 /* flags */)}; 
    TRTUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, gLogger)};

    if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "ERROR: could not parse the ONNX model: " << model_path << std::endl;
        return;
    }

    TRTUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1GB workspace

    // Enable FP16 precision if platform supports it
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // For implicit batching, set max batch size
    // builder->setMaxBatchSize(1); 

    // For dynamic shapes, add optimization profiles
    // nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    // profile->setDimensions("input_name", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, 224, 224});
    // profile->setDimensions("input_name", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, 3, 224, 224});
    // profile->setDimensions("input_name", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{4, 3, 224, 224});
    // config->addOptimizationProfile(profile);

    TRTUniquePtr<nvinfer1::IHostMemory> serializedModel{builder->buildSerializedNetwork(*network, *config)};
    if (!serializedModel) {
        std::cerr << "ERROR: Failed to build serialized network." << std::endl;
        return;
    }

    // Optionally save the serialized engine to disk for later use
    // std::ofstream planFile("engine.plan", std::ios::binary);
    // planFile.write(static_cast<const char*>(serializedModel->data()), serializedModel->size());
    // planFile.close();

    TRTUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger)};
    engine.reset(runtime->deserializeCudaEngine(serializedModel->data(), serializedModel->size()));
    if (!engine) {
        std::cerr << "ERROR: Failed to deserialize CUDA engine." << std::endl;
        return;
    }

    context.reset(engine->createExecutionContext());
    if (!context) {
        std::cerr << "ERROR: Failed to create execution context." << std::endl;
        return;
    }
    std::cout << "TensorRT engine built and context created successfully." << std::endl;
}
```

### Preprocessing Example (using OpenCV CUDA, simplified)

This example assumes `dims` is the expected input `Dims` from the engine, e.g., `Dims3{3, 224, 224}`.

```cpp
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp> // For GPU operations
#include <vector>

void preprocessImage(const std::string& image_path, float* gpu_input, const nvinfer1::Dims& dims) {
    cv::Mat frame = cv::imread(image_path);
    if (frame.empty()) {
        std::cerr << "Input image " << image_path << " load failed\n";
        return;
    }

    cv::cuda::GpuMat gpu_frame;
    gpu_frame.upload(frame); // Upload to GPU

    int input_height = dims.d; // Assuming NCHW, H is dims.d or dims.d depending on N dims
    int input_width = dims.d;  // Assuming NCHW, W is dims.d or dims.d
    int channels = dims.d;     // Assuming NCHW, C is dims.d or dims.d

    cv::Size input_size(input_width, input_height);

    cv::cuda::GpuMat resized;
    cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_LINEAR); // Resize

    cv::cuda::GpuMat flt_image;
    resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f); // Normalize to 0-1

    // Mean and Std Dev for ImageNet, example
    cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
    cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);

    // Convert HWC to NCHW format on GPU
    std::vector<cv::cuda::GpuMat> chw(channels);
    for (int i = 0; i < channels; ++i) {
        chw[i] = cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + i * input_width * input_height);
    }
    cv::cuda::split(flt_image, chw); // Split into channels and copy to contiguous GPU memory
}
```

### Inference Execution

```cpp
#include <cuda_runtime.h> // For cudaMalloc, cudaMemcpy, cudaFree

void runInference(const std::string& image_path, int batch_size) {
    if (!engine || !context) {
        std::cerr << "Engine or context not initialized." << std::endl;
        return;
    }

    std::vector<nvinfer1::Dims> input_dims;
    std::vector<nvinfer1::Dims> output_dims;
    std::vector<void*> buffers(engine->getNbBindings());

    for (int i = 0; i < engine->getNbBindings(); ++i) {
        nvinfer1::Dims current_dims = engine->getBindingDimensions(i);
        size_t binding_size = getSizeByDim(current_dims) * batch_size * sizeof(float); // Assuming float output type
        cudaMalloc(&buffers[i], binding_size); // Allocate GPU memory

        if (engine->bindingIsInput(i)) {
            input_dims.emplace_back(current_dims);
            // If using dynamic batching, set input shape here
            // context->setInputShape(engine->getBindingName(i), nvinfer1::Dims4{batch_size, ...});
        } else {
            output_dims.emplace_back(current_dims);
        }
    }

    if (input_dims.empty() || output_dims.empty()) {
        std::cerr << "Expected at least one input and one output for network." << std::endl;
        return;
    }

    // Preprocess image and copy to GPU input buffer
    preprocessImage(image_path, (float*)buffers, input_dims);

    // Perform inference
    context->enqueue(batch_size, buffers.data(), 0, nullptr); // Stream 0, no event

    // Post-process results
    // Assumes postprocessResults is defined
    // postprocessResults((float*)buffers, output_dims, batch_size); 

    // Free GPU memory
    for (void* buf : buffers) {
        cudaFree(buf);
    }
}
```

### Postprocessing Example (simplified)

```cpp
#include <numeric> // For std::iota, std::accumulate
#include <algorithm> // For std::transform, std::sort
#include <cmath> // For std::exp
#include <fstream> // For std::ifstream

// Helper to get class names from a file
std::vector<std::string> getClassNames(const std::string& filepath) {
    std::vector<std::string> classes;
    std::ifstream file(filepath);
    std::string line;
    while (std::getline(file, line)) {
        classes.push_back(line);
    }
    return classes;
}

void postprocessResults(float* gpu_output, const nvinfer1::Dims& dims, int batch_size) {
    auto classes = getClassNames("imagenet_classes.txt"); // Assuming ImageNet classes

    // Copy results from GPU to CPU
    std::vector<float> cpu_output(getSizeByDim(dims) * batch_size);
    cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Assuming a single output and batch_size 1 for simplicity
    // Calculate softmax (if not already done by the model)
    std::transform(cpu_output.begin(), cpu_output.end(), cpu_output.begin(), [](float val) { return std::exp(val); });
    float sum = std::accumulate(cpu_output.begin(), cpu_output.end(), 0.0f);

    std::vector<int> indices(cpu_output.size());
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ...

    // Sort indices by confidence in descending order
    std::sort(indices.begin(), indices.end(), [&cpu_output](int i1, int i2) {
        return cpu_output[i1] > cpu_output[i2];
    });

    // Print top predicted classes
    std::cout << "Top Predictions:" << std::endl;
    for (int i = 0; i < std::min((int)indices.size(), 5); ++i) { // Print top 5
        float confidence = 100 * cpu_output[indices[i]] / sum;
        if (confidence > 0.5f) { // Example threshold
            std::cout << "  Class: " << classes[indices[i]] 
                      << " | Confidence: " << confidence << "% | Index: " << indices[i] << std::endl;
        }
    }
}
```


- Builder
    - create network
        - can provide flags when creating
            - batch size
            - GPU memory to be used for conversion
            - fp16 support
        - default flags - implicit batch dimension
    - parser
        - parse network
    - generate engine, context
        - engine is optimized to hardware, software
        - initialization takes lot of time
        - save engine in serialized format for reuse
- inference pipeline
    - create context, engine
    - find dimensions of inputs, outputs, allocate memory
        - Need to specify precision
        - trt.volume (size in bytes, uses get_binding_shape) for inputs.
        - trt.pagelocked_empty(volume in bytes) - wont be swapped to disk
    - Creating CUDA streams
        - CUDA functions can be executed in streams. All commands within stream executed sequentially, but streams can run out of order
        - Default stream = nullstream
    - Preprocessing
        - With python APIs, preprocessing can be reused, host_input -> device_input (CUDA). Need to place data in contigous memory, use page locked memory as much as possible
        - Copy data from host to device
        - Execute inference
        - copy result from device to host
- Results
    - Compare avg inference time b/w pytorch and tensorrt execution
    - CUDA initializes and caches some data. So, first call is always slower. Use avg time across multiple cycles
    - 4-6 times using FP16 mode, 2-3 times using FP32 mode


- Tensorrt C++ API
    - pre-requisite
        - libnvinfer, libnvonnxparser, libnvparser
        - CMake
        - NVIDIA CUDA, CUDNN, 
        - model in onnx format
    - pre-processing & post-processing
        - Allocate memory outside, resize, normalize, toTensor
        - 
    - TRTLogger
        - inherits from nvinfer1::ILogger
    - TRTDestroy
        - Custom destructor for templated unique_ptr implementation
    - Process
        - parse onnx model
        - ```cpp
            void parseOnnxModel(const std::string& model_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine,
                                TRTUniquePtr< nvinfer1::IExecutionContext >& context)
            {
                TRTUniquePtr< nvinfer1::IBuilder > builder{nvinfer1::createInferBuilder(gLogger)};
                TRTUniquePtr< nvinfer1::INetworkDefinition > network{builder->createNetwork()};
                TRTUniquePtr< nvonnxparser::IParser > parser{nvonnxparser::createParser(*network, gLogger)};
                // parse ONNX
                if (!parser->parseFromFile(model_path.c_str(), static_cast< int >(nvinfer1::ILogger::Severity::kINFO)))
                {
                    std::cerr << "ERROR: could not parse the model.\n";
                    return;
                }
            }

            // create context and generate engine
            engine.reset(builder->buildEngineWithConfig(*network, *config));
            context.reset(engine->createExecutionContext());
        ```
        - For each input, output, need to create buffers. Also, need to create names for binded outputs
        - Preprocess input
        - Do inference
        - Post process output
        - free buffers

- Torch-Tensorrt for pytorch support, deployment to python runtime
- ONNX to tensorrt conversion is all or nothing 
- Nsight for converting onnx model to tensorrt engine using GUI
- Make sure all operators are supported by ONNX and Tensorrt (else, might need to implement custom Tensorrt plugin)


## References
- [Tensorrt CPP API](https://learnopencv.com/how-to-run-inference-using-tensorrt-c-api/)
- [Tensorrt Python API](https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/)
- [NVIDIA Tensorrt official documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/overview.html#installing-pycuda)
- [Tensorrt CPP Github implementation](https://github.com/cyrusbehr/tensorrt-cpp-api)
- [Veoctorization MS blog post](https://learn.microsoft.com/en-gb/archive/blogs/nativeconcurrency/what-is-vectorization)
- [Vectorization Stackoverflow post](https://stackoverflow.com/questions/1422149/what-is-vectorization)
- [Model quantization speeds up inference](https://www.reddit.com/r/learnmachinelearning/comments/zgzh6r/whyhow_does_model_quantization_speed_up_inference/)
- [ONNX version spec](https://github.com/onnx/onnx/tree/main?tab=readme-ov-file)
- [Tensorrt custom plugin](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html#extending-custom-layers)
- [ONNX tensorrt supported operators](https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md)
- [Tensorrt plugin](https://github.com/NVIDIA/TensorRT/tree/main/plugin)


### GPU programming resources
- https://www.coursera.org/specializations/gpu-programming/
- https://nvidia.github.io/cccl/cub/
- https://github.com/NVIDIA/cutlass
- https://developer.nvidia.com/blog/even-easier-introduction-cuda/
- https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-AC-04+V2
- https://www.reddit.com/r/MachineLearning/comments/w52iev/d_what_are_some_good_resources_to_learn_cuda/
- https://www.youtube.com/playlist?list=PLPJwWVtf19Wgx_bupSDDSStSv-tOGGWRO
- https://www.olcf.ornl.gov/cuda-training-series/
- https://www.udemy.com/course/mastering-gpu-parallel-programming-with-cuda/
