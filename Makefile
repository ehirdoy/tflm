SRCS := \
tensorflow/lite/experimental/micro/micro_error_reporter.cc tensorflow/lite/experimental/micro/micro_mutable_op_resolver.cc tensorflow/lite/experimental/micro/debug_log.cc tensorflow/lite/experimental/micro/debug_log_numbers.cc tensorflow/lite/experimental/micro/simple_tensor_allocator.cc tensorflow/lite/experimental/micro/micro_interpreter.cc tensorflow/lite/experimental/micro/micro_allocator.cc tensorflow/lite/experimental/micro/kernels/pack.cc tensorflow/lite/experimental/micro/kernels/logical.cc tensorflow/lite/experimental/micro/kernels/elementwise.cc tensorflow/lite/experimental/micro/kernels/comparisons.cc tensorflow/lite/experimental/micro/kernels/depthwise_conv.cc tensorflow/lite/experimental/micro/kernels/split.cc tensorflow/lite/experimental/micro/kernels/conv.cc tensorflow/lite/experimental/micro/kernels/prelu.cc tensorflow/lite/experimental/micro/kernels/softmax.cc tensorflow/lite/experimental/micro/kernels/pooling.cc tensorflow/lite/experimental/micro/kernels/ceil.cc tensorflow/lite/experimental/micro/kernels/arg_min_max.cc tensorflow/lite/experimental/micro/kernels/add.cc tensorflow/lite/experimental/micro/kernels/neg.cc tensorflow/lite/experimental/micro/kernels/floor.cc tensorflow/lite/experimental/micro/kernels/unpack.cc tensorflow/lite/experimental/micro/kernels/svdf.cc tensorflow/lite/experimental/micro/kernels/all_ops_resolver.cc tensorflow/lite/experimental/micro/kernels/fully_connected.cc tensorflow/lite/experimental/micro/kernels/maximum_minimum.cc tensorflow/lite/experimental/micro/kernels/reshape.cc tensorflow/lite/experimental/micro/kernels/strided_slice.cc tensorflow/lite/experimental/micro/kernels/round.cc tensorflow/lite/c/c_api_internal.c tensorflow/lite/core/api/error_reporter.cc tensorflow/lite/core/api/flatbuffer_conversions.cc tensorflow/lite/core/api/op_resolver.cc tensorflow/lite/core/api/tensor_utils.cc tensorflow/lite/kernels/kernel_util.cc tensorflow/lite/kernels/internal/quantization_util.cc  tensorflow/lite/experimental/micro/examples/hello_world/main.cc tensorflow/lite/experimental/micro/examples/hello_world/sine_model_data.cc tensorflow/lite/experimental/micro/examples/hello_world/output_handler.cc tensorflow/lite/experimental/micro/examples/hello_world/constants.cc
SRCS += mnist_model_data.cc

OBJS := \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(SRCS)))

CXXFLAGS += -O3 -DNDEBUG -std=c++11 -g -DTF_LITE_STATIC_MEMORY -I. -I./third_party/gemmlowp -I./third_party/flatbuffers/include -I./third_party/kissfft
CCFLAGS +=  -DNDEBUG -g -DTF_LITE_STATIC_MEMORY -I. -I./third_party/gemmlowp -I./third_party/flatbuffers/include -I./third_party/kissfft

LDFLAGS +=  -lm

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

hello_world : $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

all: hello_world

tflite: train-mnist-tflite.py
	python3 train-mnist-tflite.py

mnist_model_data.cc: train-mnist-tflite.py
	xxd -i mnist_model.tflite > $@
	sed -i 's/^unsigned char/extern const unsigned char/' $@
	sed -i 's/\[\]/\[\] __attribute__((aligned(4)))/' $@
	sed -i '/^unsigned int/d' $@

cc: mnist_model_data.cc

# digit '7' bitmap 28 * 28
input_data.h:
	wget https://gist.github.com/neil-tan/0e032be578181ec0e3d9a47e1e24d011/raw/888d098683318d030b3c4f6f4b375a64e7ad0017/input_data.h
