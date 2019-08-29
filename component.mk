#
# "main" pseudo-component makefile.
#
# (Uses default behaviour of compiling all source files in directory, adding 'include' to include path.)

CXXFLAGS += -O3 -DNDEBUG -std=c++11 -g -DTF_LITE_STATIC_MEMORY
CCFLAGS +=  -DNDEBUG -g -DTF_LITE_STATIC_MEMORY
COMPONENT_ADD_LDFLAGS += -lm

COMPONENT_ADD_INCLUDEDIRS := . \
	third_party/gemmlowp \
	third_party/flatbuffers/include \
	third_party/kissfft

COMPONENT_SRCDIRS := . \
	tensorflow/lite/c \
	tensorflow/lite/core/api \
	tensorflow/lite/experimental/micro \
	tensorflow/lite/experimental/micro/kernels \
	tensorflow/lite/kernels \
	tensorflow/lite/kernels/internal \
	tensorflow/lite/experimental/micro/examples/hello_world

COMPONENT_SUBMODULES := tflm

