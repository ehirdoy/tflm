/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <stdio.h>

#ifndef ESP_PLATFORM
#include <unistd.h>
#endif

#include "tensorflow/lite/experimental/micro/examples/hello_world/constants.h"
#include "tensorflow/lite/experimental/micro/examples/hello_world/output_handler.h"
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

extern const unsigned char mnist_model_tflite[];

static void dump_tensor(TfLiteTensor *input)
{
    printf("%s: TfLiteType=%s %zu\n",
           input->name, TfLiteTypeGetName(input->type), input->bytes);
    printf("dim[%d]: %d %d %d\n", input->dims->size,
           input->dims->data[0], input->dims->data[1], input->dims->data[2]);
}

int _main(int argc, char* argv[]) {
  // Set up logging
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = ::tflite::GetModel(mnist_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
    return 1;
  }

  // This pulls in all the operation implementations we need
  tflite::ops::micro::AllOpsResolver resolver;

  // Create an area of memory to use for input, output, and intermediate arrays.
  // Finding the minimum value for your model may require some trial and error.
  const int tensor_arena_size = 6 * 1024;
  uint8_t tensor_arena[tensor_arena_size];

  // Build an interpreter to run the model with
  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       tensor_arena_size, error_reporter);

  // Allocate memory from the tensor_arena for the model's tensors
  interpreter.AllocateTensors();

  // Obtain pointers to the model's input and output tensors
  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  dump_tensor(input);
  dump_tensor(output);

#ifndef ESP_PLATFORM
  unsigned char input_data[28 * 28];
  size_t n = read(0, input_data, sizeof(input_data));
  assert(n == (28 * 28));
#endif

  for (int i = 0; i < (28 * 28); i++) {
#ifdef ESP_PLATFORM
    float val = (unsigned char)argv[0][i] / 255.0;
#else
    float val = input_data[i];
#endif
    input->data.f[i] = val;
    printf("%s%s", (val ? "." : " "), ((i % 28) == 27 ? "\n" : ""));
  }

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed\n");
    return 0;
  }
  for (int i = 0; i < 10; i++)
    printf("%d: %f\n", i, output->data.f[i]);
  return 0;
}

#ifdef ESP_PLATFORM
extern "C" void tflm(unsigned char *d) { _main(0, (char **)&d);}
#else
int main(int argc, char* argv[]) {_main(0, NULL);}
#endif
