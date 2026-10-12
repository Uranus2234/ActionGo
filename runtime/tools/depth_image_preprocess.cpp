#include "action_go/runtime/depth_image_preprocessor.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: depth_image_preprocess raw_60x106.f32 output_58x87.f32");
    std::vector<float> raw(60 * 106);
    std::ifstream input(argv[1], std::ios::binary);
    if (!input.read(reinterpret_cast<char*>(raw.data()), raw.size() * sizeof(float)))
      throw std::runtime_error("cannot read depth input");
    const auto processed = action_go::runtime::preprocess_php_depth(raw);
    std::ofstream output(argv[2], std::ios::binary);
    if (!output.write(reinterpret_cast<const char*>(processed.data()), processed.size() * sizeof(float)))
      throw std::runtime_error("cannot write depth output");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
