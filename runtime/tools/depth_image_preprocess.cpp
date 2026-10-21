#include "action_go/runtime/depth_image_preprocessor.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  try {
    const bool pbfm = argc == 5 && std::string(argv[1]) == "--profile" && std::string(argv[2]) == "pbfm";
    if (argc != 3 && !pbfm) throw std::runtime_error("usage: depth_image_preprocess [--profile pbfm] raw.f32 output.f32");
    const int offset = pbfm ? 2 : 0;
    std::vector<float> raw(pbfm ? 36 * 64 : 60 * 106);
    std::ifstream input(argv[1 + offset], std::ios::binary);
    if (!input.read(reinterpret_cast<char*>(raw.data()), raw.size() * sizeof(float)))
      throw std::runtime_error("cannot read depth input");
    const auto processed = pbfm ? action_go::runtime::preprocess_pbfm_depth(raw) :
                                  action_go::runtime::preprocess_php_depth(raw);
    std::ofstream output(argv[2 + offset], std::ios::binary);
    if (!output.write(reinterpret_cast<const char*>(processed.data()), processed.size() * sizeof(float)))
      throw std::runtime_error("cannot write depth output");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
