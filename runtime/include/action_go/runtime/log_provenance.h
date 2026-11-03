#pragma once

#include <openssl/evp.h>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>

namespace action_go::runtime {

inline std::string log_file_sha256(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return {};
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) return {};
  char buffer[65536];
  while (input.read(buffer, sizeof(buffer)) || input.gcount()) {
    if (EVP_DigestUpdate(context.get(), buffer, input.gcount()) != 1) return {};
  }
  if (!input.eof()) return {};
  unsigned char digest[EVP_MAX_MD_SIZE];
  unsigned int length = 0;
  if (EVP_DigestFinal_ex(context.get(), digest, &length) != 1) return {};
  std::ostringstream result;
  for (unsigned int index = 0; index < length; ++index) {
    result << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned int>(digest[index]);
  }
  return result.str();
}

}
