#pragma once

#include <cerrno>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace action_go::runtime {

inline bool signal_file_exists(const std::string& path) {
  struct stat info {};
  return !path.empty() && lstat(path.c_str(), &info) == 0 &&
         S_ISREG(info.st_mode) && info.st_uid == geteuid() && info.st_nlink == 1;
}

inline void write_signal_file(const std::string& path, const std::string& contents,
                              bool replace_existing = false) {
  if (path.empty()) return;
  const auto file = std::filesystem::absolute(path).lexically_normal();
  int directory = open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  for (const auto& component : file.parent_path().relative_path()) {
    if (directory < 0) break;
    const int next = openat(directory, component.c_str(),
                            O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    close(directory);
    directory = next;
  }
  struct stat info {};
  if (directory < 0 || fstat(directory, &info) != 0 ||
      info.st_uid != geteuid() || (info.st_mode & 0022) != 0) {
    if (directory >= 0) close(directory);
    throw std::runtime_error("signal file requires an owned, non-writable directory: " + path);
  }
  if (replace_existing) {
    const int existing = fstatat(directory, file.filename().c_str(), &info, AT_SYMLINK_NOFOLLOW);
    if ((existing == 0 && (!S_ISREG(info.st_mode) || info.st_uid != geteuid() || info.st_nlink != 1)) ||
        (existing != 0 && errno != ENOENT)) {
      close(directory);
      throw std::runtime_error("cannot safely replace signal file: " + path);
    }
  }
  std::string output_name = file.filename().string();
  int output = -1;
  static std::atomic<unsigned long long> sequence{0};
  for (int attempt = 0; attempt < 128; ++attempt) {
    if (replace_existing) {
      output_name = ".action_go_signal_" + std::to_string(getpid()) + "_" +
                    std::to_string(sequence.fetch_add(1));
    }
    output = openat(directory, output_name.c_str(),
                    O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (output >= 0 || !replace_existing || errno != EEXIST) break;
  }
  const int open_error = errno;
  if (output < 0) {
    const bool existing = !replace_existing && open_error == EEXIST &&
        fstatat(directory, file.filename().c_str(), &info, AT_SYMLINK_NOFOLLOW) == 0 &&
        S_ISREG(info.st_mode) && info.st_uid == geteuid() && info.st_nlink == 1;
    close(directory);
    if (existing) return;
    throw std::runtime_error("cannot safely create signal file: " + path);
  }
  bool ok = true;
  size_t offset = 0;
  while (offset < contents.size()) {
    const auto written = write(output, contents.data() + offset, contents.size() - offset);
    if (written < 0 && errno == EINTR) continue;
    if (written <= 0) { ok = false; break; }
    offset += static_cast<size_t>(written);
  }
  if (close(output) != 0) ok = false;
  if (ok && replace_existing &&
      renameat(directory, output_name.c_str(), directory, file.filename().c_str()) != 0) ok = false;
  if (!ok) unlinkat(directory, output_name.c_str(), 0);
  close(directory);
  if (!ok) throw std::runtime_error("failed to write signal file: " + path);
}

}  // namespace action_go::runtime
