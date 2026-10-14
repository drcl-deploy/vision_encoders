#pragma once

// Socket operations for the length-prefixed camera stream. The reader thread
// owns and closes the fd; shutdown only clears running, checked every 100 ms.
// A connection or complete frame gets two seconds, including partial reads.

#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace vision_encoders
{
namespace camera_tcp
{

using Clock = std::chrono::steady_clock;
inline constexpr auto TIMEOUT = std::chrono::seconds(2);
inline constexpr auto POLL_INTERVAL = std::chrono::milliseconds(100);

inline bool wait_ready(
  int fd, short events, const std::atomic<bool> & running, Clock::time_point deadline)
{
  while (running.load()) {
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
      deadline - Clock::now());
    if (remaining.count() <= 0) return false;
    pollfd item{fd, events, 0};
    const int result = ::poll(&item, 1, static_cast<int>(
        std::min(remaining, POLL_INTERVAL).count()));
    if (result < 0 && errno == EINTR) continue;
    if (result < 0) return false;
    if (result > 0) return (item.revents & events) != 0;
  }
  return false;
}

inline void retry_delay(const std::atomic<bool> & running)
{
  const auto deadline = Clock::now() + TIMEOUT;
  while (running.load() && Clock::now() < deadline) {
    std::this_thread::sleep_for(POLL_INTERVAL);
  }
}

// Returns a nonblocking fd owned by the caller, or -1 on failure/cancellation.
inline int connect(
  const std::string & host, uint16_t port, const std::atomic<bool> & running)
{
  if (!running.load()) return -1;
  const int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (fd < 0) return -1;
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  if (::inet_pton(AF_INET, host.c_str(), &address.sin_addr) == 1) {
    const int result = ::connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address));
    if (result == 0) return fd;
    if (errno == EINPROGRESS && wait_ready(fd, POLLOUT, running, Clock::now() + TIMEOUT)) {
      int error = 0;
      socklen_t size = sizeof(error);
      if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &size) == 0 && error == 0) {
        return fd;
      }
    }
  }
  ::close(fd);
  return -1;
}

inline bool recv_exact(
  int fd, uint8_t * data, size_t size, const std::atomic<bool> & running,
  Clock::time_point deadline)
{
  size_t received = 0;
  while (received < size && running.load()) {
    if (Clock::now() >= deadline) return false;
    const ssize_t count = ::recv(fd, data + received, size - received, 0);
    if (count > 0) {
      received += static_cast<size_t>(count);
    } else if (count == 0) {
      return false;
    } else if (errno == EINTR) {
      continue;
    } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
      return false;
    } else if (!wait_ready(fd, POLLIN, running, deadline)) {
      return false;
    }
  }
  return received == size && running.load();
}

inline bool read_frame(
  int fd, std::vector<uint8_t> & payload, const std::atomic<bool> & running)
{
  const auto deadline = Clock::now() + TIMEOUT;
  uint8_t header[4];
  if (!recv_exact(fd, header, sizeof(header), running, deadline)) return false;
  const uint32_t size = uint32_t(header[0]) | (uint32_t(header[1]) << 8) |
    (uint32_t(header[2]) << 16) | (uint32_t(header[3]) << 24);
  // Retain the observation node's limit for all consumers, before allocation.
  if (size == 0 || size > 16 * 1024 * 1024) return false;
  payload.resize(size);
  return recv_exact(fd, payload.data(), payload.size(), running, deadline);
}

}  // namespace camera_tcp
}  // namespace vision_encoders
