// Copyright 2026 Tier IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef AUTOWARE__TENSORRT_E2E__ENGINE_CACHE_HPP_
#define AUTOWARE__TENSORRT_E2E__ENGINE_CACHE_HPP_

#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>

#include <chrono>
#include <filesystem>
#include <optional>
#include <string>

namespace autoware::tensorrt_e2e
{

/**
 * @brief Delete a cached TensorRT engine that is older than the ONNX beside it.
 *
 * TrtCommon reuses `<model>.engine` whenever it deserializes and the plan's TensorRT
 * version matches. Its two freshness checks -- validateNetworkIO() and
 * validateProfileDims() -- both return success when the caller passes nothing, and log
 * only "Network IO is empty, skipping validation. It might lead to undefined behavior".
 * This node cannot declare a static IO list the way autoware_bevfusion does, because it
 * is model-agnostic and reads its bindings out of whatever engine it is given. So a
 * re-exported ONNX dropped next to a stale engine would silently keep running the old
 * weights, and obey_graph_precision() would be silently inert as well, since the layer
 * precisions it pins are applied to a network that a cached plan never builds.
 *
 * Comparing mtimes is enough for that: the exporter writes a new ONNX, the engine beside
 * it is then older, and it is rebuilt. It intentionally does not hash the graph -- an
 * engine that is merely *newer* than its ONNX is left alone, which is the normal state
 * after a build.
 *
 * @param onnx_path Path to the ONNX the engine was built from.
 * @param engine_path Path to the cached engine (TrtCommonConfig derives it from the ONNX
 *                    by replacing the extension when it is not given explicitly).
 * @param logger Logger for the one-line notice when a stale engine is removed.
 * @return true when a stale engine was deleted.
 */
inline bool drop_stale_engine(
  const std::string & onnx_path, const std::string & engine_path, const rclcpp::Logger & logger)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  if (!fs::exists(engine_path, ec) || !fs::exists(onnx_path, ec)) {
    return false;
  }
  const auto engine_time = fs::last_write_time(engine_path, ec);
  if (ec) {
    return false;
  }
  const auto onnx_time = fs::last_write_time(onnx_path, ec);
  if (ec || engine_time >= onnx_time) {
    return false;
  }
  fs::remove(engine_path, ec);
  if (ec) {
    RCLCPP_WARN(
      logger, "%s is older than %s but could not be removed (%s); it will be reused as is.",
      engine_path.c_str(), onnx_path.c_str(), ec.message().c_str());
    return false;
  }
  RCLCPP_INFO(
    logger, "%s is older than %s: removed, the engine will be rebuilt.", engine_path.c_str(),
    onnx_path.c_str());
  return true;
}

/**
 * @brief What TrtCommon is about to do with an engine, said before it starts, under the
 * node's own name.
 *
 * TrtCommon deserializes `<model>.engine` when it can and builds it otherwise, and while it
 * builds it says only "Please wait for a few minutes" every five seconds, through the
 * TensorRT logger, under no node name. The build runs inside this node's constructor, so for
 * its whole duration the node is not spinning: no callback runs, no timer fires, no
 * `inference_status` is published. On the first launch after a new graph, GPU or TensorRT
 * version that is minutes to tens of minutes -- 27 min for the ResWorld planner on a vehicle
 * whose GPU the sensing stack was using at the same time (2026-09-14) -- and from outside it
 * is indistinguishable from a node that has hung or failed silently. So: say here that a
 * build is starting and what it means, and let report_engine() say how long it took and
 * whether it was a build or a load. The engine's mtime tells the two apart, because TrtCommon
 * also rebuilds a cached engine it cannot deserialize (another TensorRT version, another GPU)
 * without saying so.
 */
struct EngineNotice
{
  std::string onnx_path;
  std::string engine_path;
  std::chrono::steady_clock::time_point started;
  //! The engine's mtime when TrtCommon started; empty when no engine was cached.
  std::optional<std::filesystem::file_time_type> cached_mtime;
};

inline EngineNotice announce_engine(
  const std::string & onnx_path, const std::string & engine_path, const rclcpp::Logger & logger)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  EngineNotice notice{onnx_path, engine_path, std::chrono::steady_clock::now(), std::nullopt};
  if (fs::exists(engine_path, ec)) {
    const auto mtime = fs::last_write_time(engine_path, ec);
    if (!ec) {
      notice.cached_mtime = mtime;
    }
  }
  if (notice.cached_mtime) {
    RCLCPP_INFO(logger, "Loading the cached TensorRT engine %s", engine_path.c_str());
  } else {
    RCLCPP_WARN(
      logger,
      "No cached TensorRT engine at %s: building it from %s now. A build takes minutes to tens "
      "of minutes on a GPU the sensing stack shares, and it holds this node's constructor: "
      "until it finishes the node is not spinning -- no subscription callback runs and no "
      "inference_status is published -- and TensorRT prints only \"Please wait\". This is a "
      "wait, not a failure. To take it out of the vehicle's start-up, build the engines ahead "
      "of time with build_only:=true.",
      engine_path.c_str(), onnx_path.c_str());
  }
  return notice;
}

inline void report_engine(const EngineNotice & notice, const rclcpp::Logger & logger)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  const double seconds =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - notice.started).count();
  bool built = !notice.cached_mtime;
  if (!built && fs::exists(notice.engine_path, ec)) {
    const auto mtime = fs::last_write_time(notice.engine_path, ec);
    built = !ec && mtime != *notice.cached_mtime;
  }
  if (built) {
    RCLCPP_WARN(
      logger, "TensorRT engine %s built in %.0f s%s", notice.engine_path.c_str(), seconds,
      notice.cached_mtime ? " (the cached engine could not be deserialized and was rebuilt)"
                          : "");
  } else {
    RCLCPP_INFO(logger, "TensorRT engine %s loaded in %.1f s", notice.engine_path.c_str(), seconds);
  }
}

}  // namespace autoware::tensorrt_e2e

#endif  // AUTOWARE__TENSORRT_E2E__ENGINE_CACHE_HPP_
