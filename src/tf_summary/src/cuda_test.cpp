#include <iostream>
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"

#include <numeric>
#include "tensorflow/core/util/cuda_kernel_helper.h"
using namespace std;
using namespace stream_executor;

class stream_executor_test
{
public:
    stream_executor_test() {}
    std::unique_ptr<stream_executor::StreamExecutor> NewStreamExecutor(){
        Platform* platform = MultiPlatformManager::PlatformWithName("Host").ConsumeValueOrDie();
        StreamExecutorConfig config(0);
        return platform->GetUncachedExecutor(config).ConsumeValueOrDie();
    }
};

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
__global__ void SetOutbufZero(CudaLaunchConfig config, int* outbuf) {
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) { outbuf[x] = 0; }
}

// counting number of jobs by using atomic +1
__global__ void Count1D(CudaLaunchConfig config, int bufsize, int* outbuf) {
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  // x might overflow when testing extreme case
      break;
    }
    atomicAdd(&outbuf[x % bufsize], 1);
  }
}
__global__ void Count2D(Cuda2DLaunchConfig config, int bufsize, int* outbuf) {
  CUDA_AXIS_KERNEL_LOOP(x, config.virtual_thread_count.x, X) {
    if (x < 0) {  // x might overflow when testing extreme case
      break;
    }
    CUDA_AXIS_KERNEL_LOOP(y, config.virtual_thread_count.y, Y) {
      if (y < 0) {  // y might overflow when testing extreme case
        break;
      }
      int idx = x * config.virtual_thread_count.y + y;
      atomicAdd(&outbuf[idx % bufsize], 1);
    }
  }
}
__global__ void Count3D(Cuda3DLaunchConfig config, int bufsize, int* outbuf) {
  CUDA_AXIS_KERNEL_LOOP(x, config.virtual_thread_count.x, X) {
    if (x < 0) {  // x might overflow when testing extreme case
      break;
    }
    CUDA_AXIS_KERNEL_LOOP(y, config.virtual_thread_count.y, Y) {
      if (y < 0) {  // y might overflow when testing extreme case
        break;
      }
      CUDA_AXIS_KERNEL_LOOP(z, config.virtual_thread_count.z, Z) {
        if (z < 0) {  // z might overflow when testing extreme case
          break;
        }
        int idx =
            x * config.virtual_thread_count.y * config.virtual_thread_count.z +
            y * config.virtual_thread_count.z + z;
        atomicAdd(&outbuf[idx % bufsize], 1);
      }
    }
  }
}

__global__ void CudaShuffleGetSrcLaneTest(unsigned* failure_count) {
  unsigned lane_id = CudaLaneId();
  for (int width = warpSize; width > 1; width /= 2) {
    auto check_result = [&](const char* op_name, int param, unsigned actual,
                            unsigned expected) {
      if (actual != expected) {
        printf("Cuda%sGetSrcLane(%d, %d) for lane %d returned %d, not %d\n",
               op_name, param, width, lane_id, actual, expected);
        CudaAtomicAdd(failure_count, 1);
      }
    };
    for (int src_lane = -warpSize; src_lane <= warpSize; ++src_lane) {
      unsigned actual_lane = detail::CudaShuffleGetSrcLane(src_lane, width);
      unsigned expect_lane =
          CudaShuffleSync(kCudaWarpAll, lane_id, src_lane, width);
      check_result("Shuffle", src_lane, actual_lane, expect_lane);
    }
    for (unsigned delta = 0; delta <= warpSize; ++delta) {
      unsigned actual_lane = detail::CudaShuffleUpGetSrcLane(delta, width);
      unsigned expect_lane =
          CudaShuffleUpSync(kCudaWarpAll, lane_id, delta, width);
      check_result("ShuffleUp", delta, actual_lane, expect_lane);
    }
    for (unsigned delta = 0; delta <= warpSize; ++delta) {
      unsigned actual_lane = detail::CudaShuffleDownGetSrcLane(delta, width);
      unsigned expect_lane =
          CudaShuffleDownSync(kCudaWarpAll, lane_id, delta, width);
      check_result("ShuffleDown", delta, actual_lane, expect_lane);
    }
    for (int lane_lane = warpSize; lane_lane > 0; lane_lane /= 2) {
      unsigned actual_lane = detail::CudaShuffleXorGetSrcLane(lane_lane, width);
      unsigned expect_lane =
          CudaShuffleXorSync(kCudaWarpAll, lane_id, lane_lane, width);
      check_result("ShuffleXor", lane_lane, actual_lane, expect_lane);
    }
  }
}


class CudaLaunchConfigTest
{
public:
    CudaLaunchConfigTest() {}
    const int bufsize =1024;
    int* outbuf = nullptr;
    Eigen::CudaStreamDevice stream;
    Eigen::GpuDevice d = Eigen::GpuDevice(&stream);

//    virtual void SetUp()
//    {
//    }
};



int main()
{
    stream_executor_test test;
   // std::unique_ptr<stream_executor::StreamExecutor> executor = test.NewStreamExecutor();
    stream_executor::Stream stream(test.NewStreamExecutor().get());
    cout<<"bool:"<<stream.ok()<<endl;

    return 0;
}
#endif
