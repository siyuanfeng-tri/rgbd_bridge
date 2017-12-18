#include "rgbd_bridge/rgbd_bridge.h"
#include <boost/make_shared.hpp>

namespace rgbd_bridge {

std::string ImageTypeToString(const ImageType type) {
  switch (type) {
  case ImageType::RGB:
    return "RGB";
  case ImageType::DEPTH:
    return "DEPTH";
  case ImageType::IR:
    return "INFRARED";
  /*
  case ImageType::RECT_RGB:
    return "RECTIFIED_RGB";
  case ImageType::RECT_RGB_ALIGNED_DEPTH:
    return "DEPTH_ALIGNED_TO_RECTIFIED_RGB";
  case ImageType::DEPTH_ALIGNED_RGB:
    return "RGB_ALIGNED_TO_DEPTH";
  */
  default:
    throw std::runtime_error("Unknown ImageType");
  }
}

bool is_color_image(const ImageType type) {
  switch (type) {
  case ImageType::RGB:
  //case ImageType::RECT_RGB:
  //case ImageType::DEPTH_ALIGNED_RGB:
    return true;
  case ImageType::IR:
  case ImageType::DEPTH:
  //case ImageType::RECT_RGB_ALIGNED_DEPTH:
    return false;
  default:
    throw std::runtime_error("Unknown ImageType");
  }
}

bool is_depth_image(const ImageType type) {
  switch (type) {
  case ImageType::IR:
  case ImageType::RGB:
  //case ImageType::RECT_RGB:
  //case ImageType::DEPTH_ALIGNED_RGB:
    return false;
  case ImageType::DEPTH:
  //case ImageType::RECT_RGB_ALIGNED_DEPTH:
    return true;
  default:
    throw std::runtime_error("Unknown ImageType");
  }
}

bool RGBDSensor::supports(const ImageType type) const {
  return std::find(supported_types_.begin(), supported_types_.end(), type) !=
         supported_types_.end();
}

boost::shared_ptr<const cv::Mat>
RGBDSensor::GetLatestImage(const ImageType type, uint64_t *timestamp) const {
  return DoGetLatestImage(type, timestamp);
}

pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr
RGBDSensor::GetLatestPointCloud(uint64_t *timestamp) const {
  return DoGetLatestPointCloud(timestamp);
}

} // namespace rgbd_bridge
