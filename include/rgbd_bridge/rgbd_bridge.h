#pragma once

#include <map>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace rgbd_bridge {

enum class ImageType {
  RGB = 0,
  DEPTH,
  IR,
  //RECT_RGB,
  //RECT_RGB_ALIGNED_DEPTH,
  //DEPTH_ALIGNED_RGB,
};

std::string ImageTypeToString(const ImageType type);
bool is_color_image(const ImageType type);
bool is_depth_image(const ImageType type);

struct Intrinsics {
  float fx{};
  float fy{};
  float ppx{};
  float ppy{};
};

class RGBDSensor {
public:
  virtual ~RGBDSensor() {}

  virtual void Start(const std::vector<ImageType> &types,
                     const ImageType cloud_camera) = 0;
  virtual void Stop() = 0;

  /**
   * For rgb image, the channels are in RGB order.
   * For depth image, each element is 16bits, in units of mm.
   */
  boost::shared_ptr<const cv::Mat> GetLatestImage(const ImageType type,
                                                  uint64_t *timestamp) const;

  /**
   * Units of m.
   */
  pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr
  GetLatestPointCloud(uint64_t *timestamp) const;

  const std::vector<ImageType> &get_supported_image_types() const {
    return supported_types_;
  }

  bool supports(const ImageType type) const;

  virtual Eigen::Vector2f Project(const ImageType type,
                                  const Eigen::Vector3f &xyz) const = 0;
  // depth is in m. uv is in pixel.
  virtual Eigen::Vector3f Deproject(const ImageType type,
                                    const Eigen::Vector2i &uv,
                                    float depth) const = 0;

  virtual bool is_enabled(const ImageType type) const = 0;

  virtual void set_intrinsics(const ImageType type, const Intrinsics& intrinsics) {
    throw std::runtime_error("not implemented");
  }
  virtual Intrinsics get_intrinsics(const ImageType type) const = 0;

protected:
  template <typename DataType> struct TimeStampedData {
    DataType data{};
    uint64_t timestamp{0};
    uint64_t count{0};
  };

  RGBDSensor(const std::vector<ImageType> &supported_types)
      : supported_types_(supported_types) {}

  typedef TimeStampedData<boost::shared_ptr<const cv::Mat>> TimeStampedImage;
  typedef TimeStampedData<pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr>
      TimeStampedCloud;

  virtual boost::shared_ptr<const cv::Mat>
  DoGetLatestImage(const ImageType type, uint64_t *timestamp) const = 0;
  virtual pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr
  DoGetLatestPointCloud(uint64_t *timestamp) const = 0;

private:
  const std::vector<ImageType> supported_types_;
};

} // namespace rgbd_bridge
