#include "rgbd_bridge/real_sense_sr300.h"
#include "librealsense2/rsutil.h"

#include <boost/make_shared.hpp>

namespace rgbd_bridge {
namespace {

boost::shared_ptr<cv::Mat> MakeImg(const void *src, int width, int height,
                                   rs2_format pixel_type) {
  int cv_type = -1;
  switch (pixel_type) {
  case RS2_FORMAT_Z16:
  case RS2_FORMAT_DISPARITY16:
  case RS2_FORMAT_Y16:
  case RS2_FORMAT_RAW16:
    cv_type = CV_16UC1;
    break;
  case RS2_FORMAT_Y8:
  case RS2_FORMAT_RAW8:
    cv_type = CV_8UC1;
    break;
  case RS2_FORMAT_RGB8:
  case RS2_FORMAT_BGR8:
    cv_type = CV_8UC3;
    break;
  case RS2_FORMAT_RGBA8:
  case RS2_FORMAT_BGRA8:
    cv_type = CV_8UC4;
    break;
  default:
    throw std::runtime_error("Doesn't support this type");
  }

  boost::shared_ptr<cv::Mat> img =
      boost::make_shared<cv::Mat>(height, width, cv_type);
  if (cv_type == CV_8UC3) {
    memcpy(img->data, src, sizeof(uint8_t) * 3 * width * height);
  } else if (cv_type == CV_8UC4) {
    memcpy(img->data, src, sizeof(uint8_t) * 4 * width * height);
  } else if (cv_type == CV_8UC1) {
    memcpy(img->data, src, sizeof(uint8_t) * width * height);
  } else if (cv_type == CV_16UC1) {
    memcpy(img->data, src, sizeof(uint16_t) * width * height);
  } else {
    throw std::runtime_error("Doesn't support this type");
  }

  return img;
}

template <typename Scalar> void ScaleImg(cv::Mat *img, float scale) {
  for (int r = 0; r < img->rows; r++) {
    for (int c = 0; c < img->cols; c++) {
      img->at<Scalar>(r, c) *= scale;
    }
  }
}

void ScaleImgInPlace(cv::Mat *img, float scale) {
  switch (img->type()) {
  case CV_8UC4:
    ScaleImg<cv::Vec4b>(img, scale);
    break;
  case CV_8UC3:
    ScaleImg<cv::Vec3b>(img, scale);
    break;
  case CV_8UC1:
    ScaleImg<uint8_t>(img, scale);
    break;
  case CV_16UC1:
    ScaleImg<uint16_t>(img, scale);
    break;
  default:
    throw std::runtime_error("Doesn't support this type");
  }
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr
MakePointCloud(const RealSenseSR300 &camera, const rs2_intrinsics &depth_intrin,
               const rs2_intrinsics &color_intrin,
               const rs2_extrinsics &depth_to_color,
               const rs2_extrinsics &depth_to_desired, float depth_scale,
               const uint8_t *color_image, const uint16_t *depth_image) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
      boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

  for (int dy = 0; dy < depth_intrin.height; ++dy) {
    for (int dx = 0; dx < depth_intrin.width; ++dx) {
      // Retrieve the 16-bit depth value and map it into a depth in meters
      uint16_t depth_value = depth_image[dy * depth_intrin.width + dx];
      float depth_in_meters = depth_value * depth_scale;

      // Skip over pixels with a depth value of zero, which is used to
      // indicate no data
      if (depth_value == 0 || depth_in_meters > 1 || depth_in_meters < 0.11)
        continue;

      // Map from pixel coordinates in the depth image to pixel coordinates in
      // the color image
      float depth_pixel[2] = {(float)dx, (float)dy};
      float depth_point[3];
      rs2_deproject_pixel_to_point(depth_point, &depth_intrin, depth_pixel, depth_in_meters);

      /*
      rs::float3 color_point = depth_to_color.transform(depth_point);
      rs::float2 color_pixel = color_intrin.project(color_point);
      rs::float3 desired_point = depth_to_desired.transform(depth_point);
      */
      float color_point[3];
      rs2_transform_point_to_point(color_point, &depth_to_color, depth_point);
      float color_pixel[2];
      rs2_project_point_to_pixel(color_pixel, &color_intrin, color_point);
      float desired_point[3];
      rs2_transform_point_to_point(desired_point, &depth_to_desired, depth_point);

      // Use the color from the nearest color pixel, or pure white if this
      // point falls outside the color image
      const int cx = (int)std::round(color_pixel[0]),
                cy = (int)std::round(color_pixel[1]);
      pcl::PointXYZRGB point;
      point.x = desired_point[0];
      point.y = desired_point[1];
      point.z = desired_point[2];
      if (cx < 0 || cy < 0 || cx >= color_intrin.width ||
          cy >= color_intrin.height) {
        point.r = 255;
        point.g = 165;
        point.b = 0;
      } else {
        point.r = color_image[(cy * color_intrin.width + cx) * 3];
        point.g = color_image[(cy * color_intrin.width + cx) * 3 + 1];
        point.b = color_image[(cy * color_intrin.width + cx) * 3 + 2];
      }
      cloud->push_back(point);
    }
  }
  return cloud;
}
}

rs2::context RealSenseSR300::context_;

RealSenseSR300::RealSenseSR300(int camera_id)
    : RGBDSensor({ImageType::RGB, ImageType::DEPTH, ImageType::IR}),
                  //ImageType::RECT_RGB, ImageType::RECT_RGB_ALIGNED_DEPTH,
                  //ImageType::DEPTH_ALIGNED_RGB}),
      pipeline_(context_) {
  rs2::device_list devices = context_.query_devices();

  const int num_cameras = devices.size();
  if (num_cameras <= camera_id) {
    throw std::runtime_error("Device " + std::to_string(camera_id) +
                             " not detected. Is it plugged in?");
  }
  camera_ = devices[camera_id];

  // Want these streams.
  config_.enable_stream(RS2_STREAM_COLOR, RS2_FORMAT_RGB8, 30);
  config_.enable_stream(RS2_STREAM_DEPTH, RS2_FORMAT_Z16, 30);
  config_.enable_stream(RS2_STREAM_INFRARED, RS2_FORMAT_Y16, 30);
  rs2::pipeline_profile pipeline_profile = config_.resolve(pipeline_);
  std::vector<rs2::stream_profile> profiles = pipeline_profile.get_streams();
  for (const auto& profile : profiles) {
    stream_profiles_[profile.stream_type()] = profile;
    intrinsics_[profile.stream_type()] = profile.as<rs2::video_stream_profile>().get_intrinsics();
  }

  std::vector<rs2::sensor> sensors = camera_.query_sensors();
  for (const rs2::sensor& sensor : sensors) {
    if (rs2::depth_sensor dpt_sensor = sensor.as<rs2::depth_sensor>()) {
      depth_scale_ = dpt_sensor.get_depth_scale();
    }
  }

  /*
  // Get supported ImageType.
  const auto &supported_types = get_supported_image_types();
  camera_.enable_stream(rs2_stream::color, rs::preset::best_quality);
  camera_.enable_stream(rs2_stream::depth, rs::preset::best_quality);
  camera_.enable_stream(rs2_stream::infrared, rs::preset::best_quality);
  for (const auto &type : supported_types) {
    rs2_stream stream = ImageTypeToStreamType(type);
    intrinsics_[stream] = camera_.get_stream_intrinsics(stream);
  }
  */
}

bool RealSenseSR300::IsObjectInGrasp(const cv::Mat &raw_depth,
                                     double depth_thresh_m) const {
  // This is 7cm.
  const uint16_t depth_thresh = (uint16_t)(depth_thresh_m * 1000);
  for (int h = 479; h >= 450; h--) {
    int ctr = 0;
    for (int w = 46; w < 320; w++) {
      uint16_t pixel = raw_depth.at<uint16_t>(h, w);
      if (pixel < depth_thresh && pixel != 0) {
        ctr++;
      } else {
        ctr = 0;
      }
    }
    // Has at least 5 continuous pixel that are not empty and
    // closer than depth thresh.
    if (ctr < 5)
      return false;
  }
  return true;
  // std::cout << "Close pixel count: " << ctr << "\n";
}

void RealSenseSR300::set_laser_projector_power(int level) {
  // sfeng: for some wtf reaseon, i need to set this in a tight loop to get it
  // to work.

  /*
  for (int i = 0; i < 3; i++) {
    camera_.set_option(rs::option::f200_laser_power, level);
  }
  assert(camera_.get_option(rs::option::f200_laser_power) == level);
  */
}

Eigen::Vector2f RealSenseSR300::Project(const ImageType type,
                                        const Eigen::Vector3f &xyz) const {
  const rs2_stream stream = ImageTypeToStreamType(type);
  const rs2_intrinsics& intrinsics = intrinsics_.at(stream);
  float xyz1[3] = {xyz(0), xyz(1), xyz(2)};
  float pixel[2];
  rs2_project_point_to_pixel(pixel, &intrinsics, xyz1);
  return Eigen::Vector2f(pixel[0], pixel[1]);
}

Eigen::Vector3f RealSenseSR300::Deproject(const ImageType type,
                                          const Eigen::Vector2i &uv,
                                          float depth) const {
  const rs2_stream stream = ImageTypeToStreamType(type);
  const rs2_intrinsics& intrinsics = intrinsics_.at(stream);
  float pixel[2] = {(float)uv(0), (float)uv(1)};
  float xyz[3];
  rs2_deproject_pixel_to_point(xyz, &intrinsics, pixel, depth);
  return Eigen::Vector3f(xyz[0], xyz[1], xyz[2]);
}

int RealSenseSR300::get_number_of_cameras() {
  return context_.query_devices().size();
}

void RealSenseSR300::Start(const std::vector<ImageType> &types,
                           const ImageType cloud_base) {
  if (run_) {
    return;
  }

  run_ = true;
  std::string cam_name = camera_.get_info(RS2_CAMERA_INFO_NAME);
  std::string serial = camera_.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);

  std::cout << cam_name << " #" << serial << " starting.\n";

  /////////////////////////////////////////////////////////////////////////////
  // Config params.
  // set_mode(rs_ivcam_preset::RS_IVCAM_PRESET_DEFAULT);
  cloud_base_ = ImageTypeToStreamType(cloud_base);

  /////////////////////////////////////////////////////////////////////////////
  // Reset shared data.
  for (const ImageType type : types) {
    images_[type] = TimeStampedImage();
    std::cout << ImageTypeToString(type) << "\n";
    auto intr = intrinsics_.at(ImageTypeToStreamType(type));
    std::cout << "fx: " << intr.fx << "\n";
    std::cout << "fy: " << intr.fy << "\n";
    std::cout << "ppx: " << intr.ppx << "\n";
    std::cout << "ppy: " << intr.ppy << "\n";
    std::cout << "coeffs[0]: " << intr.coeffs[0] << "\n";
    std::cout << "coeffs[1]: " << intr.coeffs[1] << "\n";
    std::cout << "coeffs[2]: " << intr.coeffs[2] << "\n";
    std::cout << "coeffs[3]: " << intr.coeffs[3] << "\n";
    std::cout << "coeffs[4]: " << intr.coeffs[4] << "\n";
  }

  /////////////////////////////////////////////////////////////////////////////
  // Start thread.
  auto pipeline_profile = pipeline_.start(config_);

  thread_ = std::thread(&RealSenseSR300::PollingThread, this);
}

void RealSenseSR300::Stop() {
  run_ = false;
  thread_.join();
  pipeline_.stop();
  // Clean up.
  images_.clear();
  cloud_ = TimeStampedCloud();
}

/*
void RealSenseSR300::set_mode(const rs_ivcam_preset mode) {
  apply_ivcam_preset(camera_, mode);
}
*/

rs2_stream RealSenseSR300::ImageTypeToStreamType(const ImageType type) const {
  switch (type) {
  case ImageType::RGB:
    return RS2_STREAM_COLOR;
  case ImageType::DEPTH:
    return RS2_STREAM_DEPTH;
  case ImageType::IR:
    return RS2_STREAM_INFRARED;
  /*
  case ImageType::RECT_RGB:
    return RS2_STREAM_rectified_color;
  case ImageType::RECT_RGB_ALIGNED_DEPTH:
    return RS2_STREAM_depth_aligned_to_rectified_color;
  case ImageType::DEPTH_ALIGNED_RGB:
    return RS2_STREAM_color_aligned_to_depth;
  */
  default:
    throw std::logic_error("Unsupported ImageType");
  }
}

bool RealSenseSR300::is_enabled(const ImageType type) const {
  // todo
  return true;
  // return camera_.is_stream_enabled(ImageTypeToStreamType(type));
}

Intrinsics RealSenseSR300::get_intrinsics(const ImageType type) const {
  Intrinsics ret;
  rs2_stream stream = ImageTypeToStreamType(type);
  const rs2_intrinsics& intrinsics = intrinsics_.at(stream);
  ret.fx = intrinsics.fx;
  ret.fy = intrinsics.fy;
  ret.ppx = intrinsics.ppx;
  ret.ppy = intrinsics.ppy;

  return ret;
}

void RealSenseSR300::PollingThread() {
  rs2_intrinsics depth_intrin = intrinsics_.at(RS2_STREAM_DEPTH);
  rs2_intrinsics color_intrin = intrinsics_.at(RS2_STREAM_COLOR);
  rs2_extrinsics depth_to_color =
      stream_profiles_.at(RS2_STREAM_DEPTH).get_extrinsics_to(stream_profiles_.at(RS2_STREAM_COLOR));

  rs2_extrinsics depth_to_desired =
      stream_profiles_.at(RS2_STREAM_DEPTH).get_extrinsics_to(stream_profiles_.at(cloud_base_));

  std::map<const ImageType, TimeStampedImage> images = images_;

  rs2::frameset frameset;
  std::map<rs2_stream, rs2::frame> frames;

  while (run_) {
    // Block until all frames have arrived.
    frameset = pipeline_.wait_for_frames();

    if (frameset.size() != stream_profiles_.size())
      continue;

    for (auto &pair : stream_profiles_) {
      rs2_stream stream = pair.first;
      frames[stream] = frameset.first(stream);
    }

    // Raw color and depth img.
    const uint8_t *color_image = (const uint8_t *)frames.at(RS2_STREAM_COLOR).get_data();
    const uint16_t *depth_image = (const uint16_t *)frames.at(RS2_STREAM_DEPTH).get_data();

    // Make point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
        MakePointCloud(*this, depth_intrin, color_intrin, depth_to_color,
                       depth_to_desired, depth_scale_, color_image, depth_image);

    // Make cv::Mat of the images.
    uint64_t depth_time;
    uint64_t depth_ctr;
    for (auto &pair : images) {
      rs2_stream stream = ImageTypeToStreamType(pair.first);
      const rs2::video_frame frame = frames.at(stream).as<rs2::video_frame>();
      pair.second.timestamp = (uint64_t)frame.get_timestamp();
      pair.second.count = frame.get_frame_number();
      boost::shared_ptr<cv::Mat> img = MakeImg(
          frame.get_data(), frame.get_width(), frame.get_height(),
          stream_profiles_.at(stream).format());
      // Scale depth image to units of mm.
      if (stream == RS2_STREAM_DEPTH) {
        ScaleImgInPlace(img.get(), depth_scale_ * 1e3);
        depth_time = pair.second.timestamp;
        depth_ctr = pair.second.count;
      }

      pair.second.data = img;
    }

    // Lock and do pointer assignment.
    std::unique_lock<std::mutex> lock2(lock_);
    images_ = images;

    cloud_.timestamp = depth_time;
    cloud_.count = depth_ctr;
    cloud_.data = cloud;
  }
}

boost::shared_ptr<const cv::Mat>
RealSenseSR300::DoGetLatestImage(const ImageType type,
                                 uint64_t *timestamp) const {
  std::unique_lock<std::mutex> lock2(lock_);
  auto it = images_.find(type);
  if (it == images_.end()) {
    throw std::runtime_error("ImageType not initialized.");
  }
  const TimeStampedImage &image = it->second;

  if (image.count == 0) {
    *timestamp = 0;
    return nullptr;
  }
  *timestamp = image.timestamp;
  return image.data;
}

pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr
RealSenseSR300::DoGetLatestPointCloud(uint64_t *timestamp) const {
  std::unique_lock<std::mutex> lock2(lock_);
  if (cloud_.count == 0) {
    *timestamp = 0;
    return nullptr;
  }
  *timestamp = cloud_.timestamp;
  return cloud_.data;
}

} // namespace rgbd_bridge
