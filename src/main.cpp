#include "LIVMapper.h"
#include <csignal>

static std::atomic<bool> g_shutdown{false};

void signal_handler(int sig) {
  (void)sig;
  g_shutdown.store(true);
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  // Ensure SIGINT/SIGTERM reliably set our flag
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(true);
  options.automatically_declare_parameters_from_overrides(true);

  rclcpp::Node::SharedPtr nh;
  image_transport::ImageTransport it_(nh);
  LIVMapper mapper(nh, "laserMapping", options);
  mapper.shutdown_flag = &g_shutdown;
  mapper.initializeSubscribersAndPublishers(nh, it_);
  mapper.run(nh);
  rclcpp::shutdown();
  return 0;
}
