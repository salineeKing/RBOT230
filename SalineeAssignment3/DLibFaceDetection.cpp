#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui.hpp>

#include <dlib/opencv.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>

#include <librealsense2/rsutil.h>
#include <librealsense2/rs.hpp> 

using namespace cv;
using namespace std;
using namespace dlib;

// The human face detector network (source: https://github.com/davisking/dlib-models/blob/master/mmod_human_face_detector.dat.bz2)
// The shape predictor face landmark network (source: https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2)
template <long num_filters, typename SUBNET> using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;

template <typename SUBNET> using downsampler = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5 = relu<affine<con5<45, SUBNET>>>;

using net_type = loss_mmod<con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

int main() try
{
#pragma region Realsense Setup

	rs2::log_to_console(RS2_LOG_SEVERITY_ERROR);

	// Declare RealSense pipeline, encapsulating the actual device and sensors
	rs2::pipeline pipe;

	// Create a configuration for configuring the pipeline with a non default profile
	rs2::config cfg;

	// Add desired streams to configuration
	cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);

	// Start streaming with default recommended configuration
	pipe.start(cfg);

	// Camera warmup - dropping several first frames to let auto-exposure stabilize
	for (int i = 0; i < 30; i++)
	{
		pipe.wait_for_frames();
	}

#pragma endregion

#pragma region Defined Initial Settings

	image_window win1;
	image_window win2;
	cv::TickMeter timer;

	// Load face detection and pose estimation models
	net_type net;
	deserialize("../mmod_human_face_detector.dat") >> net;

	// Load Load face detection and pose estimation models for point landmarking model 
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;
	deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

#pragma endregion

	// Grab and process frames until the main window is closed by the user
	while (!win1.is_closed() && !win2.is_closed())
	{
		// Full Timer Resolution
		timer.reset();
		timer.start();

		// Wait for next set of frames from the camera
		rs2::frameset data = pipe.wait_for_frames();
		rs2::frame color = data.get_color_frame();

		// Query frame size (width and height)
		const int w = color.as<rs2::video_frame>().get_width();
		const int h = color.as<rs2::video_frame>().get_height();

		// Create OpenCV matrix of size (w,h) from the colorized depth data
		Mat liveFrame(Size(w, h), CV_8UC3, (void*)color.get_data(), Mat::AUTO_STEP);

		// Convert OpenCV image format to Dlib's image format
		cv_image<bgr_pixel> dlibIm(liveFrame);
		matrix<rgb_pixel> matrix;
		assign_image(matrix, dlibIm);

		// Run the detector on the image and show the output
		auto dets = net(matrix);
		win1.clear_overlay();
		win1.set_image(matrix);
		for (auto&& d : dets)
		{
			// Measure performance
			timer.stop();
			cv::String strTime1 = cv::format("Processed Time: %4.3f msec", timer.getTimeMilli());

			win1.add_overlay(d, rgb_pixel(255, 0, 0), strTime1);
		}

		// Detect faces 
		std::vector<dlib::rectangle> faces = detector(dlibIm);

		// Find the pose of each face.
		std::vector<full_object_detection> shapes;
		for (unsigned long i = 0; i < faces.size(); ++i)
			shapes.push_back(pose_model(dlibIm, faces[i]));

		// Display it all on the screen
		win2.clear_overlay();
		win2.set_image(dlibIm);
		win2.add_overlay(render_face_detections(shapes));
	}

	return EXIT_SUCCESS;
}
catch (std::exception& e)
{
	cout << e.what() << endl;
}