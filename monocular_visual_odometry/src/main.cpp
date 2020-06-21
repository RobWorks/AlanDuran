/*
 * main.cpp
 *
 *  Created on: 31 ene 2020
 *      Author: alan
 */

#include <iostream>
#include <fstream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video.hpp>

#include "distortion_utils.h"
#include "graph_utils.h"
#include "point_mgmt.h"
#include "rot2euler.h"

// I/O parameters
#define USE_VIDEO 			1
#define INITIAL_FRAME 		20
#define STEP				1
#define SHOW_PLOTS			0

#define DRAW_SCALE			0.75
#define LINE_SIZE			1
#define CIRCLE_SIZE			5

// ALG parameters
#define MAX_DISTANCE 		7.5
#define MAX_DIR_ANGLE 		7.5
#define MAX_SKIPPED_FRAMES	7 //actual number is -2
#define MAX_ROTATION		5
#define N_FEATURES 			100

using namespace cv;
using namespace std;

//Input Files
// 2 different image input format options: video or image list, selected with USE_VIDEO macro
const string img_lst_path = " "; 
const string video_path = "../input_data/li_imx219_pitch.avi";
const string camera_matrix_path = "../input_data/li_imx219_camera_matrix.xml"; 

//Output Files
const string report_file = "../data/jetson/reports/ip7.txt";
const string results_file = "../data/jetson/reports/results.csv";
int framecount = 0;
vector<string> image_list;
VideoCapture capture(video_path);

//Function prototipes
static Mat get_next_img();
static bool readStringList( const string& filename, vector<string>& l );


int main(int argc, char** argv)
{
	/********************** open video *************************************************/
#if USE_VIDEO
	//string filename = argc > 1 ? argv[1] : video_path;

	if (!capture.isOpened()) //error in opening the video input
	{
		cerr << "Unable to open file!" << endl;
		return 0;
	}
	capture.set(CAP_PROP_POS_FRAMES, capture.get(CAP_PROP_POS_FRAMES) + INITIAL_FRAME);
#else
	readStringList(img_lst_path, image_list);
#endif

	/********************** Read camera matrix *****************************************/
	const string input_settings_file = argc > 2 ? argv[2] : camera_matrix_path;
	FileStorage fs(input_settings_file, FileStorage::READ); // Read the settings

	if (!fs.isOpened()) {
		cout << "Could not open camera matrix file: \"" << input_settings_file << "\"" << endl;
		return -1;
	}

	Mat camera_matrix, dist_coeffs;
	fs["camera_matrix"] >> camera_matrix;
	fs["distortion_coefficients"] >> dist_coeffs;

	/************************ Initializations *****************************************/
	// Create some random colors
	vector<Scalar> colors;
	RNG rng;
	for (int i = 0; i < N_FEATURES; i++) {
		int r = rng.uniform(0, 256);
		int g = rng.uniform(0, 256);
		int b = rng.uniform(0, 256);
		colors.push_back(Scalar(r, g, b));
	}

	Mat frames[MAX_SKIPPED_FRAMES], cap_frame;
	vector<Point2f> points[MAX_SKIPPED_FRAMES];

	// Take first frame
	Mat features_img, tracking_img, epilines_img, epilines_filtered_img;
	cap_frame = get_next_img();

	if (cap_frame.empty()) {
		printf("No image data \n");
		return -1;
	}

	undistort_n_grayscale(cap_frame, frames[0], features_img, camera_matrix, dist_coeffs);
	// Create a mask image for drawing purposes
	Mat feat_mask;
	Size frame_size(cap_frame.cols, cap_frame.rows);
	feat_mask = create_mask(frame_size, camera_matrix, dist_coeffs);

	//other initializations
	int index = 0;
	int skp_total = 0, skp_frames_klt = 0, skp_frames_pose = 0, skp_frames_ep=0;
	int m_frames_klt = 0, m_frames_pose = 0, m_frames_gft = 0, m_frames_ep=0;
	double max = 0, min = 0;
	Mat R_f = Mat::eye(3, 3, CV_64F);
	Mat T_f = Mat::zeros(3, 1, CV_64F);
	Vec3f angles(0, 0, 0), acc_angles(0, 0, 0);

	fstream log;
	log.open(results_file, fstream::trunc | fstream::out);
	log << "Pitch (x), Yaw (y), Row (z), T (x), T (y), T (z)\n";
	log.close();

	while (true)
	{
		/********************************** define features to track *******************************************/

		int last_index = (index + MAX_SKIPPED_FRAMES - 1) % MAX_SKIPPED_FRAMES;
		goodFeaturesToTrack(frames[index], points[index], N_FEATURES, 0.1, 10, feat_mask, 10, false);
		update_feature_positions(&points[last_index], &points[index], N_FEATURES, 10);
		printf("\n-------------------------------------------\ngft = %d, ", (int)points[index].size());

#if SHOW_PLOTS
		draw_features(features_img, points[index], points[index].size(), colors, CIRCLE_SIZE, DRAW_SCALE);
#endif

		if (points[index].size() < 8)
		{
			m_frames_gft++;
			//TODO: Is clearing the vectors necessary?
			if (skp_total == 0) {
				points[last_index].clear();
				cap_frame = get_next_img();
				if (cap_frame.empty()) break;
				undistort_n_grayscale(cap_frame, frames[index], features_img, camera_matrix, dist_coeffs);
			}
			else {
				points[index].clear();
				index = (index + 1) % MAX_SKIPPED_FRAMES;
				skp_total = (skp_total + MAX_SKIPPED_FRAMES - 1) % MAX_SKIPPED_FRAMES;
			}

			log_result(results_file, acc_angles, T_f);
			printf("\nNot enough good features to track, missed frame %d\n", framecount);
			continue;
		}
		//capture.set(CAP_PROP_POS_FRAMES , capture.get(CAP_PROP_POS_FRAMES) + STEP);
		cap_frame = get_next_img();
		if (cap_frame.empty())
			break;

		/************** calculate optical flow with lucas-kanade ****************************************************/

		int new_index = (index + skp_total + 1) % MAX_SKIPPED_FRAMES;
		undistort_n_grayscale(cap_frame, frames[new_index], features_img, camera_matrix, dist_coeffs);
		features_img.copyTo(tracking_img);
		features_img.copyTo(epilines_img);
		features_img.copyTo(epilines_filtered_img);
		vector<uchar> status;
		vector<float> err;
		TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 30, 0.01);
		//OPTFLOW_LK_GET_MIN_EIGENVALS
		calcOpticalFlowPyrLK(frames[index], frames[new_index], points[index], points[new_index],
			status, err, Size(15, 15), 3, criteria, 0, .1e-3);

		//getting rid of points for which the KLT tracking failed or those who have gone outside the frame
		check_klt(&points[index], &points[new_index], status, MAX_DIR_ANGLE);

#if SHOW_PLOTS
		draw_tracking(tracking_img, points[index], points[new_index], points[index].size(), colors,
			CIRCLE_SIZE, LINE_SIZE, DRAW_SCALE);
#endif

		/************** filter some points with epipolar lines *******************************************************/

		printf("klt = %d, ", (int)points[new_index].size());
		if (points[index].size() < 8)
		{
			skp_frames_klt++;
			skp_total++;
			printf("\nNot enough tracked features with LK, curr skipped %d, skipped klt %d, current frame %d\n",
				skp_total, skp_frames_klt, framecount);
			if (skp_total >= MAX_SKIPPED_FRAMES - 2) {
				m_frames_klt++;
				skp_total--;;
				index = (index + 1) % MAX_SKIPPED_FRAMES;
				int total = m_frames_ep + m_frames_gft + m_frames_klt + m_frames_pose;
				printf("Exceded max number of skipped frames, missing last, total %d\n", total);
			}
			log_result(results_file, acc_angles, T_f);
			continue;
		}

		// Example. Estimation of fundamental matrix using the RANSAC algorithm
		vector<Vec3f> epilines2;
		Mat F = findFundamentalMat(points[index], points[new_index], FM_8POINT, 0.999);
		computeCorrespondEpilines(points[new_index], 2, F, epilines2);

#if SHOW_PLOTS
		draw_epilines(epilines_img, &points[new_index], &epilines2, points[new_index].size(),
			colors, CIRCLE_SIZE, LINE_SIZE, DRAW_SCALE);
#endif

		check_epilines(&points[index], &points[new_index], &epilines2, MAX_DISTANCE);

#if SHOW_PLOTS
		draw_epilines(epilines_filtered_img, &points[new_index], &epilines2, points[new_index].size(),
			colors, CIRCLE_SIZE, LINE_SIZE, DRAW_SCALE, "Epilines filtered with distance");
#endif
		/******************** recover pose ***************************************************************************/

		bool keep_frame = false, is_pose = true;
		printf("epipolar = %d\n", (int)points[new_index].size());
		if (points[new_index].size() > 5)
		{
			Mat R, T, mask;
			Mat E = findEssentialMat(points[index], points[new_index], camera_matrix, RANSAC, 0.999, 2, noArray());
			recoverPose(E, points[index], points[new_index], camera_matrix, R, T, mask);

			//Angle conversion and selection based on sign change
			angles = rotationMatrixToEulerAngles(R) * (180 / PI);
			acc_angles = rotationMatrixToEulerAngles(R_f) * (180 / PI);
			
			if (fabs(angles[0]) < MAX_ROTATION && fabs(angles[1]) < MAX_ROTATION
				&& fabs(angles[2]) < MAX_ROTATION) {
				R_f = R * R_f;
				T_f = T_f + (R_f * T);
				index = (index + skp_total + 1) % MAX_SKIPPED_FRAMES;
				skp_total = 0;
			}
			else {
				skp_frames_pose++;
				keep_frame = true;
				printf("\nNoisy result. curr skipped %d, skp pose = %d, current frame = %d\n",
					skp_total + 1, skp_frames_pose, framecount);
			}
		
			printf("\nEstimated current R:\t\t\tAccumulated estimation of R:\n");
			print_rot_matrix(R, R_f);
			printf("\nAngle: (x = pitch, y = yaw, z = roll)\n");
			print_vector(angles, acc_angles);
		}
		else
		{
			skp_frames_ep++;
			keep_frame = true;
			is_pose = false;
			printf("\nNot enough features to recover essential matrix\n");
			printf("curr skipped% d, skipped ep% d, curr frame% d\n", skp_total + 1, skp_frames_ep, framecount);
		}

		if (keep_frame) {
			skp_total++;
			if (skp_total > MAX_SKIPPED_FRAMES - 2) {
				if (is_pose) m_frames_pose++;
				else m_frames_ep++;
				skp_total--;
				index = (index + 1) % MAX_SKIPPED_FRAMES;
				int total = m_frames_ep + m_frames_gft + m_frames_klt + m_frames_pose;
				printf("Exceded max number of skipped frames, missing last, total %d\n", total);
			}
		}
		else {
			if (acc_angles.val[0] < min)
				min = acc_angles.val[0];
			if (acc_angles.val[0] > max)
				max = acc_angles.val[0];
		}
		log_result(results_file, acc_angles, T_f);

#if SHOW_PLOTS
		//if (!is_pose && keep_frame)
		char key = char(waitKey(0));
#endif
	}

	int m_frames = m_frames_gft + m_frames_klt + m_frames_pose + m_frames_ep;
	double perc = 100 * m_frames / framecount;
	printf("m_frames = %d, klt_frames = %d, pose_frames = %d\npmissed = %f, n_frames = %d\n", m_frames, 
		m_frames_klt, m_frames_pose, perc, framecount);
	printf("min = %f, max = %f\n", min, max);

	//TODO: Consider creating a logger file
	log.open(report_file, fstream::app);
	log << "\n----------------------------------\nMAX_DISTANCE\t\t" << MAX_DISTANCE << "\n";
	log << "MAX_DIR_ANGLE\t\t" << MAX_DIR_ANGLE << "\n";
	log << "N_FEATURES\t\t" << N_FEATURES << "\n";
	log << "MAX SKIPPED\t\t" << MAX_SKIPPED_FRAMES;
	log << "\nMAX ROTATION\t\t" << MAX_ROTATION << "\n\n";
	Vec3f angle = rotationMatrixToEulerAngles(R_f) * (180/PI);
	log << angle.val[0] << " " << angle.val[1] << " " << angle.val[2];
	log << "\nskp klt = " << skp_frames_klt << "\t\tskp pose = " << skp_frames_pose << "\tskp ep = " << skp_frames_ep;
	log << "\nmissed gft = " << m_frames_gft << "\tmissed klt = " << m_frames_klt;
	log << "\nmissed pose = " << m_frames_pose << "\t\tmissed ep = " << m_frames_ep;
	log << "\ntotal_missed = "<<  m_frames << "\t\tperc = " << perc << "%\tn_frames = " << framecount << "\n";
	log << "min = " << min << "\tmax = " << max << "\n";
	log.close();

#if SHOW_PLOTS
		waitKey(0);
#endif
}

static Mat get_next_img()
{
	Mat result;
	#if USE_VIDEO
		capture >> result;
	#else
		result = imread(image_list[framecount]);
	#endif

	framecount++;
	return result;
}

static bool readStringList( const string& filename, vector<string>& l )
{
	l.clear();
	FileStorage fs(filename, FileStorage::READ);
	if( !fs.isOpened() )
		return false;
	FileNode n = fs.getFirstTopLevelNode();
	if( n.type() != FileNode::SEQ )
		return false;
	FileNodeIterator it = n.begin(), it_end = n.end();
	for( ; it != it_end; ++it )
		l.push_back((string)*it);
	return true;
}

