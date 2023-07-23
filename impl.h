#pragma once
#include "./utils/all_utils.h"
#include "consts.h"
#include <vector>
using std::vector;

struct FrameInfo {
   public:
	Buffer2D<Color> m_beauty;
	Buffer2D<float> m_depth;
	Buffer2D<Vec3> m_normal;
	Buffer2D<Vec3> m_position;
};

class Impl{
public:
	template<typename T>
	using Img_vec = vector<Buffer2D<T>>;
    Impl(const FrameInfo &first_frame);
	Buffer2D<Color> 		down_sample(const Buffer2D<Color> &image,
								const float scale);
	Img_vec<float> 			dist_kernel(const Buffer2D<Color> &image);
	Img_vec<float> 			blur_kernel(const Img_vec<float> &dist_output);
	Buffer2D<Vec2> 			merge_kernel(const Img_vec<float> &blur_output);
	Buffer2D<Color>         reproject_kernel(const Buffer2D<Vec2> &shift_vec);
    Buffer2D<Color>         process_img(const Buffer2D<Color> &image);
	FrameInfo pre_frame;
	Buffer2D<Color> acc_color;	// color accumulated in TAA process
	vector<Ivec2> single_layer_shift_vectors;
	bool is_first_frame;
};