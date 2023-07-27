#pragma once
#include <vector>

#include "./utils/all_utils.h"
#include "consts.h"
using std::vector;

struct FrameInfo {
   public:
	Buffer2D<Color> m_beauty;
	Buffer2D<float> m_depth;
	Buffer2D<Vec3> m_normal;
	Buffer2D<Vec3> m_position;
};

class Impl {
   public:
	template <typename T>
	using vec_of_img = vector<Buffer2D<T>>;
	template <typename T>
	using img_of_vec = Buffer2D<vector<T>>;
	struct BufferInOnePass {
		bool is_first_frame;
		Buffer2D<Color> scale_img;
		vec_of_img<float> dist_kernel;
		vec_of_img<float> blur_kernel;
		Buffer2D<Vec3> merge_kernel_integer;
		Buffer2D<Vec3> merge_kernel_subpixel;
		Buffer2D<Color> reproject_kernel;
	};

	Impl(const FrameInfo &first_frame);
	static Buffer2D<Color> scale_img(const Buffer2D<Color> &image, const float scale);
	vec_of_img<float> dist_kernel(const Buffer2D<Color> &image, const Buffer2D<Vec2> &base_shiftv);
	static vec_of_img<float> blur_kernel(const vec_of_img<float> &dist_output);

	Buffer2D<Vec3> merge_kernel_integer(const vec_of_img<float> &blur_output,
										const Buffer2D<Vec2> &base_shiftv);
	// for each pixel, vec3 => x: shift_vec.x (integer) y: shift_vec.y (integer)
	// z: dist (float)


	Buffer2D<Vec3> merge_kernel_subpixel(
		const vec_of_img<float> &blur_output,
		const Buffer2D<Vec3> &merge_int_output);
	// for each pixel, vec3 => x: shift_vec.x (float) y: shift_vec.y (float) z:
	// dist (float)
	
	Buffer2D<Color> reproject_kernel(const Buffer2D<Vec3> &shift_vec_and_dist,
									 const Buffer2D<Color> &image);
	
	
	array<BufferInOnePass, HIER_LEVEL> process_img(const FrameInfo &frame);
	FrameInfo pre_frame;
	Buffer2D<Color> acc_color;	// color accumulated in TAA process
	vector<Ivec2> single_layer_shift_vecs;
	bool is_first_frame;
};