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
	using vec_of_img_t = vector<Buffer2D<T>>;
	template <typename T>
	using img_of_vec_t = Buffer2D<vector<T>>;
	struct BufferInOnePass {
		bool is_first_frame;
		Buffer2D<Color> scale_img;
		vec_of_img_t<float> dist_kernel;
		vec_of_img_t<float> blur_kernel;
		Buffer2D<Vec3> merge_kernel_integer;
		Buffer2D<Vec3> merge_kernel_subpixel;
		Buffer2D<Vec2> overall_shiftv;
		Buffer2D<Color> reproject_kernel;
		Buffer2D<float> final_alpha;
	};

	using output_t = array<BufferInOnePass, HIER_LEVEL>;

	Impl(const FrameInfo &first_frame);
	template <typename T>
	static Buffer2D<T> scale_img_bilinear(const Buffer2D<T> &image, const float scale) {
		CHECK(scale > 0);
		if (scale == 1) return image;
		int w = image.m_width, h = image.m_height;
		int nw = round(w * scale), nh = round(h * scale);
		Buffer2D<T> ret = CreateBuffer2D<T>(nw, nh);
#pragma omp parallel for
		for (int i = 0; i < nw; i++) {
			for (int j = 0; j < nh; j++) {
				float target_x = i / scale + .5, target_y = j / scale + .5;
				int mnx = floor(target_x), mny = floor(target_y);
				int mxx = ceil(target_x), mxy = ceil(target_y);
				mnx = max(0, mnx), mny = max(0, mny);
				mxx = min(w - 1, mxx), mxy = min(h - 1, mxy);
				// dbg(target_x, target_y, mnx, mny, mxx, mxy);
				T c_x0y0 = image(mnx, mny), c_x1y0 = image(mxx, mny);
				T c_x0y1 = image(mnx, mxy), c_x1y1 = image(mxx, mxy);
				float tx = (target_x - mnx);
				float ty = (target_y - mny);
				// dbg(i, j, mnx, mxx, mny, mxy, tx, ty);
				ret(i, j) = lerp2d(c_x0y0, c_x1y0, c_x0y1, c_x1y1, tx, ty);
			}
		}
		return ret;
	}
	

	template <typename T>
	static Buffer2D<T> scale_img_ave(const Buffer2D<T> &image, const float scale){
		CHECK(scale > 0);
		int w = image.m_width, h = image.m_height;
		int nw = round(w * scale), nh = round(h * scale);
		auto ret = CreateBuffer2D<T>(nw, nh);
		if (scale > 1) {
			for (int i = 0; i < nw; i++){
				for (int j = 0; j < nh; j++){
					ret(i, j) = image(int(i / scale), int(j / scale));
				}
			}
		} else {
			CHECK(image.m_width % int(1 / scale) == 0 && image.m_height % int(1 / scale) == 0)
			CHECK(int(1 / scale) == (1 / scale));
			int inv_sc = int(1 / scale);
			for (int i = 0; i < nw; i++){
				for (int j = 0; j < nh; j++){
					for (int ii = 0; ii < inv_sc; ii++){
						for (int jj = 0; jj < inv_sc; jj++){
							ret(i, j) += image(int(i * inv_sc + ii), int(j * inv_sc + jj));
						}
					}
					ret(i, j) *= scale * scale;
				}
			}
		}
		return ret;
	}
	

	Buffer2D<Vec3> filter_kernel(const FrameInfo &frame);

	vec_of_img_t<float> dist_kernel(const Buffer2D<Color> &image,
							      const Buffer2D<Color> &prev_image,
								  const Buffer2D<Vec2> &base_shiftv);
	static vec_of_img_t<float> blur_kernel(const vec_of_img_t<float> &dist_output);

	Buffer2D<Vec3> merge_kernel_integer(const vec_of_img_t<float> &blur_output,
										const Buffer2D<Vec2> &base_shiftv);
	// for each pixel, vec3 => x: shift_vec.x (integer) y: shift_vec.y (integer)
	// z: dist (float)

	Buffer2D<Vec3> merge_kernel_subpixel(
		const vec_of_img_t<float> &blur_output,
		const Buffer2D<Vec2> &base_shiftv,
		const Buffer2D<Vec3> &merge_int_output);
	// for each pixel, vec3 => x: shift_vec.x (float) y: shift_vec.y (float) z:
	// dist (float)

	pair<Buffer2D<Color>, Buffer2D<float>> reproject_kernel(const Buffer2D<Vec3> &shift_vec_and_dist,
									 const Buffer2D<Color> &image, 
									 const Buffer2D<Color> &prev_image
									 );

	array<BufferInOnePass, HIER_LEVEL> process_img(const FrameInfo &frame);
	FrameInfo pre_frame;
	array<Buffer2D<Color>, HIER_LEVEL> acc_color;	// color accumulated in TAA process
	vector<Ivec2> single_layer_shift_vecs;
	bool is_first_frame;
};