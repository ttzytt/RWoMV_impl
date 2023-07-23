#pragma once
#include "impl.h"

#include <bits/stdc++.h>

#include <Eigen/Dense>
using namespace std;

Impl::Impl(const FrameInfo &first_frame) {
	pre_frame = first_frame;
	acc_color = first_frame.m_beauty;
	is_first_frame = true;
	for (int i = -MAX_SHIFT_VEC_MAG; i <= MAX_SHIFT_VEC_MAG; i++) {
		for (int j = -MAX_SHIFT_VEC_MAG; j <= MAX_SHIFT_VEC_MAG; j++) {
			single_layer_shift_vectors.push_back(Ivec2(i, j));
		}
	}
}

Buffer2D<Color> Impl::down_sample(const Buffer2D<Color> &image,
								  const float scale) {
	CHECK(scale < 1 && scale > 0);
	int w = image.m_width, h = image.m_height;
	int nw = round(w * scale), nh = round(h * scale);
	Buffer2D<Color> ret = CreateBuffer2D<Color>(nw, nh);
	for (int i = 0; i < nw; i++) {
		for (int j = 0; j < nh; j++) {
			float target_x = i / scale, target_y = j / scale;
			int mnx = floor(target_x), mny = floor(target_y);
			int mxx = ceil(target_x), mxy = ceil(target_y);
			Color c_x0y0 = image(mnx, mny), c_x1y0 = image(mxx, mny);
			Color c_x0y1 = image(mnx, mxy), c_x1y1 = image(mxx, mxy);
			float tx = target_x / (mxx - mnx);
			float ty = target_y / (mxy - mny);
			ret(i, j) = lerp2d(c_x0y0, c_x1y0, c_x0y1, c_x1y1, tx, ty);
		}
	}
	return ret;
}

vector<Buffer2D<float>> Impl::dist_kernel(const Buffer2D<Color> &image) {
	vector<Buffer2D<float>> ret(single_layer_shift_vectors.size());
	for (int i = 0; i < single_layer_shift_vectors.size(); i++)
		ret.push_back(CreateBuffer2D<float>(image.m_width, image.m_height));
	for (int sv = 0; sv < single_layer_shift_vectors.size(); sv++) {
		auto [dx, dy] = single_layer_shift_vectors[sv];
		for (int i = 0; i < image.m_width; i++) {
			for (int j = 0; j < image.m_height; j++) {
				int ni = i + dx, nj = j + dy;
				if (ni < 0 || ni >= image.m_width || nj < 0 ||
					nj >= image.m_height) {
					ret[sv](i, j) = -1;
					continue;
				}
				Color cur_c = image(i, j), shift_c = image(ni, nj);
				ret[sv](i, j) = DIS_FUNC(cur_c, shift_c, Vec2(dx, dy));
			}
		}
	}
}

vector<Buffer2D<float>> Impl::blur_kernel(const Img_vec<float> &dist_output) {
	auto ret = Img_vec<float>();
	int w = dist_output[0].m_width, h = dist_output[0].m_height;
	for (auto &cur_dis : dist_output) {
		auto cur_ret = CreateBuffer2D<float>(w, h);
		for (int cx = 0; cx < w; cx++) {
			for (int cy = 0; cy < h; cy++) {
				float sum = 0;
				float w_sum = 0;
				for (int dx = -BLUR_KERNEL_RAD; dx <= BLUR_KERNEL_RAD; dx++) {
					for (int dy = -BLUR_KERNEL_RAD; dy <= BLUR_KERNEL_RAD;
						 dy++) {
						int nx = cx + dx, ny = cy + dy;
						if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
						if (cur_dis(nx, ny) == -1) continue;
						sum += guassian(dx, dy, BLUR_KERNEL_GUASSIAN_SIGMA) *
							   cur_dis(nx, ny);
						w_sum += guassian(dx, dy, BLUR_KERNEL_GUASSIAN_SIGMA);
					}
				}
				cur_ret(cx, cy) = sum / w_sum;
			}
		}
		ret.push_back(cur_ret);
	}
	return ret;
}

Buffer2D<Vec2> Impl::merge_kernel(const Img_vec<float> &blur_output) {
	// form of quadric surface: z = ax^2 + by^2 + cxy + dx + ey + f
	// phi refer to [x^2, y^2, xy, x, y, 1]
	// least square method:
	// https://www.bilibili.com/video/BV1Uu411d72H/?spm_id_from=333.337.search-card.all.click&vd_source=4de003ee9a3815aedd7d0cb2c7a12d14

	int w = blur_output[0].m_width, h = blur_output[0].m_height;
	Buffer2D<Vec2> ret = CreateBuffer2D<Vec2>(w, h);
	Buffer2D<Vec3> mn_pts2d = CreateBuffer2D<Vec3>(w, h);
	fill(mn_pts2d.m_buffer.get(), mn_pts2d.m_buffer.get() + mn_pts2d.m_size,
		 Vec3(-1, -1, -1));
	Buffer2D<float> mn_dis = CreateBuffer2D<float>(w, h);
	// x, y => shift vector, z => distance
	// in each pixel find the minimum distance and corresponding shift vector
	for (int cx = 0; cx < w; cx++) {
		for (int cy = 0; cy < h; cy++) {
			float min_dis = 1e9;
			Vec2 min_shift;
			for (int sv = 0; sv < single_layer_shift_vectors.size(); sv++) {
				auto [dx, dy] = single_layer_shift_vectors[sv];
				float cur_dis = blur_output[sv](cx, cy);
				if (cur_dis == -1) continue;
				if (cur_dis < min_dis) {
					min_dis = cur_dis;
					min_shift = Vec2(dx, dy);
				}
			}
			mn_pts2d(cx, cy) = Vec3(min_shift.x, min_shift.y, min_dis);
		}
	}
	for (int cx = 0; cx < w; cx++) {
		for (int cy = 0; cy < h; cy++) {
			vector<Vec3> pts;
			vector<float> wts;
			for (int dx = -MERGE_KERNEL_RAD; dx <= MERGE_KERNEL_RAD; dx++) {
				for (int dy = -MERGE_KERNEL_RAD; dy <= MERGE_KERNEL_RAD; dy++) {
					int nx = cx + dx, ny = cy + dy;
					if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
					if (mn_pts2d(nx, ny).z == -1) continue;
					pts.push_back(Vec3(dx, dy, 1));
					wts.push_back(guassian(dx, dy, MERGE_KERNEL_SIGMA));
				}
			}

			if (pts.empty()){
				ret(cx, cy) = Vec2(-1, -1);
				continue;
			}

			auto quadric_params = quadric_fit(pts, wts);
			auto _quadric_eval = [&quadric_params](const Vec2 &pt) {
				return quadric_eval(quadric_params, pt);
			};	// with parameters
			Vec2 sub_pix_shiftvec = two_d_grad_descent(
				_quadric_eval, Vec2(cx, cy) - Vec2(MERGE_KERNEL_RAD),
				Vec2(cx, cy) + Vec2(MERGE_KERNEL_RAD), MERGE_KERNEL_DESC_STEP,
				MERGE_KERNEL_DESC_ITER);
			ret(cx, cy) = sub_pix_shiftvec;
		}
	}
	return ret;
}

Buffer2D<Color> Impl::reproject_kernel(const Buffer2D<Vec2> &shift_vec) {
	
}