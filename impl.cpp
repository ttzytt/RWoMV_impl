#pragma once
#include "impl.h"
#include "utils/all_utils.h"
#include <bits/stdc++.h>
#include <dbg.h>

#include <Eigen/Dense>
using namespace std;

Impl::Impl(const FrameInfo &first_frame) {
	pre_frame = first_frame;
	acc_color = first_frame.m_beauty;
	is_first_frame = true;
	for (int i = -MAX_SHIFT_VEC_MAG; i <= MAX_SHIFT_VEC_MAG; i++) {
		for (int j = -MAX_SHIFT_VEC_MAG; j <= MAX_SHIFT_VEC_MAG; j++) {
			single_layer_shift_vecs.push_back(Ivec2(i, j));
		}
	}
}

Buffer2D<Color> Impl::scale_img(const Buffer2D<Color> &image,
								const float scale) {
	CHECK(scale > 0);
	int w = image.m_width, h = image.m_height;
	int nw = round(w * scale), nh = round(h * scale);
	Buffer2D<Color> ret = CreateBuffer2D<Color>(nw, nh);
#pragma omp parallel for
	for (int i = 0; i < nw; i++) {
		for (int j = 0; j < nh; j++) {
			float target_x = i / scale + .5, target_y = j / scale + .5;
			int mnx = floor(target_x ), mny = floor(target_y);
			int mxx = ceil(target_x), mxy = ceil(target_y);
			mnx = max(0, mnx), mny = max(0, mny);
			mxx = min(w - 1, mxx), mxy = min(h - 1, mxy);
			// dbg(target_x, target_y, mnx, mny, mxx, mxy);
			Color c_x0y0 = image(mnx, mny), c_x1y0 = image(mxx, mny);
			Color c_x0y1 = image(mnx, mxy), c_x1y1 = image(mxx, mxy);
			float tx = (target_x - mnx) / (mxx - mnx);
			float ty = (target_y - mny) / (mxy - mny);
			// dbg(i, j, mnx, mxx, mny, mxy, tx, ty);
			ret(i, j) = lerp2d(c_x0y0, c_x1y0, c_x0y1, c_x1y1, tx, ty);
		}
	}
	return ret;
}

vector<Buffer2D<float>> Impl::dist_kernel(const Buffer2D<Color> &image,
										  const Buffer2D<Vec2> &base_shiftv) {
	vector<Buffer2D<float>> ret(single_layer_shift_vecs.size());
	for (int i = 0; i < single_layer_shift_vecs.size(); i++)
		ret[i] = CreateBuffer2D<float>(image.m_width, image.m_height);
#pragma omp parallel for
	for (int sv = 0; sv < single_layer_shift_vecs.size(); sv++) {
		auto [dx, dy] = single_layer_shift_vecs[sv];
		for (int i = 0; i < image.m_width; i++) {
			for (int j = 0; j < image.m_height; j++) {
				auto [ni, nj] = base_shiftv(i, j);
				ni += dx + i, nj += dy + j;
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
	return ret;
}

vector<Buffer2D<float>> Impl::blur_kernel(
	const vec_of_img<float> &dist_output) {
	auto ret = vec_of_img<float>();
	int w = dist_output[0].m_width, h = dist_output[0].m_height;
#pragma omp parallel for
	for (auto &cur_dis : dist_output) {
		auto cur_ret = CreateBuffer2D<float>(w, h);
		fill(cur_ret.m_buffer.get(), cur_ret.m_buffer.get() + cur_ret.m_size,
			 -1);
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

Buffer2D<Vec3> Impl::merge_kernel_integer(const vec_of_img<float> &blur_output,
										  const Buffer2D<Vec2> &base_shiftv) {
	int w = blur_output[0].m_width, h = blur_output[0].m_height;
	Buffer2D<Vec3> ret = CreateBuffer2D<Vec3>(w, h);
	fill(ret.m_buffer.get(), ret.m_buffer.get() + ret.m_size, Vec3(-1, -1, -1));
	// x, y => shift vector, z => distance
	// in each pixel find the minimum distance and corresponding shift vector
#pragma omp parallel for
	for (int cx = 0; cx < w; cx++) {
		for (int cy = 0; cy < h; cy++) {
			float min_dis = 1e9;
			Vec2 min_shift;
			for (int sv = 0; sv < single_layer_shift_vecs.size(); sv++) {
				auto [dx, dy] =
					single_layer_shift_vecs[sv] + base_shiftv(cx, cy);
				float cur_dis = blur_output[sv](cx, cy);
				if (cur_dis == -1) continue;
				if (cur_dis < min_dis) {
					min_dis = cur_dis;
					min_shift = Vec2(dx, dy);
				}
			}
			ret(cx, cy) = Vec3(min_shift.x, min_shift.y, min_dis);
		}
	}
	return ret;
}

Buffer2D<Vec3> Impl::merge_kernel_subpixel(
	const vec_of_img<float> &blur_output,
	const Buffer2D<Vec3> &merge_int_output) {
	// form of quadric surface: z = ax^2 + by^2 + cxy + dx + ey + f
	// phi refer to [x^2, y^2, xy, x, y, 1]
	// least square method:
	// https://www.bilibili.com/video/BV1Uu411d72H/?spm_id_from=333.337.search-card.all.click&vd_source=4de003ee9a3815aedd7d0cb2c7a12d14

	int w = blur_output[0].m_width, h = blur_output[0].m_height;
	Buffer2D<Vec3> ret = CreateBuffer2D<Vec3>(w, h);
#pragma omp parallel for
	for (int cx = 0; cx < w; cx++) {
		for (int cy = 0; cy < h; cy++) {
			vector<Vec3> pts;
			vector<float> wts;
			for (int dx = -MERGE_KERNEL_RAD; dx <= MERGE_KERNEL_RAD; dx++) {
				for (int dy = -MERGE_KERNEL_RAD; dy <= MERGE_KERNEL_RAD; dy++) {
					int nx = cx + dx, ny = cy + dy;
					if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
					if (merge_int_output(nx, ny).z == -1) continue;
					pts.push_back(merge_int_output(nx, ny));
					wts.push_back(guassian(dx, dy, MERGE_KERNEL_SIGMA));
				}
			}
			if (pts.empty()) {
				ret(cx, cy) = Vec3(-1, -1, -1);
				continue;
			}

			auto quadric_params = quadric_fit(pts, wts);
			auto _quadric_eval = [&quadric_params](const Vec2 &pt) {
				return quadric_eval(quadric_params, pt);
			};	// with parameters
			Vec3 sub_pix_shiftvec_and_val = two_d_grad_descent(
				_quadric_eval, Vec2(cx, cy) - Vec2(MERGE_KERNEL_RAD),
				Vec2(cx, cy) + Vec2(MERGE_KERNEL_RAD), MERGE_KERNEL_DESC_STEP,
				MERGE_KERNEL_DESC_ITER);
			ret(cx, cy) = sub_pix_shiftvec_and_val;
		}
	}
	return ret;
}

Buffer2D<Color> Impl::reproject_kernel(const Buffer2D<Vec3> &shift_vec_and_dist,
									   const Buffer2D<Color> &image) {
	CHECK(shift_vec_and_dist.m_width == image.m_width &&
		  shift_vec_and_dist.m_height == image.m_height);

	Buffer2D<Color> ret = CreateBuffer2D<Color>(image.m_width, image.m_height);

	int w = shift_vec_and_dist.m_width, h = shift_vec_and_dist.m_height;
#pragma omp parallel for
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < h; j++) {
			auto cur_vdis = shift_vec_and_dist(i, j);
			int dist = cur_vdis.z;
			float m = REJECT_KAPPA * dist - REJECT_ETA;
			float alpha_p = clamp01(BLEND_ALPHA * (1 - m));
			if (cur_vdis.x == -1) {
				alpha_p = 0;
			}
			const Vec2 &cur_shift = Vec2(cur_vdis.x, cur_vdis.y);
			const Color &pre_acc = acc_color(i + cur_shift.x, j + cur_shift.y);
			ret(i, j) = lerp(pre_acc, image(i, j), alpha_p);
		}
	}
	return ret;
}

array<Impl::BufferInOnePass, HIER_LEVEL> Impl::process_img(
	const FrameInfo &frame) {
	auto discard_zcomp = [](const Buffer2D<Vec3> &buffer) {
		auto ret = CreateBuffer2D<Vec2>(buffer.m_width, buffer.m_height);
		for (int i = 0; i < buffer.m_width; i++) {
			for (int j = 0; j < buffer.m_height; j++) {
				ret(i, j) = Vec2(buffer(i, j).x, buffer(i, j).y);
			}
		}
		return ret;
	};

	array<BufferInOnePass, HIER_LEVEL> ret;
	if (is_first_frame) {
		is_first_frame = false;
		BufferInOnePass tmp;
		tmp.reproject_kernel = frame.m_beauty;
		tmp.is_first_frame = true;
		ret.fill(tmp);
		return ret;
	}
	auto &image = frame.m_beauty;
	float sc = 1.0 / pow(HIER_REDUC_FACTOR, HIER_LEVEL - 1);
	for (int i = HIER_LEVEL - 1; i >= 0; i--, sc *= HIER_REDUC_FACTOR) {
		auto &cur_pass = ret[i];
		if (i == 0) {
			cur_pass.is_first_frame = false;
			cur_pass.scale_img = image;
		} else {   
			cur_pass.scale_img.Copy(scale_img(image, sc));
		}
		Buffer2D<Vec2> base_shift;
		if (i != HIER_LEVEL - 1) {
			base_shift = discard_zcomp(ret[i + 1].merge_kernel_subpixel);
		} else {
			base_shift = CreateBuffer2D<Vec2>(image.m_width, image.m_height);
		}
		cur_pass.dist_kernel = dist_kernel(cur_pass.scale_img, base_shift);
		cur_pass.blur_kernel = blur_kernel(cur_pass.dist_kernel);
		cur_pass.merge_kernel_integer =
			merge_kernel_integer(cur_pass.blur_kernel, base_shift);
		cur_pass.merge_kernel_subpixel = merge_kernel_subpixel(
			cur_pass.blur_kernel, cur_pass.merge_kernel_integer);
		cur_pass.reproject_kernel =
			reproject_kernel(cur_pass.merge_kernel_subpixel, cur_pass.scale_img);
	}
}