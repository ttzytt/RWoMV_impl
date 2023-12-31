#pragma once
#include <bits/stdc++.h>

#include <Eigen/Dense>

#include "ext/common.h"
#include "ext/mathutil.h"
using std::array;
using std::clamp;
using std::cout;
using std::function;
using std::max;
using std::min;
using std::ostream;
using std::vector;
template <typename T>
T lerp(const T &a, const T &b, const float &t) {
	return a * (1.0f - t) + b * t;
}

// template <typename T>
// T clamp01(const T &v, const float &t_mn = 0, const float &t_mx = 1) {
// 	T mn = t_mn * v, mx = t_mx * v;
// 	return clamp(v, mn, mx);
// }

template <typename T>
T lerp2d(const T &q1, const T &q2, const T &q3, const T &q4, float tx,
		 float ty) {
	// q1~q4 are ordered in y then x
	return lerp(lerp(q1, q2, tx), lerp(q3, q4, tx), ty);
}

template <typename T>
T calc_sqrlen(const T &a, const T &b) {
	return (a - b) * (a - b);
}

template <typename T>
T calc_var(const std::vector<T> &vec, const T &ave) {
	T tmp;
	for (auto &v : vec) {
		tmp += calc_sqrlen(v, ave);
	}
	return SafeSqrt(tmp / vec.size());
}

template <typename T>
T calc_ave(const std::vector<T> &vec) {
	T sum;
	for (auto &v : vec) {
		sum += v;
	}
	return sum / (float)vec.size();
}

inline float rand_float() {
	static std::random_device rd;
	static std::mt19937 gen(rd());
	static std::uniform_real_distribution<float> dis(0, 1);
	return dis(gen);
}

template <typename T>
class Tvec2 {
   public:
	T x, y;
	Tvec2(const T &x, const T &y) : x(x), y(y) {}
	explicit Tvec2(const T &v = 0) : x(v), y(v) {}
	Tvec2 operator+(const Tvec2 &v) const { return Tvec2(x + v.x, y + v.y); }
	Tvec2 &operator+=(const Tvec2 &v) {
		x += v.x;
		y += v.y;
		return *this;
	}
	Tvec2 operator-(const Tvec2 &v) const { return Tvec2(x - v.x, y - v.y); }
	Tvec2 operator-() const { return Tvec2(-x, -y); }
	Tvec2 &operator-=(const Tvec2 &v) { return *this += -v; }
	Tvec2 operator*(const T &v) const { return Tvec2(x * v, y * v); }
	Tvec2 &operator*=(const T &v) {
		x *= v;
		y *= v;
		return *this;
	}
	Tvec2 operator/(const T &v) const {
		CHECK(v != 0.0);
		float inv = 1.0f / v;
		return Tvec2(x * inv, y * inv);
	}
	Tvec2 &operator/=(const T &v) {
		CHECK(v != 0.0);
		T inv = 1.0f / v;
		x *= inv;
		y *= inv;
		return *this;
	}

	bool operator==(const Tvec2 &v) const { return x == v.x && y == v.y; }
	bool operator!=(const Tvec2 &v) const { return !(*this == v); }
	bool operator<(const Tvec2 &v) const { return x < v.x && y < v.y; }
	bool operator>(const Tvec2 &v) const { return x > v.x && y > v.y; }
	bool operator<=(const Tvec2 &v) const { return x <= v.x && y <= v.y; }
	bool operator>=(const Tvec2 &v) const { return x >= v.x && y >= v.y; }

	ostream &operator<<(ostream &os) const {
		os << "(" << x << ", " << y << ")";
		return os;
	}

	T len() { return sqrt(x * x + y * y); }
	T sqr_len() { return x * x + y * y; }

	Tvec2 unit() { return *this / len(); }

	static Tvec2 rand_unit() {
		return Tvec2(rand_float(), rand_float()).unit();
	}

	Vec3 to3() const { return Vec3(x, y, 0); }
	template <typename U>
	operator Tvec2<U>() const {
		return Tvec2<U>(static_cast<U>(x), static_cast<U>(y));
	}

	static Tvec2 min(const Tvec2 &a, const Tvec2 &b) {
		return Tvec2(std::min(a.x, b.x), std::min(a.y, b.y));
	}
	static Tvec2 max(const Tvec2 &a, const Tvec2 &b) {
		return Tvec2(std::max(a.x, b.x), std::max(a.y, b.y));
	}
};

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const Tvec2<T> &v) {
	os << "(" << v.x << ", " << v.y << ")";
	return os;
}

using Vec2 = Tvec2<float>;
using Ivec2 = Tvec2<int>;


inline float guassian(int x, int y, float sigma) {
	float coeff = 1.0 / (2 * M_PI * sigma * sigma);
	return coeff * exp(-(x * x + y * y) / (2 * sigma * sigma));
};

inline array<float, 6> quadric_fit(vector<Vec3> pts, vector<float> wts) {
	using namespace Eigen;
	CHECK(pts.size() == wts.size());
	MatrixXf A(MatrixXf::Zero(6, 6));
	float phis[6][pts.size()];	// φ_i(p_j)
	for (int i = 0; i < pts.size(); i++) {
		auto [x, y, _] = pts[i];
		phis[0][i] = x * x;
		phis[1][i] = y * y;
		phis[2][i] = x * y;
		phis[3][i] = x;
		phis[4][i] = y;
		phis[5][i] = 1;
	}

#pragma omp parallel for
	for (int par_deriv = 0; par_deriv < 6; par_deriv++) {
		// 6 equations for 6 partial derivatives because there are 6
		// parameters

		for (int quadric_term = 0; quadric_term < 6; quadric_term++) {
			// for different terms in the quadric equation
			for (int data_idx = 0; data_idx < pts.size(); data_idx++) {
				// these two terms are for calculating each entries of
				// the matrix entry = sum(w_data_idx *
				// φ_quadrict(p_data_idx) * φ_quadrict(p_data_idx))
				A(par_deriv, quadric_term) += wts[data_idx] *
											  phis[quadric_term][data_idx] *
											  phis[par_deriv][data_idx];
			}
		}
	}

	VectorXf b(VectorXf::Zero(6));
	for (int eq_idx = 0; eq_idx < 6; eq_idx++) {
		for (int data_idx = 0; data_idx < pts.size(); data_idx++) {
			b(eq_idx) +=
				wts[data_idx] * pts[data_idx].z * phis[eq_idx][data_idx];
			// pts.z <=> f(p)
		}
	}
	VectorXf quadric_params = A.fullPivHouseholderQr().solve(b);
	array<float, 6> ret;
	for (int i = 0; i < 6; i++) ret[i] = quadric_params(i);
	// assert(A * quadric_params == b);	// check if the solution is correct
	return ret;
};

inline float quadric_eval(const array<float, 6> &params, const Vec2 &pt) {
	auto [x, y] = pt;
	float a = params[0], b = params[1], c = params[2], d = params[3],
		  e = params[4], f = params[5];
	return a * x * x + b * y * y + c * x * y + d * x + e * y + f;
};

inline Vec2 two_d_deriv(const function<float(Vec2)> &func, const Vec2 &pt,
						float eps = 1e-4) {
	// 2d derivative
	float x = pt.x, y = pt.y;
	float dx = (func(Vec2(x + eps, y)) - func(Vec2(x - eps, y))) / (2 * eps);
	float dy = (func(Vec2(x, y + eps)) - func(Vec2(x, y - eps))) / (2 * eps);
	return Vec2(dx, dy);
}

inline Vec3 two_d_grad_descent(const function<float(Vec2)> &func,
							   const Vec2 &mn, const Vec2 &mx, float step,
							   int iter) {
	// gradient descent in 2d
	Vec2 cur_pt = (mn + mx) / 2;
	Vec2 mn_pt = cur_pt;
	float mn_val = func(cur_pt);
	while (iter--) {
		auto [dx, dy] = two_d_deriv(func, cur_pt);
		cur_pt -= Vec2(dx, dy) * step;
		if (cur_pt.x > mx.x || cur_pt.y > mx.y || cur_pt.x < mn.x || cur_pt.y < mn.y) {
			cur_pt = (mn + mx) / 2 + Vec2::rand_unit() * step * 5;
		}
		float cur_val = func(cur_pt);
		if (cur_val < mn_val) {
			mn_val = cur_val;
			mn_pt = cur_pt;
		}
	}
	return Vec3(mn_pt.x, mn_pt.y, mn_val);
}

template <typename T>
Tvec3<T> clamp(const Tvec3<T> &v, const Tvec3<T> &mn, const Tvec3<T> &mx) {
	return Tvec3<T>(clamp(v.x, mn.x, mx.x), clamp(v.y, mn.y, mx.y),
					clamp(v.z, mn.z, mx.z));
}

template <typename T>
Tvec2<T> clamp(const Tvec2<T> &v, const Tvec2<T> &mn, const Tvec2<T> &mx) {
	return Tvec2<T>(clamp(v.x, mn.x, mx.x), clamp(v.y, mn.y, mx.y));
}