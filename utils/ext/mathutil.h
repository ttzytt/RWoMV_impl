#pragma once

#include <cmath>
#include <cstring>
#include <iostream>
#include "./common.h"
using std::cout;
using std::ostream;

inline float Sqr(const float &v) { return v * v; }
inline float SafeSqrt(const float &v) { return std::sqrt(std::max(v, 0.f)); }
inline float SafeAcos(const float &v) {
    return std::acos(std::min(std::max(v, 0.f), 1.f));
}


template <typename T>
class Tvec3 {
  public:
    enum EType { Vector, Point };
	explicit Tvec3(const T &v = 0) : x(v), y(v), z(v) {}
	Tvec3(const T &_x, const T &_y, const T &_z) : x(_x), y(_y), z(_z) {}
	Tvec3 operator+(const Tvec3 &v) const { return Tvec3(x + v.x, y + v.y, z + v.z); }
    Tvec3 operator-(const Tvec3 &v) const { return Tvec3(x - v.x, y - v.y, z - v.z); }
    Tvec3 &operator+=(const Tvec3 &v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    Tvec3 operator*(const float &v) const { return Tvec3(x * v, y * v, z * v); }
    Tvec3 operator*=(const float &v) {
        x *= v;
        y *= v;
        z *= v;
        return *this;
    }
    Tvec3 operator*(const Tvec3 &v) const { return Tvec3(x * v.x, y * v.y, z * v.z); }
    Tvec3 operator/(const float &v) const {
        CHECK(v != 0.0);
		T inv = 1.0f / v;
		return Tvec3(x * inv, y * inv, z * inv);
    }
    Tvec3 operator/(const Tvec3 &v) const {
        CHECK(v.x != 0.0);
        CHECK(v.y != 0.0);
        CHECK(v.z != 0.0);
		return Tvec3(x / v.x, y / v.y, z / v.z);
    }
    Tvec3 &operator/=(const float &v) {
        CHECK(v != 0.0);
        float inv = 1.0f / v;
        x *= inv;
        y *= inv;
        z *= inv;
        return *this;
    }

    bool operator==(const Tvec3<T> other){
        return x == other.x && y == other.y && z == other.z;
    }
    
    ostream &operator<<(ostream &os) {
        os << "(" << x << ", " << y << ", " << z << ")";
        return os;
    }

    float x, y, z;
};

using Float3 = Tvec3<float>;

template <typename T>
inline Tvec3<T> Min(const Tvec3<T> &a, const Tvec3<T> &b) {
    return Tvec3<T>(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}

template <typename T>
inline Tvec3<T> Max(const Tvec3<T> &a, const Tvec3<T> &b) {
	return Tvec3<T>(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

template <typename T>
inline float Dot(const Tvec3<T> &a, const Tvec3<T> &b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <typename T>
inline Tvec3<T> Cross(const Tvec3<T> &a, const Tvec3<T> &b) {
	return Tvec3<T>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                  a.x * b.y - a.y * b.x);
}
template <typename T>
inline float AbsSum(const Tvec3<T> &a, const Tvec3<T> &b) {
	return std::fabs(a.x - b.x) + std::fabs(a.y - b.y) + std::fabs(a.z - b.z);
}

template <typename T>
inline Tvec3<T> Abs(const Tvec3<T> &a) {
	return Tvec3<T>(std::fabs(a.x), std::fabs(a.y), std::fabs(a.z));
}
template <typename T>
inline Tvec3<T> Sqr(const Tvec3<T> &a) { return Tvec3<T>(Sqr(a.x), Sqr(a.y), Sqr(a.z)); }
template <typename T>
inline Tvec3<T> SafeSqrt(const Tvec3<T> &a) {
    return Tvec3<T>(SafeSqrt(a.x), SafeSqrt(a.y), SafeSqrt(a.z));
}

// (1 - s) * u + s * v
template <typename T>
inline Tvec3<T> Lerp(const Tvec3<T> &u, const Tvec3<T> &v, const float &s) {
    return u + (v - u) * s;
}
template <typename T>
inline Tvec3<T> Clamp(const Tvec3<T> &v, const Tvec3<T> &l, const Tvec3<T> &r) {
    return Min(Max(v, l), r);
}
template <typename T>
inline float SqrLength(const Tvec3<T> &a) {
	return Sqr(a.x) + Sqr(a.y) + Sqr(a.z);
}

template <typename T>
inline float Length(const Tvec3<T> &a) {
	return std::sqrt(Sqr(a.x) + Sqr(a.y) + Sqr(a.z));
}
template <typename T>
inline Tvec3<T> Normalize(const Tvec3<T> &a) { return a / Length(a); }

template <typename T>
inline float SqrDistance(const Tvec3<T> &a, const Tvec3<T> &b) {
	return SqrLength(a - b);
}
template <typename T>
inline float Distance(const Tvec3<T> &a, const Tvec3<T> &b) {
	return Length(a - b);
}
template <typename T>
inline float Luminance(const Tvec3<T> &rgb) {
    return Dot(rgb, Tvec3<T>(0.2126f, 0.7152f, 0.0722f));
}
template <typename T>
inline Tvec3<T> RGB2YCoCg(const Tvec3<T> &RGB) {
	float Co = RGB.x - RGB.z;
    float tmp = RGB.z + Co / 2;
    float Cg = RGB.y - tmp;
    float Y = tmp + Cg / 2;
    return Tvec3<T>(Y, Co, Cg);
}
template <typename T>
inline Tvec3<T> YCoCg2RGB(const Tvec3<T> &YCoCg) {
	float tmp = YCoCg.x - YCoCg.z / 2;
    float G = YCoCg.z + tmp;
    float B = tmp - YCoCg.y / 2;
    float R = B + YCoCg.y;
    return Tvec3<T>(R, G, B);
}

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const Tvec3<T> &v) {
	os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return os;
}

class Matrix4x4 {
  public:
    Matrix4x4() {
        memset(m, 0, sizeof(float) * 16);
        m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1;
    }
    Matrix4x4(const float _m[4][4]) { memcpy(m, _m, sizeof(float) * 16); }
    Matrix4x4(const float _m[16]) { memcpy(m, _m, sizeof(float) * 16); }
    Matrix4x4 operator*(const float &v) const {
        Matrix4x4 ret;
        for (uint32_t i = 0; i < 4; i++) {
            for (uint32_t j = 0; j < 4; j++) {
                ret.m[i][j] = m[i][j] * v;
            }
        }
        return ret;
    }
    Matrix4x4 operator/(const float &v) const {
        CHECK(v != 0);
        float inv = 1.f / v;
        return *this * inv;
    }
    Matrix4x4 operator*(const Matrix4x4 &mat) const {
        Matrix4x4 ret;
        for (uint32_t i = 0; i < 4; i++) {
            for (uint32_t j = 0; j < 4; j++) {
                ret.m[i][j] = 0;
                for (uint32_t k = 0; k < 4; k++) {
                    ret.m[i][j] += m[i][k] * mat.m[k][j];
                }
            }
        }
        return ret;
    }

	Float3 operator()(const Float3 &p, const Float3::EType &type) const;

	float m[4][4];

  public:
    friend std::ostream &operator<<(std::ostream &os, const Matrix4x4 &mat) {
        os << mat.m[0][0] << "\t" << mat.m[0][1] << "\t" << mat.m[0][2] << "\t"
           << mat.m[0][3] << "\n"
           << mat.m[1][0] << "\t" << mat.m[1][1] << "\t" << mat.m[1][2] << "\t"
           << mat.m[1][3] << "\n"
           << mat.m[2][0] << "\t" << mat.m[2][1] << "\t" << mat.m[2][2] << "\t"
           << mat.m[2][3] << "\n"
           << mat.m[3][0] << "\t" << mat.m[3][1] << "\t" << mat.m[3][2] << "\t"
           << mat.m[3][3];
        return os;
    }

    friend Matrix4x4 Inverse(const Matrix4x4 &mat);
    friend Matrix4x4 Transpose(const Matrix4x4 &mat);
};

using Vec3 = Tvec3<float>;
using Color = Tvec3<float>;
using Ivec3 = Tvec3<int>;