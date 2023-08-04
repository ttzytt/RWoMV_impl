#pragma once

#include <dbg.h>

#include <cstring>
#include <memory>

#include "../mathutil.h"
#include "./common.h"

template <typename T>
class Buffer {
   public:
	Buffer(T *buffer, const int &size);
	void Copy(const Buffer<T> &buffer);
	T operator()(int x) const;
	T &operator()(int x);
	int m_size;
	std::shared_ptr<T[]> m_buffer = nullptr;
};

template <typename T>
inline Buffer<T>::Buffer(T *buffer, const int &size)
	: m_buffer(buffer), m_size(size) {}

template <typename T>
inline void Buffer<T>::Copy(const Buffer<T> &buffer) {
	if (m_buffer == buffer.m_buffer) {
		return;
	}
	m_size = buffer.m_size;
	m_buffer = std::shared_ptr<T[]>(new T[m_size]);
	std::memcpy(m_buffer.get(), buffer.m_buffer.get(), sizeof(T) * m_size);
}

template <typename T>
inline T Buffer<T>::operator()(int x) const {
	CHECK(x < m_size);
	return m_buffer.get()[x];
}

template <typename T>
inline T& Buffer<T>::operator()(int x) {
	CHECK(x < m_size);
	return m_buffer.get()[x];
}

template <typename T>
class Buffer2D : public Buffer<T> {
   public:
	using Buffer<T>::operator();
	
	Buffer2D();
	Buffer2D(T *buffer, const int &width, const int &height);

	void Copy(const Buffer2D<T> &buffer);

	T operator()(const int &x, const int &y) const;
	T operator()(const Ivec2 &pos) const;
	T operator()(float x, float y) const;
	T operator()(const Vec2 &pos) const;
	T &operator()(const int &x, const int &y);
	T &operator()(const Ivec2 &pos);
	Buffer2D<T>& normalize();
	int m_width, m_height;
};

template <typename T>
inline Buffer2D<T>::Buffer2D()
	: Buffer<T>(nullptr, 0), m_width(0), m_height(0) {}

template <typename T>
inline Buffer2D<T>::Buffer2D(T *buffer, const int &width, const int &height)
	: Buffer<T>(buffer, width * height), m_width(width), m_height(height) {}

template <typename T>
inline void Buffer2D<T>::Copy(const Buffer2D<T> &buffer) {
	Buffer<T>::Copy(buffer);
	m_width = buffer.m_width;
	m_height = buffer.m_height;
}

template <typename T>
inline T &Buffer2D<T>::operator()(const int &x, const int &y) {
	CHECK(m_width > 0)
	CHECK(m_height > 0)
	CHECK(0 <= x && x < m_width && 0 <= y && y < m_height);
	return this->m_buffer[y * m_width + x];
}

template <typename T>
inline T Buffer2D<T>::operator()(const Ivec2 &pos) const {
	return this->operator()(pos.x, pos.y);
}

template <typename T>
inline T &Buffer2D<T>::operator()(const Ivec2 &pos) {
	return this->operator()(pos.x, pos.y);
}

template <typename T>
inline T Buffer2D<T>::operator()(float x, float y) const {
	// throw std::runtime_error("not implemented");
	if (x == int(x) && y == int(y)) {
		return operator()(int(x), int(y));
	}
	dbg("float used");
	int x0 = floor(x), y0 = floor(y);
	int x1 = ceil(x), y1 = ceil(y);
	x0 = std::max(x0, 0), y0 = std::max(y0, 0);
	;
	x1 = std::min(x1, m_height - 1), y1 = std::min(y1, m_height - 1);
	T v00 = operator()(x0, y0), v01 = operator()(x0, y1);
	T v10 = operator()(x1, y0), v11 = operator()(x1, y1);
	float tx = x - x0, ty = y - y0;
	T ret = lerp2d(v00, v10, v01, v11, tx, ty);
	return ret;
}

template <typename T>
inline T Buffer2D<T>::operator()(const Vec2 &pos) const {
	return this->operator()(pos.x, pos.y);
}

template <typename T>
inline T Buffer2D<T>::operator()(const int &x, const int &y) const {
	if (0 <= x && x < m_width && 0 <= y && y < m_height) {
		return this->m_buffer[y * m_width + x];
	} else {
		return T(0.0);
	}
}

template <typename T>
inline Buffer2D<T> CreateBuffer2D(const int &width, const int &height) {
	T *buffer = new T[width * height]();
	return Buffer2D<T>(buffer, width, height);
}

template <>
inline Buffer2D<Vec2>& Buffer2D<Vec2>::normalize(){
	// transform all value to the range of [0, 1]
	Vec2 mn(1e9), mx(-1e9);
	for (int i = 0; i < m_width * m_height; i++) {
		mn = Vec2::min(mn, m_buffer[i]);
		mx = Vec2::max(mx, m_buffer[i]);
	}
	Vec2 diff = mx - mn;
	for (int i = 0; i < m_width * m_height; i++) {
		m_buffer[i].x= (m_buffer[i] - mn).x / diff.x;
		m_buffer[i].y= (m_buffer[i] - mn).y / diff.y;
	}
	return *this;
}

template <>
inline Buffer2D<Vec3> &Buffer2D<Vec3>::normalize() {
	// transform all value to the range of [0, 1]
	Vec3 mn(1e9), mx(-1e9);
	for (int i = 0; i < m_width * m_height; i++) {
		mn = Min(mn, m_buffer[i]);
		mx = Max(mx, m_buffer[i]);
	}
	Vec3 diff = mx - mn;
	for (int i = 0; i < m_width * m_height; i++) {
		m_buffer[i].x = (m_buffer[i] - mn).x / diff.x;
		m_buffer[i].y = (m_buffer[i] - mn).y / diff.y;
		m_buffer[i].z = (m_buffer[i] - mn).z / diff.z;
	}
	return *this;
}