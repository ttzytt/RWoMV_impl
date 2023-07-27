#pragma once

#include <cstring>
#include <memory>

#include "./common.h"
#include "../mathutil.h"

template <typename T>
class Buffer {
   public:
	Buffer(T *buffer, const int &size);
	virtual void Copy(const Buffer<T> &buffer);

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
class Buffer2D : public Buffer<T> {
   public:
	Buffer2D();
	Buffer2D(T *buffer, const int &width, const int &height);

	void Copy(const Buffer2D<T> &buffer);

	T operator()(const int &x, const int &y) const;
	T operator()(const Ivec2 &pos) const;
	T operator()(float x, float y) const;
	T operator()(const Vec2 &pos) const;
	T &operator()(const int &x, const int &y);
	T &operator()(const Ivec2 &pos); 
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
	int x0 = floor(x), y0 = floor(y);
	int x1 = ceil(x), y1 = ceil(y);
	T v00 = operator()(x0, y0), v01 = operator()(x0, y1);
	T v10 = operator()(x1, y0), v11 = operator()(x1, y1);
	float tx = x - x0, ty = y - y0;
	return lerp2d(v00, v01, v10, v11, tx, ty);
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
	T *buffer = new T[width * height];
	return Buffer2D<T>(buffer, width, height);
}