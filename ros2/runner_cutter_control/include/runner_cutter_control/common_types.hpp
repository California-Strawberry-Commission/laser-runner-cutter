#pragma once

struct LaserCoord {
  float x;
  float y;
};

struct PixelCoord {
  int u;
  int v;
};

struct NormalizedPixelCoord {
  float u;
  float v;
};

struct Position {
  float x;
  float y;
  float z;
};

struct PixelRect {
  int u;
  int v;
  int width;
  int height;
};

struct NormalizedPixelRect {
  float u;
  float v;
  float width;
  float height;
};

struct LaserRect {
  float x;
  float y;
  float width;
  float height;
};

struct FrameSize {
  int width;
  int height;
};

struct LaserColor {
  float r;
  float g;
  float b;
  float i;
};