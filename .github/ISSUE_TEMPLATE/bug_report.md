---
name: Bug report
about: An image decodes or encodes wrongly, a kernel produces wrong pixels, a file crashes the decoder, or a test fails
labels: bug
---

**The input:** the image file (attach it if you can — a decode bug is far
easier to fix with the file) or the call and buffer shape that shows it.

**What should have happened** (what another decoder, PIL/OpenCV, or a
browser shows):

**What broimage did instead** (the pixels or dimensions, the error, a crash,
or the failing `ctest --output-on-failure` output — paste it):

```
```

**Environment:**
- OS and CPU (x86-64 / arm64):
- Compiler / toolchain (MSVC / GCC / Clang):
- Build options that differ from the defaults (`BROIMAGE_WITH_TENSOR`, `BROIMAGE_WITH_JIT`, a brotensor GPU backend):
- broimage commit, and bromath / brotensor commits if built from siblings:
