#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <cstdint>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, 
           size_t seqlen, size_t nhead, size_t d, float theta) {
    size_t half_d = d / 2;
    for(size_t i = 0; i < seqlen; i ++){
        float pos = static_cast<float>(pos_ids[i]);

        for(size_t h = 0; h < nhead; h ++){
            for(size_t j = 0; j < half_d; j ++){
                float freq = pos / std::pow(theta,static_cast<float>(2 * j) / d);
                float cos_angle = std::cos(freq);
                float sin_angle = std::sin(freq);

                size_t base_idx = i * (nhead * d) + h * d;
                size_t idx_a = base_idx + j;
                size_t idx_b = base_idx + j + half_d;

                float a = llaisys::utils::cast<float>(in[idx_a]);
                float b = llaisys::utils::cast<float>(in[idx_b]);

                out[idx_a] = llaisys::utils::cast<T>(a * cos_angle - b * sin_angle);
                out[idx_b] = llaisys::utils::cast<T>(b * cos_angle + a * sin_angle);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
            llaisysDataType_t type, size_t seqlen, size_t nhead, size_t d, float theta) {
    const int64_t *p_ids = reinterpret_cast<const int64_t *>(pos_ids);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                    p_ids, seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                    p_ids, seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                    p_ids, seqlen, nhead, d, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
