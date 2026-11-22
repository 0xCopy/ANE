| MIL op | status | ok | wrong | fail | fail, no dead ops | arguments (kinds seen) | failing error classes |
|---|---|---:|---:|---:|---:|---|---|
| `abs` | supported | 6 | 0 | 0 | 0 | x:tensor |  |
| `acos` | unsupported | 0 | 0 | 4 | 4 | x:tensor | InvalidMILProgram |
| `add` | sensitive | 7072 | 9 | 8 | 2 | x:tensor; y:tensor/float/blob/int | CompilationFailure, InvalidMILProgram |
| `argsort` | sensitive | 10 | 0 | 9 | 9 | ascending:bool; axis:int; x:tensor | InvalidMILProgram |
| `asin` | unsupported | 0 | 0 | 4 | 4 | x:tensor | InvalidMILProgram |
| `atan` | supported | 4 | 0 | 0 | 0 | x:tensor |  |
| `atanh` | unsupported | 0 | 0 | 4 | 4 | x:tensor | InvalidMILProgram |
| `avg_pool` | supported | 6 | 0 | 0 | 0 | ceil_mode:bool; exclude_padding_from_average:bool; kernel_sizes:ints; pad:ints; pad_type:str; strides:ints; x:tensor |  |
| `band_part` | unsupported | 0 | 0 | 6 | 6 | lower:int; upper:int; x:tensor | InvalidMILProgram |
| `batch_norm` | supported | 1 | 0 | 0 | 0 | beta:blob; epsilon:float; gamma:blob; mean:blob; variance:blob; x:tensor |  |
| `batch_to_space` | supported | 1 | 0 | 0 | 0 | block_shape:ints; crops:tconst; x:tensor |  |
| `cast` | sensitive | 138 | 0 | 45 | 19 | dtype:str; x:tensor | CompilationFailure, InvalidMILProgram |
| `ceil` | supported | 5 | 0 | 0 | 0 | x:tensor |  |
| `clamped_relu` | supported | 1 | 0 | 0 | 0 | alpha:float; beta:float; x:tensor |  |
| `clip` | supported | 1 | 0 | 0 | 0 | alpha:float; beta:float; x:tensor |  |
| `concat` | sensitive | 22 | 0 | 1 | 1 | axis:int; interleave:bool; values:tuple | InvalidMILProgram |
| `constexpr_affine_dequantize` | sensitive | 8 | 0 | 6 | 6 |  | InvalidMILProgram |
| `constexpr_blockwise_shift_scale` | sensitive | 2 | 0 | 9 | 9 | data:qblob; offset:qblob; scale:blob | InvalidMILProgram |
| `constexpr_lut_to_dense` | supported | 9 | 0 | 0 | 0 | indices:qblob; lut:blob/tconst |  |
| `constexpr_sparse_to_dense` | supported | 3 | 0 | 0 | 0 | mask:qblob; nonzero_data:blob |  |
| `conv` | sensitive | 139 | 8 | 67 | 67 | bias:blob; dilations:ints; groups:int; pad:ints; pad_type:str; strides:ints; weight:blob/tensor; x:tensor | CompilationFailure, _ANECompiler : ANECCompile() FAILED, InvalidMILProgram |
| `conv_transpose` | supported | 6 | 0 | 0 | 0 | bias:blob; dilations:ints; groups:int; pad:ints; pad_type:str; strides:ints; weight:blob; x:tensor |  |
| `cos` | supported | 5 | 0 | 0 | 0 | x:tensor |  |
| `cosh` | unsupported | 0 | 0 | 4 | 4 | x:tensor | InvalidMILProgram |
| `crop` | supported | 2 | 0 | 0 | 0 | crop_height:ints; crop_width:ints; x:tensor |  |
| `crop_resize` | unsupported | 0 | 0 | 3 | 3 | box_coordinate_mode:str; box_indices:tensor; boxes:tensor; normalized_coordinates:bool; pad_value:float; sampling_mode:str; spatial_scale:float; target_height:int; target_width:int; x:tensor | InvalidMILProgram |
| `cumsum` | unsupported | 0 | 0 | 10 | 10 | axis:int; exclusive:bool; reverse:bool; x:tensor | InvalidMILProgram |
| `depth_to_space` | supported | 1 | 0 | 0 | 0 | block_size:int; x:tensor |  |
| `dequantize` | supported | 5 | 0 | 0 | 0 | axis:int; input:tensor; scale:blob/float; zero_point:qblob/int |  |
| `einsum` | supported | 1 | 0 | 0 | 0 | equation:str; values:tuple |  |
| `elu` | supported | 1 | 0 | 0 | 0 | alpha:float; x:tensor |  |
| `equal` | sensitive | 105 | 0 | 7 | 1 | x:tensor/blob; y:tensor/float/blob | CompilationFailure, InvalidMILProgram |
| `erf` | supported | 2 | 0 | 0 | 0 | x:tensor |  |
| `exp` | supported | 18 | 0 | 0 | 0 | x:tensor |  |
| `exp2` | supported | 1 | 0 | 0 | 0 | x:tensor |  |
| `expand_dims` | supported | 5 | 0 | 0 | 0 | axes:ints; x:tensor |  |
| `fill` | supported | 1 | 0 | 0 | 0 | shape:ints; value:float |  |
| `fill_like` | supported | 1 | 0 | 0 | 0 | ref_tensor:tensor; value:float |  |
| `flatten2d` | supported | 5 | 0 | 0 | 0 | axis:int; x:tensor |  |
| `floor` | supported | 7 | 0 | 0 | 0 | x:tensor |  |
| `floor_div` | supported | 2 | 0 | 0 | 0 | x:tensor; y:tensor |  |
| `gather` | unsupported | 0 | 0 | 19 | 19 | axis:int; batch_dims:int; indices:ints/tensor; validate_indices:bool; x:tensor | InvalidMILProgram, CompilationFailure |
| `gather_along_axis` | unsupported | 0 | 0 | 7 | 7 | axis:int; indices:tensor/tconst; validate_indices:bool; x:tensor | CompilationFailure |
| `gather_nd` | unsupported | 0 | 0 | 5 | 5 | batch_dims:int; indices:tconst/tensor; validate_indices:bool; x:tensor | InvalidMILProgram |
| `gelu` | supported | 3 | 0 | 0 | 0 | mode:str; x:tensor |  |
| `greater` | sensitive | 2 | 0 | 1 | 0 | x:tensor; y:tensor/float | CompilationFailure |
| `greater_equal` | supported | 1 | 0 | 0 | 0 | x:tensor; y:tensor |  |
| `gru` | unsupported | 0 | 0 | 9 | 9 | activation:str; bias:blob; direction:str; initial_h:tensor; output_sequence:bool; recurrent_activation:str; weight_hh:blob; weight_ih:blob; x:tensor | InvalidMILProgram |
| `identity` | sensitive | 2 | 0 | 3 | 1 | x:tensor | InvalidMILProgram, CompilationFailure |
| `instance_norm` | supported | 1 | 0 | 0 | 0 | beta:blob; epsilon:float; gamma:blob; x:tensor |  |
| `inverse` | supported | 1 | 0 | 0 | 0 | epsilon:float; x:tensor |  |
| `l2_norm` | supported | 1 | 0 | 0 | 0 | epsilon:float; x:tensor |  |
| `l2_pool` | supported | 3 | 0 | 0 | 0 | ceil_mode:bool; kernel_sizes:ints; pad:ints; pad_type:str; strides:ints; x:tensor |  |
| `layer_norm` | supported | 3 | 0 | 0 | 0 | axes:ints; beta:blob; epsilon:float; gamma:blob; x:tensor |  |
| `leaky_relu` | supported | 1 | 0 | 0 | 0 | alpha:float; x:tensor |  |
| `less` | sensitive | 1 | 0 | 3 | 0 | x:tensor; y:tensor/float/int | CompilationFailure, InvalidMILProgram |
| `less_equal` | supported | 1 | 0 | 0 | 0 | x:tensor; y:tensor |  |
| `linear` | sensitive | 5 | 0 | 2 | 2 | bias:blob; weight:blob/tensor; x:tensor | InvalidMILProgram |
| `linear_activation` | supported | 1 | 0 | 0 | 0 | alpha:float; beta:float; x:tensor |  |
| `list_length` | unsupported | 0 | 0 | 1 | 1 | ls:tensor | InvalidMILProgram |
| `list_read` | unsupported | 0 | 0 | 1 | 1 | index:int; ls:tensor | InvalidMILProgram |
| `list_write` | unsupported | 0 | 0 | 2 | 2 | index:int; ls:tensor; value:tensor | InvalidMILProgram |
| `local_response_norm` | supported | 1 | 0 | 0 | 0 | alpha:float; beta:float; k:float; size:int; x:tensor |  |
| `log` | supported | 5 | 0 | 0 | 0 | epsilon:float; x:tensor |  |
| `logical_and` | supported | 6 | 0 | 0 | 0 | x:tensor; y:tensor |  |
| `logical_not` | supported | 11 | 0 | 0 | 0 | x:tensor |  |
| `logical_or` | unsupported | 0 | 0 | 4 | 4 | x:tensor; y:tensor | InvalidMILProgram |
| `logical_xor` | unsupported | 0 | 0 | 4 | 4 | x:tensor; y:tensor | InvalidMILProgram |
| `lstm` | unsupported | 0 | 0 | 5 | 5 | activation:str; bias:blob; cell_activation:str; direction:str; initial_c:tensor; initial_h:tensor; output_sequence:bool; recurrent_activation:str; weight_hh:blob; weight_ih:blob; x:tensor | InvalidMILProgram |
| `make_list` | unsupported | 0 | 0 | 2 | 2 | dtype:str; dynamic_length:bool; elem_shape:tuple; init_length:int | InvalidMILProgram |
| `matmul` | supported | 12 | 0 | 0 | 0 | transpose_x:bool; transpose_y:bool; x:blob/tensor; y:tensor/blob |  |
| `max_pool` | supported | 6 | 0 | 0 | 0 | ceil_mode:bool; kernel_sizes:ints; pad:ints; pad_type:str; strides:ints; x:tensor |  |
| `maximum` | supported | 1 | 0 | 0 | 0 | x:tensor; y:tensor |  |
| `minimum` | supported | 1 | 0 | 0 | 0 | x:tensor; y:tensor |  |
| `mod` | unsupported | 0 | 0 | 4 | 4 | x:tensor; y:tensor | InvalidMILProgram |
| `mul` | sensitive | 5786 | 9 | 15 | 5 | x:tensor/blob; y:float/tensor/blob | CompilationFailure, InvalidMILProgram |
| `non_maximum_suppression` | unsupported | 0 | 0 | 1 | 1 | boxes:tensor; iou_threshold:float; max_boxes:int; per_class_suppression:bool; scores:tensor | InvalidMILProgram |
| `non_zero` | unsupported | 0 | 0 | 3 | 3 | x:tensor | InvalidMILProgram |
| `not_equal` | supported | 1 | 0 | 0 | 0 | x:tensor; y:tensor |  |
| `one_hot` | unsupported | 0 | 0 | 5 | 5 | axis:int; indices:tensor; off_value:float/int; on_value:float/int; one_hot_vector_size:int | InvalidMILProgram |
| `pad` | supported | 3 | 0 | 0 | 0 | constant_val:float; mode:str; pad:ints; x:tensor |  |
| `pixel_shuffle` | supported | 1 | 0 | 0 | 0 | upscale_factor:int; x:tensor |  |
| `pixel_unshuffle` | supported | 1 | 0 | 0 | 0 | downscale_factor:int; x:tensor |  |
| `pow` | supported | 1 | 0 | 0 | 0 | x:tensor; y:tensor |  |
| `prelu` | supported | 1 | 0 | 0 | 0 | alpha:blob; x:tensor |  |
| `quantize` | supported | 5 | 0 | 0 | 0 | axis:int; input:tensor; output_dtype:str; scale:blob/float; zero_point:qblob/int |  |
| `random_bernoulli` | unsupported | 0 | 0 | 1 | 1 | prob:float; seed:int; shape:ints | InvalidMILProgram |
| `random_categorical` | unsupported | 0 | 0 | 1 | 1 | mode:str; seed:int; size:int; x:tensor | InvalidMILProgram |
| `random_normal` | unsupported | 0 | 0 | 2 | 1 | mean:float; seed:int; shape:ints; stddev:float | InvalidMILProgram |
| `random_uniform` | deviates | 0 | 9 | 1 | 0 | high:float; low:float; seed:int; shape:ints | CompilationFailure |
| `range_1d` | supported | 1 | 0 | 0 | 0 | end:float; start:float; step:float |  |
| `real_div` | supported | 7 | 0 | 0 | 0 | x:tensor; y:tensor |  |
| `reduce_argmax` | unsupported | 0 | 0 | 10 | 10 | axis:int; keep_dims:bool; output_dtype:str; x:tensor | InvalidMILProgram |
| `reduce_argmin` | unsupported | 0 | 0 | 9 | 9 | axis:int; keep_dims:bool; output_dtype:str; x:tensor | InvalidMILProgram |
| `reduce_l1_norm` | supported | 17 | 0 | 0 | 0 | axes:ints; keep_dims:bool; x:tensor |  |
| `reduce_l2_norm` | supported | 17 | 0 | 0 | 0 | axes:ints; keep_dims:bool; x:tensor |  |
| `reduce_log_sum` | supported | 17 | 0 | 0 | 0 | axes:ints; keep_dims:bool; x:tensor |  |
| `reduce_log_sum_exp` | supported | 17 | 0 | 0 | 0 | axes:ints; keep_dims:bool; x:tensor |  |
| `reduce_max` | supported | 78 | 0 | 0 | 0 | axes:ints; keep_dims:bool; x:tensor |  |
| `reduce_mean` | supported | 18 | 0 | 0 | 0 | axes:ints; keep_dims:bool; x:tensor |  |
| `reduce_min` | supported | 73 | 0 | 0 | 0 | axes:ints; keep_dims:bool; x:tensor |  |
| `reduce_prod` | unsupported | 0 | 0 | 20 | 20 | axes:ints; keep_dims:bool; x:tensor | InvalidMILProgram |
| `reduce_sum` | sensitive | 22 | 0 | 5 | 1 | axes:ints; keep_dims:bool; x:tensor | InvalidMILProgram, CompilationFailure |
| `reduce_sum_square` | supported | 17 | 0 | 0 | 0 | axes:ints; keep_dims:bool; x:tensor |  |
| `relu` | sensitive | 5644 | 0 | 38 | 29 | x:tensor | CompilationFailure |
| `relu6` | supported | 1 | 0 | 0 | 0 | x:tensor |  |
| `resample` | sensitive | 10 | 0 | 2 | 2 | align_corners:bool; coordinates:tensor; coordinates_format:str; coordinates_mode:str; coordinates_type:str; padding_mode:str; padding_value:float; sampling_mode:str; x:tensor | CompilationFailure |
| `reshape` | sensitive | 16 | 0 | 1 | 1 | shape:ints; x:tensor | InvalidMILProgram |
| `resize_bilinear` | supported | 6 | 0 | 0 | 0 | sampling_mode:str; target_size_height:int; target_size_width:int; x:tensor |  |
| `resize_nearest_neighbor` | supported | 3 | 0 | 0 | 0 | target_size_height:int; target_size_width:int; x:tensor |  |
| `reverse` | supported | 7 | 0 | 0 | 0 | axes:ints; x:tensor |  |
| `reverse_sequence` | unsupported | 0 | 0 | 3 | 3 | batch_axis:int; lengths:ints; seq_axis:int; x:tensor | InvalidMILProgram |
| `rnn` | unsupported | 0 | 0 | 9 | 9 | activation:str; bias:blob; direction:str; initial_h:tensor; output_sequence:bool; weight_hh:blob; weight_ih:blob; x:tensor | InvalidMILProgram |
| `round` | supported | 9 | 0 | 0 | 0 | x:tensor |  |
| `rsqrt` | supported | 1 | 0 | 0 | 0 | epsilon:float; x:tensor |  |
| `scaled_dot_product_attention` | supported | 8 | 5 | 0 | 0 | attn_mask:other/blob/tensor; key:tensor; query:tensor; value:tensor |  |
| `scaled_tanh` | supported | 1 | 0 | 0 | 0 | alpha:float; beta:float; x:tensor |  |
| `scatter` | unsupported | 0 | 0 | 4 | 4 | axis:int; data:tensor; indices:ints; mode:str; updates:tensor; validate_indices:bool | InvalidMILProgram |
| `scatter_along_axis` | unsupported | 0 | 0 | 4 | 4 | axis:int; data:tensor; indices:tconst; mode:str; updates:tensor; validate_indices:bool | InvalidMILProgram |
| `scatter_nd` | unsupported | 0 | 0 | 2 | 2 | data:tensor; indices:tconst; mode:str; updates:tensor; validate_indices:bool | InvalidMILProgram |
| `select` | sensitive | 1 | 0 | 1 | 0 | a:tensor; b:tensor; cond:tensor | CompilationFailure |
| `shape` | unsupported | 0 | 0 | 2 | 2 | x:tensor | InvalidMILProgram |
| `sigmoid` | supported | 8 | 0 | 0 | 0 | x:tensor |  |
| `sigmoid_hard` | supported | 1 | 0 | 0 | 0 | alpha:float; beta:float; x:tensor |  |
| `sign` | supported | 5 | 0 | 0 | 0 | x:tensor |  |
| `silu` | supported | 2 | 0 | 0 | 0 | x:tensor |  |
| `sin` | supported | 5 | 0 | 0 | 0 | x:tensor |  |
| `sinh` | unsupported | 0 | 0 | 4 | 4 | x:tensor | InvalidMILProgram |
| `slice_by_index` | sensitive | 8 | 0 | 10 | 6 | begin:ints; begin_mask:bools; end:ints; end_mask:bools; squeeze_mask:bools; stride:ints; x:tensor | InvalidMILProgram |
| `slice_by_size` | supported | 3 | 0 | 0 | 0 | begin:ints; size:ints; x:tensor |  |
| `slice_update` | sensitive | 3 | 0 | 1 | 1 | begin:ints; begin_mask:bools; end:ints; end_mask:bools; stride:ints; update:tensor; x:tensor | InvalidMILProgram |
| `sliding_windows` | unsupported | 0 | 0 | 2 | 2 | axis:int; size:int; stride:int; x:tensor | InvalidMILProgram |
| `softmax` | supported | 3 | 0 | 0 | 0 | axis:int; x:tensor |  |
| `softplus` | supported | 3 | 0 | 0 | 0 | x:tensor |  |
| `softsign` | supported | 3 | 0 | 0 | 0 | x:tensor |  |
| `space_to_batch` | supported | 2 | 0 | 0 | 0 | block_shape:ints; paddings:tconst; x:tensor |  |
| `space_to_depth` | supported | 1 | 0 | 0 | 0 | block_size:int; x:tensor |  |
| `split` | supported | 6 | 0 | 0 | 0 | axis:int; num_splits:int; split_sizes:ints; x:tensor |  |
| `sqrt` | supported | 3 | 0 | 0 | 0 | x:tensor |  |
| `square` | supported | 7 | 0 | 0 | 0 | x:tensor |  |
| `squeeze` | supported | 5 | 0 | 0 | 0 | axes:ints; x:tensor |  |
| `stack` | supported | 6 | 0 | 0 | 0 | axis:int; values:tuple |  |
| `sub` | supported | 60 | 0 | 0 | 0 | x:float/tensor; y:tensor |  |
| `tan` | unsupported | 0 | 0 | 4 | 4 | x:tensor | InvalidMILProgram |
| `tanh` | supported | 7 | 0 | 0 | 0 | x:tensor |  |
| `threshold` | supported | 1 | 0 | 0 | 0 | alpha:float; x:tensor |  |
| `thresholded_relu` | supported | 1 | 0 | 0 | 0 | alpha:float; x:tensor |  |
| `tile` | supported | 4 | 0 | 0 | 0 | reps:ints; x:tensor |  |
| `topk` | unsupported | 0 | 0 | 19 | 13 | ascending:bool; axis:int; k:int; output_indices_dtype:str; return_indices:bool; sort:bool; x:tensor | InvalidMILProgram |
| `transpose` | supported | 2 | 0 | 0 | 0 | perm:ints; x:tensor |  |
| `upsample_bilinear` | supported | 3 | 0 | 0 | 0 | align_corners:bool; scale_factor_height:int; scale_factor_width:int; x:tensor |  |
| `upsample_nearest_neighbor` | supported | 2 | 0 | 0 | 0 | scale_factor_height:int; scale_factor_width:int; x:tensor |  |
