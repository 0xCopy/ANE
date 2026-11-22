"""Indexing and data-movement ops: gather*, scatter*, one_hot, band_part, fill, range_1d, tile, reverse*, sliding_windows,
space/batch, crop, expand/squeeze/flatten, non_zero, identity, concat/stack/split, slice_*.  Case names start with ix_."""

def register(env):
    case, mb, np, f16, rs, SHAPE, X, XY = env.case, env.mb, env.np, env.f16, env.rs, env.SHAPE, env.X, env.XY
    I = lambda a: np.asarray(a, np.int32)
    F = lambda a: np.asarray(a, np.float16)
    BIG = (1, 64, 1, 32)

    # ---- gather: axis sweep, static and dynamic int32 indices (4-D canonical shape)
    for ax in range(4):
        n = SHAPE[ax]
        idx = I([0, min(n - 1, 2)])
        case(f"ix_gather_ax{ax}_const", "gather", X, (lambda ax, idx: lambda x: mb.gather(x=x, indices=idx, axis=ax))(ax, idx))
        case(f"ix_gather_ax{ax}_dyn", "gather", [("x", SHAPE), ("i", (2,), "int32")],
             (lambda ax: lambda x, i: mb.gather(x=x, indices=i, axis=ax))(ax), kinds={"i": f"idx:{n}"})
    # negative constant indices, validate_indices, 2-D and 3-D data
    case("ix_gather_neg_const_ax1", "gather", X, lambda x: mb.gather(x=x, indices=I([-1, -3]), axis=1))
    case("ix_gather_validate_ax1", "gather", X, lambda x: mb.gather(x=x, indices=I([5, 9]), axis=1, validate_indices=True))
    case("ix_gather_2d_ax0_dyn", "gather", [("x", (64, 32)), ("i", (3,), "int32")], lambda x, i: mb.gather(x=x, indices=i, axis=0), kinds={"i": "idx:64"})
    case("ix_gather_2d_ax1_const", "gather", [("x", (64, 32))], lambda x: mb.gather(x=x, indices=I([0, 31, 7]), axis=1))
    case("ix_gather_3d_ax1_const", "gather", [("x", (4, 8, 16))], lambda x: mb.gather(x=x, indices=I([0, 7, 3]), axis=1))
    case("ix_gather_3d_ax2_dyn", "gather", [("x", (4, 8, 16)), ("i", (2,), "int32")], lambda x, i: mb.gather(x=x, indices=i, axis=2), kinds={"i": "idx:16"})
    case("ix_gather_3d_ax0_dyn", "gather", [("x", (4, 8, 16)), ("i", (2,), "int32")], lambda x, i: mb.gather(x=x, indices=i, axis=0), kinds={"i": "idx:4"})

    # ---- gather_nd (batch_dims 0 and 1)
    case("ix_gather_nd_bd0_const", "gather", [("x", (8, 16))], lambda x: mb.gather_nd(x=x, indices=I([[0, 1], [5, 3]])))
    case("ix_gather_nd_bd0_dyn", "gather", [("x", (8, 16)), ("i", (3, 2), "int32")], lambda x, i: mb.gather_nd(x=x, indices=i), kinds={"i": "idx:8"})
    case("ix_gather_nd_bd0_k1_const", "gather", [("x", (8, 16))], lambda x: mb.gather_nd(x=x, indices=I([[2], [6]])))
    case("ix_gather_nd_bd1_const", "gather", [("x", (4, 8, 16))],
         lambda x: mb.gather_nd(x=x, indices=I([[[0], [3]], [[1], [7]], [[2], [0]], [[3], [5]]]), batch_dims=1))
    case("ix_gather_nd_bd1_dyn", "gather", [("x", (4, 8, 16)), ("i", (4, 2, 1), "int32")],
         lambda x, i: mb.gather_nd(x=x, indices=i, batch_dims=1), kinds={"i": "idx:8"})

    # ---- gather_along_axis (axes 0-3), dynamic and const
    for ax in range(4):
        shp = list(BIG); shp[ax] = 2; shp = tuple(shp)
        n = BIG[ax]
        case(f"ix_gather_along_ax{ax}_dyn", "gather", [("x", BIG), ("i", shp, "int32")],
             (lambda ax: lambda x, i: mb.gather_along_axis(x=x, indices=i, axis=ax))(ax), kinds={"i": f"idx:{n}"})
    ic = np.stack([rs.randint(0, 32, (1, 64, 1, 4))])[0].astype(np.int32)
    case("ix_gather_along_ax3_const", "gather", X, lambda x: mb.gather_along_axis(x=x, indices=I(ic), axis=3))
    case("ix_gather_along_ax1_neg_const", "gather", X,
         lambda x: mb.gather_along_axis(x=x, indices=I(-np.abs(rs.randint(1, 64, (1, 2, 1, 32)))), axis=1))

    # ---- scatter (axis, mode update / add), scatter_nd, scatter_along_axis
    for mode in ("update", "add"):
        case(f"ix_scatter_ax1_{mode}", "scatter", [("x", SHAPE), ("u", (1, 2, 1, 32))],
             (lambda mode: lambda x, u: mb.scatter(data=x, indices=I([1, 3]), updates=u, axis=1, mode=mode))(mode))
        case(f"ix_scatter_ax3_{mode}", "scatter", [("x", SHAPE), ("u", (1, 64, 1, 4))],
             (lambda mode: lambda x, u: mb.scatter(data=x, indices=I([0, 5, 9, 31]), updates=u, axis=3, mode=mode))(mode))
        case(f"ix_scatter_nd_{mode}", "scatter", [("x", (64, 32)), ("u", (3, 32))],
             (lambda mode: lambda x, u: mb.scatter_nd(data=x, indices=I([[0], [5], [9]]), updates=u, mode=mode))(mode))
    uidx = I(np.stack([rs.permutation(32)[:4] for _ in range(64)])[None, :, None, :])
    uidx1 = I(np.stack([rs.permutation(64)[:4] for _ in range(32)], axis=1)[None, :, None, :])
    for mode in ("update", "add"):
        case(f"ix_scatter_along_ax3_{mode}", "scatter", [("x", SHAPE), ("u", (1, 64, 1, 4))],
             (lambda mode: lambda x, u: mb.scatter_along_axis(data=x, indices=uidx, updates=u, axis=3, mode=mode))(mode))
        case(f"ix_scatter_along_ax1_{mode}", "scatter", [("x", SHAPE), ("u", (1, 4, 1, 32))],
             (lambda mode: lambda x, u: mb.scatter_along_axis(data=x, indices=uidx1[:, :4, :, :].reshape(1, 4, 1, 32), updates=u, axis=1, mode=mode))(mode))

    # ---- one_hot (indices int32 input; depth 4)
    case("ix_one_hot_axis_last", "onehot", [("i", (8,), "int32")],
         lambda i: mb.one_hot(indices=i, one_hot_vector_size=4, axis=-1, on_value=f16(1.0), off_value=f16(0.0)), kinds={"i": "idx:4"}, tol=0)
    case("ix_one_hot_axis0", "onehot", [("i", (8,), "int32")],
         lambda i: mb.one_hot(indices=i, one_hot_vector_size=4, axis=0, on_value=f16(1.0), off_value=f16(0.0)), kinds={"i": "idx:4"}, tol=0)
    case("ix_one_hot_axis1_onoff", "onehot", [("i", (1, 8), "int32")],
         lambda i: mb.one_hot(indices=i, one_hot_vector_size=4, axis=1, on_value=f16(2.0), off_value=f16(-1.0)), kinds={"i": "idx:4"}, tol=0)

    # ---- band_part
    case("ix_band_part_lower_tri", "band", [("x", (64, 32))], lambda x: mb.band_part(x=x, lower=I(-1), upper=I(0)))
    case("ix_band_part_upper_tri", "band", [("x", (64, 32))], lambda x: mb.band_part(x=x, lower=I(0), upper=I(-1)))
    case("ix_band_part_band21", "band", [("x", (64, 32))], lambda x: mb.band_part(x=x, lower=I(2), upper=I(1)))
    case("ix_band_part_4d_lower", "band", X, lambda x: mb.band_part(x=x, lower=I(-1), upper=I(0)))

    # ---- fill / range_1d (no tensor input: constant generators)
    case("ix_fill_const", "fill", [("x", BIG)], lambda x: mb.add(x=x, y=mb.fill(shape=I([1, 64, 1, 32]), value=f16(0.5))))
    case("ix_range_1d_f16", "fill", [("x", (8,))], lambda x: mb.add(x=x, y=mb.range_1d(start=f16(0.0), end=f16(8.0), step=f16(1.0))))

    # ---- tile, reverse, reverse_sequence, sliding_windows
    case("ix_tile_c2", "tile", X, lambda x: mb.tile(x=x, reps=[1, 2, 1, 1]))
    case("ix_tile_w4", "tile", X, lambda x: mb.tile(x=x, reps=[1, 1, 1, 4]))
    case("ix_tile_mixed", "tile", X, lambda x: mb.tile(x=x, reps=[2, 1, 3, 1]))
    for ax in range(4):
        case(f"ix_reverse_ax{ax}", "reverse", X, (lambda ax: lambda x: mb.reverse(x=x, axes=[ax]))(ax))
    case("ix_reverse_ax13", "reverse", X, lambda x: mb.reverse(x=x, axes=[1, 3]))
    case("ix_reverse_3d_ax1", "reverse", [("x", (4, 8, 16))], lambda x: mb.reverse(x=x, axes=[1]))
    case("ix_reverse_seq_bax1_sax3", "reverse", X,
         lambda x: mb.reverse_sequence(x=x, lengths=I(rs.randint(1, 33, (64,))), seq_axis=3, batch_axis=1))
    case("ix_reverse_seq_bax0_sax1", "reverse", X,
         lambda x: mb.reverse_sequence(x=x, lengths=I([10]), seq_axis=1, batch_axis=0))
    case("ix_sliding_windows_ax3_w4s2", "windows", X, lambda x: mb.sliding_windows(x=x, axis=3, size=4, stride=2))
    case("ix_sliding_windows_ax1_w8s4", "windows", X, lambda x: mb.sliding_windows(x=x, axis=1, size=8, stride=4))

    # ---- space/batch, crop
    case("ix_space_to_batch_b12", "s2b", X, lambda x: mb.space_to_batch(x=x, block_shape=I([1, 2]), paddings=I([[0, 0], [0, 0]])))
    case("ix_space_to_batch_b14", "s2b", X, lambda x: mb.space_to_batch(x=x, block_shape=I([1, 4]), paddings=I([[0, 0], [0, 0]])))
    case("ix_batch_to_space_b14", "s2b", [("x", (4, 64, 1, 8))], lambda x: mb.batch_to_space(x=x, block_shape=I([1, 4]), crops=I([[0, 0], [0, 0]])))
    case("ix_crop_hw", "crop", X, lambda x: mb.crop(x=x, crop_height=[0, 0], crop_width=[3, 4]))
    case("ix_crop_h_real", "crop", [("x", (1, 1, 64, 32))], lambda x: mb.crop(x=x, crop_height=[1, 2], crop_width=[3, 0]))

    # ---- expand/squeeze/flatten/non_zero/identity
    for nm, axes in {"ax0": [0], "ax2": [2], "axm1": [-1]}.items():
        case(f"ix_expand_dims_{nm}", "shape", X, (lambda axes: lambda x: mb.expand_dims(x=x, axes=axes))(axes))
    case("ix_expand_dims_ax04_rank3", "shape", [("x", (4, 8, 16))], lambda x: mb.expand_dims(x=x, axes=[0, 4]))
    case("ix_squeeze_ax0", "shape", X, lambda x: mb.squeeze(x=x, axes=[0]))
    case("ix_squeeze_ax02", "shape", X, lambda x: mb.squeeze(x=x, axes=[0, 2]))
    case("ix_squeeze_all", "shape", X, lambda x: mb.squeeze(x=x))
    for ax in range(4):
        case(f"ix_flatten2d_ax{ax}", "shape", X, (lambda ax: lambda x: mb.flatten2d(x=x, axis=ax))(ax))
    case("ix_non_zero_round", "shape", X, lambda x: mb.non_zero(x=x), kinds={"x": "round"}, tol=0)
    case("ix_non_zero_2d_round", "shape", [("x", (64, 32))], lambda x: mb.non_zero(x=x), kinds={"x": "round"}, tol=0)
    case("ix_identity", "shape", X, lambda x: mb.identity(x=x))

    # ---- concat (3 inputs, interleave), stack, split
    C3 = [("x", SHAPE), ("y", SHAPE), ("z", SHAPE)]
    case("ix_concat3_ax1", "concat", C3, lambda x, y, z: mb.concat(values=(x, y, z), axis=1))
    case("ix_concat3_ax3", "concat", C3, lambda x, y, z: mb.concat(values=(x, y, z), axis=3))
    case("ix_concat3_ax3_interleave", "concat", C3, lambda x, y, z: mb.concat(values=(x, y, z), axis=3, interleave=True))
    case("ix_stack2_ax1", "concat", XY, lambda x, y: mb.stack(values=(x, y), axis=1))
    case("ix_stack2_ax0", "concat", XY, lambda x, y: mb.stack(values=(x, y), axis=0))
    case("ix_stack2_ax4", "concat", XY, lambda x, y: mb.stack(values=(x, y), axis=4))
    case("ix_split_uneven_ax1", "split", X, lambda x: mb.split(x=x, split_sizes=[10, 20, 34], axis=1))
    case("ix_split_uneven_ax3", "split", X, lambda x: mb.split(x=x, split_sizes=[5, 27], axis=3))
    case("ix_split_uneven_ax3_1_31", "split", X, lambda x: mb.split(x=x, split_sizes=[1, 31], axis=3))
    case("ix_split_uneven_ax1_13_51", "split", X, lambda x: mb.split(x=x, split_sizes=[13, 51], axis=1))

    # ---- slice_by_index / slice_by_size / slice_update
    case("ix_slice_idx_masks_stride", "slice", X,
         lambda x: mb.slice_by_index(x=x, begin=[0, 2, 0, 31], end=[1, 50, 1, 0], stride=[1, 3, 1, -1],
                                     begin_mask=[True, False, True, False], end_mask=[True, False, True, False]))
    case("ix_slice_idx_negstride_all", "slice", X,
         lambda x: mb.slice_by_index(x=x, begin=[0, 0, 0, 0], end=[0, 0, 0, 0], stride=[1, 1, 1, -1],
                                     begin_mask=[True] * 4, end_mask=[True] * 4))
    case("ix_slice_idx_negindex", "slice", X,
         lambda x: mb.slice_by_index(x=x, begin=[0, -10, 0, 0], end=[1, -1, 1, 32], stride=[1, 1, 1, 1]))
    case("ix_slice_idx_squeeze_ax1", "slice", X,
         lambda x: mb.slice_by_index(x=x, begin=[0, 4, 0, 0], end=[1, 5, 1, 32], stride=[1, 1, 1, 1], squeeze_mask=[False, True, False, False]))
    case("ix_slice_idx_stride2_ax3", "slice", X,
         lambda x: mb.slice_by_index(x=x, begin=[0, 0, 0, 0], end=[1, 64, 1, 32], stride=[1, 1, 1, 2]))
    case("ix_slice_size_fixed", "slice", X, lambda x: mb.slice_by_size(x=x, begin=[0, 4, 0, 0], size=[1, 16, 1, 8]))
    case("ix_slice_size_neg1", "slice", X, lambda x: mb.slice_by_size(x=x, begin=[0, 4, 0, 8], size=[1, -1, 1, -1]))
    case("ix_slice_update_ax1", "slice", [("x", BIG), ("u", (1, 8, 1, 32))],
         lambda x, u: mb.slice_update(x=x, update=u, begin=[0, 4, 0, 0], end=[1, 12, 1, 32], stride=[1, 1, 1, 1]))
    case("ix_slice_update_masks", "slice", [("x", BIG), ("u", (1, 8, 1, 32))],
         lambda x, u: mb.slice_update(x=x, update=u, begin=[0, 4, 0, 0], end=[1, 12, 1, 32], stride=[1, 1, 1, 1],
                                      begin_mask=[True, False, True, True], end_mask=[True, False, True, True]))
    case("ix_slice_update_stride2", "slice", [("x", BIG), ("u", (1, 4, 1, 32))],
         lambda x, u: mb.slice_update(x=x, update=u, begin=[0, 4, 0, 0], end=[1, 12, 1, 32], stride=[1, 2, 1, 1]))

    # ---- variants added to isolate the failures above (one thing changed each)
    case("ix_gather_ax1_const_retry", "gather", X, lambda x: mb.gather(x=x, indices=I([0, 2]), axis=1))
    case("ix_gather_ax2_const_retry", "gather", X, lambda x: mb.gather(x=x, indices=I([0, 0]), axis=2))
    case("ix_stack2_ax2", "concat", XY, lambda x, y: mb.stack(values=(x, y), axis=2))
    case("ix_stack2_ax3", "concat", XY, lambda x, y: mb.stack(values=(x, y), axis=3))
    case("ix_one_hot_depth2", "onehot", [("i", (8,), "int32")],
         lambda i: mb.one_hot(indices=i, one_hot_vector_size=2, axis=-1, on_value=f16(1.0), off_value=f16(0.0)), kinds={"i": "idx:2"}, tol=0)
    case("ix_band_part_small_8x8", "band", [("x", (8, 8))], lambda x: mb.band_part(x=x, lower=I(-1), upper=I(0)))
    case("ix_non_zero_1d_round", "shape", [("x", (8,))], lambda x: mb.non_zero(x=x), kinds={"x": "round"}, tol=0)
    case("ix_reverse_seq_bax0_sax3", "reverse", X, lambda x: mb.reverse_sequence(x=x, lengths=I([20]), seq_axis=3, batch_axis=0))
    case("ix_slice_update_stride1_small", "slice", [("x", BIG), ("u", (1, 4, 1, 32))],
         lambda x, u: mb.slice_update(x=x, update=u, begin=[0, 4, 0, 0], end=[1, 8, 1, 32], stride=[1, 1, 1, 1]))
