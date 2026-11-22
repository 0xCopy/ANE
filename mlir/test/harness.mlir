// Helpers shared by the check_*.mlir drivers (concatenated in their RUN lines).

// v[i,j] = amp * sin(seed + 0.7*i + 1.3*j) in f32
func.func @gen32(%r: index, %c: index, %seed: f32, %amp: f32) -> tensor<?x?xf32> {
  %e = tensor.empty(%r, %c) : tensor<?x?xf32>
  %ki = arith.constant 0.7 : f32
  %kj = arith.constant 1.3 : f32
  %R = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>], iterator_types = ["parallel", "parallel"]}
      outs(%e : tensor<?x?xf32>) {
    ^bb0(%o : f32):
      %i = linalg.index 0 : index
      %j = linalg.index 1 : index
      %ii = arith.index_cast %i : index to i32
      %ji = arith.index_cast %j : index to i32
      %fi = arith.sitofp %ii : i32 to f32
      %fj = arith.sitofp %ji : i32 to f32
      %a = arith.mulf %fi, %ki : f32
      %b = arith.mulf %fj, %kj : f32
      %ab = arith.addf %a, %b : f32
      %t = arith.addf %ab, %seed : f32
      %s = math.sin %t : f32
      %v = arith.mulf %s, %amp : f32
      linalg.yield %v : f32
    } -> tensor<?x?xf32>
  return %R : tensor<?x?xf32>
}

// the same, rounded to f16
func.func @gen(%r: index, %c: index, %seed: f32, %amp: f32) -> tensor<?x?xf16> {
  %m = call @gen32(%r, %c, %seed, %amp) : (index, index, f32, f32) -> tensor<?x?xf32>
  %h = call @to_f16(%m) : (tensor<?x?xf32>) -> tensor<?x?xf16>
  return %h : tensor<?x?xf16>
}

// the same, truncated toward zero to i8
func.func @gen8(%r: index, %c: index, %seed: f32, %amp: f32) -> tensor<?x?xi8> {
  %m = call @gen32(%r, %c, %seed, %amp) : (index, index, f32, f32) -> tensor<?x?xf32>
  %e = tensor.empty(%r, %c) : tensor<?x?xi8>
  %R = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>], iterator_types = ["parallel", "parallel"]}
      ins(%m : tensor<?x?xf32>) outs(%e : tensor<?x?xi8>) {
    ^bb0(%x : f32, %o : i8):
      %q = arith.fptosi %x : f32 to i8
      linalg.yield %q : i8
    } -> tensor<?x?xi8>
  return %R : tensor<?x?xi8>
}

func.func @gen1(%n: index, %seed: f32, %amp: f32) -> tensor<?xf16> {
  %c1 = arith.constant 1 : index
  %m = call @gen(%n, %c1, %seed, %amp) : (index, index, f32, f32) -> tensor<?x?xf16>
  %v = tensor.collapse_shape %m [[0, 1]] : tensor<?x?xf16> into tensor<?xf16>
  return %v : tensor<?xf16>
}

// rows [off, off+n) of a [C, S] matrix
func.func @rows(%m: tensor<?x?xf16>, %off: index, %n: index) -> tensor<?x?xf16> {
  %c1 = arith.constant 1 : index
  %s = tensor.dim %m, %c1 : tensor<?x?xf16>
  %r = tensor.extract_slice %m[%off, 0] [%n, %s] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
  return %r : tensor<?x?xf16>
}

func.func @transpose(%m: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %m, %c0 : tensor<?x?xf16>
  %d1 = tensor.dim %m, %c1 : tensor<?x?xf16>
  %e = tensor.empty(%d1, %d0) : tensor<?x?xf16>
  %r = linalg.transpose ins(%m : tensor<?x?xf16>) outs(%e : tensor<?x?xf16>) permutation = [1, 0]
  return %r : tensor<?x?xf16>
}

// Prints "maxabs <max |lit|> maxdiff <max |lit - nf|>".
func.func @report(%lit: tensor<?x?xf16>, %nf: tensor<?x?xf16>) {
  %z = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<f32>
  %f = linalg.fill ins(%z : f32) outs(%e : tensor<f32>) -> tensor<f32>
  %ma, %md = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> ()>,
                                             affine_map<(i, j) -> ()>], iterator_types = ["reduction", "reduction"]}
      ins(%lit, %nf : tensor<?x?xf16>, tensor<?x?xf16>) outs(%f, %f : tensor<f32>, tensor<f32>) {
    ^bb0(%a : f16, %b : f16, %accA : f32, %accD : f32):
      %af = arith.extf %a : f16 to f32
      %bf = arith.extf %b : f16 to f32
      %d = arith.subf %af, %bf : f32
      %ad = math.absf %d : f32
      %aa = math.absf %af : f32
      %r0 = arith.maximumf %accA, %aa : f32
      %r1 = arith.maximumf %accD, %ad : f32
      linalg.yield %r0, %r1 : f32, f32
    } -> (tensor<f32>, tensor<f32>)
  %va = tensor.extract %ma[] : tensor<f32>
  %vd = tensor.extract %md[] : tensor<f32>
  vector.print str "maxabs "
  vector.print %va : f32 punctuation <no_punctuation>
  vector.print str " maxdiff "
  vector.print %vd : f32
  return
}

// vertical (channel) and horizontal (spatial) concatenation of [C, S] matrices
func.func @vcat(%a: tensor<?x?xf16>, %b: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %r = tensor.concat dim(0) %a, %b : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  return %r : tensor<?x?xf16>
}
func.func @hcat(%a: tensor<?x?xf16>, %b: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %r = tensor.concat dim(1) %a, %b : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  return %r : tensor<?x?xf16>
}
