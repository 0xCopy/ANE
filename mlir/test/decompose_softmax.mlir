// Transform script: linalg.softmax has no direct lowering to loops; decompose it first.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %sm = transform.structured.match ops{["linalg.softmax"]} in %root : (!transform.any_op) -> !transform.any_op
    %d = transform.structured.decompose_interface %sm : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
