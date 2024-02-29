#map = affine_map<(d0)[s0] -> (-d0 + 32400, s0)>
#map1 = affine_map<(d0)[s0] -> (-d0 + 16, s0)>

func.func @a() {
  %c16 = arith.constant 16 : index
  %c32400 = arith.constant 32400 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %0 = vector.vscale
  %1 = arith.muli %0, %c4 : index
  scf.for %arg0 = %c0 to %c32400 step %1 {
    %2 = affine.min #map(%arg0)[%1]
    scf.for %arg1 = %c0 to %c16 step %1 {
      %3 = affine.min #map1(%arg1)[%1]
      %4 = "test.reify_scalable_bound"(%2) {type = "UB"} : (index) -> index
      %5 = "test.reify_scalable_bound"(%3) {type = "UB"} : (index) -> index
    }
  }
  return
}


#map2 = affine_map<(d0, d1)[s0] -> (-d0 + d1, s0)>
#map3 = affine_map<(d0, d1)[s0] -> (-d0 + d1, s0)>

func.func @b(%a: index, %b: index) {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %0 = vector.vscale
  %1 = arith.muli %0, %c4 : index
  scf.for %arg0 = %c0 to %a step %1 {
    %2 = affine.min #map2(%arg0)[%1, %a]
    scf.for %arg1 = %c0 to %b step %1 {
      %3 = affine.min #map3(%arg1)[%1, %b]
      %4 = "test.reify_scalable_bound"(%2) {type = "UB"} : (index) -> index
      %5 = "test.reify_scalable_bound"(%3) {type = "UB"} : (index) -> index
    }
  }
  return
}


func.func @c() {
  %vscale = vector.vscale
  %c8 = arith.constant 8 : index
  %v = arith.addi %vscale, %c8 : index
  %0 = "test.reify_scalable_bound"(%v) {type = "UB"} : (index) -> index
  return
}
