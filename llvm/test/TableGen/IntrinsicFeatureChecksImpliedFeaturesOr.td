// RUN: not llvm-tblgen -gen-intrinsic-feature-checks \
// RUN:   -intrinsic-feature-prefix=test -intrinsic-feature-namespace=TestTarget \
// RUN:   -I %p/../../include %s 2>&1 | FileCheck %s

// CHECK: error: implied features may not use '|'

include "llvm/IR/Intrinsics.td"
include "llvm/Target/TargetSubtarget.td"

def FeatureExample : SubtargetFeature<"example", "Example", "true", "">;
def FeatureOther : SubtargetFeature<"other", "Other", "true", "">;

class InvalidImpliedFeaturesGuard<string features> : FeatureGuard<features> {
  let ImpliedFeatures = "example|other";
}

let TargetPrefix = "test" in
def int_test_implied_features_or
    : GuardedDefaultIntrinsic<[], [], [IntrNoMem]> {
  let TargetGuards =
      [GuardedVariant<[InvalidImpliedFeaturesGuard<"example">]>];
}
