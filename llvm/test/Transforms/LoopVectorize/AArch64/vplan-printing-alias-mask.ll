; REQUIRES: asserts
; RUN: opt -S -mtriple=aarch64-unknown-linux-gnu -debug-only=loop-vectorize -mattr=+sve2 -passes=loop-vectorize -enable-partial-aliasing-vectorization -disable-output %s 2>&1 | FileCheck %s
; RUN: opt -S -mtriple=aarch64-unknown-linux-gnu -debug-only=loop-vectorize -mattr=+sve2 -passes=loop-vectorize -enable-partial-aliasing-vectorization -prefer-predicate-over-epilogue=predicate-dont-vectorize -disable-output %s 2>&1 | FileCheck %s --check-prefix=CHECK-TF


target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-none-unknown-elf"

define void @alias_mask(ptr noalias %a, ptr %b, ptr %c, i32 %n) {
; CHECK-LABEL: 'alias_mask'
; CHECK:      VPlan 'Initial VPlan for VF={2,4,8,16},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VF:%.]]> = VF
; CHECK-NEXT: Live-in vp<[[VFxUF:%.]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.]]> = vector-trip-count
; CHECK-NEXT: Live-in vp<[[ALIAS_MASK:%.]]> = alias-mask
; CHECK-NEXT: vp<[[TC:%.]]> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.body.preheader>:
; CHECK-NEXT:   EMIT vp<[[TC]]> = EXPAND SCEV (zext i32 %n to i64)
; CHECK-NEXT:   IR   %wide.trip.count = zext nneg i32 %n to i64
; CHECK-NEXT: Successor(s): scalar.ph, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK-NEXT:     vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>, vp<[[VF]]>
; CHECK-NEXT:     CLONE ir<%arrayidx> = getelementptr inbounds ir<%a>, vp<[[STEPS]]>
; CHECK-NEXT:     vp<[[PTR_A:%.+]]> = vector-pointer inbounds ir<%arrayidx>
; CHECK-NEXT:     WIDEN ir<%0> = load vp<[[PTR_A]]>, vp<[[ALIAS_MASK]]>
; CHECK-NEXT:     CLONE ir<%arrayidx2> = getelementptr inbounds ir<%b>, vp<[[STEPS]]>
; CHECK-NEXT:     vp<[[PTR_B:%.+]]> = vector-pointer inbounds ir<%arrayidx2>
; CHECK-NEXT:     WIDEN ir<%1> = load vp<[[PTR_B]]>, vp<[[ALIAS_MASK]]>
; CHECK-NEXT:     WIDEN ir<%add> = add ir<%1>, ir<%0>
; CHECK-NEXT:     CLONE ir<%arrayidx6> = getelementptr inbounds ir<%c>, vp<[[STEPS]]>
; CHECK-NEXT:     vp<[[PTR_C:%.+]]> = vector-pointer inbounds ir<%arrayidx6>
; CHECK-NEXT:     WIDEN store vp<[[PTR_C]]>, ir<%add>, vp<[[ALIAS_MASK]]>
; CHECK-NEXT:     EMIT vp<%index.next> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:     EMIT branch-on-count vp<%index.next>, vp<[[VEC_TC]]>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<%cmp.n> = icmp eq vp<[[TC]]>, vp<[[VEC_TC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<%cmp.n>
; CHECK-NEXT: Successor(s): ir-bb<for.cond.cleanup.loopexit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.cond.cleanup.loopexit>:
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT:   EMIT-SCALAR vp<%bc.resume.val> = phi [ vp<[[VEC_TC]]>, middle.block ], [ ir<0>, ir-bb<for.body.preheader> ]
; CHECK-NEXT: Successor(s): ir-bb<for.body>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.body>:
; CHECK-NEXT:   IR   %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ] (extra operand: vp<%bc.resume.val> from scalar.ph)
; CHECK-NEXT:   IR   %arrayidx = getelementptr inbounds i8, ptr %a, i64 %indvars.iv
; CHECK-NEXT:   IR   %0 = load i8, ptr %arrayidx, align 1
; CHECK-NEXT:   IR   %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %indvars.iv
; CHECK-NEXT:   IR   %1 = load i8, ptr %arrayidx2, align 1
; CHECK-NEXT:   IR   %add = add i8 %1, %0
; CHECK-NEXT:   IR   %arrayidx6 = getelementptr inbounds i8, ptr %c, i64 %indvars.iv
; CHECK-NEXT:   IR   store i8 %add, ptr %arrayidx6, align 1
; CHECK-NEXT:   IR   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK-NEXT:   IR   %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
; CHECK-NEXT: No successors
; CHECK-NEXT: }

; CHECK-TF-LABEL: 'alias_mask'
; CHECK-TF:      VPlan 'Initial VPlan for VF={2,4,8,16},UF>=1' {
; CHECK-TF-NEXT: Live-in vp<[[VF:%.]]> = VF
; CHECK-TF-NEXT: Live-in vp<[[VFxUF:%.]]> = VF * UF
; CHECK-TF-NEXT: Live-in vp<[[ALIAS_MASK:%.]]> = alias-mask
; CHECK-TF-NEXT: vp<[[TC:%.]]> = original trip-count
; CHECK-TF-EMPTY:
; CHECK-TF-NEXT: ir-bb<for.body.preheader>:
; CHECK-TF-NEXT:   EMIT vp<[[TC]]> = EXPAND SCEV (zext i32 %n to i64)
; CHECK-TF-NEXT:   IR   %wide.trip.count = zext nneg i32 %n to i64
; CHECK-TF-NEXT: Successor(s): scalar.ph, vector.ph
; CHECK-TF-EMPTY:
; CHECK-TF-NEXT: vector.ph:
; CHECK-TF-NEXT:   EMIT vp<%index.part.next> = VF * Part + ir<0>
; CHECK-TF-NEXT:   EMIT vp<%active.lane.mask.entry> = active lane mask vp<%index.part.next>, vp<[[TC]]>, ir<1>
; CHECK-TF-NEXT: Successor(s): vector loop
; CHECK-TF-EMPTY:
; CHECK-TF-NEXT: <x1> vector loop: {
; CHECK-TF-NEXT:   vector.body:
; CHECK-TF-NEXT:     EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK-TF-NEXT:     ACTIVE-LANE-MASK-PHI vp<[[LANE_MASK:%.+]]> = phi vp<%active.lane.mask.entry>, vp<%active.lane.mask.next>
; CHECK-TF-NEXT:     vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>, vp<[[VF]]>
; CHECK-TF-NEXT:     EMIT vp<[[MASK:%.+]]> = and vp<[[LANE_MASK]]>, vp<[[ALIAS_MASK]]>
; CHECK-TF-NEXT:     CLONE ir<%arrayidx> = getelementptr inbounds ir<%a>, vp<[[STEPS]]>
; CHECK-TF-NEXT:     vp<[[PTR_A:%.+]]> = vector-pointer inbounds ir<%arrayidx>
; CHECK-TF-NEXT:     WIDEN ir<%0> = load vp<[[PTR_A]]>, vp<[[MASK]]>
; CHECK-TF-NEXT:     CLONE ir<%arrayidx2> = getelementptr inbounds ir<%b>, vp<[[STEPS]]>
; CHECK-TF-NEXT:     vp<[[PTR_B:%.+]]> = vector-pointer inbounds ir<%arrayidx2>
; CHECK-TF-NEXT:     WIDEN ir<%1> = load vp<[[PTR_B]]>, vp<[[MASK]]>
; CHECK-TF-NEXT:     WIDEN ir<%add> = add ir<%1>, ir<%0>
; CHECK-TF-NEXT:     CLONE ir<%arrayidx6> = getelementptr inbounds ir<%c>, vp<[[STEPS]]>
; CHECK-TF-NEXT:     vp<[[PTR_C:%.+]]> = vector-pointer inbounds ir<%arrayidx6>
; CHECK-TF-NEXT:     WIDEN store vp<[[PTR_C]]>, ir<%add>, vp<[[MASK]]>
; CHECK-TF-NEXT:     EMIT vp<%index.next> = add vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-TF-NEXT:     EMIT vp<[[PART_IDX:%.+]]> = VF * Part + vp<%index.next>
; CHECK-TF-NEXT:     EMIT vp<%active.lane.mask.next> = active lane mask vp<[[PART_IDX]]>, vp<[[TC]]>, ir<1>
; CHECK-TF-NEXT:     EMIT vp<[[NOT_MASK:%.+]]> = not vp<%active.lane.mask.next>
; CHECK-TF-NEXT:     EMIT branch-on-cond vp<[[NOT_MASK]]>
; CHECK-TF-NEXT:   No successors
; CHECK-TF-NEXT: }
; CHECK-TF-NEXT: Successor(s): middle.block
; CHECK-TF-EMPTY:
; CHECK-TF-NEXT: middle.block:
; CHECK-TF-NEXT: Successor(s): ir-bb<for.cond.cleanup.loopexit>
; CHECK-TF-EMPTY:
; CHECK-TF-NEXT: ir-bb<for.cond.cleanup.loopexit>:
; CHECK-TF-NEXT: No successors
; CHECK-TF-EMPTY:
; CHECK-TF-NEXT: scalar.ph:
; CHECK-TF-NEXT:   EMIT-SCALAR vp<%bc.resume.val> = phi [ ir<0>, ir-bb<for.body.preheader> ]
; CHECK-TF-NEXT: Successor(s): ir-bb<for.body>
; CHECK-TF-EMPTY:
; CHECK-TF-NEXT: ir-bb<for.body>:
; CHECK-TF-NEXT:   IR   %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ] (extra operand: vp<%bc.resume.val> from scalar.ph)
; CHECK-TF-NEXT:   IR   %arrayidx = getelementptr inbounds i8, ptr %a, i64 %indvars.iv
; CHECK-TF-NEXT:   IR   %0 = load i8, ptr %arrayidx, align 1
; CHECK-TF-NEXT:   IR   %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %indvars.iv
; CHECK-TF-NEXT:   IR   %1 = load i8, ptr %arrayidx2, align 1
; CHECK-TF-NEXT:   IR   %add = add i8 %1, %0
; CHECK-TF-NEXT:   IR   %arrayidx6 = getelementptr inbounds i8, ptr %c, i64 %indvars.iv
; CHECK-TF-NEXT:   IR   store i8 %add, ptr %arrayidx6, align 1
; CHECK-TF-NEXT:   IR   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK-TF-NEXT:   IR   %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
; CHECK-TF-NEXT: No successors
; CHECK-TF-NEXT: }
entry:
  %cmp11 = icmp sgt i32 %n, 0
  br i1 %cmp11, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext nneg i32 %n to i64
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %a, i64 %indvars.iv
  %0 = load i8, ptr %arrayidx, align 1
  %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %indvars.iv
  %1 = load i8, ptr %arrayidx2, align 1
  %add = add i8 %1, %0
  %arrayidx6 = getelementptr inbounds i8, ptr %c, i64 %indvars.iv
  store i8 %add, ptr %arrayidx6, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
}
