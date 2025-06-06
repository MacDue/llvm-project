declare void @private_za_callee()

; Note: Proper naming TDB ("aarch64.za.generation" is kinda a bad name :))

; Possible Clang IR generation:
; Note: All intrinsics require IntrInaccessibleMemOrArgMemOnly as there are side-effects (they cannot be reordered).

define void @new_za(<vscale x 16 x i1> %pg, ptr %a, ptr %b) nounwind "aarch64_new_za" "aarch64_pstate_sm_body" {
  ; Allocate a single "aarch64.za.generation" local for the current ZA state.
  %current_za_gen = alloca target("aarch64.za.generation")
  store target("aarch64.za.generation") zeroinitializer, ptr %current_za_gen

  ; Load the current ZA "generation"
  %za_init = load target("aarch64.za.generation"), ptr %current_za_gen
  ; Increment the ZA generation (to mark that the following intrinsic with update ZA)
  %next_za_0 = call target("aarch64.za.generation") @llvm.aarch64.sme.mark.update.za.state(target("aarch64.za.generation") %za_init)
  call void @llvm.aarch64.sme.ld1b.horiz(<vscale x 16 x i1> %pg, ptr %a, i32 0, i32 0)
  ; Store the resulting ZA generation:
  store target("aarch64.za.generation") %next_za_0, ptr %current_za_gen

  ; For calls to private ZA functions we need to model the ZA clobbers. This
  ; is done with the `clobber.za.state` intrinsic.

  ; First private ZA call.
  tail call void @llvm.aarch64.sme.clobber.za.state()
  call void @private_za_callee()

  ; Second private ZA call.
  tail call void @llvm.aarch64.sme.clobber.za.state()
  call void @private_za_callee()

  ; Another use of a ZA intrinsic.
  %za_init_1 = load target("aarch64.za.generation"), ptr %current_za_gen
  %next_za_1 = call target("aarch64.za.generation") @llvm.aarch64.sme.mark.update.za.state(target("aarch64.za.generation") %za_init_1)
  call void @llvm.aarch64.sme.ld1b.horiz(<vscale x 16 x i1> %pg, ptr %a, i32 0, i32 0)
  store target("aarch64.za.generation") %next_za_1, ptr %current_za_gen

  ; Finally, store a tile slice from ZA.
  %final_gen = load target("aarch64.za.generation"), ptr %current_za_gen
  call void @llvm.aarch64.sme.mark.use.za.state(target("aarch64.za.generation") %final_gen)
  call void @llvm.aarch64.sme.st1b.horiz(<vscale x 16 x i1> %pg, ptr %b, i32 0, i32 0)

  ret void
}

; After opt -O1:
define void @new_za_opt(<vscale x 16 x i1> %pg, ptr %a, ptr %b) local_unnamed_addr #0 {
  %next_za_0 = tail call target("aarch64.za.generation") @llvm.aarch64.sme.mark.update.za.state(target("aarch64.za.generation") zeroinitializer)
  tail call void @llvm.aarch64.sme.ld1b.horiz(<vscale x 16 x i1> %pg, ptr %a, i32 0, i32 0)

  tail call void @llvm.aarch64.sme.clobber.za.state()
  tail call void @private_za_callee() #2

  tail call void @llvm.aarch64.sme.clobber.za.state()
  tail call void @private_za_callee() #2

  %next_za_1 = tail call target("aarch64.za.generation") @llvm.aarch64.sme.mark.update.za.state(target("aarch64.za.generation") %next_za_0)
  tail call void @llvm.aarch64.sme.ld1b.horiz(<vscale x 16 x i1> %pg, ptr %a, i32 0, i32 0)

  tail call void @llvm.aarch64.sme.mark.use.za.state(target("aarch64.za.generation") %next_za_1)
  tail call void @llvm.aarch64.sme.st1b.horiz(<vscale x 16 x i1> %pg, ptr %b, i32 0, i32 0)
  ret void
}

define void @inout_za(<vscale x 16 x i1> %pg, ptr %a, ptr %b) nounwind "aarch64_inout_za" "aarch64_pstate_sm_body" {
  %current_za_gen = alloca target("aarch64.za.generation")

  ; This is the same as the previous example but since this is an "inout_za" function,
  ; we first store the incoming ZA (from current.za.state) to %current_za_gen.
  %inout_za = call target("aarch64.za.generation") @llvm.aarch64.sme.current.za.state()
  store target("aarch64.za.generation") %inout_za, ptr %current_za_gen

  %za_init = load target("aarch64.za.generation"), ptr %current_za_gen
  %next_za_0 = call target("aarch64.za.generation") @llvm.aarch64.sme.mark.update.za.state(target("aarch64.za.generation") %za_init)
  call void @llvm.aarch64.sme.ld1b.horiz(<vscale x 16 x i1> %pg, ptr %a, i32 0, i32 0)
  store target("aarch64.za.generation") %next_za_0, ptr %current_za_gen

  tail call void @llvm.aarch64.sme.clobber.za.state()
  call void @private_za_callee()
  tail call void @llvm.aarch64.sme.clobber.za.state()
  call void @private_za_callee()

  %za_init_1 = load target("aarch64.za.generation"), ptr %current_za_gen
  %next_za_1 = call target("aarch64.za.generation") @llvm.aarch64.sme.mark.update.za.state(target("aarch64.za.generation") %za_init_1)
  call void @llvm.aarch64.sme.ld1b.horiz(<vscale x 16 x i1> %pg, ptr %a, i32 0, i32 0)
  store target("aarch64.za.generation") %next_za_1, ptr %current_za_gen

  %final_gen = load target("aarch64.za.generation"), ptr %current_za_gen
  call void @llvm.aarch64.sme.mark.use.za.state(target("aarch64.za.generation") %final_gen)
  call void @llvm.aarch64.sme.st1b.horiz(<vscale x 16 x i1> %pg, ptr %b, i32 0, i32 0)

  ; Finally, on (all) exits to the function use `.use.za.state` for the current
  ; ZA generation (this is needed to ensure the inout ZA value is preserved).
  %za_out = load target("aarch64.za.generation"), ptr %current_za_gen
  call void @llvm.aarch64.sme.mark.use.za.state(target("aarch64.za.generation") %za_out)

  ret void
}

; Same example, but with a loop:
define void @za_loop(<vscale x 16 x i1> %pg, ptr %a, ptr %b) nounwind "aarch64_new_za" "aarch64_pstate_sm_body" {
entry:
  %current_za_gen = alloca target("aarch64.za.generation")
  br label %for.body

for.body:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %for.body ]

  %za_init = load target("aarch64.za.generation"), ptr %current_za_gen
  %next_za_0 = call target("aarch64.za.generation") @llvm.aarch64.sme.mark.update.za.state(target("aarch64.za.generation") %za_init)
  call void @llvm.aarch64.sme.ld1b.horiz(<vscale x 16 x i1> %pg, ptr %a, i32 0, i32 %iv)
  store target("aarch64.za.generation") %next_za_0, ptr %current_za_gen

  tail call void @llvm.aarch64.sme.clobber.za.state()
  call void @private_za_callee()
  tail call void @llvm.aarch64.sme.clobber.za.state()
  call void @private_za_callee()

  %za_init_1 = load target("aarch64.za.generation"), ptr %current_za_gen
  %next_za_1 = call target("aarch64.za.generation") @llvm.aarch64.sme.mark.update.za.state(target("aarch64.za.generation") %za_init_1)
  call void @llvm.aarch64.sme.ld1b.horiz(<vscale x 16 x i1> %pg, ptr %a, i32 0, i32 %iv)
  store target("aarch64.za.generation") %next_za_1, ptr %current_za_gen

  %final_gen = load target("aarch64.za.generation"), ptr %current_za_gen
  call void @llvm.aarch64.sme.mark.use.za.state(target("aarch64.za.generation") %final_gen)
  call void @llvm.aarch64.sme.st1b.horiz(<vscale x 16 x i1> %pg, ptr %b, i32 0, i32 %iv)

  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond.not = icmp eq i32 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}
