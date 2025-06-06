declare void @private_za_callee()

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
