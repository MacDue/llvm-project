declare void @private_za_callee()

define void @new_za(<vscale x 16 x i1> %pg, ptr %a, ptr %b) nounwind "aarch64_new_za" "aarch64_pstate_sm_body" {
  ; Allocate a single "aarch64.za.generation" local for the current ZA generation.
  %current_za_gen = alloca target("aarch64.za.generation")
  store target("aarch64.za.generation") zeroinitializer, ptr %current_za_gen

  ; Note: With this the intrinsic do not need side effects (e.g. IntrInaccessibleMemOrArgMemOnly)
  ; as that as provided by the loads/stores.
  ; Load the current ZA generation before an intrinsic:
  %za_init = load target("aarch64.za.generation"), ptr %current_za_gen
  ; Use the ZA generation in the intrinsic:
  %za_gen0 = call target("aarch64.za.generation") @llvm.aarch64.sme.za.gen.ld1b.horiz(target("aarch64.za.generation") %za_init, <vscale x 16 x i1> %pg, ptr %a, i32 0, i32 0)
  ; Store the resulting ZA generation:
  store target("aarch64.za.generation") %za_gen0, ptr %current_za_gen

  ; For calls to private ZA functions we need to model the ZA clobbers. This
  ; is done with the `clobber.za.state` intrinsic. These are marked with
  ; IntrInaccessibleMemOrArgMemOnly as they cannot be reordered or CSE'd.

  ; First private ZA call.
  tail call void @llvm.aarch64.sme.clobber.za.state()
  call void @private_za_callee()

  ; Second private ZA call.
  tail call void @llvm.aarch64.sme.clobber.za.state()
  call void @private_za_callee()

  ; Another use of a ZA intrinsic.
  %za_in = load target("aarch64.za.generation"), ptr %current_za_gen
  %za_gen1 = call target("aarch64.za.generation") @llvm.aarch64.sme.za.gen.ld1b.horiz(target("aarch64.za.generation") %za_in, <vscale x 16 x i1> %pg, ptr %a, i32 0, i32 0)
  store target("aarch64.za.generation") %za_gen1, ptr %current_za_gen

  ; Finally, store a tile slice from ZA.
  %za_gen2 = load target("aarch64.za.generation"), ptr %current_za_gen
  call void @llvm.aarch64.sme.za.gen.st1b.horiz(target("aarch64.za.generation") %za_gen2, <vscale x 16 x i1> %pg, ptr %b, i32 0, i32 0)

  ret void
}
