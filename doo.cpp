void private_za_callee_a();
void private_za_callee_b();
void private_za_callee_c();
void shared(int, int, int) __arm_inout("za");
void shared2(int, int, int) __arm_agnostic("sme_za_state");

void test_lazy_save_multiple_paths(int a, int as) __arm_inout("za") {
  if (a)
    private_za_callee_b();
  else
    private_za_callee_c();
  private_za_callee_a();
  for (int i = 0; i < as; i++) {
    for (int j = 0; j < as; j++) {
      for (int k = 0; k < as; k++) {
        private_za_callee_c();
      }
    }
  }
  // shared(1, 2, 3);
}
